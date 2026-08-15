"""Tests for the fail-closed BIDS EEGLAB inventory."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg import bids_eeglab
from hunt_eeg.bids_eeglab import (
    BIDSEeglabInputs,
    _validate_geometry,
    publish_inventory,
)
from hunt_eeg.provenance import verify_provenance


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture(
    root: Path, *, flat_reference: bool = True, external_fdt: bool = False
) -> tuple[BIDSEeglabInputs, Path]:
    dataset = root / "dataset"
    eeg = dataset / "sub-public" / "eeg"
    eeg.mkdir(parents=True)
    raw_path = eeg / "sub-public_task-contrast_run-1_eeg.set"
    samples = np.arange(500)
    cz = np.zeros(500) if flat_reference else np.sin(samples / 7) * 1e-6
    raw = mne.io.RawArray(
        np.vstack(
            [
                np.sin(samples / 10) * 1e-6,
                np.cos(samples / 10) * 1e-6,
                cz,
            ]
        ),
        mne.create_info(["E1", "E2", "Cz"], 100.0, ["eeg"] * 3),
        verbose="ERROR",
    )
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={
                "E1": (-0.03, 0.05, 0.07),
                "E2": (0.03, 0.05, 0.07),
                "Cz": (0.0, 0.0, 0.1),
            },
            coord_frame="head",
        )
    )
    raw.set_annotations(
        mne.Annotations([1.0, 2.0], [0.0, 0.0], ["left_target", "right_target"])
    )
    mne.export.export_raw(
        raw_path, raw, fmt="eeglab", overwrite=True, verbose="ERROR"
    )
    if external_fdt:
        payload = loadmat(raw_path, simplify_cells=True)
        data = np.asarray(payload.pop("data"), dtype="<f4")
        fdt_path = raw_path.with_suffix(".fdt")
        data.flatten(order="F").tofile(fdt_path)
        payload["data"] = fdt_path.name
        savemat(
            raw_path,
            {key: value for key, value in payload.items() if not key.startswith("__")},
            appendmat=False,
        )

    events_tsv = eeg / "sub-public_task-contrast_run-1_events.tsv"
    pd.DataFrame(
        {
            "onset": [1.0, 2.0],
            "duration": ["n/a", "n/a"],
            "sample": [100, 200],
            "value": ["left_target", "right_target"],
        }
    ).to_csv(events_tsv, sep="\t", index=False)
    channels_tsv = eeg / "sub-public_task-contrast_run-1_channels.tsv"
    pd.DataFrame(
        {
            "name": ["E1", "E2", "Cz"],
            "type": ["EEG"] * 3,
            "units": ["uV"] * 3,
        }
    ).to_csv(channels_tsv, sep="\t", index=False)
    root_eeg_json = dataset / "task-contrast_eeg.json"
    _write_json(
        root_eeg_json,
        {
            "TaskName": "contrast",
            "SamplingFrequency": 100,
            "PowerLineFrequency": 60,
            "EEGReference": "Cz",
        },
    )
    run_eeg_json = eeg / "sub-public_task-contrast_run-1_eeg.json"
    _write_json(run_eeg_json, {"EEGChannelCount": 3})
    events_json = dataset / "task-contrast_events.json"
    _write_json(
        events_json,
        {
            "value": {
                "Levels": {
                    "left_target": "Left target",
                    "right_target": "Right target",
                }
            }
        },
    )
    description = dataset / "dataset_description.json"
    _write_json(
        description,
        {
            "Name": "Synthetic public BIDS EEG",
            "DatasetDOI": "doi:10.18112/openneuro.ds999999.v1.0.0",
        },
    )
    profile = root / "profile.json"
    _write_json(
        profile,
        {
            "schema_version": "1",
            "dataset": {
                "accession": "ds999999",
                "snapshot": "1.0.0",
                "name": "Synthetic public BIDS EEG",
            },
            "task": "contrast",
            "recording": {"participant_id": "public", "run": "1"},
            "expected": {
                "sampling_frequency_hz": 100,
                "eeg_channel_count": 3,
                "reference": "Cz",
                "power_line_frequency_hz": 60,
                "channel_unit": "uV",
                "minimum_positioned_channels": 3,
            },
            "reference_channel": {
                "name": "Cz",
                "must_be_flat": True,
                "exclude_from_qc": True,
                "exclude_from_reference_average": False,
                "post_reference_policy": "include_flat_online_reference_in_average_transform",
            },
            "processing": {
                "filter_hz": [1.0, 40.0],
                "filter_method": "firwin_zero_phase_hamming",
                "segment_guard": "half_filter_support",
                "average_reference": "all_good_eeg_including_flat_online_reference",
            },
            "events": {
                "alignment_tolerance_samples": 0,
                "epoch_definitions": {
                    "left_target": {
                        "purpose": "qc_candidate",
                        "window_seconds": [-0.2, 0.8],
                        "baseline_seconds": [-0.2, 0],
                        "minimum_count": 1,
                    },
                    "right_target": {
                        "purpose": "qc_candidate",
                        "window_seconds": [-0.2, 0.8],
                        "baseline_seconds": [-0.2, 0],
                        "minimum_count": 1,
                    },
                },
                "context_values": {},
                "excluded_values": {},
            },
        },
    )
    inputs = BIDSEeglabInputs(
        dataset_root=dataset,
        raw_set=raw_path,
    )
    return inputs, profile


class BIDSEeglabTests(unittest.TestCase):
    def test_standard_montage_geometry_is_validated(self):
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        raw = mne.io.RawArray(
            np.zeros((129, 10)),
            mne.create_info(montage.ch_names, 100, "eeg"),
            verbose="ERROR",
        )
        raw.set_montage(montage)
        controls = {
            "standard_montage": "GSN-HydroCel-129",
            "maximum_similarity_rmse_mm": 0.1,
            "minimum_pairwise_distance_correlation": 0.9999,
        }

        result = _validate_geometry(raw, controls)

        self.assertTrue(result["standard_template_mapping_validated"])
        self.assertLess(result["similarity_rmse_mm"], 0.1)

        angle = np.deg2rad(31)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        translation = np.asarray([0.012, -0.021, 0.008])
        for channel in raw.info["chs"]:
            channel["loc"][:3] = 0.92 * channel["loc"][:3] @ rotation + translation
        transformed = _validate_geometry(raw, controls)
        self.assertLess(transformed["similarity_rmse_mm"], 0.1)

        raw.info["chs"][0]["loc"][:3] += 0.02
        with self.assertRaisesRegex(ValueError, "geometry"):
            _validate_geometry(raw, controls)

    def test_ordered_standard_montage_subset_is_validated_explicitly(self):
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        names = ["E1", "E2", "Cz"]
        positions = montage.get_positions()["ch_pos"]
        raw = mne.io.RawArray(
            np.zeros((3, 10)),
            mne.create_info(names, 100, "eeg"),
            verbose="ERROR",
        )
        raw.set_montage(
            mne.channels.make_dig_montage(
                ch_pos={name: positions[name] for name in names},
                coord_frame="head",
            )
        )
        controls = {
            "standard_montage": "GSN-HydroCel-129",
            "allow_montage_subset": True,
            "maximum_similarity_rmse_mm": 0.1,
            "minimum_pairwise_distance_correlation": 0.9999,
        }

        result = _validate_geometry(raw, controls)

        self.assertTrue(result["standard_template_mapping_validated"])
        self.assertTrue(result["montage_subset"])

    def test_inventory_is_deterministic_and_exact_set_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            first = publish_inventory(inputs, profile, root / "first")
            second = publish_inventory(inputs, profile, root / "second")
            first_provenance = verify_provenance(first)
            second_provenance = verify_provenance(second)
            self.assertEqual(
                (first / "audit.json").read_bytes(),
                (second / "audit.json").read_bytes(),
            )
            self.assertEqual(first_provenance, second_provenance)
            audit = json.loads((first / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["events"]["rows"], 2)
            self.assertTrue(audit["channels"]["reference_channel_flat"])
            self.assertFalse(audit["publication_allowed"])

    def test_event_mismatch_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            events_tsv = next(inputs.raw_set.parent.glob("*_events.tsv"))
            table = pd.read_csv(events_tsv, sep="\t")
            table.loc[0, "value"] = "wrong_target"
            table.to_csv(events_tsv, sep="\t", index=False)
            output = root / "failed"
            with self.assertRaisesRegex(ValueError, "event values"):
                publish_inventory(inputs, profile, output)
            self.assertFalse(output.exists())

    def test_nonflat_declared_reference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root, flat_reference=False)
            with self.assertRaisesRegex(ValueError, "reference channel is not flat"):
                publish_inventory(inputs, profile, root / "failed")

    def test_external_fdt_is_bound_and_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root, external_fdt=True)
            fdt_path = inputs.raw_set.with_suffix(".fdt")
            original_summary = bids_eeglab._amplitude_summary

            def mutate_after_data_access(raw, reference):
                summary = original_summary(raw, reference)
                with fdt_path.open("ab") as stream:
                    stream.write(b"drift")
                return summary

            with mock.patch.object(
                bids_eeglab, "_amplitude_summary", side_effect=mutate_after_data_access
            ), self.assertRaisesRegex(ValueError, "inputs changed"):
                publish_inventory(inputs, profile, root / "failed")
            self.assertFalse((root / "failed").exists())

    def test_ordinary_in_dataset_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            target = inputs.raw_set.with_name("ordinary-target.set")
            inputs.raw_set.replace(target)
            inputs.raw_set.symlink_to(target.name)
            with self.assertRaisesRegex(ValueError, "not a bounded git-annex"):
                publish_inventory(inputs, profile, root / "failed")

    def test_nonfinite_event_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            events_tsv = next(inputs.raw_set.parent.glob("*_events.tsv"))
            table = pd.read_csv(events_tsv, sep="\t")
            table["onset"] = table["onset"].astype(object)
            table.loc[0, "onset"] = "nan"
            table.to_csv(events_tsv, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                publish_inventory(inputs, profile, root / "failed")

    def test_ambiguous_inheritance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            duplicate = inputs.dataset_root / "sub-public_eeg.json"
            _write_json(duplicate, {"TaskName": "contrast"})
            with self.assertRaisesRegex(ValueError, "Ambiguous BIDS inheritance"):
                publish_inventory(inputs, profile, root / "failed")

    def test_dangling_output_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile = _fixture(root)
            output = root / "dangling-output"
            output.symlink_to(root / "missing-target", target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_inventory(inputs, profile, output)
            self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
