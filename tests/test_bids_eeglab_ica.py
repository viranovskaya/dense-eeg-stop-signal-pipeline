"""Tests for provenance-bound BIDS/EEGLAB ICA review packages."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_bids_eeglab import _fixture
from test_bids_eeglab_review import _completed_tables

from hunt_eeg import (
    bids_eeglab_ica,
    bids_eeglab_ica_decisions,
    bids_eeglab_preprocess,
)
from hunt_eeg.bids_eeglab import publish_inventory
from hunt_eeg.bids_eeglab_ica import publish_ica_review
from hunt_eeg.bids_eeglab_ica_decisions import finalize_ica_decisions
from hunt_eeg.bids_eeglab_preprocess import (
    _epoch_condition,
    publish_reviewed_preprocessing,
)
from hunt_eeg.bids_eeglab_qc import publish_qc
from hunt_eeg.bids_eeglab_review import finalize_review_decisions
from hunt_eeg.provenance import verify_provenance


def _ica_fixture(root: Path):
    inputs, profile = _fixture(root)
    samples = np.arange(3000)
    raw = mne.io.RawArray(
        np.vstack(
            [
                np.sin(samples / 11.0) * 2e-6
                + np.sin(samples / 37.0) * 0.4e-6,
                np.cos(samples / 17.0) * 1.5e-6
                + np.sin(samples / 29.0) * 0.3e-6,
                np.sin(samples / 19.0) * 1.2e-6
                + np.cos(samples / 31.0) * 0.2e-6,
                np.zeros_like(samples, dtype=float),
            ]
        ),
        mne.create_info(["E1", "E2", "E3", "Cz"], 100.0, "eeg"),
        verbose="ERROR",
    )
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    positions = montage.get_positions()["ch_pos"]
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={name: positions[name] for name in raw.ch_names},
            coord_frame="head",
        )
    )
    raw.set_annotations(
        mne.Annotations(
            [1.0, 5.0, 10.0, 15.0],
            [0.0, 0.0, 0.0, 0.0],
            ["left_target", "right_target", "left_target", "right_target"],
        )
    )
    mne.export.export_raw(
        inputs.raw_set, raw, fmt="eeglab", overwrite=True, verbose="ERROR"
    )
    pd.DataFrame(
        {
            "onset": [1.0, 5.0, 10.0, 15.0],
            "duration": ["n/a"] * 4,
            "sample": [100, 500, 1000, 1500],
            "value": [
                "left_target",
                "right_target",
                "left_target",
                "right_target",
            ],
        }
    ).to_csv(
        inputs.raw_set.with_name(
            inputs.raw_set.name.replace("_eeg.set", "_events.tsv")
        ),
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        {
            "name": ["E1", "E2", "E3", "Cz"],
            "type": ["EEG"] * 4,
            "units": ["uV"] * 4,
        }
    ).to_csv(
        inputs.raw_set.with_name(
            inputs.raw_set.name.replace("_eeg.set", "_channels.tsv")
        ),
        sep="\t",
        index=False,
    )
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["expected"]["power_line_frequency_hz"] = 40.0
    payload["expected"]["eeg_channel_count"] = 4
    payload["expected"]["minimum_positioned_channels"] = 4
    payload["qc"] = {
        "screen_filter_hz": [1.0, 40.0],
        "full_recording_window_seconds": 5.0,
        "robust_z_threshold": 5.0,
        "flat_fraction_threshold": 0.1,
        "line_noise_ratio_db_threshold": 20.0,
    }
    payload["geometry_validation"] = {
        "standard_montage": "GSN-HydroCel-129",
        "allow_montage_subset": True,
        "maximum_similarity_rmse_mm": 0.1,
        "minimum_pairwise_distance_correlation": 0.9999,
    }
    profile.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    eeg_json = inputs.dataset_root / "task-contrast_eeg.json"
    metadata = json.loads(eeg_json.read_text(encoding="utf-8"))
    metadata["PowerLineFrequency"] = 40.0
    metadata["EEGChannelCount"] = 4
    eeg_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_eeg_json = inputs.raw_set.with_suffix(".json")
    run_metadata = json.loads(run_eeg_json.read_text(encoding="utf-8"))
    run_metadata["EEGChannelCount"] = 4
    run_eeg_json.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory = publish_inventory(inputs, profile, root / "inventory")
    qc = publish_qc(inputs, profile, inventory, root / "qc")
    channel_path, segment_path = _completed_tables(qc, root)
    channel = pd.read_csv(
        channel_path, sep="\t", dtype=str, keep_default_na=False
    )
    if "E2" in set(channel["channel"]):
        channel.loc[channel["channel"] == "E2", "decision"] = "interpolate"
        channel.loc[channel["channel"] == "E2", "evidence"] = (
            "full_trace;neighbour_trace"
        )
        channel.loc[channel["channel"] == "E2", "notes"] = (
            "Synthetic bad-channel decision used to exercise interpolation."
        )
    else:
        channel.loc[len(channel)] = {
            "channel": "E2",
            "candidate_reason": "manual_addition",
            "decision": "interpolate",
            "reviewer": "Test reviewer",
            "reviewed_at": "2026-08-15",
            "evidence": "full_trace;neighbour_trace",
            "notes": "Synthetic bad-channel decision used to exercise interpolation.",
        }
    channel.to_csv(channel_path, sep="\t", index=False)
    segment = pd.read_csv(
        segment_path, sep="\t", dtype=str, keep_default_na=False
    )
    segment.loc[len(segment)] = {
        "candidate_id": "manual-0001",
        "channel": "all",
        "prompt_onset_s": "",
        "prompt_duration_s": "",
        "candidate_reason": "manual_addition",
        "decision": "exclude",
        "refined_onset_s": "20.0",
        "refined_duration_s": "0.2",
        "scope": "ica",
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-08-15",
        "evidence": "full_trace;all_channels",
        "notes": "Synthetic transient exclusion used to exercise filter guards.",
    }
    segment.loc[len(segment)] = {
        "candidate_id": "manual-0002",
        "channel": "all",
        "prompt_onset_s": "",
        "prompt_duration_s": "",
        "candidate_reason": "manual_addition",
        "decision": "exclude",
        "refined_onset_s": "9.9",
        "refined_duration_s": "0.2",
        "scope": "epochs",
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-08-15",
        "evidence": "full_trace;all_channels",
        "notes": "Synthetic epoch exclusion used to exercise drop accounting.",
    }
    segment.to_csv(segment_path, sep="\t", index=False)
    bundle = finalize_review_decisions(
        inputs,
        qc,
        channel_path,
        segment_path,
        "public",
        root / "bundle",
    )
    return inputs, profile, qc, bundle


class BIDSEeglabICATests(unittest.TestCase):
    def test_epoch_minimum_is_enforced_after_annotation_rejection(self):
        raw = mne.io.RawArray(
            np.zeros((1, 300)),
            mne.create_info(["E1"], 100.0, "eeg"),
            verbose="ERROR",
        )
        raw.set_annotations(
            mne.Annotations(
                [0.8, 1.0],
                [0.5, 0.0],
                ["BAD_review_epochs", "left_target"],
            )
        )
        definition = {
            "window_seconds": [-0.2, 0.8],
            "baseline_seconds": [-0.2, 0.0],
            "minimum_count": 1,
        }
        source_rows = pd.DataFrame(
            [
                {
                    "source_bids_row_index": 0,
                    "onset": "1.0",
                    "duration": "n/a",
                    "sample": "100",
                    "value": "left_target",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "Too few retained 'left_target'"):
            _epoch_condition(raw, "left_target", definition, 1, source_rows)

    def test_synthetic_ica_review_is_deterministic_and_exact_set_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)

            first = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-first"
            )
            second = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-second"
            )

            first_provenance = verify_provenance(first)
            second_provenance = verify_provenance(second)
            self.assertEqual(first_provenance, second_provenance)
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            summary = json.loads((first / "ica_review_summary.json").read_text())
            self.assertEqual(summary["status"], "manual_component_review_required")
            self.assertTrue(summary["fit_stopping_rule_met"])
            self.assertFalse(summary["automatic_component_exclusion"])
            self.assertFalse(summary["publication_allowed"])
            self.assertEqual(
                len(summary["processing"]["guarded_filter_intervals"]), 1
            )
            self.assertFalse(
                summary["auxiliary_correlation_cues"]["eog_channel_available"]
            )
            self.assertFalse(
                summary["auxiliary_correlation_cues"]["ecg_channel_available"]
            )
            self.assertEqual(
                len(list((first / "figures").glob("component-*_properties.png"))),
                summary["components"],
            )
            self.assertGreater(
                summary["variance_denominator"]["bad_samples_omitted"], 0
            )

    def test_output_inside_source_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)

            with self.assertRaisesRegex(ValueError, "outside source"):
                publish_ica_review(
                    inputs,
                    profile,
                    qc,
                    bundle,
                    "public",
                    inputs.dataset_root / "derived-ica",
                )

    def test_ica_config_is_captured_once_and_bound_to_the_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)
            original = bids_eeglab_ica._captured_json_sha
            with mock.patch.object(
                bids_eeglab_ica, "_captured_json_sha", wraps=original
            ) as captured:
                output = publish_ica_review(
                    inputs, profile, qc, bundle, "public", root / "ica"
                )

            config_path = Path(bids_eeglab_ica.DEFAULT_ICA_CONFIG).absolute()
            config_reads = [
                call
                for call in captured.call_args_list
                if Path(call.args[0]).absolute() == config_path
            ]
            self.assertEqual(len(config_reads), 1)
            provenance = verify_provenance(output)
            self.assertEqual(
                provenance["controls"]["ica_config_sha256"],
                bids_eeglab_ica.sha256_file(config_path),
            )

    def test_complete_ica_decisions_are_finalized_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)
            ica_review = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-review"
            )
            decisions = pd.read_csv(
                ica_review / "ica_decision_template.csv",
                dtype=str,
                keep_default_na=False,
            )
            decisions["decision"] = "keep"
            decisions["reason"] = "No reproducible artifact pattern was identified."
            decisions["reviewer"] = "Test reviewer"
            decisions["reviewed_at"] = "2026-08-15"
            decisions["evidence"] = "topography;time_course;spectrum"
            decision_path = root / "completed_ica_decisions.csv"
            decisions.to_csv(decision_path, index=False)

            first = finalize_ica_decisions(
                inputs,
                qc,
                bundle,
                ica_review,
                decision_path,
                "public",
                root / "ica-decisions-first",
            )
            second = finalize_ica_decisions(
                inputs,
                qc,
                bundle,
                ica_review,
                decision_path,
                "public",
                root / "ica-decisions-second",
            )

            self.assertEqual(verify_provenance(first), verify_provenance(second))
            self.assertEqual(
                (first / "ica_decision_summary.json").read_bytes(),
                (second / "ica_decision_summary.json").read_bytes(),
            )
            summary = json.loads(
                (first / "ica_decision_summary.json").read_text()
            )
            self.assertEqual(summary["components"], summary["kept_components"])
            self.assertEqual(summary["excluded_components"], [])
            self.assertFalse(summary["automatic_component_exclusion"])

    def test_ica_decision_publication_rechecks_upstream_after_temp_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)
            ica_review = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-review"
            )
            decisions = pd.read_csv(
                ica_review / "ica_decision_template.csv",
                dtype=str,
                keep_default_na=False,
            )
            decisions["decision"] = "keep"
            decisions["reason"] = "No reproducible artifact pattern was identified."
            decisions["reviewer"] = "Test reviewer"
            decisions["reviewed_at"] = "2026-08-15"
            decisions["evidence"] = "topography;time_course;spectrum"
            decision_path = root / "mutable_ica_decisions.csv"
            decisions.to_csv(decision_path, index=False)
            output = root / "mutated-decisions"
            original = bids_eeglab_ica_decisions.verify_provenance
            mutated = False

            def mutate_after_temp_verify(path):
                nonlocal mutated
                result = original(path)
                if Path(path).name.startswith(".mutated-decisions-") and not mutated:
                    decision_path.write_bytes(decision_path.read_bytes() + b"\n")
                    mutated = True
                return result

            with mock.patch.object(
                bids_eeglab_ica_decisions,
                "verify_provenance",
                side_effect=mutate_after_temp_verify,
            ), self.assertRaisesRegex(ValueError, "decision input changed"):
                finalize_ica_decisions(
                    inputs,
                    qc,
                    bundle,
                    ica_review,
                    decision_path,
                    "public",
                    output,
                )

            self.assertTrue(mutated)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".mutated-decisions-*")), [])

    def test_reviewed_preprocessing_exports_deterministic_condition_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)
            ica_review = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-review"
            )
            decisions = pd.read_csv(
                ica_review / "ica_decision_template.csv",
                dtype=str,
                keep_default_na=False,
            )
            decisions["decision"] = "keep"
            decisions["reason"] = "No reproducible artifact pattern was identified."
            decisions["reviewer"] = "Test reviewer"
            decisions["reviewed_at"] = "2026-08-15"
            decisions["evidence"] = "topography;time_course;spectrum"
            decisions.loc[0, "decision"] = "exclude"
            decisions.loc[0, "reason"] = (
                "Synthetic exclusion used only to exercise exact ICA application."
            )
            decision_path = root / "completed_ica_decisions.csv"
            decisions.to_csv(decision_path, index=False)
            decision_bundle = finalize_ica_decisions(
                inputs,
                qc,
                bundle,
                ica_review,
                decision_path,
                "public",
                root / "ica-decisions",
            )

            first = publish_reviewed_preprocessing(
                inputs,
                profile,
                qc,
                bundle,
                ica_review,
                decision_bundle,
                "public",
                root / "processed-first",
            )
            second = publish_reviewed_preprocessing(
                inputs,
                profile,
                qc,
                bundle,
                ica_review,
                decision_bundle,
                "public",
                root / "processed-second",
            )

            self.assertEqual(verify_provenance(first), verify_provenance(second))
            first_files = {
                path.relative_to(first).as_posix(): path.read_bytes()
                for path in first.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second).as_posix(): path.read_bytes()
                for path in second.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            summary = json.loads((first / "preprocessing_summary.json").read_text())
            self.assertEqual(summary["epochs"]["left_target"]["input_events"], 2)
            self.assertEqual(summary["epochs"]["left_target"]["retained_epochs"], 1)
            self.assertEqual(summary["epochs"]["left_target"]["dropped_epochs"], 1)
            self.assertEqual(summary["epochs"]["right_target"]["input_events"], 2)
            self.assertEqual(summary["epochs"]["right_target"]["retained_epochs"], 2)
            self.assertFalse(summary["ica"]["automatic_component_exclusion"])
            self.assertEqual(summary["ica"]["excluded_components"], [0])
            self.assertEqual(summary["interpolated_channels"], ["E2"])
            lineage = pd.read_csv(first / "left_target_epoch_lineage.csv")
            self.assertEqual(lineage["retained"].tolist(), [True, False])
            self.assertEqual(lineage["source_bids_row_index"].tolist(), [0, 2])
            epochs = mne.read_epochs(first / "left_target-epo.fif", verbose="ERROR")
            baseline = (epochs.times >= -0.2) & (epochs.times <= 0.0)
            np.testing.assert_allclose(
                epochs.get_data()[:, :, baseline].mean(axis=2),
                0.0,
                rtol=0,
                atol=1e-12,
            )
            original_capture = bids_eeglab_preprocess._capture
            events_path = inputs.raw_set.with_name(
                inputs.raw_set.name.replace("_eeg.set", "_events.tsv")
            )
            transient_output = root / "transient-events-preprocessing"
            mutation_attempted = False

            def transient_event_mutation(path, expected_sha256):
                nonlocal mutation_attempted
                path = Path(path)
                if path.name.endswith("_events.tsv") and not mutation_attempted:
                    self.assertEqual(path.resolve(), events_path.resolve())
                    original_bytes = path.read_bytes()
                    changed = original_bytes.replace(b"1.0\t", b"1.1\t", 1)
                    self.assertNotEqual(changed, original_bytes)
                    path.write_bytes(changed)
                    mutation_attempted = True
                    try:
                        return original_capture(path, expected_sha256)
                    finally:
                        path.write_bytes(original_bytes)
                return original_capture(path, expected_sha256)

            with mock.patch.object(
                bids_eeglab_preprocess,
                "_capture",
                side_effect=transient_event_mutation,
            ), self.assertRaisesRegex(ValueError, "input changed"):
                publish_reviewed_preprocessing(
                    inputs,
                    profile,
                    qc,
                    bundle,
                    ica_review,
                    decision_bundle,
                    "public",
                    transient_output,
                )

            self.assertTrue(mutation_attempted)
            self.assertFalse(transient_output.exists())
            self.assertEqual(list(root.glob(".transient-events-preprocessing-*")), [])

    def test_preprocessing_rechecks_upstream_after_temp_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _ica_fixture(root)
            ica_review = publish_ica_review(
                inputs, profile, qc, bundle, "public", root / "ica-review"
            )
            decisions = pd.read_csv(
                ica_review / "ica_decision_template.csv",
                dtype=str,
                keep_default_na=False,
            )
            decisions["decision"] = "keep"
            decisions["reason"] = "No reproducible artifact pattern was identified."
            decisions["reviewer"] = "Test reviewer"
            decisions["reviewed_at"] = "2026-08-15"
            decisions["evidence"] = "topography;time_course;spectrum"
            decision_path = root / "completed_ica_decisions.csv"
            decisions.to_csv(decision_path, index=False)
            decision_bundle = finalize_ica_decisions(
                inputs,
                qc,
                bundle,
                ica_review,
                decision_path,
                "public",
                root / "ica-decisions",
            )
            output = root / "mutated-preprocessing"
            original = bids_eeglab_preprocess.verify_provenance
            temp_verifications = 0
            mutated = False

            def mutate_after_publication_verify(path):
                nonlocal temp_verifications, mutated
                result = original(path)
                if Path(path).name.startswith(".mutated-preprocessing-"):
                    temp_verifications += 1
                    if temp_verifications == 2:
                        summary_path = decision_bundle / "ica_decision_summary.json"
                        summary_path.write_bytes(summary_path.read_bytes() + b"\n")
                        mutated = True
                return result

            with mock.patch.object(
                bids_eeglab_preprocess,
                "verify_provenance",
                side_effect=mutate_after_publication_verify,
            ), self.assertRaises(ValueError):
                publish_reviewed_preprocessing(
                    inputs,
                    profile,
                    qc,
                    bundle,
                    ica_review,
                    decision_bundle,
                    "public",
                    output,
                )

            self.assertTrue(mutated)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".mutated-preprocessing-*")), [])


if __name__ == "__main__":
    unittest.main()
