"""Tests for controlled BIDS/EEGLAB channel and temporal QC."""

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

from hunt_eeg import bids_eeglab_qc
from hunt_eeg.bids_eeglab import publish_inventory
from hunt_eeg.bids_eeglab_qc import _channel_review_template, publish_qc
from hunt_eeg.bids_eeglab_review_pack import (
    _minmax_envelope,
    _save_channel_review_figures,
    build_review_pack,
)
from hunt_eeg.provenance import canonical_sha256, verify_provenance


class BIDSEeglabQCTests(unittest.TestCase):
    def _prepared(self, root: Path, *, robust_z_threshold: float = 5.0):
        inputs, profile = _fixture(root)
        payload = json.loads(profile.read_text(encoding="utf-8"))
        payload["expected"]["power_line_frequency_hz"] = 40.0
        payload["qc"] = {
            "screen_filter_hz": [1.0, 40.0],
            "full_recording_window_seconds": 2.0,
            "robust_z_threshold": robust_z_threshold,
            "flat_fraction_threshold": 0.1,
            "line_noise_ratio_db_threshold": 20.0,
        }
        profile.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        eeg_json = inputs.dataset_root / "task-contrast_eeg.json"
        eeg_payload = json.loads(eeg_json.read_text(encoding="utf-8"))
        eeg_payload["PowerLineFrequency"] = 40.0
        eeg_json.write_text(
            json.dumps(eeg_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inventory = publish_inventory(inputs, profile, root / "inventory")
        return inputs, profile, inventory

    def test_qc_is_deterministic_and_requires_manual_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(root)
            first = publish_qc(inputs, profile, inventory, root / "first")
            second = publish_qc(inputs, profile, inventory, root / "second")
            first_provenance = verify_provenance(first)
            second_provenance = verify_provenance(second)
            self.assertEqual(first_provenance, second_provenance)
            self.assertEqual(
                (first / "qc_summary.json").read_bytes(),
                (second / "qc_summary.json").read_bytes(),
            )
            summary = json.loads(
                (first / "qc_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["scalp_channels_screened"], 2)
            self.assertEqual(summary["automatic_bad_channel_decisions"], 0)
            self.assertFalse(summary["publication_allowed"])
            self.assertTrue((first / "channel_review_template.tsv").is_file())
            self.assertTrue((first / "figures" / "line_noise_review.png").is_file())

    def test_inventory_mismatch_fails_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(root)
            payload = json.loads(profile.read_text(encoding="utf-8"))
            payload["qc"]["robust_z_threshold"] = 4.0
            profile.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "failed"
            with self.assertRaisesRegex(ValueError, "profile"):
                publish_qc(inputs, profile, inventory, output)
            self.assertFalse(output.exists())

    def test_review_pack_has_one_deterministic_panel_per_segment_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(
                root, robust_z_threshold=0.5
            )
            qc = publish_qc(inputs, profile, inventory, root / "qc")
            first = build_review_pack(inputs, profile, qc, root / "pack-a")
            second = build_review_pack(inputs, profile, qc, root / "pack-b")
            first_provenance = verify_provenance(first)
            second_provenance = verify_provenance(second)
            self.assertEqual(first_provenance, second_provenance)
            template = pd.read_csv(qc / "segment_review_template.tsv", sep="\t")
            index = pd.read_csv(first / "segment_figure_index.csv")
            channel_template = pd.read_csv(
                qc / "channel_review_template.tsv", sep="\t"
            )
            channel_index = pd.read_csv(first / "channel_figure_index.csv")
            self.assertGreater(len(template), 0)
            self.assertEqual(
                index["candidate_id"].tolist(), template["candidate_id"].tolist()
            )
            self.assertEqual(len(index), len(set(index["figure_path"])))
            self.assertIn("plotted_onset_s", index.columns)
            self.assertIn("plotted_duration_s", index.columns)
            for relative in index["figure_path"]:
                self.assertTrue((first / relative).is_file())
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )
            self.assertEqual(
                channel_index["channel"].tolist(),
                channel_template["channel"].tolist(),
            )
            for relative in channel_index["figure_path"]:
                self.assertTrue((first / relative).is_file())
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )
            summary = json.loads(
                (first / "review_pack_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["automatic_decisions"], 0)
            self.assertEqual(summary["decision_status"], "pending")
            self.assertFalse(summary["publication_allowed"])
            review_html = (first / "review_index.html").read_text(encoding="utf-8")
            self.assertIn("Edit only those copies", review_html)
            self.assertIn("Red dashed lines mark the prompt", review_html)

    def test_full_trace_envelope_preserves_short_extrema(self):
        values = pd.Series([0.0] * 1001).to_numpy()
        values[503] = 42.0
        time, envelope = _minmax_envelope(values, sfreq=100.0, maximum_bins=100)

        self.assertIn(42.0, envelope)
        self.assertIn(5.03, time)

    def test_channel_figure_names_do_not_collide_after_sanitizing(self):
        raw = mne.io.RawArray(
            np.zeros((2, 500)),
            mne.create_info(["A/B", "A_B"], 100.0, "eeg"),
            verbose="ERROR",
        )
        prompts = pd.DataFrame(
            {
                "channel": ["A/B", "A_B"],
                "candidate_reason": ["line_noise_concentration"] * 2,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            index = _save_channel_review_figures(
                Path(directory), raw, raw.copy(), prompts, 40.0
            )
            self.assertEqual(len(set(index["figure_path"])), 2)
            self.assertTrue(all((Path(directory) / Path(path).name).is_file() for path in index["figure_path"]))

    def test_review_pack_rejects_stale_full_profile_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(
                root, robust_z_threshold=0.5
            )
            qc = publish_qc(inputs, profile, inventory, root / "qc")
            payload = json.loads(profile.read_text(encoding="utf-8"))
            payload["processing"]["segment_guard"] = "changed_after_qc"
            profile.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale"):
                build_review_pack(inputs, profile, qc, root / "failed")
            self.assertFalse((root / "failed").exists())

    def test_dangling_output_symlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(root)
            output = root / "dangling"
            output.symlink_to(root / "missing", target_is_directory=True)
            with self.assertRaises(FileExistsError):
                publish_qc(inputs, profile, inventory, output)
            self.assertTrue(output.is_symlink())

    def test_line_noise_only_candidate_requires_channel_review(self):
        temporal = pd.DataFrame(
            [{"channel": "E1", "channel_requires_review": False}]
        )
        line = pd.DataFrame(
            [{"channel": "E1", "review_high_line_noise": True}]
        )

        template = _channel_review_template(temporal, line)

        self.assertEqual(template["channel"].tolist(), ["E1"])
        self.assertEqual(template["decision"].tolist(), ["pending"])
        self.assertIn("line_noise_concentration", template.iloc[0]["candidate_reason"])

    def test_stale_inventory_source_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(root)
            path = inventory / "provenance.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["controls"]["source_manifest"]["sha256"] = "0" * 64
            core = {
                key: value
                for key, value in payload.items()
                if key not in {"core_sha256", "outputs"}
            }
            payload["core_sha256"] = canonical_sha256(core)
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source context is stale"):
                publish_qc(inputs, profile, inventory, root / "failed")

    def test_mutation_after_qc_run_is_rejected_before_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, inventory = self._prepared(root)
            original = bids_eeglab_qc._run_qc

            def mutate_after_run(*args, **kwargs):
                summary = original(*args, **kwargs)
                profile.write_text("{}\n", encoding="utf-8")
                return summary

            output = root / "failed"
            with mock.patch.object(
                bids_eeglab_qc,
                "_run_qc",
                side_effect=mutate_after_run,
            ), self.assertRaisesRegex(ValueError, "profile changed before publication"):
                publish_qc(inputs, profile, inventory, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
