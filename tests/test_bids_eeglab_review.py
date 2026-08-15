"""Tests for immutable BIDS/EEGLAB human-decision bundles."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_bids_eeglab import _fixture

from hunt_eeg import bids_eeglab_review
from hunt_eeg.bids_eeglab import publish_inventory
from hunt_eeg.bids_eeglab_qc import publish_qc
from hunt_eeg.bids_eeglab_review import (
    CHANNEL_COLUMNS,
    SEGMENT_COLUMNS,
    _effective_intervals,
    _merged_intervals,
    _validate_channel_decisions,
    _validate_segment_decisions,
    finalize_review_decisions,
)
from hunt_eeg.provenance import verify_provenance


def _prepared(root: Path):
    inputs, profile = _fixture(root)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["expected"]["power_line_frequency_hz"] = 40.0
    payload["qc"] = {
        "screen_filter_hz": [1.0, 40.0],
        "full_recording_window_seconds": 2.0,
        "robust_z_threshold": 5.0,
        "flat_fraction_threshold": 0.1,
        "line_noise_ratio_db_threshold": 20.0,
    }
    profile.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    eeg_json = inputs.dataset_root / "task-contrast_eeg.json"
    metadata = json.loads(eeg_json.read_text(encoding="utf-8"))
    metadata["PowerLineFrequency"] = 40.0
    eeg_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory = publish_inventory(inputs, profile, root / "inventory")
    qc = publish_qc(inputs, profile, inventory, root / "qc")
    return inputs, qc


def _completed_tables(qc: Path, root: Path) -> tuple[Path, Path]:
    channel = pd.read_csv(
        qc / "channel_review_template.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    channel["decision"] = "keep"
    channel["reviewer"] = "Test reviewer"
    channel["reviewed_at"] = "2026-08-15"
    channel["evidence"] = "raw_trace;raw_psd;neighbour_trace"
    channel["notes"] = "Raw trace and spectrum reviewed."
    channel_path = root / "channel.tsv"
    channel.to_csv(channel_path, sep="\t", index=False)

    segment = pd.read_csv(
        qc / "segment_review_template.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    segment["decision"] = "keep"
    segment["reviewer"] = "Test reviewer"
    segment["reviewed_at"] = "2026-08-15"
    segment["evidence"] = "raw_trace;filtered_trace"
    segment["notes"] = "Prompt checked against the raw trace."
    segment_path = root / "segment.tsv"
    segment.to_csv(segment_path, sep="\t", index=False)
    return channel_path, segment_path


class BIDSEeglabReviewTests(unittest.TestCase):
    def test_pending_channel_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            qc = Path(directory)
            template = pd.DataFrame(
                [
                    {
                        "channel": "E1",
                        "candidate_reason": "line_noise_concentration",
                        "decision": "pending",
                        "reviewer": "",
                        "reviewed_at": "",
                        "evidence": "",
                        "notes": "",
                    }
                ],
                columns=CHANNEL_COLUMNS,
            )
            template.to_csv(qc / "channel_review_template.tsv", sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "Invalid channel decisions"):
                _validate_channel_decisions(qc, template)

    def test_finalized_decisions_are_deterministic_and_exact_set_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, qc = _prepared(root)
            channel, segment = _completed_tables(qc, root)

            first = finalize_review_decisions(
                inputs, qc, channel, segment, "public", root / "first"
            )
            second = finalize_review_decisions(
                inputs, qc, channel, segment, "public", root / "second"
            )

            self.assertEqual(verify_provenance(first), verify_provenance(second))
            self.assertEqual(
                (first / "decision_summary.json").read_bytes(),
                (second / "decision_summary.json").read_bytes(),
            )
            summary = json.loads((first / "decision_summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["automatic_decisions"], 0)
            self.assertFalse(summary["publication_allowed"])

    def test_manual_additions_are_bound_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, qc = _prepared(root)
            channel_path, segment_path = _completed_tables(qc, root)
            channel = pd.read_csv(
                channel_path, sep="\t", dtype=str, keep_default_na=False
            )
            screened = pd.read_csv(qc / "line_noise_metrics.csv")["channel"].astype(str)
            manual_channel = next(
                name for name in screened if name not in set(channel["channel"])
            )
            channel.loc[len(channel)] = {
                "channel": manual_channel,
                "candidate_reason": "manual_addition",
                "decision": "interpolate",
                "reviewer": "Test reviewer",
                "reviewed_at": "2026-08-15",
                "evidence": "full_trace;neighbour_trace",
                "notes": "Persistent attenuation found outside the screen.",
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
                "refined_onset_s": "0.6",
                "refined_duration_s": "0.2",
                "scope": "both",
                "reviewer": "Test reviewer",
                "reviewed_at": "2026-08-15",
                "evidence": "full_trace;all_channels",
                "notes": "Brief common-mode transient found manually.",
            }
            segment.to_csv(segment_path, sep="\t", index=False)

            output = finalize_review_decisions(
                inputs, qc, channel_path, segment_path, "public", root / "decisions"
            )
            summary = json.loads((output / "decision_summary.json").read_text())
            self.assertEqual(summary["channel_decisions"]["manual_addition_rows"], 1)
            self.assertEqual(summary["segment_decisions"]["manual_addition_rows"], 1)
            self.assertEqual(
                summary["segment_decisions"]["ica_exclusion_intervals"][0][
                    "candidate_ids"
                ],
                ["manual-0001"],
            )
            self.assertEqual(
                summary["segment_decisions"]["exclusion_application"],
                "global_recording_intervals_all_channels",
            )

    def test_segment_exclusion_requires_refined_bound_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qc = root / "qc"
            qc.mkdir()
            (qc / "qc_summary.json").write_text(
                json.dumps({"recording_duration_s": 10.0}), encoding="utf-8"
            )
            template = pd.DataFrame(
                [
                    {
                        "candidate_id": "segment-0001",
                        "channel": "E1",
                        "prompt_onset_s": "2.0",
                        "prompt_duration_s": "2.0",
                        "candidate_reason": "high_filtered_range",
                        "decision": "pending",
                        "refined_onset_s": "",
                        "refined_duration_s": "",
                        "scope": "",
                        "reviewer": "",
                        "reviewed_at": "",
                        "evidence": "",
                        "notes": "",
                    }
                ],
                columns=SEGMENT_COLUMNS,
            )
            template.to_csv(qc / "segment_review_template.tsv", sep="\t", index=False)
            completed = template.copy()
            completed.loc[0, ["decision", "refined_onset_s", "refined_duration_s"]] = [
                "exclude",
                "2.25",
                "0.5",
            ]
            completed.loc[
                0, ["scope", "reviewer", "reviewed_at", "evidence", "notes"]
            ] = [
                "both",
                "Reviewer",
                "2026-08-15",
                "raw_trace;filtered_trace",
                "Transient confirmed in the raw trace.",
            ]

            validated, duration = _validate_segment_decisions(qc, completed)
            intervals = _merged_intervals(validated)
            ica_intervals = _effective_intervals(validated, "ica")
            epoch_intervals = _effective_intervals(validated, "epochs")

            self.assertEqual(duration, 10.0)
            self.assertEqual(intervals[0]["onset_s"], 2.25)
            self.assertEqual(intervals[0]["duration_s"], 0.5)
            self.assertEqual(intervals[0]["scope"], "both")
            self.assertEqual(ica_intervals[0]["scope"], "ica")
            self.assertEqual(epoch_intervals[0]["scope"], "epochs")

            completed.loc[0, "refined_duration_s"] = "20"
            with self.assertRaisesRegex(ValueError, "outside the recording"):
                _validate_segment_decisions(qc, completed)

            completed.loc[0, ["refined_onset_s", "refined_duration_s"]] = ["7", "1"]
            with self.assertRaisesRegex(ValueError, "must overlap"):
                _validate_segment_decisions(qc, completed)

            completed.loc[0, ["refined_onset_s", "refined_duration_s", "notes"]] = [
                "2.25",
                "0.5",
                "",
            ]
            with self.assertRaisesRegex(ValueError, "rationale"):
                _validate_segment_decisions(qc, completed)

    def test_identity_path_and_input_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, qc = _prepared(root)
            channel, segment = _completed_tables(qc, root)

            with self.assertRaisesRegex(ValueError, "Participant ID"):
                finalize_review_decisions(
                    inputs, qc, channel, segment, "wrong", root / "wrong-participant"
                )
            with self.assertRaisesRegex(ValueError, "outside the source dataset"):
                finalize_review_decisions(
                    inputs,
                    qc,
                    channel,
                    segment,
                    "public",
                    inputs.dataset_root / "derived-decisions",
                )

            linked = root / "linked-channel.tsv"
            linked.symlink_to(channel)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                finalize_review_decisions(
                    inputs, qc, linked, segment, "public", root / "linked-input"
                )

    def test_decision_mutation_after_capture_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, qc = _prepared(root)
            channel, segment = _completed_tables(qc, root)
            output = root / "published"
            original = bids_eeglab_review._validate_channel_decisions

            def mutate_after_capture(*args):
                result = original(*args)
                channel.write_text(channel.read_text() + "\n", encoding="utf-8")
                return result

            with (
                mock.patch.object(
                    bids_eeglab_review,
                    "_validate_channel_decisions",
                    side_effect=mutate_after_capture,
                ),
                self.assertRaisesRegex(ValueError, "Decision inputs changed"),
            ):
                finalize_review_decisions(
                    inputs, qc, channel, segment, "public", output
                )
            self.assertFalse(output.exists())

    def test_qc_package_is_immutable_and_failed_publication_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, qc = _prepared(root)
            channel, segment = _completed_tables(qc, root)
            with self.assertRaisesRegex(ValueError, "outside the immutable QC"):
                finalize_review_decisions(
                    inputs,
                    qc,
                    channel,
                    segment,
                    "public",
                    qc / "decisions",
                )

            output = root / "published"
            original = bids_eeglab_review.verify_provenance

            def fail_after_rename(path):
                if Path(path) == output:
                    raise ValueError("post-rename verification failed")
                return original(path)

            with (
                mock.patch.object(
                    bids_eeglab_review,
                    "verify_provenance",
                    side_effect=fail_after_rename,
                ),
                self.assertRaisesRegex(ValueError, "post-rename"),
            ):
                finalize_review_decisions(
                    inputs,
                    qc,
                    channel,
                    segment,
                    "public",
                    output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
