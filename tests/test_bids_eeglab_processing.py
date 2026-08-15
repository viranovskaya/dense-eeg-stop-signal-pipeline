"""Tests for fail-closed BIDS/EEGLAB decision consumption."""

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

from test_bids_eeglab_review import _completed_tables, _prepared

from hunt_eeg.bids_eeglab_processing import (
    ProcessingControls,
    VerifiedReviewBundle,
    _fir_guard_seconds,
    _guarded_intervals,
    load_verified_review_bundle,
    prepare_filtered_reviewed_raw,
    prepare_reviewed_raw,
)
from hunt_eeg.bids_eeglab_review import finalize_review_decisions


def _bundle(root: Path, *, interpolate: bool = False):
    inputs, qc = _prepared(root)
    channel_path, segment_path = _completed_tables(qc, root)
    channel = pd.read_csv(channel_path, sep="\t", dtype=str, keep_default_na=False)
    if interpolate:
        if channel.empty:
            channel.loc[len(channel)] = {
                "channel": "E1",
                "candidate_reason": "manual_addition",
                "decision": "interpolate",
                "reviewer": "Test reviewer",
                "reviewed_at": "2026-08-15",
                "evidence": "full_trace;neighbour_trace",
                "notes": "Persistent artifact confirmed in the trace.",
            }
        else:
            channel.loc[0, "decision"] = "interpolate"
            channel.loc[0, "notes"] = "Persistent artifact confirmed in the trace."
        channel.to_csv(channel_path, sep="\t", index=False)
    segment = pd.read_csv(segment_path, sep="\t", dtype=str, keep_default_na=False)
    segment.loc[len(segment)] = {
        "candidate_id": "manual-0001",
        "channel": "all",
        "prompt_onset_s": "",
        "prompt_duration_s": "",
        "candidate_reason": "manual_addition",
        "decision": "exclude",
        "refined_onset_s": "0.5",
        "refined_duration_s": "0.2",
        "scope": "both",
        "reviewer": "Test reviewer",
        "reviewed_at": "2026-08-15",
        "evidence": "full_trace;all_channels",
        "notes": "Transient confirmed in the raw and filtered trace.",
    }
    segment.to_csv(segment_path, sep="\t", index=False)
    bundle = finalize_review_decisions(
        inputs, qc, channel_path, segment_path, "public", root / "bundle"
    )
    return inputs, root / "profile.json", qc, bundle


class BIDSEeglabProcessingTests(unittest.TestCase):
    def test_verified_bundle_prepares_target_specific_global_bad_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _bundle(root)

            review = load_verified_review_bundle(inputs, qc, bundle, "public")
            raw, prepared_review = prepare_reviewed_raw(
                inputs, profile, qc, bundle, "public", "ica"
            )

            self.assertEqual(review.identity, prepared_review.identity)
            self.assertEqual(raw.info["bads"], [])
            self.assertIn("BAD_review_ica", set(raw.annotations.description))
            self.assertNotIn("BAD_review_epochs", set(raw.annotations.description))
            self.assertEqual(len(review.ica_intervals), 1)
            self.assertEqual(len(review.epoch_intervals), 1)

    def test_interpolation_requires_validated_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _bundle(root, interpolate=True)

            with self.assertRaisesRegex(ValueError, "validated standard montage"):
                prepare_reviewed_raw(inputs, profile, qc, bundle, "public", "epochs")

    def test_bundle_tampering_and_wrong_target_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _bundle(root)
            with self.assertRaisesRegex(ValueError, "target"):
                prepare_reviewed_raw(inputs, profile, qc, bundle, "public", "analysis")

            summary = bundle / "decision_summary.json"
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["automatic_decisions"] = 1
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_verified_review_bundle(inputs, qc, bundle, "public")

    def test_filter_guard_uses_half_support_and_merges_overlaps(self):
        controls = ProcessingControls(
            filter_hz=(1.0, 40.0),
            filter_method="firwin_zero_phase_hamming",
            segment_guard="half_filter_support",
            average_reference="all_good_eeg_including_flat_online_reference",
        )
        guard = _fir_guard_seconds(100.0, controls)
        taps = mne.filter.create_filter(
            None,
            100.0,
            1.0,
            40.0,
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            verbose="ERROR",
        )
        self.assertEqual(guard, (len(taps) - 1) / 200.0)
        merged = _guarded_intervals(
            (
                {
                    "onset_s": 1.0,
                    "stop_s": 1.2,
                    "duration_s": 0.2,
                    "candidate_ids": ["a"],
                },
                {
                    "onset_s": 1.3,
                    "stop_s": 1.4,
                    "duration_s": 0.1,
                    "candidate_ids": ["b"],
                },
            ),
            5.0,
            guard,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["onset_s"], 0.0)
        self.assertEqual(merged[0]["candidate_ids"], ["a", "b"])

    def test_filtered_preparation_reconstructs_flat_online_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, profile, qc, bundle = _bundle(root)

            raw, review, summary = prepare_filtered_reviewed_raw(
                inputs, profile, qc, bundle, "public", "ica"
            )

            self.assertEqual(review.identity, summary["review_bundle"])
            self.assertGreater(float(np.ptp(raw.get_data(picks=["Cz"]))), 0.0)
            good = [name for name in raw.ch_names if name not in raw.info["bads"]]
            np.testing.assert_allclose(
                raw.get_data(picks=good).mean(axis=0),
                0.0,
                rtol=0,
                atol=1e-18,
            )
            self.assertIn("BAD_review_ica", set(raw.annotations.description))
            self.assertIn("BAD_filter_guard_ica", set(raw.annotations.description))
            self.assertGreater(summary["filter_guard_seconds"], 0.0)

    def test_average_reference_excludes_reviewed_bad_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "reference_channel": {
                            "name": "Cz",
                            "exclude_from_reference_average": False,
                            "post_reference_policy": (
                                "include_flat_online_reference_in_average_transform"
                            ),
                        },
                        "processing": {
                            "filter_hz": [1.0, 40.0],
                            "filter_method": "firwin_zero_phase_hamming",
                            "segment_guard": "half_filter_support",
                            "average_reference": (
                                "all_good_eeg_including_flat_online_reference"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            samples = np.arange(1000)
            signal = np.sin(samples / 13.0) * 1e-6
            raw = mne.io.RawArray(
                np.vstack([signal, signal * 100, np.zeros_like(signal)]),
                mne.create_info(["E1", "E2", "Cz"], 100.0, "eeg"),
                verbose="ERROR",
            )
            raw.info["bads"] = ["E2"]
            review = VerifiedReviewBundle(
                participant_id="public",
                bad_channels=("E2",),
                ica_intervals=(),
                epoch_intervals=(),
                channel_decisions=(),
                segment_decisions=(),
                identity={"core_sha256": "test"},
            )
            with (
                mock.patch(
                    "hunt_eeg.bids_eeglab_processing.prepare_reviewed_raw",
                    return_value=(raw, review),
                ),
                mock.patch(
                    "hunt_eeg.bids_eeglab_processing.load_verified_review_bundle",
                    return_value=review,
                ),
            ):
                prepared, _, summary = prepare_filtered_reviewed_raw(
                    mock.sentinel.inputs,
                    profile,
                    root / "qc",
                    root / "bundle",
                    "public",
                    "ica",
                )
            e1, e2, cz = prepared.get_data(picks=["E1", "E2", "Cz"])
            np.testing.assert_allclose(cz, -e1, rtol=1e-10, atol=1e-18)
            self.assertGreater(float(np.max(np.abs(e2))), 20 * float(np.max(np.abs(e1))))
            self.assertEqual(summary["bad_channels_excluded_from_reference"], ["E2"])

    def test_filter_guard_contains_a_reviewed_transient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "reference_channel": {
                            "name": "Cz",
                            "exclude_from_reference_average": False,
                            "post_reference_policy": (
                                "include_flat_online_reference_in_average_transform"
                            ),
                        },
                        "processing": {
                            "filter_hz": [1.0, 40.0],
                            "filter_method": "firwin_zero_phase_hamming",
                            "segment_guard": "half_filter_support",
                            "average_reference": (
                                "all_good_eeg_including_flat_online_reference"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            samples = np.arange(2000)
            baseline = np.vstack(
                [
                    np.sin(samples / 13.0) * 1e-6,
                    np.cos(samples / 17.0) * 1e-6,
                    np.zeros_like(samples, dtype=float),
                ]
            )
            contaminated = baseline.copy()
            contaminated[0, 800:820] += 1000e-6
            review = VerifiedReviewBundle(
                participant_id="public",
                bad_channels=(),
                ica_intervals=(
                    {
                        "onset_s": 8.0,
                        "stop_s": 8.2,
                        "duration_s": 0.2,
                        "candidate_ids": ["transient"],
                    },
                ),
                epoch_intervals=(),
                channel_decisions=(),
                segment_decisions=(),
                identity={"core_sha256": "test"},
            )

            def process(data):
                raw = mne.io.RawArray(
                    data.copy(),
                    mne.create_info(["E1", "E2", "Cz"], 100.0, "eeg"),
                    verbose="ERROR",
                )
                with (
                    mock.patch(
                        "hunt_eeg.bids_eeglab_processing.prepare_reviewed_raw",
                        return_value=(raw, review),
                    ),
                    mock.patch(
                        "hunt_eeg.bids_eeglab_processing.load_verified_review_bundle",
                        return_value=review,
                    ),
                ):
                    return prepare_filtered_reviewed_raw(
                        mock.sentinel.inputs,
                        profile,
                        root / "qc",
                        root / "bundle",
                        "public",
                        "ica",
                    )

            artifact_raw, _, summary = process(contaminated)
            control_raw, _, _ = process(baseline)
            guarded = summary["guarded_filter_intervals"][0]
            times = artifact_raw.times
            outside = (times < guarded["onset_s"]) | (times >= guarded["stop_s"])
            inside = (times >= 8.0) & (times < 8.2)
            np.testing.assert_allclose(
                artifact_raw.get_data()[:, outside],
                control_raw.get_data()[:, outside],
                rtol=0,
                atol=1e-15,
            )
            self.assertGreater(
                float(
                    np.max(
                        np.abs(
                            artifact_raw.get_data()[:, inside]
                            - control_raw.get_data()[:, inside]
                        )
                    )
                ),
                100e-6,
            )


if __name__ == "__main__":
    unittest.main()
