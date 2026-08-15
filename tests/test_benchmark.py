"""Tests for the public synthetic corruption benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.benchmark import (
    _binary_metrics,
    Corruption,
    SyntheticBenchmarkConfig,
    inject_corruptions,
    make_clean_recording,
    preprocess_with_known_decisions,
    score_detection,
    score_band_power,
    score_line_noise_detection,
    score_preservation,
    score_task_signal,
    summarize_band_errors,
    summarize_evaluation_units,
    truth_by_window,
    verify_brainvision_round_trip,
    write_json,
)


class SyntheticBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = SyntheticBenchmarkConfig(
            sampling_frequency_hz=200.0,
            duration_seconds=40.0,
            eeg_channel_count=16,
            window_seconds=20.0,
            random_seeds={
                "calibration": (1,),
                "held_out": (2,),
                "stress": (3,),
            },
            corruptions=(
                Corruption("persistent_noise", "primary", 2, 0.0, 40.0, 50.0),
                Corruption("flat_segment", "primary", 7, 20.0, 40.0, 0.0),
            ),
        )

    def test_generation_and_corruption_are_deterministic(self):
        clean_first = make_clean_recording(self.config, 17)
        clean_second = make_clean_recording(self.config, 17)
        corrupted_first, truth_first = inject_corruptions(clean_first, self.config, 17)
        corrupted_second, truth_second = inject_corruptions(
            clean_second, self.config, 17
        )

        np.testing.assert_array_equal(clean_first.get_data(), clean_second.get_data())
        np.testing.assert_array_equal(
            corrupted_first.get_data(), corrupted_second.get_data()
        )
        pd.testing.assert_frame_equal(truth_first, truth_second)
        self.assertEqual(len(clean_first.copy().pick("eeg").ch_names), 16)
        self.assertIn("EOG", clean_first.ch_names)
        self.assertIn("ECG", clean_first.ch_names)

    def test_truth_grid_matches_overlapping_windows(self):
        clean = make_clean_recording(self.config, 5)
        _, truth = inject_corruptions(clean, self.config, 5)
        grid = truth_by_window(
            truth,
            clean.copy().pick("eeg").ch_names,
            clean.n_times,
            float(clean.info["sfreq"]),
            self.config.window_seconds,
        )

        noisy_channel = clean.ch_names[2]
        flat_channel = clean.ch_names[7]
        self.assertEqual(
            grid.loc[grid["channel"] == noisy_channel, "truth_positive"].tolist(),
            [True, True],
        )
        self.assertEqual(
            grid.loc[grid["channel"] == flat_channel, "truth_positive"].tolist(),
            [False, True],
        )

    def test_detection_metrics_use_channel_window_pairs(self):
        truth = pd.DataFrame(
            {
                "window_index": [1, 1, 2, 2],
                "channel": ["A", "B", "A", "B"],
                "truth_positive": [True, False, True, False],
                "truth_families": ["noise", "", "flat", ""],
                "truth_roles": ["primary", "", "primary", ""],
            }
        )
        predicted = pd.DataFrame(
            {
                "window_index": [1, 1, 2, 2],
                "channel": ["A", "B", "A", "B"],
                "window_flag": [True, True, False, False],
                "flat_flag": [False, False, False, False],
                "flag_reason": ["high", "high", "", ""],
            }
        )

        comparison, metrics = score_detection(predicted, truth)

        self.assertEqual(len(comparison), 4)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_truth_grid_keeps_primary_and_stress_cases_separate(self):
        config = SyntheticBenchmarkConfig(
            sampling_frequency_hz=200.0,
            duration_seconds=40.0,
            eeg_channel_count=16,
            window_seconds=20.0,
            random_seeds=self.config.random_seeds,
            corruptions=(
                Corruption("persistent_noise", "primary", 2, 0.0, 40.0, 50.0),
                Corruption("electrode_pop", "stress", 4, 5.0, 5.5, 100.0),
            ),
        )
        clean = make_clean_recording(config, 9)
        _, truth = inject_corruptions(clean, config, 9)
        grid = truth_by_window(
            truth,
            clean.copy().pick("eeg").ch_names,
            clean.n_times,
            float(clean.info["sfreq"]),
            config.window_seconds,
        )

        primary = grid.loc[grid["truth_roles"] == "primary"]
        stress = grid.loc[grid["truth_roles"] == "stress"]
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(stress), 1)
        self.assertTrue(primary["truth_positive"].all())
        self.assertTrue(stress["truth_positive"].all())

    def test_primary_denominator_excludes_stress_case(self):
        temporal = pd.DataFrame({"truth_positive": [True, True, False, False]})
        primary = temporal.iloc[[0, 2, 3]].copy()
        stress = temporal.iloc[[1]].copy()

        units = summarize_evaluation_units(temporal, primary, stress)

        self.assertEqual(units["temporal_channel_windows"], 4)
        self.assertEqual(units["primary_channel_windows"], 3)
        self.assertEqual(units["primary_positive_channel_windows"], 1)
        self.assertEqual(units["stress_positive_channel_windows"], 1)

    def test_binary_metrics_retain_exact_confusion_counts(self):
        truth = pd.Series([True, True, False, False])
        predicted = pd.Series([True, False, True, False])
        metrics = _binary_metrics(truth, predicted)
        self.assertEqual(
            {key: metrics[key] for key in (
                "true_positive", "false_positive", "false_negative", "true_negative"
            )},
            {
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_negative": 1,
            },
        )

    def test_preservation_keeps_clean_samples_separate(self):
        clean = make_clean_recording(self.config, 31)
        corrupted, truth = inject_corruptions(clean, self.config, 31)
        preservation = score_preservation(clean, corrupted, truth)
        noisy_channel = clean.ch_names[2]
        unaffected_channel = clean.ch_names[0]

        noisy = preservation.loc[
            (preservation["channel"] == noisy_channel)
            & (preservation["region"] == "corrupted_samples")
        ].iloc[0]
        unaffected = preservation.loc[
            (preservation["channel"] == unaffected_channel)
            & (preservation["region"] == "clean_samples")
        ].iloc[0]
        self.assertGreater(noisy["rmse_uv"], 10.0)
        self.assertAlmostEqual(unaffected["rmse_uv"], 0.0)
        self.assertAlmostEqual(unaffected["correlation"], 1.0)

    def test_line_noise_uses_raw_psd_detector(self):
        config = SyntheticBenchmarkConfig(
            sampling_frequency_hz=200.0,
            duration_seconds=40.0,
            eeg_channel_count=16,
            window_seconds=20.0,
            random_seeds=self.config.random_seeds,
            corruptions=(Corruption("line_noise", "primary", 4, 0.0, 40.0, 80.0),),
        )
        clean = make_clean_recording(config, 44)
        corrupted, truth = inject_corruptions(clean, config, 44)

        comparison, metrics = score_line_noise_detection(
            corrupted,
            truth,
            line_frequency_hz=50.0,
            threshold_db=20.0,
        )

        target = comparison.loc[comparison["channel"] == clean.ch_names[4]].iloc[0]
        self.assertTrue(target["truth_positive"])
        self.assertTrue(target["predicted_positive"])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)

    def test_json_writer_accepts_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_json(path, {"count": np.int64(3), "score": np.float64(0.5)})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"count": 3, "score": 0.5})

    def test_known_decision_workflow_preserves_task_measure(self):
        clean = make_clean_recording(self.config, 52)
        corrupted, truth = inject_corruptions(clean, self.config, 52)
        reference = preprocess_with_known_decisions(clean, [], (1.0, 40.0))
        bads = truth.loc[
            truth["family"].isin({"persistent_noise", "flat_segment"}),
            "channel",
        ].tolist()
        cleaned = preprocess_with_known_decisions(corrupted, bads, (1.0, 40.0))
        task = score_task_signal(reference, cleaned)
        band_power = score_band_power(reference, cleaned)

        self.assertEqual(task["channel"].tolist(), ["C3", "C4"])
        self.assertTrue((task["epochs"] > 0).all())
        self.assertTrue(np.isfinite(task["absolute_amplitude_error_uv"]).all())
        self.assertTrue(task["reference_peak_latency_ms"].between(200.0, 450.0).all())
        self.assertEqual(set(band_power["band"]), {"delta", "theta", "alpha", "beta"})
        self.assertTrue(np.isfinite(band_power["absolute_log_ratio_db"]).all())

    def test_band_error_summary_keeps_channel_groups_and_upper_tail(self):
        table = pd.DataFrame(
            {
                "channel": ["A", "B", "C", "D"],
                "band": ["alpha"] * 4,
                "absolute_log_ratio_db": [0.01, 0.10, 1.00, 3.00],
            }
        )

        classified, summary = summarize_band_errors(
            table,
            oracle_channels={"C"},
            corrupted_channels={"B", "C"},
        )

        self.assertEqual(
            classified.set_index("channel")["channel_group"].to_dict(),
            {
                "A": "unaffected",
                "B": "non_oracle_corrupted",
                "C": "oracle_interpolated",
                "D": "unaffected",
            },
        )
        self.assertEqual(summary["all_channel_band_values"]["values"], 4)
        self.assertEqual(summary["oracle_interpolated"]["values"], 1)
        self.assertEqual(
            summary["all_channel_band_values"]["maximum_absolute_error_db"],
            3.0,
        )

    def test_synthetic_markers_form_complete_go_and_stop_trials(self):
        clean = make_clean_recording(self.config, 62)
        from hunt_eeg.events import (
            annotations_to_markers,
            classify_trials,
            reconcile_trials,
        )

        markers = annotations_to_markers(clean.annotations)
        trials = classify_trials(markers)
        accounting = reconcile_trials(markers, trials)

        self.assertEqual(accounting["detected_trial_starts"], len(trials))
        self.assertTrue(accounting["accounting_complete"])
        self.assertFalse(
            trials["classification_status"].isin({"ambiguous", "invalid"}).any()
        )
        self.assertFalse(trials["classification_status"].eq("incomplete").any())

    def test_brainvision_round_trip_preserves_public_fixture(self):
        clean = make_clean_recording(self.config, 71)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.vhdr"
            mne.export.export_raw(
                path,
                clean,
                fmt="brainvision",
                overwrite=False,
                verbose="ERROR",
            )
            reread = mne.io.read_raw_brainvision(
                path,
                preload=True,
                verbose="ERROR",
            )
            diagnostics = verify_brainvision_round_trip(clean, reread)

        self.assertEqual(diagnostics["channels"], 18)
        self.assertLessEqual(diagnostics["maximum_marker_onset_error_samples"], 1.0)


if __name__ == "__main__":
    unittest.main()
