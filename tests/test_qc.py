"""Tests for complete temporal coverage in raw-channel QC."""

from __future__ import annotations

import os
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

from hunt_eeg.config import load_analysis_config
from hunt_eeg.qc import (
    _full_recording_window_metrics,
    _save_candidate_window_traces,
    _select_candidate_review_windows,
    _summarize_full_recording_windows,
    _window_qc_status,
)


class TemporalQCTests(unittest.TestCase):
    @staticmethod
    def _filtered(raw, analysis):
        filtered = raw.copy()
        filtered.filter(
            *analysis.filter_hz,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            phase="zero",
            verbose="ERROR",
        )
        return filtered

    def test_late_intermittent_channel_artifact_is_not_missed(self):
        sfreq = 100.0
        seconds = 65
        rng = np.random.default_rng(41)
        data = rng.normal(scale=2e-6, size=(16, int(sfreq * seconds)))
        time = np.arange(data.shape[1]) / sfreq
        data += np.sin(2 * np.pi * 10 * time) * np.linspace(
            4e-6,
            8e-6,
            16,
        )[:, np.newaxis]
        late = (time >= 55) & (time < 60)
        data[3, late] += 2e-3 * np.sin(2 * np.pi * 6 * time[late])
        info = mne.create_info(
            [f"E{index:02d}" for index in range(16)],
            sfreq,
            "eeg",
        )
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        analysis = load_analysis_config()
        filtered = self._filtered(raw, analysis)

        metrics = _full_recording_window_metrics(
            raw,
            filtered,
            analysis.qc,
        )
        summary = _summarize_full_recording_windows(metrics).set_index("channel")

        self.assertEqual(metrics["window_index"].nunique(), 4)
        final_window = metrics.loc[metrics["window_index"] == 4]
        self.assertAlmostEqual(final_window["start_s"].iloc[0], 60.0)
        self.assertAlmostEqual(final_window["stop_s"].iloc[0], 65.0)
        self.assertTrue(final_window["is_partial"].all())
        self.assertGreater(summary.loc["E03", "flagged_window_count"], 0)
        self.assertEqual(summary.loc["E00", "flagged_window_count"], 0)

        coverage = metrics[
            ["window_index", "start_sample", "stop_sample_exclusive"]
        ].drop_duplicates()
        self.assertEqual(coverage["start_sample"].tolist(), [0, 2000, 4000, 6000])
        self.assertEqual(
            coverage["stop_sample_exclusive"].tolist(),
            [2000, 4000, 6000, 6500],
        )

    def test_last_partial_window_and_boundary_transient_are_scanned(self):
        sfreq = 100.0
        seconds = 65
        rng = np.random.default_rng(7)
        data = rng.normal(scale=2e-6, size=(16, int(sfreq * seconds)))
        time = np.arange(data.shape[1]) / sfreq
        partial = (time >= 62) & (time < 64)
        boundary = (time >= 19.5) & (time < 20.5)
        data[4, partial] += 3e-3 * np.sin(2 * np.pi * 7 * time[partial])
        data[5, boundary] += 3e-3 * np.sin(2 * np.pi * 7 * time[boundary])
        raw = mne.io.RawArray(
            data,
            mne.create_info([f"E{i:02d}" for i in range(16)], sfreq, "eeg"),
            verbose="ERROR",
        )
        analysis = load_analysis_config()
        metrics = _full_recording_window_metrics(
            raw,
            self._filtered(raw, analysis),
            analysis.qc,
        )

        partial_rows = metrics[(metrics["channel"] == "E04") & metrics["window_flag"]]
        boundary_rows = metrics[(metrics["channel"] == "E05") & metrics["window_flag"]]
        self.assertEqual(partial_rows["window_index"].tolist(), [4])
        self.assertEqual(boundary_rows["window_index"].tolist(), [1, 2])

    def test_zero_mad_is_unscorable_and_output_is_deterministic(self):
        sfreq = 100.0
        time = np.arange(2000) / sfreq
        shared = 5e-6 * np.sin(2 * np.pi * 10 * time)
        data = np.repeat(shared[np.newaxis, :], 8, axis=0)
        raw = mne.io.RawArray(
            data,
            mne.create_info([f"E{i:02d}" for i in range(8)], sfreq, "eeg"),
            verbose="ERROR",
        )
        analysis = load_analysis_config()
        filtered = self._filtered(raw, analysis)

        first = _full_recording_window_metrics(raw, filtered, analysis.qc)
        second = _full_recording_window_metrics(raw, filtered, analysis.qc)

        self.assertTrue((first["qc_status"] == "unscorable_zero_mad").all())
        self.assertTrue(first["z_log_filtered_std"].isna().all())
        self.assertFalse(first["window_flag"].any())
        self.assertTrue(first.equals(second))

    def test_one_available_metric_keeps_window_partially_scorable(self):
        status, std_scorable, range_scorable = _window_qc_status(
            np.array([np.nan, np.nan]),
            np.array([0.0, 6.0]),
        )

        self.assertEqual(status, "partially_scorable_zero_mad")
        self.assertFalse(std_scorable)
        self.assertTrue(range_scorable)

    def test_flat_channel_is_kept_as_a_separate_review_reason(self):
        sfreq = 100.0
        rng = np.random.default_rng(12)
        data = rng.normal(scale=3e-6, size=(16, 2000))
        data[2] = 0.0
        raw = mne.io.RawArray(
            data,
            mne.create_info([f"E{i:02d}" for i in range(16)], sfreq, "eeg"),
            verbose="ERROR",
        )
        analysis = load_analysis_config()

        metrics = _full_recording_window_metrics(
            raw,
            self._filtered(raw, analysis),
            analysis.qc,
        )
        row = metrics.loc[metrics["channel"] == "E02"].iloc[0]

        self.assertTrue(row["flat_flag"])
        self.assertFalse(row["window_flag"])
        self.assertIn("flat_segment", row["flag_reason"])

    def test_candidate_trace_figure_supports_sparse_array_adjacency(self):
        sfreq = 100.0
        names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T7",
            "T8",
            "P7",
            "P8",
        ]
        rng = np.random.default_rng(3)
        raw = mne.io.RawArray(
            rng.normal(scale=2e-6, size=(len(names), 2000)),
            mne.create_info(names, sfreq, "eeg"),
            verbose="ERROR",
        )
        raw.set_montage("standard_1020", verbose="ERROR")
        filtered = raw.copy()
        candidate = pd.DataFrame(
            [
                {
                    "channel": "Fp1",
                    "window_index": 1,
                    "start_sample": 0,
                    "stop_sample_exclusive": 2000,
                    "start_s": 0.0,
                    "stop_s": 20.0,
                    "z_log_filtered_std": 7.0,
                    "z_log_filtered_range": 6.0,
                    "flat_fraction": 0.0,
                    "window_flag": True,
                    "flat_flag": False,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _save_candidate_window_traces(output, raw, filtered, candidate)
            figures = list((output / "figures" / "window_reviews").glob("*.png"))

        self.assertEqual(len(figures), 1)

    def test_flat_review_selects_the_most_flat_window(self):
        metrics = pd.DataFrame(
            [
                {
                    "channel": "E01",
                    "window_index": 1,
                    "z_log_filtered_std": -1.0,
                    "z_log_filtered_range": -0.5,
                    "flat_fraction": 0.25,
                    "window_flag": False,
                    "flat_flag": True,
                },
                {
                    "channel": "E01",
                    "window_index": 2,
                    "z_log_filtered_std": -4.0,
                    "z_log_filtered_range": -3.0,
                    "flat_fraction": 0.80,
                    "window_flag": False,
                    "flat_flag": True,
                },
            ]
        )

        selected = _select_candidate_review_windows(metrics)

        self.assertEqual(selected["window_index"].tolist(), [2])

    def test_review_can_keep_amplitude_and_flat_exemplars(self):
        metrics = pd.DataFrame(
            [
                {
                    "channel": "E01",
                    "window_index": 1,
                    "z_log_filtered_std": 7.0,
                    "z_log_filtered_range": 6.0,
                    "flat_fraction": 0.0,
                    "window_flag": True,
                    "flat_flag": False,
                },
                {
                    "channel": "E01",
                    "window_index": 2,
                    "z_log_filtered_std": -3.0,
                    "z_log_filtered_range": -2.0,
                    "flat_fraction": 0.75,
                    "window_flag": False,
                    "flat_flag": True,
                },
            ]
        )

        selected = _select_candidate_review_windows(metrics)

        self.assertEqual(selected["window_index"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
