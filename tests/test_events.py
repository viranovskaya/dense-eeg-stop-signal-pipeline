"""Regression tests for the recovered task marker logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.events import classify_trials, marker_counts, normalize_marker


def marker(onset: float, code: str) -> dict:
    return {"onset_s": onset, "duration_s": 0.0, "marker": code}


class EventLogicTests(unittest.TestCase):
    def test_brainvision_marker_normalization(self):
        self.assertEqual(normalize_marker("Stimulus/S  17"), "S17")
        self.assertEqual(normalize_marker("S  5"), "S5")

    def test_go_and_stop_outcomes(self):
        markers = [
            marker(0.0, "S17"),
            marker(0.5, "S6"),
            marker(1.0, "S18"),
            marker(1.7, "S4"),
            marker(2.0, "S1"),
            marker(2.25, "S19"),
            marker(2.42, "S5"),
            marker(3.0, "S2"),
            marker(3.30, "S19"),
            marker(5.0, "S17"),
            marker(5.55, "S6"),
        ]
        trials = classify_trials(markers)
        self.assertEqual(
            trials["trial_class"].tolist(),
            [
                "go_correct",
                "go_incorrect_or_slow",
                "stop_failed",
                "stop_successful",
                "go_correct",
            ],
        )
        self.assertAlmostEqual(trials.iloc[0]["response_time_ms"], 500.0)
        self.assertAlmostEqual(trials.iloc[2]["stop_signal_delay_ms"], 250.0)

    def test_s7_after_stop_signal_is_unclassified(self):
        markers = [
            marker(0.0, "S1"),
            marker(0.2, "S19"),
            marker(0.4, "S7"),
            marker(1.0, "S17"),
            marker(1.5, "S6"),
        ]

        trials = classify_trials(markers)

        self.assertEqual(trials.iloc[0]["trial_class"], "stop_unclassified")
        self.assertEqual(trials.iloc[0]["outcome_code"], "S7")
        self.assertTrue(pd.isna(trials.iloc[0]["response_time_ms"]))
        self.assertEqual(trials.iloc[1]["trial_class"], "go_correct")

    def test_marker_counts_are_sorted(self):
        counts = marker_counts(
            [marker(0.0, "S19"), marker(0.2, "S6"), marker(0.4, "S19")]
        )
        self.assertEqual(
            counts.to_dict("records"),
            [
                {"marker": "S6", "count": 1},
                {"marker": "S19", "count": 2},
            ],
        )


if __name__ == "__main__":
    unittest.main()
