"""Regression tests for the recovered task marker logic."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.events import classify_trials, marker_counts, normalize_marker
from hunt_eeg.qc import _read_brainvision_compat


def marker(onset: float, code: str) -> dict:
    return {"onset_s": onset, "duration_s": 0.0, "marker": code}


FIXTURES = PROJECT_ROOT / "tests" / "fixtures"


class EventLogicTests(unittest.TestCase):
    def test_brainvision_marker_normalization(self):
        self.assertEqual(normalize_marker("Stimulus/S  17"), "S17")
        self.assertEqual(normalize_marker("S  5"), "S5")

    def test_go_and_stop_outcomes(self):
        markers = pd.read_csv(FIXTURES / "stop_signal_markers.csv").to_dict("records")
        trials = classify_trials(markers)
        self.assertEqual(
            trials["trial_class"].tolist(),
            [
                "go_correct",
                "go_incorrect_or_slow",
                "stop_failed",
                "stop_successful",
                "stop_unclassified",
                "go_correct",
            ],
        )
        self.assertAlmostEqual(trials.iloc[0]["response_time_ms"], 500.0)
        self.assertAlmostEqual(trials.iloc[2]["stop_signal_delay_ms"], 250.0)
        self.assertEqual(trials.iloc[4]["outcome_code"], "S7")

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

    def test_brainvision_reader_uses_temporary_marker_when_date_is_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "sample.vhdr"
            marker_file = root / "sample.vmrk"
            data_file = root / "sample.eeg"
            data_file.write_bytes(b"")
            header.write_text(
                "Brain Vision Data Exchange Header File Version 1.0\n"
                "[Common Infos]\n"
                "DataFile=sample.eeg\n"
                "MarkerFile=sample.vmrk\n",
                encoding="utf-8",
            )
            marker_file.write_text(
                "Brain Vision Data Exchange Marker File, Version 1.0\n"
                "[Common Infos]\n"
                "DataFile=sample.eeg\n"
                "[Marker Infos]\n"
                "Mk1=New Segment,,1,1,0,202001011200001234567\n",
                encoding="utf-8",
            )
            calls = []

            def fake_reader(path, preload=False, verbose=None):
                calls.append(Path(path))
                if len(calls) == 1:
                    raise ValueError("unconverted data remains: 7")
                rewritten_marker = calls[1].with_suffix(".vmrk")
                self.assertIn(
                    "00000000000000000000",
                    rewritten_marker.read_text(encoding="utf-8"),
                )
                self.assertIn(
                    f"DataFile={data_file.resolve()}",
                    rewritten_marker.read_text(encoding="utf-8"),
                )
                return "raw"

            with patch("mne.io.read_raw_brainvision", side_effect=fake_reader):
                self.assertEqual(_read_brainvision_compat(header), "raw")

            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
