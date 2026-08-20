"""Validation tests for executable project configuration."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.config import load_analysis_config, load_event_codebook
from hunt_eeg.events import classify_trials


def marker(onset: float, code: str) -> dict:
    return {"onset_s": onset, "duration_s": 0.0, "marker": code}


class ConfigurationTests(unittest.TestCase):
    def test_project_configuration_is_valid(self):
        analysis = load_analysis_config()
        codebook = load_event_codebook()

        self.assertEqual(analysis.filter_hz, (1.0, 40.0))
        self.assertEqual(analysis.ica_filter_hz, (1.0, 40.0))
        self.assertEqual(analysis.erp_filter_hz, (0.2, 30.0))
        self.assertEqual(analysis.qc.full_recording_window_seconds, 20.0)
        self.assertEqual(analysis.epochs_seconds["stop"], (-2.0, 2.0))
        self.assertEqual(
            codebook.outcome_classes("go")["S6"],
            "go_correct",
        )
        self.assertEqual(
            codebook.inferred_outcome("stop"),
            ("stop_successful", "medium"),
        )

    def test_codebook_controls_trial_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_codebook.csv"
            with (PROJECT_ROOT / "config" / "event_codebook.csv").open(
                encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row["marker"] == "S6":
                    row["classification"] = "go_confirmed"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            codebook = load_event_codebook(path)
            trials = classify_trials(
                [marker(0.0, "S17"), marker(0.4, "S6")],
                codebook=codebook,
            )

        self.assertEqual(trials.iloc[0]["trial_class"], "go_confirmed")

    def test_duplicate_codebook_markers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_codebook.csv"
            original = (PROJECT_ROOT / "config" / "event_codebook.csv").read_text(
                encoding="utf-8"
            )
            duplicate = original.splitlines()[1]
            path.write_text(original + duplicate + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate markers"):
                load_event_codebook(path)

    def test_outcome_class_must_match_trial_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_codebook.csv"
            original = (PROJECT_ROOT / "config" / "event_codebook.csv").read_text(
                encoding="utf-8"
            )
            path.write_text(
                original.replace("S6,outcome,go,go_correct", "S6,outcome,go,stop_failed"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must start with"):
                load_event_codebook(path)

    def test_inferred_outcome_requires_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event_codebook.csv"
            original = (PROJECT_ROOT / "config" / "event_codebook.csv").read_text(
                encoding="utf-8"
            )
            path.write_text(
                original.replace(
                    "S19,stop_signal,stop,,stop_successful,medium",
                    "S19,stop_signal,stop,,stop_successful,",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be paired"):
                load_event_codebook(path)

    def test_epoch_window_must_span_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            payload = json.loads(
                (PROJECT_ROOT / "config" / "analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["epochs_seconds"]["go"] = [0.0, 3.0]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "span time zero"):
                load_analysis_config(path)

    def test_qc_thresholds_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            payload = json.loads(
                (PROJECT_ROOT / "config" / "analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["qc"]["flat_fraction_threshold"] = 1.5
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_analysis_config(path)

    def test_full_recording_window_length_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            payload = json.loads(
                (PROJECT_ROOT / "config" / "analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["qc"]["full_recording_window_seconds"] = 0
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_analysis_config(path)


if __name__ == "__main__":
    unittest.main()
