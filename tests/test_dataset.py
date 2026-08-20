"""Tests for dataset-level preprocessing accounting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hunt_eeg.dataset import build_dataset_aggregate
from run_dataset_preprocess import package_participant_id


class DatasetAccountingTests(unittest.TestCase):
    def test_published_participant_directory_has_exact_fixed_study_id(self):
        self.assertEqual(package_participant_id(Path("sub-901")), "901")
        for name in ("901", "sub-91", "sub-901-extra"):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "Invalid participant package directory"
            ):
                package_participant_id(Path(name))

    def test_trials_epochs_decisions_and_qc_are_aggregated(self):
        table = pd.DataFrame(
            [
                {
                    "participant_id": "001",
                    "decision_record_complete": True,
                    "interpolated_channel_count": 1,
                    "detected_trial_starts": 3,
                    "go_events_proposed": 2,
                    "go_epochs_retained": 1,
                    "go_epochs_dropped": 1,
                    "stop_events_proposed": 1,
                    "stop_epochs_retained": 1,
                    "stop_epochs_dropped": 0,
                    "before_median_filtered_std_uv": 10.0,
                    "after_median_filtered_std_uv": 8.0,
                    "before_median_filtered_robust_range_uv": 40.0,
                    "after_median_filtered_robust_range_uv": 32.0,
                    "before_maximum_flat_fraction": 0.2,
                    "after_maximum_flat_fraction": 0.1,
                    "before_median_line_noise_ratio_db": 4.0,
                    "after_median_line_noise_ratio_db": -2.0,
                }
            ]
        )
        summaries = {
            "001": {
                "trial_reconciliation": {
                    "status_counts": {"classified": 2, "inferred": 1}
                },
                "go_epoch_accounting": {"drop_reasons": {"BAD_motion": 1}},
                "stop_epoch_accounting": {"drop_reasons": {}},
            }
        }

        aggregate = build_dataset_aggregate(table, summaries)

        self.assertEqual(aggregate["detected_trial_starts_total"], 3)
        self.assertEqual(
            aggregate["trial_status_totals"],
            {"classified": 2, "inferred": 1},
        )
        self.assertEqual(
            aggregate["epochs"]["go"]["drop_reasons"],
            {"BAD_motion": 1},
        )
        self.assertEqual(
            aggregate["before_after_qc"]["median_filtered_std_uv"],
            {"before_dataset_median": 10.0, "after_dataset_median": 8.0},
        )


if __name__ == "__main__":
    unittest.main()
