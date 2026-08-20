from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.ica_validation import (
    auxiliary_correlation_summary,
    signal_preservation_summary,
    verified_fixture_seed,
    verify_matched_ica_processing,
    verify_validation_context,
)


def validation_recordings() -> tuple[mne.io.RawArray, mne.io.RawArray]:
    sfreq = 100.0
    times = np.arange(1000) / sfreq
    eog = np.sin(2 * np.pi * times)
    ecg = np.sin(2 * np.pi * 1.3 * times)
    task = np.sin(2 * np.pi * 10 * times)
    reference_data = np.vstack(
        [
            (task + 0.5 * eog) * 1e-6,
            (task + 0.4 * ecg) * 1e-6,
            eog * 1e-6,
            ecg * 1e-6,
        ]
    )
    observed_data = np.vstack(
        [
            (task + 0.05 * eog) * 1e-6,
            (task + 0.04 * ecg) * 1e-6,
        ]
    )
    reference = mne.io.RawArray(
        reference_data,
        mne.create_info(
            ["C3", "C4", "EOG", "ECG"],
            sfreq,
            ["eeg", "eeg", "eog", "ecg"],
        ),
        verbose="ERROR",
    )
    observed = mne.io.RawArray(
        observed_data,
        mne.create_info(["C3", "C4"], sfreq, ["eeg", "eeg"]),
        verbose="ERROR",
    )
    reference.set_annotations(
        mne.Annotations(onset=[2.0, 5.0], duration=[0.0, 0.0], description=["S1", "S2"])
    )
    observed.set_annotations(reference.annotations.copy())
    return reference, observed


class ICAValidationTests(unittest.TestCase):
    def test_fixture_seed_must_match_verified_package(self):
        fixture = {"workflow": "public_synthetic_ica_fixture"}
        with self.assertRaisesRegex(ValueError, "does not match"):
            verified_fixture_seed(fixture, {"seed": 4401}, 9999)
        self.assertEqual(verified_fixture_seed(fixture, {"seed": 4401}, 4401), 4401)

    def test_validation_context_rejects_current_source_drift(self):
        review = {
            "software": {"source_manifest": {"sha256": "old"}},
            "configuration": {"analysis_sha256": "analysis"},
            "ica_configuration": {"config_sha256": "ica"},
        }
        processed = {
            "software": {"source_manifest": {"sha256": "old"}},
            "configuration": {"analysis_sha256": "analysis"},
        }
        with self.assertRaisesRegex(ValueError, "source"):
            verify_validation_context(
                review,
                processed,
                processed,
                current_source_manifest_sha256="current",
                analysis_config_sha256="analysis",
                ica_config_sha256="ica",
            )

    def test_auxiliary_correlations_report_reduction(self):
        reference, observed = validation_recordings()
        summary = auxiliary_correlation_summary(reference, observed, reference)
        self.assertLess(
            summary["eog"]["maximum_absolute_correlation_reviewed"],
            summary["eog"]["maximum_absolute_correlation_control"],
        )
        self.assertLess(
            summary["ecg"]["maximum_absolute_correlation_reviewed"],
            summary["ecg"]["maximum_absolute_correlation_control"],
        )

    def test_signal_preservation_keeps_denominators_visible(self):
        reference, observed = validation_recordings()
        summary = signal_preservation_summary(reference, observed)
        self.assertEqual(summary["task_signal"]["channels"], ["C3", "C4"])
        self.assertEqual(summary["band_power"]["channel_band_values"], 8)
        self.assertTrue(
            np.isfinite(summary["band_power"]["maximum_absolute_change_db"])
        )

    def test_validation_context_accepts_exact_reviewed_snapshot(self):
        review = {
            "software": {"source_manifest": {"sha256": "source"}},
            "configuration": {"analysis_sha256": "analysis"},
            "ica_configuration": {"config_sha256": "ica"},
        }
        processed = {
            "software": {"source_manifest": {"sha256": "source"}},
            "configuration": {"analysis_sha256": "analysis"},
        }
        verify_validation_context(
            review,
            processed,
            processed,
            current_source_manifest_sha256="source",
            analysis_config_sha256="analysis",
            ica_config_sha256="ica",
        )

    def test_matched_processing_rejects_nonempty_control_exclusions(self):
        shared = {
            "participant_id": "synthetic",
            "inputs": [{"role": "raw", "sha256": "input"}],
            "configuration": {"analysis_sha256": "analysis"},
            "bad_channel_decisions": [{"decision": "none"}],
            "interval_review": {"complete": True},
            "software": {"source_manifest": {"sha256": "source"}},
            "ica_inputs": {
                "solution_sha256": "solution",
                "review_provenance_sha256": "review",
                "decision_table_sha256": "control-decisions",
                "automatic_exclusion": False,
            },
        }
        reviewed = {
            **shared,
            "ica_inputs": {
                **shared["ica_inputs"],
                "decision_table_sha256": "reviewed-decisions",
            },
        }
        control_summary = {
            "participant_id": "synthetic",
            "erp_filter_hz": [0.2, 30.0],
            "reference": "average EEG only",
            "interpolated_bad_channels": [],
            "trial_reconciliation": {"accounting_complete": True},
            "go_epoch_accounting": {"retained_epochs": 1},
            "stop_epoch_accounting": {"retained_epochs": 1},
            "interval_review": {"complete": True},
            "ica": {
                "components": 2,
                "excluded_components": [0],
                "decision_record_complete": True,
                "solution_sha256": "solution",
                "decision_table_sha256": "control-decisions",
            },
        }
        reviewed_summary = {
            **control_summary,
            "ica": {
                "components": 2,
                "excluded_components": [1],
                "decision_record_complete": True,
                "solution_sha256": "solution",
                "decision_table_sha256": "reviewed-decisions",
            },
        }
        with self.assertRaisesRegex(ValueError, "retain every component"):
            verify_matched_ica_processing(
                shared,
                reviewed,
                control_summary,
                reviewed_summary,
            )
        control_summary["ica"]["excluded_components"] = []
        verify_matched_ica_processing(
            shared,
            reviewed,
            control_summary,
            reviewed_summary,
        )


if __name__ == "__main__":
    unittest.main()
