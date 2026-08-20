from __future__ import annotations

import unittest
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from hunt_eeg.stop_signal_endpoint import (
    ENDPOINT_OUTPUT_FILES,
    exact_sign_flip_p,
    participant_behavioral_diagnostics,
    participant_endpoint,
    summarize_endpoint,
    verify_endpoint_package,
)
from hunt_eeg.provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_SCRIPT = PROJECT_ROOT / "scripts" / "run_stop_signal_endpoint.py"


def synthetic_stop_epochs(
    absent: int = 8,
    present: int = 8,
) -> tuple[mne.EpochsArray, pd.DataFrame]:
    sfreq = 100.0
    times = np.arange(-0.8, 0.81, 1 / sfreq)
    classes = ["stop_successful"] * absent + ["stop_failed"] * present
    data = np.zeros((len(classes), 3, len(times)))
    info = mne.create_info(["FC1", "FC2", "FCz"], sfreq, ch_types="eeg")
    events = np.column_stack(
        [
            np.arange(len(classes)) * 200,
            np.zeros(len(classes), dtype=int),
            np.ones(len(classes), dtype=int),
        ]
    )
    metadata = pd.DataFrame(
        {
            "trial_index": np.arange(1, len(classes) + 1),
            "trial_class": classes,
            "stimulus_to_stop_signal_ms": np.repeat(300.0, len(classes)),
        }
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.8,
        event_id={"stop": 1},
        metadata=metadata,
        verbose="ERROR",
    )
    with epochs.info._unlock():
        epochs.info["highpass"] = 0.2
        epochs.info["lowpass"] = 30.0
    measurement = (epochs.times >= 0.25) & (epochs.times <= 0.45)
    epochs._data[:absent, :, measurement] = 2e-6
    epochs._data[absent:, :, measurement] = 0.5e-6
    lineage = metadata.copy()
    lineage["retained"] = True
    lineage["output_epoch_index"] = np.arange(len(classes))
    return epochs, lineage


def write_synthetic_dataset(root: Path) -> Path:
    """Write ten minimal verified participant packages for CLI integration."""
    dataset = root / "dataset"
    dataset.mkdir()
    participant_records = []
    interval_manifest_sha256 = "a" * 64
    for number in range(1, 11):
        participant_id = f"{number:03d}"
        package = dataset / f"sub-{participant_id}"
        package.mkdir()
        epochs, lineage = synthetic_stop_epochs()
        epochs.save(
            package / f"sub-{participant_id}_stop-epo.fif",
            overwrite=True,
            verbose="ERROR",
        )
        lineage.to_csv(package / "stop_epoch_lineage.csv", index=False)
        trials = pd.DataFrame(
            [
                {
                    "trial_type": "stop",
                    "trial_class": "stop_successful",
                    "outcome_code": "",
                    "stimulus_to_stop_signal_ms": 250.0,
                    "stimulus_to_outcome_ms": "",
                }
                for _ in range(10)
            ]
            + [
                {
                    "trial_type": "stop",
                    "trial_class": "stop_failed",
                    "outcome_code": "S5",
                    "stimulus_to_stop_signal_ms": 350.0,
                    "stimulus_to_outcome_ms": 500.0,
                }
                for _ in range(10)
            ]
            + [
                {
                    "trial_type": "go",
                    "trial_class": "go_correct",
                    "outcome_code": "S6",
                    "stimulus_to_stop_signal_ms": "",
                    "stimulus_to_outcome_ms": 600.0,
                }
                for _ in range(20)
            ]
        )
        trials.to_csv(package / "reconstructed_trials.csv", index=False)
        base_core = {
            "schema_version": "1",
            "participant_id": participant_id,
            "configuration": {"erp_filter_hz": [0.2, 30.0]},
            "interval_review": {
                "identity": {
                    "manifest_sha256": interval_manifest_sha256,
                    "participant_decisions": [
                        {
                            "participant_id": participant_id,
                            "interval_id": "none",
                            "decision": "none",
                        }
                    ],
                    "participant_decisions_sha256": "b" * 64,
                },
                "application_target": "epochs",
                "complete": True,
                "automatic_decisions": False,
                "applied_intervals": [],
            },
            "ica_inputs": {
                "solution_sha256": "c" * 64,
                "review_provenance_sha256": "d" * 64,
                "decision_table_sha256": "e" * 64,
                "automatic_exclusion": False,
            },
        }
        core = {**base_core, "core_sha256": canonical_sha256(base_core)}
        write_provenance(package, core)
        provenance = verify_provenance(package)
        participant_records.append(
            {
                "participant_id": participant_id,
                "provenance_sha256": sha256_file(package / "provenance.json"),
                "core_sha256": provenance["core_sha256"],
            }
        )
    (dataset / "dataset_provenance.json").write_text(
        json.dumps(
            {
                "source_manifest": source_manifest(),
                "fixed_study_interval_manifest_sha256": (
                    interval_manifest_sha256
                ),
                "participants": participant_records,
                "outputs": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class StopSignalEndpointTests(unittest.TestCase):
    def test_behavioral_diagnostics_report_ssd_and_race_ordering(self):
        trials = pd.DataFrame(
            {
                "trial_type": ["stop", "stop", "go", "go", "go"],
                "trial_class": [
                    "stop_successful",
                    "stop_failed",
                    "go_correct",
                    "go_incorrect_or_slow",
                    "go_unclassified",
                ],
                "outcome_code": ["", "S5", "S6", "S4", "S5"],
                "stimulus_to_stop_signal_ms": [250.0, 350.0, "", "", ""],
                "stimulus_to_outcome_ms": ["", 450.0, 550.0, 650.0, 10.0],
            }
        )
        diagnostics = participant_behavioral_diagnostics(trials)
        self.assertEqual(diagnostics["response_present_probability"], 0.5)
        self.assertEqual(diagnostics["stop_trials_total"], 2)
        self.assertEqual(diagnostics["unclassified_stop_trials"], 0)
        self.assertEqual(diagnostics["ssd_present_minus_absent_ms"], 100.0)
        self.assertEqual(diagnostics["mean_go_response_rt_ms"], 600.0)
        self.assertTrue(diagnostics["failed_stop_rt_shorter_than_go_rt"])

        trials.loc[0, "stimulus_to_stop_signal_ms"] = "missing"
        with self.assertRaisesRegex(ValueError, "positive finite SSD"):
            participant_behavioral_diagnostics(trials)

    def test_participant_endpoint_uses_fixed_roi_window_and_minimum(self):
        epochs, lineage = synthetic_stop_epochs()
        row = participant_endpoint(epochs, lineage, participant_id="001")
        self.assertTrue(row["eligible"])
        self.assertAlmostEqual(row["contrast_absent_minus_present_uv"], 1.5)

        epochs, lineage = synthetic_stop_epochs(absent=7)
        row = participant_endpoint(epochs, lineage, participant_id="002")
        self.assertFalse(row["eligible"])
        self.assertIsNone(row["contrast_absent_minus_present_uv"])

    def test_lineage_mismatch_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        lineage.loc[0, "trial_index"] = 999
        with self.assertRaisesRegex(ValueError, "trial_index"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_integer_valued_float_lineage_indices_are_accepted(self):
        epochs, lineage = synthetic_stop_epochs()
        lineage["output_epoch_index"] = lineage["output_epoch_index"].map(
            lambda value: f"{float(value):.1f}"
        )
        row = participant_endpoint(epochs, lineage, participant_id="001")
        self.assertTrue(row["eligible"])

    def test_fractional_lineage_index_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        lineage["output_epoch_index"] = lineage["output_epoch_index"].astype(float)
        lineage.loc[0, "output_epoch_index"] = 0.5
        with self.assertRaisesRegex(ValueError, "finite integers"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_invalid_retained_value_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        lineage["retained"] = lineage["retained"].astype(object)
        lineage.loc[0, "retained"] = "yes"
        with self.assertRaisesRegex(ValueError, "only true or false"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_unsupported_trial_class_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        assert epochs.metadata is not None
        epochs.metadata.loc[0, "trial_class"] = "stop_unclassified"
        lineage.loc[0, "trial_class"] = "stop_unclassified"
        with self.assertRaisesRegex(ValueError, "unsupported trial class"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_roi_channel_must_be_eeg(self):
        epochs, lineage = synthetic_stop_epochs()
        epochs.set_channel_types({"FC1": "misc"}, verbose="ERROR")
        with self.assertRaisesRegex(ValueError, "typed as EEG"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_wrong_filter_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        with epochs.info._unlock():
            epochs.info["highpass"] = 1.0
        with self.assertRaisesRegex(ValueError, "0.2-30 Hz"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_missing_stop_delay_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        assert epochs.metadata is not None
        epochs.metadata.loc[0, "stimulus_to_stop_signal_ms"] = np.nan
        lineage.loc[0, "stimulus_to_stop_signal_ms"] = np.nan
        with self.assertRaisesRegex(ValueError, "positive finite stop delay"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_truncated_pre_go_baseline_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        assert epochs.metadata is not None
        epochs.metadata["stimulus_to_stop_signal_ms"] = 700.0
        lineage["stimulus_to_stop_signal_ms"] = 700.0
        with self.assertRaisesRegex(ValueError, "complete 200 ms pre-go baseline"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_truncated_measurement_window_fails_closed(self):
        epochs, lineage = synthetic_stop_epochs()
        epochs.crop(tmax=0.4)
        with self.assertRaisesRegex(ValueError, "complete 250-450 ms"):
            participant_endpoint(epochs, lineage, participant_id="001")

    def test_group_summary_is_deterministic_and_exact(self):
        rows = [
            {
                "eligible": eligible,
                "exclusion_reason": (
                    ""
                    if eligible
                    else "fewer_than_8_retained_epochs_in_one_or_both_conditions"
                ),
                "response_absent_epochs": 10 if eligible else 7,
                "response_present_epochs": 10 if eligible else 9,
                "contrast_absent_minus_present_uv": contrast,
                "response_present_probability": probability,
                "ssd_present_minus_absent_ms": difference,
                "failed_stop_rt_shorter_than_go_rt": ordering,
                "stop_trials_total": 40,
                "unclassified_stop_trials": 1,
                "classified_stop_fraction": 39 / 40,
            }
            for eligible, contrast, probability, difference, ordering in (
                (True, 1.0, 0.40, 80.0, True),
                (True, 2.0, 0.50, 100.0, True),
                (True, 3.0, 0.60, 120.0, False),
                (False, None, 0.50, 90.0, True),
            )
        ]
        first = summarize_endpoint(rows)
        second = summarize_endpoint(rows)
        self.assertEqual(first, second)
        self.assertEqual(first["participants_eligible"], 3)
        self.assertEqual(first["participants_excluded"], 1)
        self.assertEqual(
            first["participant_exclusion_reasons"],
            {"fewer_than_8_retained_epochs_in_one_or_both_conditions": 1},
        )
        self.assertEqual(
            first["epoch_accounting"],
            {
                "all_participants": {
                    "response_absent": 37,
                    "response_present": 39,
                },
                "eligible_participants": {
                    "response_absent": 30,
                    "response_present": 30,
                },
                "excluded_participants": {
                    "response_absent": 7,
                    "response_present": 9,
                },
            },
        )
        self.assertEqual(first["participant_contrast_sd_uv"], 1.0)
        self.assertEqual(
            first["behavioral_diagnostics"]["unclassified_stop_trials"], 4
        )
        self.assertEqual(
            first["behavioral_diagnostics"]["failed_stop_rt_shorter_than_go_rt"],
            {
                "passed": 3,
                "assessed": 4,
                "interpretation": (
                    "Necessary ordering diagnostic for the independent race model; "
                    "not proof that all model assumptions hold"
                ),
            },
        )
        self.assertEqual(exact_sign_flip_p(np.array([1.0, 2.0])), 0.5)

    def test_endpoint_package_detects_tampering_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ENDPOINT_OUTPUT_FILES:
                (root / name).write_bytes(name.encode("ascii"))
            outputs = [
                {
                    "path": name,
                    "size_bytes": (root / name).stat().st_size,
                    "sha256": sha256_file(root / name),
                }
                for name in sorted(ENDPOINT_OUTPUT_FILES)
            ]
            core = {"schema_version": "1", "outputs": outputs}
            payload = {**core, "core_sha256": canonical_sha256(core)}
            (root / "endpoint_provenance.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            verify_endpoint_package(root)
            (root / "endpoint_summary.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_endpoint_package(root)

    def test_cli_runs_fixed_ten_recording_endpoint_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_synthetic_dataset(root)
            first = root / "endpoint-first"
            second = root / "endpoint-second"
            environment = {
                **os.environ,
                "HOME": str(root / "home"),
                "MPLCONFIGDIR": str(root / "matplotlib"),
            }
            for output in (first, second):
                subprocess.run(
                    [
                        sys.executable,
                        str(ENDPOINT_SCRIPT),
                        "--dataset-output",
                        str(dataset),
                        "--output",
                        str(output),
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                verify_endpoint_package(output)
            self.assertEqual(tree_hashes(first), tree_hashes(second))
            summary = json.loads(
                (first / "endpoint_summary.json").read_text(encoding="utf-8")
            )
            provenance = json.loads(
                (first / "endpoint_provenance.json").read_text(encoding="utf-8")
            )
            runtime = provenance["runtime"]
            self.assertEqual(runtime["python"], platform.python_version())
            self.assertEqual(runtime["platform"], platform.platform())
            self.assertEqual(
                runtime["dependency_file_sha256"],
                {
                    name: sha256_file(PROJECT_ROOT / name)
                    for name in ("requirements.txt", "constraints-ci.txt")
                },
            )
            self.assertEqual(summary["participants_total"], 10)
            self.assertEqual(summary["participants_eligible"], 10)
            self.assertAlmostEqual(summary["mean_contrast_uv"], 1.5)
            diagnostics = summary["behavioral_diagnostics"]
            self.assertEqual(diagnostics["mean_response_present_probability"], 0.5)
            self.assertEqual(
                diagnostics["mean_within_participant_ssd_present_minus_absent_ms"],
                100.0,
            )
            self.assertEqual(
                diagnostics["failed_stop_rt_shorter_than_go_rt"]["passed"], 10
            )

    def test_cli_rejects_output_inside_preprocessing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_synthetic_dataset(root)
            output = dataset / "endpoint"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENDPOINT_SCRIPT),
                    "--dataset-output",
                    str(dataset),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "HOME": str(root / "home"),
                    "MPLCONFIGDIR": str(root / "matplotlib"),
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("outside the preprocessing dataset", completed.stderr)
            self.assertFalse(output.exists())

    def test_cli_rejects_undeclared_dataset_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_synthetic_dataset(root)
            (dataset / "stale-result.txt").write_text("stale", encoding="utf-8")
            output = root / "endpoint"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENDPOINT_SCRIPT),
                    "--dataset-output",
                    str(dataset),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                env={
                    **os.environ,
                    "HOME": str(root / "home"),
                    "MPLCONFIGDIR": str(root / "matplotlib"),
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("top-level output set is not exact", completed.stderr)
            self.assertFalse(output.exists())

    def test_cli_requires_reviewed_intervals_and_ica_for_every_participant(self):
        for field, message in (
            ("interval_review", "Completed fixed-study interval review"),
            ("ica_inputs", "Reviewed ICA application"),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                dataset = write_synthetic_dataset(root)
                package = dataset / "sub-001"
                provenance_path = package / "provenance.json"
                payload = json.loads(provenance_path.read_text(encoding="utf-8"))
                outputs = payload.pop("outputs")
                payload.pop("core_sha256")
                payload.pop(field)
                payload["core_sha256"] = canonical_sha256(payload)
                payload["outputs"] = outputs
                provenance_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                dataset_path = dataset / "dataset_provenance.json"
                dataset_payload = json.loads(
                    dataset_path.read_text(encoding="utf-8")
                )
                first = dataset_payload["participants"][0]
                first["provenance_sha256"] = sha256_file(provenance_path)
                first["core_sha256"] = payload["core_sha256"]
                dataset_path.write_text(
                    json.dumps(dataset_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                output = root / "endpoint"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ENDPOINT_SCRIPT),
                        "--dataset-output",
                        str(dataset),
                        "--output",
                        str(output),
                    ],
                    cwd=PROJECT_ROOT,
                    env={
                        **os.environ,
                        "HOME": str(root / "home"),
                        "MPLCONFIGDIR": str(root / "matplotlib"),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
