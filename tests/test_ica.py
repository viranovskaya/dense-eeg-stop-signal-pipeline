from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.ica import (
    ICA_DECISION_COLUMNS,
    apply_reviewed_ica,
    component_diagnostics,
    fit_ica,
    ica_decimation,
    load_ica_config,
    load_ica_decisions,
    prepare_ica_input,
    run_ica_review,
)
from hunt_eeg.fixed_study_intervals import load_fixed_study_interval_manifest
from hunt_eeg.provenance import verify_provenance


class _Sources:
    def __init__(self, data: np.ndarray):
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


class ICAReviewCLITests(unittest.TestCase):
    def test_review_cli_forces_headless_matplotlib_backend(self):
        script = PROJECT_ROOT / "scripts" / "run_ica_review.py"
        code = (
            "import importlib.util, matplotlib; "
            f"spec=importlib.util.spec_from_file_location('review_cli', {str(script)!r}); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "assert matplotlib.get_backend().lower() == 'agg'"
        )
        environment = {**os.environ, "MPLBACKEND": "MacOSX"}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class _FakeICA:
    n_components_ = 3
    n_iter_ = 42
    ch_names = ("F3", "F4")

    def __init__(self):
        self.exclude = []

    def score_sources(self, raw, target, score_func, verbose):
        del raw, score_func, verbose
        if target == "EOG":
            return np.array([0.1, 0.7, -0.2])
        return np.array([0.05, 0.1, -0.8])

    def get_sources(self, raw):
        del raw
        return _Sources(
            np.array(
                [
                    [1.0, -1.0, 1.0, -1.0],
                    [2.0, -2.0, 2.0, -2.0],
                    [3.0, -3.0, 3.0, -3.0],
                ]
            )
        )

    def apply(self, raw, exclude, verbose):
        del verbose
        self.exclude = list(exclude)
        raw._data[0] += len(exclude)
        return raw

    def save(self, path, overwrite, verbose):
        del overwrite, verbose
        Path(path).write_bytes(b"fixed ICA solution")


class _NonConvergedICA:
    n_iter_ = 1000

    def fit(self, *args, **kwargs):
        del args, kwargs
        return self


def _raw() -> mne.io.RawArray:
    info = mne.create_info(
        ["F3", "F4", "EOG", "ECG"],
        100.0,
        ch_types=["eeg", "eeg", "eog", "ecg"],
    )
    return mne.io.RawArray(np.zeros((4, 100)), info, verbose="ERROR")


def _decision_table(path: Path, solution_sha256: str) -> None:
    rows = []
    for component in range(3):
        rows.append(
            {
                "participant_id": "test",
                "ica_solution_sha256": solution_sha256,
                "component": str(component),
                "decision": "exclude" if component == 1 else "keep",
                "reason": "ocular pattern" if component == 1 else "neural pattern",
                "reviewer": "reviewer",
                "reviewed_at": "2026-08-12",
                "evidence": "topography; time course; spectrum",
            }
        )
    pd.DataFrame(rows, columns=ICA_DECISION_COLUMNS).to_csv(path, index=False)


def _brainvision_stub(root: Path) -> Path:
    header = root / "private-name.vhdr"
    (root / "private-name.eeg").write_bytes(b"signal")
    (root / "private-name.vmrk").write_text("marker", encoding="utf-8")
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "DataFile=private-name.eeg\n"
        "MarkerFile=private-name.vmrk\n",
        encoding="utf-8",
    )
    return header


class ICAWorkflowTests(unittest.TestCase):
    def test_ica_input_uses_the_declared_ica_filter(self):
        raw = _raw()
        decisions = [
            {
                "channel": "F3",
                "decision": "keep",
                "reason": "reviewed and retained",
                "reviewer": "reviewer",
                "reviewed_at": "2026-08-17",
                "evidence_windows": "full recording",
            }
        ]
        with patch("hunt_eeg.ica._read_brainvision_compat", return_value=raw):
            prepared = prepare_ica_input(Path("unused.vhdr"), decisions)

        self.assertEqual(prepared.info["highpass"], 1.0)
        self.assertEqual(prepared.info["lowpass"], 40.0)

    def test_config_is_explicit_and_does_not_enable_automatic_exclusion(self):
        config = load_ica_config()

        self.assertEqual(config.method, "infomax")
        self.assertEqual(
            config.fit_params,
            {"extended": True, "n_small_angle": 20, "w_change": 0.0},
        )
        self.assertGreater(config.random_state, 0)

    def test_diagnostics_are_cues_not_decisions(self):
        diagnostics = component_diagnostics(_FakeICA(), _raw(), load_ica_config())

        self.assertEqual(diagnostics["candidate_review"].tolist(), [False, True, True])
        self.assertFalse(diagnostics["automatic_exclusion"].any())
        self.assertAlmostEqual(diagnostics["source_variance_fraction"].sum(), 1.0)

    def test_extended_infomax_fit_uses_reviewed_good_eeg_rank(self):
        sfreq = 100.0
        times = np.arange(1200) / sfreq
        rng = np.random.default_rng(19)
        names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "EOG", "ECG"]
        data = np.vstack(
            [
                np.sin(2 * np.pi * (7 + index * 0.3) * times) * 1e-6
                + rng.normal(scale=0.2e-6, size=len(times))
                for index in range(6)
            ]
            + [
                4e-6 * np.sin(2 * np.pi * 0.8 * times),
                3e-6 * np.sin(2 * np.pi * 1.2 * times),
            ]
        )
        raw = mne.io.RawArray(
            data,
            mne.create_info(
                names,
                sfreq,
                ch_types=["eeg"] * 6 + ["eog", "ecg"],
            ),
            verbose="ERROR",
        )
        raw.info["bads"] = ["Fp2"]
        raw.filter(None, 40.0, picks="data", verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

        ica = fit_ica(raw, load_ica_config())

        self.assertNotIn("Fp2", ica.ch_names)
        self.assertNotIn("EOG", ica.ch_names)
        self.assertNotIn("ECG", ica.ch_names)
        self.assertLessEqual(ica.n_components_, 5)
        self.assertGreater(ica.n_components_, 1)

    def test_fit_stops_when_small_angle_rule_is_not_reached(self):
        raw = _raw()
        with raw.info._unlock():
            raw.info["lowpass"] = 40.0
        with (
            patch("hunt_eeg.ica.mne.compute_rank", return_value={"eeg": 2}),
            patch(
                "hunt_eeg.ica.mne.preprocessing.ICA",
                return_value=_NonConvergedICA(),
            ),
            self.assertRaisesRegex(RuntimeError, "small-angle stopping rule"),
        ):
            fit_ica(raw, load_ica_config())

    def test_decimation_preserves_margin_above_the_lowpass(self):
        raw = mne.io.RawArray(
            np.zeros((2, 1000)),
            mne.create_info(["F3", "F4"], 1000.0, ch_types="eeg"),
            verbose="ERROR",
        )
        with raw.info._unlock():
            raw.info["lowpass"] = 40.0

        decim = ica_decimation(raw, load_ica_config())

        self.assertEqual(decim, 10)
        self.assertEqual(raw.info["sfreq"] / decim, 100.0)

    def test_every_component_requires_a_complete_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            _decision_table(path, "a" * 64)
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            table = table.iloc[:-1]
            table.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "exactly 3 rows"):
                load_ica_decisions(
                    path,
                    participant_id="test",
                    solution_sha256="a" * 64,
                    components=3,
                )

    def test_decisions_are_bound_to_the_exact_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            _decision_table(path, "a" * 64)

            with self.assertRaisesRegex(ValueError, "solution SHA-256"):
                load_ica_decisions(
                    path,
                    participant_id="test",
                    solution_sha256="b" * 64,
                    components=3,
                )

    def test_review_solution_must_match_the_current_preprocessing_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "ica_solution.fif"
            solution.write_bytes(b"solution")
            decisions = root / "decisions.csv"
            decisions.write_text("unused\n", encoding="utf-8")
            with (
                patch(
                    "hunt_eeg.ica.verify_provenance",
                    return_value={
                        "workflow": "ica_component_review",
                        "participant_id": "test",
                        "inputs": [{"sha256": "old"}],
                        "configuration": {"analysis_sha256": "analysis"},
                        "bad_channel_decisions": [],
                        "software": {"source_manifest": {"sha256": "source"}},
                    },
                ),
                self.assertRaisesRegex(ValueError, "does not match"),
            ):
                apply_reviewed_ica(
                    _raw(),
                    participant_id="test",
                    solution_path=solution,
                    decision_path=decisions,
                    expected_context={
                        "inputs": [{"sha256": "current"}],
                        "analysis_sha256": "analysis",
                        "bad_channel_decisions": [],
                        "source_manifest_sha256": "source",
                    },
                )

    def test_reviewed_decisions_are_the_only_exclusions_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "ica_solution.fif"
            solution.write_bytes(b"fixed ICA solution")
            (root / "provenance.json").write_text("{}\n", encoding="utf-8")
            solution_sha256 = hashlib.sha256(solution.read_bytes()).hexdigest()
            decisions = root / "decisions.csv"
            _decision_table(decisions, solution_sha256)
            original = _raw()
            fake = _FakeICA()

            with (
                patch("mne.preprocessing.read_ica", return_value=fake),
                patch(
                    "hunt_eeg.ica.verify_provenance",
                    return_value={
                        "workflow": "ica_component_review",
                        "participant_id": "test",
                        "inputs": [],
                        "configuration": {"analysis_sha256": "analysis"},
                        "bad_channel_decisions": [],
                        "software": {"source_manifest": {"sha256": "source"}},
                    },
                ),
                patch(
                    "hunt_eeg.ica.sha256_file",
                    side_effect=lambda path: (
                        solution_sha256
                        if Path(path).name == "ica_solution.fif"
                        else hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    ),
                ),
            ):
                cleaned, summary = apply_reviewed_ica(
                    original,
                    participant_id="test",
                    solution_path=solution,
                    decision_path=decisions,
                    expected_context={
                        "inputs": [],
                        "analysis_sha256": "analysis",
                        "bad_channel_decisions": [],
                        "source_manifest_sha256": "source",
                    },
                )

        self.assertEqual(fake.exclude, [1])
        self.assertEqual(summary["excluded_components"], [1])
        self.assertTrue(summary["decision_record_complete"])
        self.assertFalse(summary["automatic_exclusion"])
        self.assertTrue(np.all(original.get_data() == 0))
        self.assertTrue(np.all(cleaned.get_data(picks=["F3"]) == 1))

    def test_review_solution_must_match_fixed_interval_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "ica_solution.fif"
            solution.write_bytes(b"solution")
            decisions = root / "decisions.csv"
            decisions.write_text("unused\n", encoding="utf-8")
            base = {
                "workflow": "ica_component_review",
                "participant_id": "test",
                "inputs": [],
                "configuration": {"analysis_sha256": "analysis"},
                "bad_channel_decisions": [],
                "software": {"source_manifest": {"sha256": "source"}},
                "interval_review": {"identity": {"manifest_sha256": "old"}},
            }
            with (
                patch("hunt_eeg.ica.verify_provenance", return_value=base),
                self.assertRaisesRegex(ValueError, "does not match"),
            ):
                apply_reviewed_ica(
                    _raw(),
                    participant_id="test",
                    solution_path=solution,
                    decision_path=decisions,
                    expected_context={
                        "inputs": [],
                        "analysis_sha256": "analysis",
                        "bad_channel_decisions": [],
                        "source_manifest_sha256": "source",
                        "interval_review_identity": {"manifest_sha256": "new"},
                    },
                )

    def test_review_package_is_atomic_private_and_provenance_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = _brainvision_stub(root)
            output = root / "ica-review"
            decisions = [
                {
                    "participant_id": "test",
                    "channel": "",
                    "decision": "none",
                    "reason": "no persistent bad channels",
                    "reviewer": "reviewer",
                    "reviewed_at": "2026-08-12",
                    "evidence_windows": "0-20",
                }
            ]
            prepared = _raw()
            prepared.filter(None, 40.0, picks="data", verbose="ERROR")
            interval_path = root / "intervals.csv"
            interval_path.write_text(
                "participant_id,interval_id,decision,start_s,stop_s,scope,reason,reviewer,reviewed_at,evidence\n"
                "test,artifact01,exclude,0.2,0.4,both,reviewed transient,reviewer,2026-08-18,raw zoom\n",
                encoding="utf-8",
            )
            interval_manifest = load_fixed_study_interval_manifest(interval_path)
            with (
                patch("hunt_eeg.ica.prepare_ica_input", return_value=prepared),
                patch("hunt_eeg.ica.fit_ica", return_value=_FakeICA()),
                patch("hunt_eeg.ica.save_review_figures", return_value=[]),
            ):
                summary = run_ica_review(
                    header,
                    output,
                    participant_id="test",
                    bad_channel_decisions=decisions,
                    interval_manifest=interval_manifest,
                )
            provenance = verify_provenance(output)

        self.assertEqual(summary["components"], 3)
        self.assertIn("Private derived output", summary["privacy"])
        self.assertEqual(provenance["workflow"], "ica_component_review")
        self.assertEqual(
            provenance["interval_review"]["identity"]["manifest_sha256"],
            interval_manifest.sha256,
        )
        self.assertEqual(
            provenance["interval_review"]["application_target"], "ica"
        )
        self.assertNotIn("private-name", str(provenance))


if __name__ == "__main__":
    unittest.main()
