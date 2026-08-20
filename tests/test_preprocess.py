from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.preprocess import _event_array, preprocess_recording
from hunt_eeg.fixed_study_intervals import load_fixed_study_interval_manifest


def synthetic_recording() -> mne.io.RawArray:
    sfreq = 100.0
    duration_s = 30.0
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "EOG", "ECG"]
    times = np.arange(int(sfreq * duration_s)) / sfreq
    rng = np.random.default_rng(20260713)
    data = []
    for index, _ in enumerate(channels):
        signal = (
            (1.0 + 0.1 * index) * 1e-6 * np.sin(2 * np.pi * (6 + index % 3) * times)
        )
        data.append(signal + rng.normal(scale=0.2e-6, size=len(times)))

    info = mne.create_info(channels, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.asarray(data), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[3.0, 3.4, 8.0, 8.5, 13.0, 13.25, 13.6, 20.0, 20.25, 25.0, 25.4],
            duration=[0.0] * 11,
            description=[
                "Stimulus/S 17",
                "Stimulus/S 6",
                "Stimulus/S 18",
                "Stimulus/S 4",
                "Stimulus/S 1",
                "Stimulus/S 19",
                "Stimulus/S 5",
                "Stimulus/S 2",
                "Stimulus/S 19",
                "Stimulus/S 17",
                "Stimulus/S 6",
            ],
        )
    )
    return raw


def brainvision_stub(root: Path) -> Path:
    header = root / "synthetic.vhdr"
    marker_file = root / "synthetic.vmrk"
    signal_file = root / "synthetic.eeg"
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "DataFile=synthetic.eeg\n"
        "MarkerFile=synthetic.vmrk\n",
        encoding="utf-8",
    )
    marker_file.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n",
        encoding="utf-8",
    )
    signal_file.write_bytes(b"synthetic signal placeholder")
    return header


class PreprocessingTests(unittest.TestCase):
    def test_empty_event_array_keeps_mne_shape(self):
        raw = synthetic_recording()
        events, event_id, metadata = _event_array(
            raw,
            pd.DataFrame(columns=["trial_class", "stimulus_onset_s"]),
            "stimulus_onset_s",
            ["go_correct"],
        )

        self.assertEqual(events.shape, (0, 3))
        self.assertEqual(event_id, {"go_correct": 1})
        self.assertTrue(metadata.empty)

    def test_repeated_synthetic_run_reproduces_output_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output_one = root / "processed-one"
            output_two = root / "processed-two"
            recordings = [synthetic_recording(), synthetic_recording()]
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat",
                side_effect=recordings,
            ):
                preprocess_recording(
                    header,
                    output_one,
                    participant_id="test",
                    export_eeglab=False,
                )
                preprocess_recording(
                    header,
                    output_two,
                    participant_id="test",
                    export_eeglab=False,
                )

            first = json.loads(
                (output_one / "provenance.json").read_text(encoding="utf-8")
            )
            second = json.loads(
                (output_two / "provenance.json").read_text(encoding="utf-8")
            )

        self.assertEqual(first["core_sha256"], second["core_sha256"])
        self.assertEqual(first["outputs"], second["outputs"])

    def test_existing_output_files_are_not_mixed_into_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "processed"
            output.mkdir()
            (output / "old-result.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "new output path"):
                preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    export_eeglab=False,
                )

    def test_participant_id_cannot_redirect_output_paths(self):
        invalid_ids = ["", ".", "..", "../escape", "a/b", "a\\b", "/absolute", "é"]
        for participant_id in invalid_ids:
            with (
                self.subTest(participant_id=participant_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "processed"
                with (
                    patch("hunt_eeg.preprocess._read_brainvision_compat") as reader,
                    self.assertRaisesRegex(
                        ValueError,
                        "Participant ID must be",
                    ),
                ):
                    preprocess_recording(
                        Path("synthetic.vhdr"),
                        output,
                        participant_id=participant_id,
                        export_eeglab=False,
                    )
                reader.assert_not_called()
                self.assertFalse(output.exists())

    def test_failed_run_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "processed"
            with (
                patch(
                    "hunt_eeg.preprocess._read_brainvision_compat",
                    side_effect=RuntimeError("reader failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "reader failed"),
            ):
                preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    export_eeglab=False,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".processed.tmp-*")), [])

    def test_unverified_temporary_package_is_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "processed"
            with (
                patch(
                    "hunt_eeg.preprocess._read_brainvision_compat",
                    return_value=synthetic_recording(),
                ),
                patch(
                    "hunt_eeg.preprocess.verify_provenance",
                    side_effect=ValueError("verification failed"),
                ),
                self.assertRaisesRegex(ValueError, "verification failed"),
            ):
                preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    export_eeglab=False,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".processed.tmp-*")), [])

    def test_ica_solution_and_decisions_are_required_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "processed"
            solution = root / "ica_solution.fif"
            solution.write_bytes(b"solution")

            with self.assertRaisesRegex(ValueError, "provided together"):
                preprocess_recording(
                    root / "unused.vhdr",
                    output,
                    participant_id="test",
                    ica_solution_path=solution,
                    export_eeglab=False,
                )

            self.assertFalse(output.exists())

    def test_reviewed_ica_runs_before_ecg_drop_and_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "processed"
            solution = root / "ica_solution.fif"
            solution.write_bytes(b"solution")
            (root / "provenance.json").write_text("{}\n", encoding="utf-8")
            decisions = root / "ica_decisions.csv"
            decisions.write_text("reviewed decisions\n", encoding="utf-8")
            bad_channel_decisions = [
                {
                    "channel": "Fp2",
                    "decision": "interpolate",
                    "reason": "persistent artifact",
                    "reviewer": "reviewer",
                    "reviewed_at": "2026-08-12",
                    "evidence_windows": "0-20",
                }
            ]

            def apply_ica(raw, **kwargs):
                self.assertIn("ECG", raw.ch_names)
                self.assertIn("Fp2", raw.info["bads"])
                self.assertEqual(kwargs["participant_id"], "test")
                return raw.copy(), {
                    "status": "applied reviewed decisions",
                    "excluded_components": [1],
                    "decision_record_complete": True,
                    "automatic_exclusion": False,
                }

            with (
                patch(
                    "hunt_eeg.preprocess._read_brainvision_compat",
                    return_value=synthetic_recording(),
                ),
                patch(
                    "hunt_eeg.preprocess.apply_reviewed_ica",
                    side_effect=apply_ica,
                ),
            ):
                summary = preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    bad_channel_decisions=bad_channel_decisions,
                    ica_solution_path=solution,
                    ica_decision_path=decisions,
                    export_eeglab=False,
                )

            continuous = mne.io.read_raw_fif(
                output / "sub-test_continuous_0.2-30Hz_avgref_raw.fif",
                preload=False,
                verbose="ERROR",
            )

        self.assertEqual(summary["ica"]["excluded_components"], [1])
        self.assertNotIn("ECG", continuous.ch_names)
        self.assertEqual(summary["interpolated_bad_channels"], ["Fp2"])

    def test_keep_decision_channel_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "processed"
            decisions = [
                {
                    "channel": "TYPO",
                    "decision": "keep",
                    "reason": "reviewed",
                    "reviewer": "reviewer",
                    "reviewed_at": "2026-08-11",
                    "evidence_windows": "0-10 s",
                }
            ]
            with (
                patch(
                    "hunt_eeg.preprocess._read_brainvision_compat",
                    return_value=synthetic_recording(),
                ),
                self.assertRaisesRegex(ValueError, "not present"),
            ):
                preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    bad_channel_decisions=decisions,
                    export_eeglab=False,
                )
            self.assertFalse(output.exists())

    def test_synthetic_recording_runs_through_preprocessing(self):
        raw = synthetic_recording()
        with tempfile.TemporaryDirectory() as directory:
            header = brainvision_stub(Path(directory))
            output = Path(directory) / "processed"
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat",
                return_value=raw,
            ):
                summary = preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    bad_channels=["Fp2"],
                    export_eeglab=True,
                )

            self.assertTrue(summary["ecg_removed"])
            self.assertTrue(summary["eog_retained_and_excluded_from_reference"])
            self.assertEqual(summary["interpolated_bad_channels"], ["Fp2"])
            self.assertEqual(
                summary["go_epochs"],
                {"go_correct": 2, "go_incorrect_or_slow": 1},
            )
            self.assertEqual(
                summary["stop_epochs"],
                {"stop_successful": 1, "stop_failed": 1},
            )
            self.assertEqual(summary["go_epoch_accounting"]["proposed_events"], 3)
            self.assertEqual(summary["go_epoch_accounting"]["retained_epochs"], 3)
            self.assertTrue(summary["go_epoch_accounting"]["accounting_complete"])

            continuous = mne.io.read_raw_fif(
                output / "sub-test_continuous_0.2-30Hz_avgref_raw.fif",
                preload=True,
                verbose="ERROR",
            )
            self.assertNotIn("ECG", continuous.ch_names)
            self.assertEqual(continuous.get_channel_types(picks=["EOG"]), ["eog"])
            self.assertEqual(continuous.info["bads"], [])
            self.assertAlmostEqual(continuous.info["highpass"], 0.2)
            self.assertAlmostEqual(continuous.info["lowpass"], 30.0)
            eeg = continuous.get_data(picks="eeg")
            self.assertTrue(np.allclose(eeg.mean(axis=0), 0.0, atol=1e-10))

            go_epochs = mne.read_epochs(
                output / "sub-test_go-epo.fif",
                preload=False,
                verbose="ERROR",
            )
            stop_epochs = mne.read_epochs(
                output / "sub-test_stop-epo.fif",
                preload=False,
                verbose="ERROR",
            )
            self.assertEqual(len(go_epochs), 3)
            self.assertEqual(len(stop_epochs), 2)
            self.assertIsNotNone(go_epochs.metadata)
            self.assertIn("trial_index", go_epochs.metadata.columns)

            for filename in [
                "sub-test_continuous_0.2-30Hz_avgref.set",
                "sub-test_go.set",
                "sub-test_stop.set",
                "figures/c3_c4_go_lowpass12.png",
                "figures/c3_c4_stop_lowpass12.png",
                "preprocessing_summary.json",
                "reconstructed_trials.csv",
                "go_epoch_lineage.csv",
                "stop_epoch_lineage.csv",
                "provenance.json",
            ]:
                self.assertTrue((output / filename).is_file(), filename)
            provenance = json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertNotIn(str(Path(directory)), json.dumps(provenance))
            self.assertNotIn(
                "provenance.json", {item["path"] for item in provenance["outputs"]}
            )
            self.assertEqual(len(provenance["core_sha256"]), 64)

    def test_bad_annotations_are_preserved_for_epoch_rejection(self):
        raw = synthetic_recording()
        raw.annotations.append(3.1, 0.2, "BAD_motion")
        with tempfile.TemporaryDirectory() as directory:
            header = brainvision_stub(Path(directory))
            output = Path(directory) / "processed"
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat",
                return_value=raw,
            ):
                summary = preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    export_eeglab=False,
                )

        accounting = summary["go_epoch_accounting"]
        self.assertEqual(accounting["proposed_events"], 3)
        self.assertEqual(accounting["retained_epochs"], 2)
        self.assertEqual(accounting["dropped_epochs"], 1)
        self.assertEqual(accounting["drop_reasons"], {"BAD_motion": 1})

    def test_bad_annotation_containing_marker_text_is_preserved(self):
        raw = synthetic_recording()
        raw.annotations.append(3.1, 0.2, "BAD_S17_motion")
        with tempfile.TemporaryDirectory() as directory:
            header = brainvision_stub(Path(directory))
            output = Path(directory) / "processed"
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat", return_value=raw
            ):
                summary = preprocess_recording(
                    header, output, participant_id="test", export_eeglab=False
                )
        self.assertEqual(
            summary["go_epoch_accounting"]["drop_reasons"],
            {"BAD_S17_motion": 1},
        )

    def test_reviewed_interval_and_fir_support_drop_every_overlapping_epoch(self):
        raw = synthetic_recording()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            interval_path = root / "fixed_intervals.csv"
            interval_path.write_text(
                "participant_id,interval_id,decision,start_s,stop_s,scope,reason,reviewer,reviewed_at,evidence\n"
                "test,artifact01,exclude,2.9,3.1,epochs,reviewed transient,reviewer,2026-08-18,raw zoom\n",
                encoding="utf-8",
            )
            manifest = load_fixed_study_interval_manifest(interval_path)
            output = root / "processed"
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat",
                return_value=raw,
            ):
                summary = preprocess_recording(
                    header,
                    output,
                    participant_id="test",
                    interval_manifest=manifest,
                    export_eeglab=False,
                )
            provenance = json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            )
            go_lineage = pd.read_csv(
                output / "go_epoch_lineage.csv", keep_default_na=False
            )

        self.assertEqual(summary["go_epoch_accounting"]["retained_epochs"], 1)
        self.assertEqual(summary["go_epoch_accounting"]["dropped_epochs"], 2)
        self.assertEqual(summary["stop_epoch_accounting"]["retained_epochs"], 1)
        self.assertEqual(summary["stop_epoch_accounting"]["dropped_epochs"], 1)
        self.assertIn(
            "BAD_fixed_epochs_artifact01",
            summary["go_epoch_accounting"]["drop_reasons"],
        )
        self.assertEqual(
            set(
                go_lineage.loc[
                    ~go_lineage["retained"], "reviewed_interval_reasons"
                ]
            ),
            {"artifact01: reviewed transient"},
        )
        self.assertEqual(
            provenance["interval_review"]["identity"]["manifest_sha256"],
            manifest.sha256,
        )
        self.assertGreater(
            summary["interval_review"]["fir_half_support_guard_s"], 8.0
        )


if __name__ == "__main__":
    unittest.main()
