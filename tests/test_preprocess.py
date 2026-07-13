from __future__ import annotations

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


sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.preprocess import preprocess_recording


def synthetic_recording() -> mne.io.RawArray:
    sfreq = 100.0
    duration_s = 30.0
    channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "EOG", "ECG"]
    times = np.arange(int(sfreq * duration_s)) / sfreq
    rng = np.random.default_rng(20260713)
    data = []
    for index, _ in enumerate(channels):
        signal = (1.0 + 0.1 * index) * 1e-6 * np.sin(
            2 * np.pi * (6 + index % 3) * times
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


class PreprocessingTests(unittest.TestCase):
    def test_synthetic_recording_runs_through_preprocessing(self):
        raw = synthetic_recording()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed"
            with patch(
                "hunt_eeg.preprocess._read_brainvision_compat",
                return_value=raw,
            ):
                summary = preprocess_recording(
                    Path("synthetic.vhdr"),
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

            continuous = mne.io.read_raw_fif(
                output / "sub-test_continuous_1-40Hz_avgref_raw.fif",
                preload=True,
                verbose="ERROR",
            )
            self.assertNotIn("ECG", continuous.ch_names)
            self.assertEqual(continuous.get_channel_types(picks=["EOG"]), ["eog"])
            self.assertEqual(continuous.info["bads"], [])
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

            for filename in [
                "sub-test_continuous_1-40Hz_avgref.set",
                "sub-test_go.set",
                "sub-test_stop.set",
                "figures/c3_c4_go_lowpass12.png",
                "figures/c3_c4_stop_lowpass12.png",
                "preprocessing_summary.json",
                "reconstructed_trials.csv",
            ]:
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
