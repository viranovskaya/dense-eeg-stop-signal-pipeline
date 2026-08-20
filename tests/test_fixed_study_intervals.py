from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np

from hunt_eeg.fixed_study_intervals import (
    INTERVAL_COLUMNS,
    apply_reviewed_intervals,
    fir_half_support_seconds,
    load_fixed_study_interval_manifest,
)


def _write_manifest(path: Path, rows: list[list[str]]) -> Path:
    lines = [",".join(INTERVAL_COLUMNS), *(",".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _exclude(
    participant: str = "001",
    interval_id: str = "artifact01",
    start: str = "5.0",
    stop: str = "6.0",
    scope: str = "both",
) -> list[str]:
    return [
        participant,
        interval_id,
        "exclude",
        start,
        stop,
        scope,
        "reviewed transient",
        "reviewer",
        "2026-08-18",
        "raw panel and zoomed trace",
    ]


def _none(participant: str = "002") -> list[str]:
    return [
        participant,
        "none",
        "none",
        "",
        "",
        "",
        "no reviewed temporal exclusions",
        "reviewer",
        "2026-08-18",
        "full candidate review",
    ]


class FixedStudyIntervalTests(unittest.TestCase):
    def test_current_1000_hz_filters_have_declared_half_support(self):
        self.assertAlmostEqual(
            fir_half_support_seconds(1000.0, (1.0, 40.0)), 1.65
        )
        self.assertAlmostEqual(
            fir_half_support_seconds(1000.0, (0.2, 30.0)), 8.25
        )

    def test_complete_manifest_is_captured_and_bound_to_exact_participants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                Path(directory) / "intervals.csv",
                [_exclude(), _none()],
            )
            manifest = load_fixed_study_interval_manifest(
                path, expected_participants=["001", "002"]
            )

            identity = manifest.identity("002")

        self.assertEqual(manifest.participants, ("001", "002"))
        self.assertIsNone(identity["participant_decisions"][0]["start_s"])
        self.assertEqual(len(identity["manifest_sha256"]), 64)
        self.assertEqual(len(identity["participant_decisions_sha256"]), 64)

    def test_manifest_rejects_overlap_and_incomplete_participant_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlap = _write_manifest(
                root / "overlap.csv",
                [
                    _exclude(stop="7.0"),
                    _exclude(interval_id="artifact02", start="6.9", stop="8.0"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_fixed_study_interval_manifest(overlap)

            incomplete = _write_manifest(root / "incomplete.csv", [_exclude()])
            with self.assertRaisesRegex(ValueError, "missing=.*002"):
                load_fixed_study_interval_manifest(
                    incomplete, expected_participants=["001", "002"]
                )

    def test_none_row_cannot_be_mixed_with_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                Path(directory) / "mixed.csv",
                [_none("001"), _exclude()],
            )
            with self.assertRaisesRegex(ValueError, "cannot mix"):
                load_fixed_study_interval_manifest(path)

    def test_manifest_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(Path(directory) / "intervals.csv", [_exclude()])
            manifest = load_fixed_study_interval_manifest(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "changed during the run"):
                manifest.verify_unchanged()

    def test_scope_and_fir_half_support_control_applied_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                Path(directory) / "intervals.csv",
                [_exclude(scope="ica")],
            )
            manifest = load_fixed_study_interval_manifest(path)
            raw = mne.io.RawArray(
                np.zeros((2, 2000)),
                mne.create_info(["F3", "F4"], 100.0, ch_types="eeg"),
                verbose="ERROR",
            )

            application = apply_reviewed_intervals(
                raw,
                manifest,
                participant_id="001",
                target="ica",
                filter_hz=(1.0, 40.0),
            )
            guard = fir_half_support_seconds(100.0, (1.0, 40.0))

            self.assertAlmostEqual(application["fir_half_support_guard_s"], guard)
            self.assertAlmostEqual(
                application["applied_intervals"][0]["applied_start_s"],
                5.0 - guard,
            )
            self.assertEqual(
                list(raw.annotations.description), ["BAD_fixed_ica_artifact01"]
            )

            untouched = raw.copy().set_annotations(mne.Annotations([], [], []))
            epochs_application = apply_reviewed_intervals(
                untouched,
                manifest,
                participant_id="001",
                target="epochs",
                filter_hz=(0.2, 30.0),
            )
            self.assertEqual(epochs_application["applied_intervals"], [])
            self.assertEqual(len(untouched.annotations), 0)

    def test_interval_cannot_extend_beyond_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_manifest(
                Path(directory) / "intervals.csv",
                [_exclude(start="19.0", stop="21.0")],
            )
            manifest = load_fixed_study_interval_manifest(path)
            raw = mne.io.RawArray(
                np.zeros((1, 2000)),
                mne.create_info(["F3"], 100.0, ch_types="eeg"),
                verbose="ERROR",
            )
            with self.assertRaisesRegex(ValueError, "exceeds recording duration"):
                apply_reviewed_intervals(
                    raw,
                    manifest,
                    participant_id="001",
                    target="epochs",
                    filter_hz=(0.2, 30.0),
                )


if __name__ == "__main__":
    unittest.main()
