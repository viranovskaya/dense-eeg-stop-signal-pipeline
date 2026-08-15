"""Determinism and privacy checks for run provenance."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.config import load_analysis_config
from hunt_eeg.provenance import (
    build_provenance_core,
    verify_provenance,
    write_provenance,
)


def brainvision_stub(root: Path) -> Path:
    header = root / "private-name_003_recording.vhdr"
    marker_file = root / "private-name_003_recording.vmrk"
    signal_file = root / "private-name_003_recording.eeg"
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={signal_file.name}\n"
        f"MarkerFile={marker_file.name}\n",
        encoding="utf-8",
    )
    marker_file.write_text("synthetic markers\n", encoding="utf-8")
    signal_file.write_bytes(b"synthetic signal")
    return header


class ProvenanceTests(unittest.TestCase):
    def test_output_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "output"
            output.mkdir()
            artifact = output / "artifact.txt"
            artifact.write_text("original", encoding="utf-8")
            core = build_provenance_core(
                vhdr=header,
                participant_id="001",
                analysis=load_analysis_config(),
                bad_channel_decisions=[],
            )
            write_provenance(output, core)
            verify_provenance(output)
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output mismatch"):
                verify_provenance(output)

    def test_extra_output_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "output"
            output.mkdir()
            (output / "artifact.txt").write_text("original", encoding="utf-8")
            core = build_provenance_core(
                vhdr=header,
                participant_id="001",
                analysis=load_analysis_config(),
                bad_channel_decisions=[],
            )
            write_provenance(output, core)
            (output / "injected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output set"):
                verify_provenance(output)

    def test_unsafe_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "output"
            output.mkdir()
            (output / "artifact.txt").write_text("original", encoding="utf-8")
            core = build_provenance_core(
                vhdr=header,
                participant_id="001",
                analysis=load_analysis_config(),
                bad_channel_decisions=[],
            )
            provenance_path = write_provenance(output, core)
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            payload["outputs"][0]["path"] = "../artifact.txt"
            provenance_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_provenance(output)

    def test_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "output"
            output.mkdir()
            external = root / "external.txt"
            external.write_text("outside", encoding="utf-8")
            link = output / "link.txt"
            link.symlink_to(external)
            core = build_provenance_core(
                vhdr=header,
                participant_id="001",
                analysis=load_analysis_config(),
                bad_channel_decisions=[],
            )
            write_provenance(output, core)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_provenance(output)

    def test_symlink_directory_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            output = root / "output"
            output.mkdir()
            (output / "artifact.txt").write_text("original", encoding="utf-8")
            core = build_provenance_core(
                vhdr=header,
                participant_id="001",
                analysis=load_analysis_config(),
                bad_channel_decisions=[],
            )
            write_provenance(output, core)
            external = root / "external"
            external.mkdir()
            (output / "linked-directory").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_provenance(output)

    def test_symlink_output_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "linked-output"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output root symlink"):
                verify_provenance(link)
    def test_core_is_stable_and_does_not_expose_source_names_or_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = brainvision_stub(root)
            arguments = {
                "vhdr": header,
                "participant_id": "003",
                "analysis": load_analysis_config(),
                "bad_channel_decisions": [
                    {
                        "channel": "E42",
                        "decision": "interpolate",
                        "reason": "persistent flat trace",
                        "reviewer": "reviewer-1",
                        "reviewed_at": "2026-08-11",
                        "evidence_windows": "0-20;120-140;240-260",
                    }
                ],
            }
            first = build_provenance_core(**arguments)
            second = build_provenance_core(**arguments)

            encoded = json.dumps(first, sort_keys=True)
            self.assertEqual(first, second)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("private-name", encoded)
            self.assertTrue(first["decision_record_complete"])
            self.assertIn("git_dirty", first["software"])
            self.assertIn("Version", first["GeneratedBy"][0])
            self.assertEqual(len(first["core_sha256"]), 64)
            self.assertEqual(
                [source["role"] for source in first["inputs"]],
                ["header", "marker", "signal"],
            )

    def test_explicit_no_bad_channels_counts_as_a_complete_review(self):
        with tempfile.TemporaryDirectory() as directory:
            header = brainvision_stub(Path(directory))
            core = build_provenance_core(
                vhdr=header,
                participant_id="003",
                analysis=load_analysis_config(),
                bad_channel_decisions=[
                    {
                        "channel": "",
                        "decision": "none",
                        "reason": "no persistent bad channels",
                        "reviewer": "reviewer-1",
                        "reviewed_at": "2026-08-11",
                        "evidence_windows": "0-20;120-140;240-260",
                    }
                ],
            )

        self.assertTrue(core["decision_record_complete"])


if __name__ == "__main__":
    unittest.main()
