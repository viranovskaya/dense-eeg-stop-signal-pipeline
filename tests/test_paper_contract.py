"""Consistency checks for the manuscript evidence boundary."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json_pointer(payload, pointer: str):
    value = payload
    for token in pointer.removeprefix("/").split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _safe_repository_artifact(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Repository evidence path must be safe and relative")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Repository evidence path must not contain symlinks")
    resolved_root = root.resolve(strict=True)
    artifact = candidate.resolve(strict=True)
    if not artifact.is_relative_to(resolved_root) or not artifact.is_file():
        raise ValueError("Repository evidence must be a regular file inside the root")
    return artifact


class PaperContractTests(unittest.TestCase):
    def test_validation_matrix_has_closed_evidence_references(self):
        path = PROJECT_ROOT / "docs" / "paper_validation_matrix.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        scaffold = (PROJECT_ROOT / "docs" / "manuscript_scaffold.md").read_text(
            encoding="utf-8"
        )
        manuscript_path = _safe_repository_artifact(
            PROJECT_ROOT, Path(payload["manuscript_draft"])
        )
        manuscript_bytes = manuscript_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(manuscript_bytes).hexdigest(),
            payload["manuscript_draft_sha256"],
        )
        manuscript = re.sub(r"\s+", " ", manuscript_bytes.decode("utf-8"))
        assets_manifest_path = _safe_repository_artifact(
            PROJECT_ROOT, Path(payload["paper_assets_manifest"])
        )
        assets_manifest_bytes = assets_manifest_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(assets_manifest_bytes).hexdigest(),
            payload["paper_assets_manifest_sha256"],
        )
        assets_manifest = json.loads(assets_manifest_bytes)
        for relative, expected_sha256 in assets_manifest["sources"].items():
            source = _safe_repository_artifact(PROJECT_ROOT, Path(relative))
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), expected_sha256
            )
        for record in assets_manifest["outputs"]:
            asset = _safe_repository_artifact(
                PROJECT_ROOT,
                Path(payload["paper_assets_manifest"]).parent / record["path"],
            )
            self.assertEqual(asset.stat().st_size, record["size_bytes"])
            self.assertEqual(
                hashlib.sha256(asset.read_bytes()).hexdigest(), record["sha256"]
            )
        partitions = payload["evidence_partitions"]
        partition_ids = [item["id"] for item in partitions]
        self.assertEqual(len(partition_ids), len(set(partition_ids)))

        claim_ids = [item["id"] for item in payload["claims"]]
        self.assertEqual(len(claim_ids), len(set(claim_ids)))
        for claim in payload["claims"]:
            self.assertTrue(set(claim["evidence"]).issubset(partition_ids))
            if claim["status"] == "blocked":
                self.assertTrue(claim["blockers"])
            if claim["status"] == "not_supported":
                self.assertTrue(claim["prohibited_wording"])

        for partition in partitions:
            for key in ("repository_result", "repository_contract"):
                if key not in partition:
                    continue
                relative = Path(partition[key])
                artifact = _safe_repository_artifact(PROJECT_ROOT, relative)
                observed_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
                self.assertEqual(
                    observed_sha256,
                    partition[f"{key}_sha256"],
                )
                if key == "repository_result":
                    result = json.loads(artifact.read_text(encoding="utf-8"))
                    for binding in partition.get("metric_bindings", []):
                        self.assertEqual(
                            _json_pointer(result, binding["json_pointer"]),
                            binding["expected"],
                        )
                        self.assertIn(binding["scaffold_text"], scaffold)
                        self.assertIn(binding["scaffold_text"], manuscript)

        heldout = next(item for item in partitions if item["id"] == "synthetic_qc_benchmark")
        result = json.loads(
            (PROJECT_ROOT / heldout["repository_result"]).read_text(encoding="utf-8")
        )
        recordings = int(result["recordings"])
        units = result["evaluation_units_per_recording"]
        primary_total = recordings * int(units["primary_channel_windows"])
        primary_positives = recordings * int(
            units["primary_positive_channel_windows"]
        )
        temporal_total = recordings * int(units["temporal_channel_windows"])
        temporal_positives = recordings * (
            int(units["primary_positive_channel_windows"])
            + int(units["stress_positive_channel_windows"])
        )
        self.assertIn(
            f"{primary_positives} positive pairs among {primary_total:,} primary pairs",
            scaffold,
        )
        self.assertIn(
            f"{temporal_positives} positives among {temporal_total:,} pairs",
            scaffold,
        )

    def test_repository_artifacts_reject_file_and_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            file_link = root / "file-link.json"
            file_link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _safe_repository_artifact(root, Path("file-link.json"))

            target_dir = root / "target-dir"
            target_dir.mkdir()
            (target_dir / "result.json").write_text("{}\n", encoding="utf-8")
            directory_link = root / "directory-link"
            directory_link.symlink_to(target_dir, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _safe_repository_artifact(
                    root, Path("directory-link") / "result.json"
                )


if __name__ == "__main__":
    unittest.main()
