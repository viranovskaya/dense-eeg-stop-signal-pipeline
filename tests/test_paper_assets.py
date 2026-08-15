"""Reproducibility checks for public aggregate manuscript assets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT_ROOT / "scripts" / "build_paper_assets.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_paper_assets", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load paper asset builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaperAssetTests(unittest.TestCase):
    def test_tracked_assets_are_reproduced_exactly(self):
        builder = _load_builder()
        tracked_package = PROJECT_ROOT / "docs" / "paper_assets"
        tracked_manifest_path = tracked_package / "manifest.json"
        tracked_manifest = json.loads(tracked_manifest_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            output_package = Path(directory) / "paper_assets"
            generated_manifest = builder.build_assets(output_package)
            self.assertEqual(generated_manifest, tracked_manifest)
            self.assertEqual(
                (output_package / "manifest.json").read_bytes(),
                tracked_manifest_path.read_bytes(),
            )
            for record in tracked_manifest["outputs"]:
                relative = Path(record["path"])
                generated = output_package / relative
                tracked = tracked_package / relative
                self.assertEqual(generated.read_bytes(), tracked.read_bytes())
                self.assertEqual(
                    hashlib.sha256(generated.read_bytes()).hexdigest(),
                    record["sha256"],
                )
            builder._verify_package(output_package, tracked_manifest)
            self.assertEqual(
                {
                    path.relative_to(output_package).as_posix()
                    for path in output_package.rglob("*")
                    if path.is_file()
                },
                {"manifest.json"} | {item["path"] for item in tracked_manifest["outputs"]},
            )

    def test_qc_figure_exposes_scope_and_log_axis(self):
        builder = _load_builder()
        payload = (
            PROJECT_ROOT / "docs" / "paper_assets" / "figures" / "synthetic_qc_validation.svg"
        ).read_text(encoding="utf-8")
        self.assertIn("public synthetic known-truth benchmark only", payload)
        self.assertIn("Logarithmic y-axis", payload)
        self.assertIn("3,405 pairs; 39 positives", payload)
        self.assertIn("3,408 pairs; 42 positives", payload)
        result = json.loads((PROJECT_ROOT / builder.QC_RESULT_PATH).read_text())
        changed = deepcopy(result)
        changed["recordings"] = 2
        changed["temporal_primary_detection"]["true_positive"] = 26
        changed["temporal_detection"]["true_positive"] = 26
        changed["temporal_detection"]["false_negative"] = 2
        changed_payload = builder._qc_svg(changed).decode("utf-8")
        self.assertIn("2,270 pairs; 26 positives", changed_payload)
        self.assertIn("2,272 pairs; 28 positives", changed_payload)
        self.assertNotIn("3,405 pairs; 39 positives", changed_payload)

    def test_log_axis_fails_closed_outside_declared_range(self):
        builder = _load_builder()
        result = json.loads((PROJECT_ROOT / builder.QC_RESULT_PATH).read_text())
        invalid = deepcopy(result)
        invalid["signal_preservation"]["band_power_absolute_error_db"]["unaffected"][
            "maximum_absolute_error_db"
        ] = 101.0
        with self.assertRaisesRegex(ValueError, "declared 0.01-100 dB axis"):
            builder._qc_svg(invalid)

    def test_qc_rows_use_source_confusion_counts(self):
        builder = _load_builder()
        result = json.loads((PROJECT_ROOT / builder.QC_RESULT_PATH).read_text())
        rows = builder._qc_rows(result)
        primary = next(
            row for row in rows
            if row["group"] == "primary" and row["metric"] == "recall"
        )
        inclusive = next(
            row for row in rows
            if row["group"] == "stress_inclusive" and row["metric"] == "recall"
        )
        self.assertEqual(
            (primary["true_positives"], primary["false_positives"], primary["false_negatives"]),
            (39, 0, 0),
        )
        self.assertEqual(
            (inclusive["true_positives"], inclusive["false_positives"], inclusive["false_negatives"]),
            (39, 0, 3),
        )
        invalid = deepcopy(result)
        invalid["temporal_detection"]["false_negative"] = 2
        with self.assertRaisesRegex(ValueError, "declared positives"):
            builder._qc_rows(invalid)

    def test_repository_sources_reject_unsafe_and_symlinked_paths(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            (root / "real.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                builder._safe_repository_file(Path("../real.json"), root)
            (root / "file-link.json").symlink_to(root / "real.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                builder._safe_repository_file(Path("file-link.json"), root)
            target = root / "target"
            target.mkdir()
            (target / "value.json").write_text("{}\n", encoding="utf-8")
            (root / "dir-link").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                builder._safe_repository_file(Path("dir-link/value.json"), root)

    def test_source_drift_cleans_temporary_package(self):
        builder = _load_builder()
        observed = 0
        original = builder._verify_capture

        def fail_after_generation(capture):
            nonlocal observed
            observed += 1
            if observed == 1:
                raise ValueError("simulated source drift")
            return original(capture)

        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            output = parent / "paper_assets"
            with mock.patch.object(builder, "_verify_capture", side_effect=fail_after_generation):
                with self.assertRaisesRegex(ValueError, "simulated source drift"):
                    builder.build_assets(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.glob(".paper_assets.tmp-*")), [])

    def test_output_parent_symlink_is_rejected(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "ancestry"):
                builder.build_assets(link / "paper_assets")

    def test_package_verifier_detects_tamper(self):
        builder = _load_builder()
        tracked = PROJECT_ROOT / "docs" / "paper_assets"
        manifest = json.loads((tracked / "manifest.json").read_text())
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            copy = Path(directory) / "paper_assets"
            shutil.copytree(tracked, copy)
            target = copy / manifest["outputs"][0]["path"]
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                builder._verify_package(copy, manifest)

    def test_interrupt_and_failed_temporary_verification_leave_no_package(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            output = parent / "paper_assets"
            with mock.patch.object(builder, "_write_bytes_at", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    builder.build_assets(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.glob(".paper_assets.tmp-*")), [])

            with mock.patch.object(
                builder,
                "_verify_package_fd",
                side_effect=ValueError("simulated temporary tamper"),
            ):
                with self.assertRaisesRegex(ValueError, "temporary tamper"):
                    builder.build_assets(output)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.glob(".paper_assets.tmp-*")), [])

    def test_existing_package_is_never_replaced(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            output = Path(directory) / "paper_assets"
            output.mkdir()
            marker = output / "owner.txt"
            marker.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                builder.build_assets(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing\n")

    def test_publish_race_never_replaces_or_removes_foreign_package(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            output = Path(directory) / "paper_assets"

            def lose_publish_race(parent_fd, source, destination):
                os.mkdir(destination, dir_fd=parent_fd)
                destination_fd = os.open(
                    destination,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                try:
                    descriptor = os.open(
                        "owner.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=destination_fd,
                    )
                    os.write(descriptor, b"foreign\n")
                    os.close(descriptor)
                finally:
                    os.close(destination_fd)
                raise FileExistsError("simulated publish race")

            with mock.patch.object(
                builder, "_rename_directory_noreplace", side_effect=lose_publish_race
            ):
                with self.assertRaises(FileExistsError):
                    builder.build_assets(output)
            self.assertEqual((output / "owner.txt").read_text(), "foreign\n")

    def test_parent_replacement_cannot_redirect_publication(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            output = parent / "paper_assets"
            moved = root / "original-parent"
            original = builder._prepare_output_parent

            def replace_after_open(requested):
                result = original(requested)
                parent.rename(moved)
                parent.mkdir()
                return result

            with mock.patch.object(
                builder, "_prepare_output_parent", side_effect=replace_after_open
            ):
                with self.assertRaisesRegex(ValueError, "parent identity changed"):
                    builder.build_assets(output)
            self.assertFalse(output.exists())
            self.assertFalse((moved / "paper_assets").exists())
            self.assertEqual(list(moved.glob(".paper_assets.tmp-*")), [])

    def test_cleanup_does_not_enter_replacement_directory(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            parent = Path(directory)
            victim = parent / "victim"
            victim.mkdir()
            (victim / "ours.txt").write_text("ours\n", encoding="utf-8")
            details = victim.stat()
            expected = (details.st_dev, details.st_ino)
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            real_open = os.open
            swapped = False

            def swap_before_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if not swapped and path == "victim" and kwargs.get("dir_fd") == parent_fd:
                    swapped = True
                    victim.rename(parent / "original-victim")
                    victim.mkdir()
                    (victim / "foreign.txt").write_text("foreign\n", encoding="utf-8")
                return real_open(path, flags, *args, **kwargs)

            try:
                with mock.patch.object(builder.os, "open", side_effect=swap_before_open):
                    builder._remove_tree_at(parent_fd, "victim", expected)
            finally:
                os.close(parent_fd)
            self.assertEqual((victim / "foreign.txt").read_text(), "foreign\n")
            self.assertEqual(
                (parent / "original-victim" / "ours.txt").read_text(), "ours\n"
            )


if __name__ == "__main__":
    unittest.main()
