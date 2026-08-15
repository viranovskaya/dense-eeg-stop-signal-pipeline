#!/usr/bin/env python3
"""Build deterministic public aggregate assets for the methods manuscript."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import html
import io
import json
import math
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_RESULT_PATH = Path("docs/benchmark_results/heldout_seed_summary.json")
ICA_RESULT_PATH = Path("docs/benchmark_results/ica_seed_4401_summary.json")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CapturedFile:
    path: Path
    relative_path: str
    payload: bytes
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repository_file(relative: Path, root: Path = PROJECT_ROOT) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Repository source path must be safe and relative")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Repository source path must not contain symlinks")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("Repository source must be a regular file inside the root")
    return resolved


def _capture_repository_file(relative: Path) -> CapturedFile:
    path = _safe_repository_file(relative)
    payload = path.read_bytes()
    return CapturedFile(
        path=path,
        relative_path=relative.as_posix(),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _verify_capture(capture: CapturedFile) -> None:
    if capture.path.is_symlink() or not capture.path.is_file():
        raise ValueError(f"Captured source changed type: {capture.relative_path}")
    observed = _sha256(capture.path)
    if observed != capture.sha256:
        raise ValueError(f"Captured source changed: {capture.relative_path}")


def _prepare_output_parent(output: Path) -> tuple[str, Path, int]:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output package must be new: {output}")
    parent = output.parent
    if not parent.is_dir():
        raise ValueError("Output parent must already exist")
    current_fd = os.open(parent.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parent.parts[1:]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
    except OSError as error:
        os.close(current_fd)
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError("Output ancestry must not contain symlinks") from error
        raise
    except BaseException:
        os.close(current_fd)
        raise
    return output.name, parent, current_fd


def _write_bytes_at(root_fd: int, relative: Path, payload: bytes) -> None:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Paper asset output path must be safe and relative")
    directory_fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            relative.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Atomically publish a directory without replacing an existing path."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000004
        )  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_fd, source_bytes, parent_fd, destination_bytes, 0x00000001
        )
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-replace directory publish is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _read_bytes_at(directory_fd: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _scan_package_fd(directory_fd: int, prefix: Path = Path()) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for name in sorted(os.listdir(directory_fd)):
        details = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        relative = prefix / name
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("Paper asset package must not contain symlinks")
        if stat.S_ISDIR(details.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                records.update(_scan_package_fd(child_fd, relative))
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(details.st_mode):
            payload = _read_bytes_at(directory_fd, name)
            records[relative.as_posix()] = (
                len(payload), hashlib.sha256(payload).hexdigest()
            )
        else:
            raise ValueError(f"Unsupported paper asset type: {relative}")
    return records


def _verify_package_fd(root_fd: int, manifest: dict) -> None:
    observed = _scan_package_fd(root_fd)
    expected = {
        record["path"]: (record["size_bytes"], record["sha256"])
        for record in manifest["outputs"]
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected[MANIFEST_NAME] = (
        len(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest()
    )
    if observed != expected:
        raise ValueError("Paper asset package exact-set or hash verification failed")


def _remove_tree_at(parent_fd: int, name: str, expected: tuple[int, int]) -> None:
    try:
        details = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(details.st_mode) or (details.st_dev, details.st_ino) != expected:
        return
    directory_fd = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
    )
    try:
        opened = os.fstat(directory_fd)
        if (opened.st_dev, opened.st_ino) != expected:
            return
        for child in os.listdir(directory_fd):
            child_details = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_details.st_mode):
                _remove_tree_at(
                    directory_fd,
                    child,
                    (child_details.st_dev, child_details.st_ino),
                )
            else:
                os.unlink(child, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) == expected:
        os.rmdir(name, dir_fd=parent_fd)


def _verify_package(root: Path, manifest: dict) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Paper asset package must be a real directory")
    expected = {MANIFEST_NAME}
    for record in manifest["outputs"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe paper asset path")
        expected.add(relative.as_posix())
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"Missing or unsafe paper asset: {relative}")
        if candidate.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Paper asset size mismatch: {relative}")
        if _sha256(candidate) != record["sha256"]:
            raise ValueError(f"Paper asset hash mismatch: {relative}")
    actual = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Paper asset package must not contain symlinks")
        if candidate.is_file():
            actual.add(candidate.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(
            f"Paper asset exact-set mismatch: expected {sorted(expected)}, "
            f"observed {sorted(actual)}"
        )
    manifest_path = root / MANIFEST_NAME
    observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed_manifest != manifest:
        raise ValueError("Paper asset manifest content mismatch")


def _csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _qc_rows(result: dict) -> list[dict]:
    rows: list[dict] = []
    primary = result["temporal_primary_detection"]
    inclusive = result["temporal_detection"]
    recordings = int(result["recordings"])
    units = result["evaluation_units_per_recording"]
    primary_pairs = recordings * int(units["primary_channel_windows"])
    primary_positives = recordings * int(units["primary_positive_channel_windows"])
    temporal_pairs = recordings * int(units["temporal_channel_windows"])
    temporal_positives = recordings * (
        int(units["primary_positive_channel_windows"])
        + int(units["stress_positive_channel_windows"])
    )
    for metric in ("precision", "recall", "f1"):
        for group, payload, pairs, positives in (
            ("primary", primary, primary_pairs, primary_positives),
            ("stress_inclusive", inclusive, temporal_pairs, temporal_positives),
        ):
            true_positives = int(payload["true_positive"])
            false_positives = int(payload["false_positive"])
            false_negatives = int(payload["false_negative"])
            if true_positives + false_negatives != positives:
                raise ValueError("Confusion counts do not match declared positives")
            rows.append({
                "panel": "temporal_detection",
                "group": group,
                "metric": metric,
                "value": repr(float(payload[metric])),
                "display_value": f"{float(payload[metric]):.3f}",
                "unit": "proportion",
                "fixture_count": recordings,
                "evaluation_pairs": pairs,
                "truth_positives": positives,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "channels_or_values_aggregated": "",
            })

    groups = result["signal_preservation"]["band_power_absolute_error_db"]
    labels = {
        "unaffected": "unaffected",
        "oracle_interpolated": "oracle_interpolated",
        "non_oracle_corrupted": "other_corrupted",
        "all_channel_band_values": "all_values",
    }
    metrics = (
        ("median_absolute_error_db", "median"),
        ("p95_absolute_error_db", "p95"),
        ("maximum_absolute_error_db", "maximum"),
    )
    for source_group, display_group in labels.items():
        payload = groups[source_group]
        for source_metric, display_metric in metrics:
            value = float(payload[source_metric])
            rows.append(
                {
                    "panel": "band_power_preservation",
                    "group": display_group,
                    "metric": display_metric,
                    "value": repr(value),
                    "display_value": f"{value:.3f}",
                    "unit": "dB absolute error",
                    "fixture_count": recordings,
                    "evaluation_pairs": "",
                    "truth_positives": "",
                    "true_positives": "",
                    "false_positives": "",
                    "false_negatives": "",
                    "channels_or_values_aggregated": int(payload["values"]),
                }
            )
    return rows


def _ica_rows(result: dict) -> list[dict]:
    checks = result["checks"]
    correlation = checks["auxiliary_correlation"]
    band = checks["signal_preservation"]["band_power"]
    task = checks["signal_preservation"]["task_signal"]
    channels_compared = int(correlation["eog"]["channels_compared"])
    task_channels = len(task["channels"])
    values = [
        ("fit", "iterations", result["ica"]["fit_iterations"], "iterations", ""),
        (
            "eog",
            "maximum_absolute_correlation_before",
            correlation["eog"]["maximum_absolute_correlation_before"],
            "correlation",
            channels_compared,
        ),
        (
            "eog",
            "maximum_absolute_correlation_after",
            correlation["eog"]["maximum_absolute_correlation_after"],
            "correlation",
            channels_compared,
        ),
        (
            "ecg",
            "maximum_absolute_correlation_before",
            correlation["ecg"]["maximum_absolute_correlation_before"],
            "correlation",
            channels_compared,
        ),
        (
            "ecg",
            "maximum_absolute_correlation_after",
            correlation["ecg"]["maximum_absolute_correlation_after"],
            "correlation",
            channels_compared,
        ),
        (
            "band_power",
            "median_absolute_change",
            band["median_absolute_change_db"],
            "dB",
            band["channel_band_values"],
        ),
        (
            "band_power",
            "p95_absolute_change",
            band["p95_absolute_change_db"],
            "dB",
            band["channel_band_values"],
        ),
        (
            "band_power",
            "maximum_absolute_change",
            band["maximum_absolute_change_db"],
            "dB",
            band["channel_band_values"],
        ),
        (
            "task_signal",
            "maximum_peak_amplitude_change",
            task["maximum_peak_amplitude_change_uv"],
            "microvolts",
            task_channels,
        ),
        (
            "task_signal",
            "maximum_peak_latency_change",
            task["maximum_peak_latency_change_ms"],
            "ms",
            task_channels,
        ),
    ]
    return [
        {
            "group": group,
            "metric": metric,
            "value": repr(float(value)),
            "display_value": f"{float(value):.3f}",
            "unit": unit,
            "fixture_count": 1,
            "channels_or_values_aggregated": n,
        }
        for group, metric, value, unit, n in values
    ]


def _qc_svg(result: dict) -> bytes:
    primary = result["temporal_primary_detection"]
    inclusive = result["temporal_detection"]
    preservation = result["signal_preservation"]["band_power_absolute_error_db"]
    groups = [
        ("Unaffected", preservation["unaffected"]),
        ("Oracle", preservation["oracle_interpolated"]),
        ("Other", preservation["non_oracle_corrupted"]),
        ("All", preservation["all_channel_band_values"]),
    ]
    width, height = 1100, 560
    left_x, right_x = 70, 600
    top, plot_height = 100, 330
    colors = {"primary": "#287a8d", "inclusive": "#d28b17"}
    recordings = int(result["recordings"])
    units = result["evaluation_units_per_recording"]
    primary_pairs = recordings * int(units["primary_channel_windows"])
    primary_positives = int(primary["true_positive"]) + int(primary["false_negative"])
    temporal_pairs = recordings * int(units["temporal_channel_windows"])
    temporal_positives = int(inclusive["true_positive"]) + int(
        inclusive["false_negative"]
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Public synthetic QC validation summary</title>",
        "<desc>Primary and stress-inclusive temporal detection metrics, plus logarithmic band-power error summaries for unaffected, oracle-interpolated, other corrupted and all channel-band values.</desc>",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        "<style>text{font-family:Arial,sans-serif;fill:#173047}.axis{stroke:#526b7a;stroke-width:1}.grid{stroke:#dbe4e9;stroke-width:1}.small{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:700}.value{font-size:11px;font-weight:700}</style>",
        '<text class="title" x="70" y="38">A  Temporal channel-window detection</text>',
        '<text class="title" x="600" y="38">B  Band-power absolute error</text>',
        '<text class="small" x="600" y="60">Logarithmic y-axis; public synthetic known-truth benchmark only</text>',
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        parts.extend(
            [
                f'<line class="grid" x1="{left_x}" x2="520" y1="{y:.1f}" y2="{y:.1f}"/>',
                f'<text class="small" x="58" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>',
            ]
        )
    parts.append(
        f'<line class="axis" x1="{left_x}" x2="{left_x}" y1="{top}" y2="{top + plot_height}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left_x}" x2="520" y1="{top + plot_height}" y2="{top + plot_height}"/>'
    )
    metric_names = (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"))
    for index, (label, key) in enumerate(metric_names):
        center = 145 + index * 145
        for offset, source, group in (
            (-24, primary, "primary"),
            (24, inclusive, "inclusive"),
        ):
            value = float(source[key])
            bar_height = plot_height * value
            x = center + offset - 18
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x}" y="{y:.1f}" width="36" height="{bar_height:.1f}" fill="{colors[group]}"/>'
            )
            parts.append(
                f'<text class="value" x="{center + offset}" y="{max(88, y - 6):.1f}" text-anchor="middle">{value:.3f}</text>'
            )
        parts.append(
            f'<text class="label" x="{center}" y="454" text-anchor="middle">{html.escape(label)}</text>'
        )
    parts.extend(
        [
            '<rect x="100" y="492" width="18" height="12" fill="#287a8d"/>',
            f'<text class="small" x="125" y="503">Primary: {primary_pairs:,} pairs; {primary_positives:,} positives</text>',
            '<rect x="310" y="492" width="18" height="12" fill="#d28b17"/>',
            f'<text class="small" x="335" y="503">Stress-inclusive: {temporal_pairs:,} pairs; {temporal_positives:,} positives</text>',
        ]
    )

    log_min, log_max = -2.0, 2.0
    right_width = 430

    plotted_values = [
        float(payload[key])
        for _, payload in groups
        for key in (
            "median_absolute_error_db",
            "p95_absolute_error_db",
            "maximum_absolute_error_db",
        )
    ]
    if any(not math.isfinite(value) or not 0.01 <= value <= 100 for value in plotted_values):
        raise ValueError("Band-power values must be finite and within the declared 0.01-100 dB axis")

    def log_y(value: float) -> float:
        return top + plot_height * (log_max - math.log10(value)) / (log_max - log_min)

    for exponent in range(-2, 3):
        value = 10**exponent
        y = log_y(value)
        label = {0.01: "0.01", 0.1: "0.1", 1: "1", 10: "10", 100: "100"}[value]
        parts.extend(
            [
                f'<line class="grid" x1="{right_x}" x2="{right_x + right_width}" y1="{y:.1f}" y2="{y:.1f}"/>',
                f'<text class="small" x="{right_x - 12}" y="{y + 4:.1f}" text-anchor="end">{label}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{right_x}" x2="{right_x}" y1="{top}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{right_x}" x2="{right_x + right_width}" y1="{top + plot_height}" y2="{top + plot_height}"/>',
            f'<text class="label" x="{right_x - 52}" y="{top + plot_height / 2}" transform="rotate(-90 {right_x - 52} {top + plot_height / 2})" text-anchor="middle">Absolute error (dB)</text>',
        ]
    )
    marker_colors = {"median": "#287a8d", "p95": "#7a5ca8", "maximum": "#b33d3d"}
    for index, (label, payload) in enumerate(groups):
        x = right_x + 62 + index * 105
        values = {
            "median": float(payload["median_absolute_error_db"]),
            "p95": float(payload["p95_absolute_error_db"]),
            "maximum": float(payload["maximum_absolute_error_db"]),
        }
        parts.append(
            f'<line x1="{x}" x2="{x}" y1="{log_y(values["median"]):.1f}" y2="{log_y(values["maximum"]):.1f}" stroke="#9aaab3" stroke-width="2"/>'
        )
        for metric, value in values.items():
            y = log_y(value)
            parts.append(
                f'<circle cx="{x}" cy="{y:.1f}" r="6" fill="{marker_colors[metric]}" stroke="white" stroke-width="1.5"/>'
            )
        parts.extend(
            [
                f'<text class="label" x="{x}" y="454" text-anchor="middle">{html.escape(label)}</text>',
                f'<text class="small" x="{x}" y="472" text-anchor="middle">n={int(payload["values"]):,}</text>',
            ]
        )
    legend_x = 660
    for offset, metric, label in (
        (0, "median", "Median"),
        (100, "p95", "95th percentile"),
        (250, "maximum", "Maximum"),
    ):
        parts.extend(
            [
                f'<circle cx="{legend_x + offset}" cy="505" r="6" fill="{marker_colors[metric]}"/>',
                f'<text class="small" x="{legend_x + offset + 12}" y="509">{label}</text>',
            ]
        )
    parts.append("</svg>\n")
    return "".join(parts).encode("utf-8")


def build_assets(output_package: Path) -> dict:
    output_name, resolved_parent, parent_fd = _prepare_output_parent(output_package)
    parent_details = os.fstat(parent_fd)
    parent_identity = (parent_details.st_dev, parent_details.st_ino)
    builder_capture = _capture_repository_file(Path("scripts/build_paper_assets.py"))
    qc_capture = _capture_repository_file(QC_RESULT_PATH)
    ica_capture = _capture_repository_file(ICA_RESULT_PATH)
    qc = json.loads(qc_capture.payload.decode("utf-8"))
    ica = json.loads(ica_capture.payload.decode("utf-8"))

    assets = {
        "figures/synthetic_qc_validation.svg": _qc_svg(qc),
        "tables/synthetic_qc_validation.csv": _csv_bytes(
            _qc_rows(qc),
            [
                "panel", "group", "metric", "value", "display_value", "unit",
                "fixture_count", "evaluation_pairs", "truth_positives",
                "true_positives", "false_positives", "false_negatives",
                "channels_or_values_aggregated",
            ],
        ),
        "tables/synthetic_ica_validation.csv": _csv_bytes(
            _ica_rows(ica),
            [
                "group", "metric", "value", "display_value", "unit",
                "fixture_count", "channels_or_values_aggregated",
            ],
        ),
    }
    manifest = {
        "schema_version": "1",
        "scope": "Public aggregate synthetic manuscript assets only.",
        "builder": {
            "path": builder_capture.relative_path,
            "sha256": builder_capture.sha256,
        },
        "sources": {
            qc_capture.relative_path: qc_capture.sha256,
            ica_capture.relative_path: ica_capture.sha256,
        },
        "outputs": [
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for relative, payload in sorted(assets.items())
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name = f".{output_name}.tmp-{secrets.token_hex(12)}"
    temporary_fd: int | None = None
    temporary_identity: tuple[int, int] | None = None
    published = False
    try:
        os.mkdir(temporary_name, 0o700, dir_fd=parent_fd)
        temporary_fd = os.open(
            temporary_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        details = os.fstat(temporary_fd)
        temporary_identity = (details.st_dev, details.st_ino)
        for relative, payload in assets.items():
            _write_bytes_at(temporary_fd, Path(relative), payload)
        _write_bytes_at(temporary_fd, Path(MANIFEST_NAME), manifest_bytes)
        _verify_package_fd(temporary_fd, manifest)
        for capture in (builder_capture, qc_capture, ica_capture):
            _verify_capture(capture)
        current_parent = resolved_parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise ValueError("Output parent identity changed before publication")
        _rename_directory_noreplace(parent_fd, temporary_name, output_name)
        published = True
        current_parent = resolved_parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != parent_identity:
            raise ValueError("Output parent identity changed during publication")
        return manifest
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if not published and temporary_identity is not None:
            _remove_tree_at(parent_fd, temporary_name, temporary_identity)
        os.close(parent_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-package",
        type=Path,
        default=PROJECT_ROOT / "docs" / "paper_assets",
        help="New atomic package directory receiving figures/, tables/ and manifest.json",
    )
    args = parser.parse_args()
    manifest = build_assets(args.output_package)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
