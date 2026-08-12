"""Small deterministic provenance records for private preprocessing runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

from .config import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_EVENT_CODEBOOK,
    AnalysisConfig,
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    """Return the SHA-256 of a canonical JSON representation."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _brainvision_sources(vhdr: Path) -> list[dict]:
    """Identify a BrainVision triplet without exposing local paths or names."""
    vhdr = Path(vhdr).expanduser().resolve()
    header = vhdr.read_text(encoding="utf-8-sig", errors="replace")
    data_match = re.search(r"^DataFile=(.+?)\r?$", header, flags=re.MULTILINE)
    marker_match = re.search(r"^MarkerFile=(.+?)\r?$", header, flags=re.MULTILINE)
    if not data_match or not marker_match:
        raise ValueError(f"Could not identify the BrainVision triplet for {vhdr.name}")
    sources = [
        ("header", vhdr),
        ("marker", (vhdr.parent / marker_match.group(1).strip()).resolve()),
        ("signal", (vhdr.parent / data_match.group(1).strip()).resolve()),
    ]
    missing = [role for role, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "BrainVision source files are missing for roles: " + ", ".join(missing)
        )
    return [
        {
            "role": role,
            "format": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in sources
    ]


def _git_state() -> dict[str, str | bool]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"revision": revision, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": True}


def source_manifest() -> dict:
    """Bind a run to the executable project files, including dirty changes."""
    project_root = Path(__file__).resolve().parents[2]
    roots = ("src", "scripts", "config")
    paths = sorted(
        path
        for root in roots
        for path in (project_root / root).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"files": files, "sha256": canonical_sha256({"files": files})}


def package_versions() -> dict[str, str]:
    """Return versions of packages that can change numerical outputs."""
    versions = {}
    for package in (
        "mne",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "h5py",
        "eeglabio",
        "pybv",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_provenance_core(
    *,
    vhdr: Path,
    participant_id: str,
    analysis: AnalysisConfig,
    bad_channel_decisions: list[dict],
    analysis_config_path: Path = DEFAULT_ANALYSIS_CONFIG,
    event_codebook_path: Path = DEFAULT_EVENT_CODEBOOK,
) -> dict:
    """Build the stable part of a run record before outputs are written."""
    normalized_decisions = sorted(
        (
            {
                str(key): value
                for key, value in decision.items()
                if key != "participant_id"
            }
            for decision in bad_channel_decisions
        ),
        key=lambda decision: (
            str(decision.get("channel", "")),
            str(decision.get("decision", "")),
        ),
    )
    git_state = _git_state()
    payload = {
        "schema_version": "1",
        "participant_id": participant_id,
        "inputs": _brainvision_sources(vhdr),
        "configuration": {
            "analysis_sha256": sha256_file(analysis_config_path),
            "event_codebook_sha256": sha256_file(event_codebook_path),
            "filter_hz": list(analysis.filter_hz),
            "line_frequency_hz": analysis.line_frequency_hz,
            "montage": analysis.montage,
            "reference": analysis.reference,
            "epochs_seconds": {
                condition: list(window)
                for condition, window in sorted(analysis.epochs_seconds.items())
            },
            "event_integrity_limits_ms": dict(
                sorted(analysis.event_integrity_limits_ms.items())
            ),
            "qc": {
                "representative_window_count": analysis.qc.representative_window_count,
                "representative_window_seconds": analysis.qc.representative_window_seconds,
                "full_recording_window_seconds": (
                    analysis.qc.full_recording_window_seconds
                ),
                "robust_z_threshold": analysis.qc.robust_z_threshold,
                "flat_fraction_threshold": analysis.qc.flat_fraction_threshold,
                "line_noise_ratio_db_threshold": analysis.qc.line_noise_ratio_db_threshold,
            },
        },
        "bad_channel_decisions": normalized_decisions,
        "decision_record_complete": bool(normalized_decisions)
        and all(
            all(
                decision.get(field)
                for field in (
                    "decision",
                    "reason",
                    "reviewer",
                    "reviewed_at",
                    "evidence_windows",
                )
            )
            and (decision.get("decision") == "none" or bool(decision.get("channel")))
            for decision in normalized_decisions
        ),
        "software": {
            "git_revision": git_state["revision"],
            "git_dirty": git_state["dirty"],
            "python": platform.python_version(),
            "packages": package_versions(),
            "source_manifest": source_manifest(),
        },
        "GeneratedBy": [
            {
                "Name": "dense-eeg-stop-signal-pipeline",
                "Version": (
                    f"{git_state['revision']}+dirty"
                    if git_state["dirty"]
                    else str(git_state["revision"])
                ),
                "CodeURL": "https://github.com/viranovskaya/dense-eeg-stop-signal-pipeline",
            }
        ],
    }
    payload["core_sha256"] = canonical_sha256(payload)
    return payload


def write_provenance(output: Path, core: dict) -> Path:
    """Add output digests and write provenance without hashing itself."""
    output = Path(output).resolve()
    provenance_path = output / "provenance.json"
    files = [
        path for path in output.rglob("*") if path.is_file() and path != provenance_path
    ]
    payload = {
        **core,
        "outputs": [
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(
                files, key=lambda item: item.relative_to(output).as_posix()
            )
        ],
    }
    provenance_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance_path


def verify_input_sources(vhdr: Path, expected: list[dict]) -> None:
    """Reject a source triplet that changed after run initialization."""
    if _brainvision_sources(vhdr) != expected:
        raise ValueError("BrainVision source files changed during the run")


def verify_provenance(output: Path) -> dict:
    """Verify a run core and every output declared by its provenance record."""
    output = Path(output).resolve()
    path = output / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"core_sha256", "outputs"}
    }
    if canonical_sha256(core) != payload.get("core_sha256"):
        raise ValueError(f"Provenance core hash mismatch in {path}")
    records = payload.get("outputs", [])
    declared_paths: list[str] = []
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe provenance output path: {record['path']}")
        normalized = relative.as_posix()
        declared_paths.append(normalized)
        candidate = output / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(output):
            raise ValueError(f"Unsafe provenance output path: {record['path']}")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != record["size_bytes"]
            or sha256_file(candidate) != record["sha256"]
        ):
            raise ValueError(f"Provenance output mismatch: {record['path']}")
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("Provenance contains duplicate output paths")
    actual_paths = {
        candidate.relative_to(output).as_posix()
        for candidate in output.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if actual_paths != set(declared_paths):
        raise ValueError("Provenance output set does not match the run directory")
    return payload
