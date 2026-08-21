#!/usr/bin/env python3
"""Compute the single fixed stop-signal ERP endpoint from completed outputs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import pandas as pd

from hunt_eeg.provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
)
from hunt_eeg.stop_signal_endpoint import (
    ENDPOINT_OUTPUT_FILES,
    participant_behavioral_diagnostics,
    participant_endpoint,
    summarize_endpoint,
    verify_endpoint_package,
)


EXPECTED_RECORDINGS = 10
DEPENDENCY_FILES = ("requirements.txt", "constraints-ci.txt")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_identity() -> dict:
    """Bind numerical and rendered outputs to the exact local runtime."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "dependency_file_sha256": {
            name: sha256_file(PROJECT_ROOT / name) for name in DEPENDENCY_FILES
        },
    }


def _verify_dependency_context(expected: dict[str, str]) -> None:
    """Reject dependency-control drift during endpoint calculation."""
    current = {name: sha256_file(PROJECT_ROOT / name) for name in DEPENDENCY_FILES}
    if current != expected:
        raise RuntimeError("Endpoint dependency controls changed during calculation")


def _participant_id(package: Path) -> str:
    if not package.name.startswith("sub-") or len(package.name) <= 4:
        raise ValueError(f"Invalid participant package name: {package.name}")
    return package.name[4:]


def _verify_dataset_context(dataset: Path, source_sha256: str) -> dict:
    """Verify dataset-level files and participant provenance identities."""
    if dataset.is_symlink() or not dataset.is_dir():
        raise ValueError("Dataset preprocessing package must be a real directory")
    if any(candidate.is_symlink() for candidate in dataset.rglob("*")):
        raise ValueError("Dataset preprocessing package must not contain symlinks")
    provenance_path = dataset / "dataset_provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if payload.get("source_manifest", {}).get("sha256") != source_sha256:
        raise ValueError("Dataset preprocessing source does not match endpoint source")
    packages = sorted(path for path in dataset.glob("sub-*") if path.is_dir())
    if {path.name for path in packages} != {
        path.name for path in dataset.iterdir() if path.is_dir()
    }:
        raise ValueError("Dataset participant directory set is not exact")
    discovered = {_participant_id(path) for path in packages}
    if len(discovered) != EXPECTED_RECORDINGS:
        raise ValueError(
            f"Fixed study requires exactly {EXPECTED_RECORDINGS} recordings"
        )
    records = payload.get("participants", [])
    expected = {str(record["participant_id"]) for record in records}
    if discovered != expected or len(records) != len(expected):
        raise ValueError("Dataset and participant provenance sets do not match")
    records_by_participant = {
        str(record["participant_id"]): record for record in records
    }
    interval_manifest_sha256 = payload.get(
        "fixed_study_interval_manifest_sha256", ""
    )
    if not _is_sha256(interval_manifest_sha256):
        raise ValueError("Dataset does not bind a completed fixed-study interval manifest")
    for package in packages:
        participant_id = _participant_id(package)
        participant = verify_provenance(package)
        record = records_by_participant[participant_id]
        if (
            sha256_file(package / "provenance.json")
            != record["provenance_sha256"]
            or participant["core_sha256"] != record["core_sha256"]
        ):
            raise ValueError(f"Dataset participant binding failed: {package.name}")
        interval_review = participant.get("interval_review", {})
        if (
            interval_review.get("identity", {}).get("manifest_sha256")
            != interval_manifest_sha256
            or interval_review.get("application_target") != "epochs"
            or interval_review.get("complete") is not True
            or interval_review.get("automatic_decisions") is not False
        ):
            raise ValueError(
                f"Completed fixed-study interval review is missing for {package.name}"
            )
        ica_inputs = participant.get("ica_inputs", {})
        required_ica_hashes = {
            "solution_sha256",
            "review_provenance_sha256",
            "decision_table_sha256",
        }
        if (
            not required_ica_hashes.issubset(ica_inputs)
            or any(not _is_sha256(ica_inputs[key]) for key in required_ica_hashes)
            or ica_inputs.get("automatic_exclusion") is not False
        ):
            raise ValueError(
                f"Reviewed ICA application is missing for {package.name}"
            )
    declared_outputs = {
        str(record["path"]): record for record in payload.get("outputs", [])
    }
    actual_outputs = {
        candidate.name
        for candidate in dataset.iterdir()
        if candidate.is_file() and candidate != provenance_path
    }
    if actual_outputs != set(declared_outputs):
        raise ValueError("Dataset top-level output set is not exact")
    for record in declared_outputs.values():
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("Dataset provenance contains an unsafe output path")
        candidate = dataset / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"Dataset output is missing or unsafe: {relative}")
        if (
            candidate.stat().st_size != record["size_bytes"]
            or sha256_file(candidate) != record["sha256"]
        ):
            raise ValueError(f"Dataset output verification failed: {relative}")
    return payload


def _verify_publication_context(
    *,
    dataset: Path,
    dataset_provenance_sha256: str,
    source_sha256: str,
    packages: list[Path],
    participant_inputs: dict[str, dict[str, str]],
    dependency_file_sha256: dict[str, str],
) -> None:
    """Recheck every upstream identity used by the endpoint calculation."""
    dataset_provenance = dataset / "dataset_provenance.json"
    if sha256_file(dataset_provenance) != dataset_provenance_sha256:
        raise RuntimeError("Dataset provenance changed during endpoint calculation")
    if source_manifest()["sha256"] != source_sha256:
        raise RuntimeError("Endpoint source changed during calculation")
    _verify_dependency_context(dependency_file_sha256)
    _verify_dataset_context(dataset, source_sha256)
    for package in packages:
        participant_id = _participant_id(package)
        provenance = verify_provenance(package)
        expected = participant_inputs[participant_id]
        if (
            sha256_file(package / "provenance.json")
            != expected["provenance_sha256"]
            or provenance["core_sha256"] != expected["core_sha256"]
        ):
            raise RuntimeError(f"Participant package changed: {package.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset_output.expanduser().resolve()
    requested_output = args.output.expanduser().resolve()
    if requested_output.exists() or requested_output.is_symlink():
        raise SystemExit("Endpoint calculation requires a new output path")
    if requested_output.is_relative_to(dataset):
        raise SystemExit("Endpoint output must be outside the preprocessing dataset")
    dataset_provenance = dataset / "dataset_provenance.json"
    if not dataset_provenance.is_file():
        raise SystemExit("Completed dataset_provenance.json is required")
    initial_dataset_sha = sha256_file(dataset_provenance)
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    _verify_dataset_context(dataset, initial_source["sha256"])

    packages = sorted(path for path in dataset.glob("sub-*") if path.is_dir())
    if not packages:
        raise SystemExit("No participant preprocessing packages found")
    participant_inputs = {}
    rows = []
    for package in packages:
        participant_id = _participant_id(package)
        provenance = verify_provenance(package)
        if provenance.get("participant_id") != participant_id:
            raise ValueError(f"Participant identity mismatch for {package.name}")
        if provenance.get("configuration", {}).get("erp_filter_hz") != [0.2, 30.0]:
            raise ValueError(f"ERP filter mismatch for {package.name}")
        participant_inputs[participant_id] = {
            "provenance_sha256": sha256_file(package / "provenance.json"),
            "core_sha256": provenance["core_sha256"],
        }
        epoch_path = package / f"sub-{participant_id}_stop-epo.fif"
        lineage_path = package / "stop_epoch_lineage.csv"
        trials_path = package / "reconstructed_trials.csv"
        if not all(path.is_file() for path in (epoch_path, lineage_path, trials_path)):
            raise FileNotFoundError(
                f"Stop epochs, lineage, or reconstructed trials missing for {package.name}"
            )
        epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
        lineage = pd.read_csv(lineage_path, keep_default_na=False)
        trials = pd.read_csv(trials_path, keep_default_na=False)
        row = participant_endpoint(epochs, lineage, participant_id=participant_id)
        row.update(participant_behavioral_diagnostics(trials))
        rows.append(row)

    summary = summarize_endpoint(rows)
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_output.name}.tmp-",
            dir=requested_output.parent,
        )
    )
    cleanup = lambda: shutil.rmtree(temporary, ignore_errors=True)
    atexit.register(cleanup)
    table = pd.DataFrame(rows).sort_values("participant_id")
    table.to_csv(temporary / "participant_endpoint.csv", index=False)
    (temporary / "endpoint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    mean = summary["mean_contrast_uv"]
    lower, upper = summary["bootstrap_95_ci_uv"]
    axis.errorbar(
        [0],
        [mean],
        yerr=[[mean - lower], [upper - mean]],
        fmt="o",
        color="#253858",
        capsize=6,
    )
    axis.axhline(0.0, color="#666666", linewidth=1)
    axis.set(
        xticks=[0],
        xticklabels=["Response absent - present"],
        ylabel="Mean FC1/FC2/FCz amplitude (uV)",
        title="Stop-signal endpoint (250-450 ms)",
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(temporary / "endpoint_mean_ci.png", dpi=180)
    plt.close(figure)

    outputs = [
        {
            "path": name,
            "size_bytes": (temporary / name).stat().st_size,
            "sha256": sha256_file(temporary / name),
        }
        for name in sorted(ENDPOINT_OUTPUT_FILES)
    ]
    core = {
        "schema_version": "1",
        "workflow": "fixed_stop_signal_erp_endpoint",
        "dataset_provenance_sha256": initial_dataset_sha,
        "participant_inputs": dict(sorted(participant_inputs.items())),
        "source_manifest": initial_source,
        "runtime": initial_runtime,
        "outputs": outputs,
        "privacy": (
            "Private derived package: participant_endpoint.csv contains "
            "pseudonymous participant-level values and is not publication-ready."
        ),
    }
    provenance = {**core, "core_sha256": canonical_sha256(core)}
    (temporary / "endpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verify_endpoint_package(temporary)

    _verify_publication_context(
        dataset=dataset,
        dataset_provenance_sha256=initial_dataset_sha,
        source_sha256=initial_source["sha256"],
        packages=packages,
        participant_inputs=participant_inputs,
        dependency_file_sha256=initial_runtime["dependency_file_sha256"],
    )

    temporary.replace(requested_output)
    try:
        verify_endpoint_package(requested_output)
        _verify_publication_context(
            dataset=dataset,
            dataset_provenance_sha256=initial_dataset_sha,
            source_sha256=initial_source["sha256"],
            packages=packages,
            participant_inputs=participant_inputs,
            dependency_file_sha256=initial_runtime["dependency_file_sha256"],
        )
    except BaseException:
        shutil.rmtree(requested_output, ignore_errors=True)
        raise
    atexit.unregister(cleanup)


if __name__ == "__main__":
    main()
