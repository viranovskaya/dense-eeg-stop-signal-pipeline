"""Immutable human decisions for BIDS/EEGLAB ICA solutions."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from .bids_eeglab import BIDSEeglabInputs, input_identities, resolve_inputs
from .bids_eeglab_processing import load_verified_review_bundle
from .bids_eeglab_review import _nearest_existing_ancestor, _runtime_identity
from .ica import _validate_ica_decisions
from .provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


def _captured_csv(path: Path) -> tuple[pd.DataFrame, str]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("ICA decisions must be a regular non-symlink file")
    data = path.read_bytes()
    table = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False)
    return table, hashlib.sha256(data).hexdigest()


def _ica_review_context(root: Path) -> tuple[dict, dict, dict]:
    root = Path(root).expanduser().absolute()
    provenance = verify_provenance(root)
    if provenance.get("workflow") != "bids_eeglab_ica_review":
        raise ValueError("ICA package has the wrong workflow")
    summary_path = root / "ica_review_summary.json"
    solution_path = root / "ica_solution.fif"
    template_path = root / "ica_decision_template.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_sha256(summary) != provenance.get("summary_sha256"):
        raise ValueError("ICA summary does not match its provenance")
    if sha256_file(solution_path) != provenance.get("solution_sha256"):
        raise ValueError("ICA solution does not match its provenance")
    if summary.get("publication_allowed") is not False:
        raise ValueError("ICA package does not retain the controlled boundary")
    identity = {
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
        "summary_sha256": sha256_file(summary_path),
        "solution_sha256": sha256_file(solution_path),
        "decision_template_sha256": sha256_file(template_path),
    }
    return summary, provenance, identity


def ica_decision_context(root: Path) -> tuple[dict, dict, dict]:
    """Return a verified immutable ICA-decision package and its identity."""
    root = Path(root).expanduser().absolute()
    provenance = verify_provenance(root)
    if provenance.get("workflow") != "bids_eeglab_ica_decisions":
        raise ValueError("ICA decision package has the wrong workflow")
    summary_path = root / "ica_decision_summary.json"
    decisions_path = root / "ica_decisions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_sha256(summary) != provenance.get("summary_sha256"):
        raise ValueError("ICA decision summary does not match its provenance")
    if summary.get("status") != "complete" or summary.get(
        "publication_allowed"
    ) is not False:
        raise ValueError("ICA decisions are incomplete or outside the boundary")
    identity = {
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
        "summary_sha256": sha256_file(summary_path),
        "decisions_sha256": sha256_file(decisions_path),
    }
    return summary, provenance, identity


def _verify_finalization_context(
    inputs: BIDSEeglabInputs,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    decision_path: Path,
    participant_id: str,
    initial_inputs: list[dict],
    initial_source: dict,
    initial_runtime: dict,
    manual_review_identity: dict,
    ica_summary: dict,
    ica_provenance: dict,
    ica_identity: dict,
    decision_sha256: str,
) -> None:
    if sha256_file(decision_path) != decision_sha256:
        raise ValueError("ICA decision input changed during finalization")
    if input_identities(resolve_inputs(inputs)) != initial_inputs:
        raise ValueError("Dataset inputs changed during ICA decision finalization")
    if source_manifest() != initial_source or _runtime_identity() != initial_runtime:
        raise ValueError("Source or runtime changed during ICA finalization")
    reloaded_review = load_verified_review_bundle(
        inputs, qc_path, review_bundle_path, participant_id
    )
    if reloaded_review.identity != manual_review_identity:
        raise ValueError("Manual review bundle changed during ICA finalization")
    final_summary, final_provenance, final_identity = _ica_review_context(
        ica_review_path
    )
    if (
        final_summary != ica_summary
        or final_provenance != ica_provenance
        or final_identity != ica_identity
    ):
        raise ValueError("ICA review package changed during finalization")


def finalize_ica_decisions(
    inputs: BIDSEeglabInputs,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    decision_path: Path,
    participant_id: str,
    output: Path,
) -> Path:
    """Finalize one complete component decision table atomically."""
    qc_path = Path(qc_path).expanduser().absolute()
    review_bundle_path = Path(review_bundle_path).expanduser().absolute()
    ica_review_path = Path(ica_review_path).expanduser().absolute()
    decision_path = Path(decision_path).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("ICA decision finalization requires a new output path")
    resolved = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved)
    protected = [
        resolved.dataset_root.resolve(strict=True),
        qc_path.resolve(strict=True),
        review_bundle_path.resolve(strict=True),
        ica_review_path.resolve(strict=True),
    ]
    nearest = _nearest_existing_ancestor(output.parent).resolve(strict=True)
    if any(nearest.is_relative_to(root) for root in protected):
        raise ValueError("ICA decisions must remain outside source and input packages")

    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    manual_review = load_verified_review_bundle(
        inputs, qc_path, review_bundle_path, participant_id
    )
    ica_summary, ica_provenance, ica_identity = _ica_review_context(ica_review_path)
    if ica_summary.get("participant_id") != participant_id:
        raise ValueError("ICA package participant does not match the request")
    if ica_provenance.get("inputs") != initial_inputs:
        raise ValueError("ICA package does not match the selected dataset inputs")
    if ica_provenance.get("decision_bundle") != manual_review.identity:
        raise ValueError("ICA package does not match the manual review bundle")
    if ica_provenance.get("controls", {}).get("source_manifest") != initial_source:
        raise ValueError("ICA package source context is stale")
    if ica_provenance.get("runtime") != initial_runtime:
        raise ValueError("ICA package runtime context is incompatible")

    table, decision_sha256 = _captured_csv(decision_path)
    table = _validate_ica_decisions(
        table,
        participant_id=participant_id,
        solution_sha256=ica_identity["solution_sha256"],
        components=int(ica_summary["components"]),
    )
    excluded = table.loc[table["decision"] == "exclude", "component"].tolist()
    summary = {
        "schema_version": "1",
        "status": "complete",
        "participant_id": participant_id,
        "components": len(table),
        "kept_components": int((table["decision"] == "keep").sum()),
        "excluded_components": [int(value) for value in excluded],
        "reviewers": sorted(table["reviewer"].unique().tolist()),
        "automatic_component_exclusion": False,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core = {
        "schema_version": "1",
        "workflow": "bids_eeglab_ica_decisions",
        "participant_id": participant_id,
        "inputs": initial_inputs,
        "manual_review_bundle": dict(manual_review.identity),
        "ica_review": ica_identity,
        "decision_input_sha256": decision_sha256,
        "source_manifest": initial_source,
        "runtime": initial_runtime,
        "summary_sha256": canonical_sha256(summary),
        "automatic_component_exclusion": False,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    published = False
    try:
        table.to_csv(temporary / "ica_decisions.csv", index=False)
        (temporary / "ica_decision_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        context_args = (
            inputs,
            qc_path,
            review_bundle_path,
            ica_review_path,
            decision_path,
            participant_id,
            initial_inputs,
            initial_source,
            initial_runtime,
            manual_review.identity,
            ica_summary,
            ica_provenance,
            ica_identity,
            decision_sha256,
        )
        _verify_finalization_context(*context_args)
        write_provenance(temporary, core)
        verify_provenance(temporary)
        _verify_finalization_context(*context_args)
        temporary.replace(output)
        published = True
        verify_provenance(output)
        _verify_finalization_context(*context_args)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
