"""Reviewed preprocessing and epoch export for BIDS/EEGLAB recordings."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import mne
import pandas as pd

from .bids_eeglab import (
    BIDSEeglabInputs,
    _capture,
    _captured_json_sha,
    input_identities,
    resolve_inputs,
)
from .bids_eeglab_ica_decisions import (
    _captured_csv,
    _ica_review_context,
    ica_decision_context,
)
from .bids_eeglab_processing import (
    load_verified_review_bundle,
    prepare_filtered_reviewed_raw,
)
from .bids_eeglab_review import _nearest_existing_ancestor, _runtime_identity
from .ica import _validate_ica_decisions
from .provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


def _epoch_condition(
    raw: mne.io.BaseRaw,
    condition: str,
    definition: dict,
    event_code: int,
    source_rows: pd.DataFrame,
) -> tuple[mne.Epochs, pd.DataFrame, dict]:
    events, _ = mne.events_from_annotations(
        raw,
        event_id={condition: event_code},
        verbose="ERROR",
    )
    minimum = int(definition["minimum_count"])
    if len(events) < minimum:
        raise ValueError(f"Too few {condition!r} events for reviewed epoching")
    window = tuple(float(value) for value in definition["window_seconds"])
    baseline = tuple(float(value) for value in definition["baseline_seconds"])
    epochs = mne.Epochs(
        raw,
        events,
        event_id={condition: event_code},
        tmin=window[0],
        tmax=window[1],
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        on_missing="raise",
        verbose="ERROR",
    )
    if len(source_rows) != len(events):
        raise RuntimeError("Epoch events cannot be mapped to their BIDS rows")
    rows = []
    reasons = Counter()
    for index, ((_, source), event, drop_reasons) in enumerate(
        zip(source_rows.iterrows(), events, epochs.drop_log, strict=True)
    ):
        source_sample = int(float(source["sample"]))
        if int(event[0]) != source_sample:
            raise RuntimeError("Epoch event sample does not match its BIDS row")
        retained = not drop_reasons
        for reason in drop_reasons:
            reasons[str(reason)] += 1
        rows.append(
            {
                "condition_event_index": index,
                "source_bids_row_index": int(source["source_bids_row_index"]),
                "source_onset_s": float(source["onset"]),
                "source_duration": str(source["duration"]),
                "source_sample": source_sample,
                "event_sample": int(event[0]),
                "condition": condition,
                "retained": retained,
                "drop_reasons": ";".join(str(reason) for reason in drop_reasons),
            }
        )
    accounting = {
        "input_events": len(events),
        "retained_epochs": len(epochs),
        "dropped_epochs": len(events) - len(epochs),
        "drop_reasons": dict(sorted(reasons.items())),
        "window_seconds": list(window),
        "baseline_seconds": list(baseline),
    }
    if len(epochs) < minimum:
        raise ValueError(
            f"Too few retained {condition!r} epochs after reviewed annotation "
            "rejection"
        )
    return epochs, pd.DataFrame(rows), accounting


def _verify_context_unchanged(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    ica_decision_path: Path,
    participant_id: str,
    initial: dict,
) -> None:
    final = _context(
        inputs,
        profile_path,
        qc_path,
        review_bundle_path,
        ica_review_path,
        ica_decision_path,
        participant_id,
    )
    identity_keys = (
        "profile_sha256",
        "inputs",
        "source_manifest",
        "runtime",
        "ica_identity",
        "ica_decision_identity",
    )
    if any(final[key] != initial[key] for key in identity_keys):
        raise ValueError("Preprocessing context changed before publication")
    if final["manual_review"].identity != initial["manual_review"].identity:
        raise ValueError("Manual review bundle changed before publication")


def _context(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    ica_decision_path: Path,
    participant_id: str,
) -> dict:
    profile, profile_sha256 = _captured_json_sha(profile_path)
    initial_inputs = input_identities(resolve_inputs(inputs))
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    manual_review = load_verified_review_bundle(
        inputs, qc_path, review_bundle_path, participant_id
    )
    ica_summary, ica_provenance, ica_identity = _ica_review_context(ica_review_path)
    decision_summary, decision_provenance, decision_identity = ica_decision_context(
        ica_decision_path
    )
    if ica_provenance.get("inputs") != initial_inputs:
        raise ValueError("ICA review does not match the selected dataset inputs")
    if ica_provenance.get("decision_bundle") != manual_review.identity:
        raise ValueError("ICA review does not match the manual review bundle")
    if ica_provenance.get("controls", {}).get("profile_sha256") != profile_sha256:
        raise ValueError("ICA review does not match the processing profile")
    if ica_provenance.get("controls", {}).get("source_manifest") != initial_source:
        raise ValueError("ICA review source context is stale")
    if ica_provenance.get("runtime") != initial_runtime:
        raise ValueError("ICA review runtime context is incompatible")
    if decision_provenance.get("inputs") != initial_inputs:
        raise ValueError("ICA decisions do not match the selected dataset inputs")
    if decision_provenance.get("manual_review_bundle") != manual_review.identity:
        raise ValueError("ICA decisions do not match the manual review bundle")
    if decision_provenance.get("ica_review") != ica_identity:
        raise ValueError("ICA decisions do not match the ICA solution package")
    if decision_provenance.get("source_manifest") != initial_source:
        raise ValueError("ICA decision source context is stale")
    if decision_provenance.get("runtime") != initial_runtime:
        raise ValueError("ICA decision runtime context is incompatible")
    return {
        "profile": profile,
        "profile_sha256": profile_sha256,
        "inputs": initial_inputs,
        "source_manifest": initial_source,
        "runtime": initial_runtime,
        "manual_review": manual_review,
        "ica_summary": ica_summary,
        "ica_provenance": ica_provenance,
        "ica_identity": ica_identity,
        "ica_decision_summary": decision_summary,
        "ica_decision_provenance": decision_provenance,
        "ica_decision_identity": decision_identity,
    }


def _run_preprocess(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    ica_decision_path: Path,
    participant_id: str,
    context: dict,
    output: Path,
) -> dict:
    raw, manual_review, processing = prepare_filtered_reviewed_raw(
        inputs,
        profile_path,
        qc_path,
        review_bundle_path,
        participant_id,
        "epochs",
        expected_profile_sha256=context["profile_sha256"],
    )
    if manual_review.identity != context["manual_review"].identity:
        raise ValueError("Manual review bundle changed during preprocessing")
    if processing.get("review_bundle") != context["manual_review"].identity:
        raise ValueError("Prepared data does not match the initial review bundle")
    output.mkdir(parents=True, exist_ok=False)
    solution_path = ica_review_path / "ica_solution.fif"
    solution_bytes = solution_path.read_bytes()
    captured_solution = output / ".ica_solution_input.fif"
    captured_solution.write_bytes(solution_bytes)
    if sha256_file(captured_solution) != context["ica_identity"]["solution_sha256"]:
        raise ValueError("ICA solution changed before preprocessing")
    ica = mne.preprocessing.read_ica(captured_solution, verbose="ERROR")
    captured_solution.unlink()
    missing = sorted(set(ica.ch_names) - set(raw.ch_names))
    if missing:
        raise ValueError(f"ICA solution channels are absent from the input: {missing}")
    decision_table, decision_sha256 = _captured_csv(
        ica_decision_path / "ica_decisions.csv"
    )
    if decision_sha256 != context["ica_decision_identity"]["decisions_sha256"]:
        raise ValueError("ICA decisions changed before preprocessing")
    decisions = _validate_ica_decisions(
        decision_table,
        participant_id=participant_id,
        solution_sha256=context["ica_identity"]["solution_sha256"],
        components=int(ica.n_components_),
    )
    excluded = (
        decisions.loc[decisions["decision"] == "exclude", "component"]
        .astype(int)
        .tolist()
    )
    if excluded != context["ica_decision_summary"]["excluded_components"]:
        raise ValueError("ICA decision summary does not match the applied exclusions")
    ica.apply(raw, exclude=excluded, verbose="ERROR")
    interpolated = list(manual_review.bad_channels)
    if interpolated:
        raw.interpolate_bads(reset_bads=True, method={"eeg": "spline"}, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    raw.save(output / "cleaned_raw.fif", overwrite=False, fmt="single", verbose="ERROR")
    resolved = resolve_inputs(inputs)
    input_by_role = {item["role"]: item for item in context["inputs"]}
    event_table = pd.read_csv(
        BytesIO(
            _capture(
                resolved.events_tsv,
                input_by_role["events_tsv"]["sha256"],
            )
        ),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    event_table.insert(0, "source_bids_row_index", range(len(event_table)))
    epoch_summaries = {}
    for event_code, (condition, definition) in enumerate(
        sorted(context["profile"]["events"]["epoch_definitions"].items()),
        start=1,
    ):
        source_rows = event_table.loc[event_table["value"] == condition]
        epochs, lineage, accounting = _epoch_condition(
            raw, condition, definition, event_code, source_rows
        )
        epochs.save(
            output / f"{condition}-epo.fif",
            overwrite=False,
            fmt="single",
            verbose="ERROR",
        )
        lineage.to_csv(output / f"{condition}_epoch_lineage.csv", index=False)
        epoch_summaries[condition] = accounting

    summary = {
        "schema_version": "1",
        "status": "complete_reviewed_preprocessing",
        "participant_id": participant_id,
        "processing": processing,
        "ica": {
            "solution_sha256": context["ica_identity"]["solution_sha256"],
            "components": int(ica.n_components_),
            "excluded_components": excluded,
            "decision_record_complete": True,
            "automatic_component_exclusion": False,
        },
        "interpolated_channels": interpolated,
        "post_interpolation_reference": "average",
        "epochs": epoch_summaries,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    core = {
        "schema_version": "1",
        "workflow": "bids_eeglab_reviewed_preprocessing",
        "participant_id": participant_id,
        "inputs": context["inputs"],
        "controls": {
            "profile_sha256": context["profile_sha256"],
            "source_manifest": context["source_manifest"],
        },
        "runtime": context["runtime"],
        "manual_review_bundle": dict(manual_review.identity),
        "ica_review": context["ica_identity"],
        "ica_decisions": context["ica_decision_identity"],
        "processing_sha256": canonical_sha256(processing),
        "summary_sha256": canonical_sha256(summary),
        "automatic_component_exclusion": False,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)
    write_provenance(output, core)
    verify_provenance(output)
    return summary


def publish_reviewed_preprocessing(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    review_bundle_path: Path,
    ica_review_path: Path,
    ica_decision_path: Path,
    participant_id: str,
    output: Path,
) -> Path:
    """Publish reviewed continuous and epoch outputs atomically."""
    paths = [
        Path(path).expanduser().absolute()
        for path in (
            profile_path,
            qc_path,
            review_bundle_path,
            ica_review_path,
            ica_decision_path,
        )
    ]
    profile_path, qc_path, review_bundle_path, ica_review_path, ica_decision_path = (
        paths
    )
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Reviewed preprocessing requires a new output path")
    resolved = resolve_inputs(inputs)
    protected = [
        resolved.dataset_root.resolve(strict=True),
        qc_path.resolve(strict=True),
        review_bundle_path.resolve(strict=True),
        ica_review_path.resolve(strict=True),
        ica_decision_path.resolve(strict=True),
    ]
    nearest = _nearest_existing_ancestor(output.parent).resolve(strict=True)
    if any(nearest.is_relative_to(root) for root in protected):
        raise ValueError("Preprocessing output must remain outside source and inputs")
    initial = _context(
        inputs,
        profile_path,
        qc_path,
        review_bundle_path,
        ica_review_path,
        ica_decision_path,
        participant_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    temporary.rmdir()
    published = False
    try:
        _run_preprocess(
            inputs,
            profile_path,
            qc_path,
            review_bundle_path,
            ica_review_path,
            ica_decision_path,
            participant_id,
            initial,
            temporary,
        )
        _verify_context_unchanged(
            inputs,
            profile_path,
            qc_path,
            review_bundle_path,
            ica_review_path,
            ica_decision_path,
            participant_id,
            initial,
        )
        verify_provenance(temporary)
        _verify_context_unchanged(
            inputs,
            profile_path,
            qc_path,
            review_bundle_path,
            ica_review_path,
            ica_decision_path,
            participant_id,
            initial,
        )
        temporary.replace(output)
        published = True
        verify_provenance(output)
        _verify_context_unchanged(
            inputs,
            profile_path,
            qc_path,
            review_bundle_path,
            ica_review_path,
            ica_decision_path,
            participant_id,
            initial,
        )
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
