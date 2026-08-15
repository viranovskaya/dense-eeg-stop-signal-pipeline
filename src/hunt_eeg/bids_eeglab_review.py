"""Finalize human channel and segment decisions for one BIDS/EEGLAB QC package."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import tempfile
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd

from .bids_eeglab import BIDSEeglabInputs, input_identities, resolve_inputs
from .provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)

CHANNEL_COLUMNS = [
    "channel",
    "candidate_reason",
    "decision",
    "reviewer",
    "reviewed_at",
    "evidence",
    "notes",
]
SEGMENT_COLUMNS = [
    "candidate_id",
    "channel",
    "prompt_onset_s",
    "prompt_duration_s",
    "candidate_reason",
    "decision",
    "refined_onset_s",
    "refined_duration_s",
    "scope",
    "reviewer",
    "reviewed_at",
    "evidence",
    "notes",
]


def _captured_tsv(path: Path) -> tuple[pd.DataFrame, str]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Decision input must be a regular non-symlink file")
    data = path.read_bytes()
    table = pd.read_csv(BytesIO(data), sep="\t", dtype=str, keep_default_na=False)
    return table, hashlib.sha256(data).hexdigest()


def _strip_columns(table: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    if table.columns.tolist() != columns:
        raise ValueError(f"{label} columns do not match the review template")
    table = table.copy()
    for column in columns:
        table[column] = table[column].str.strip()
    return table


def _validate_review_metadata(table: pd.DataFrame, label: str) -> None:
    required = ["reviewer", "reviewed_at", "evidence"]
    empty = [column for column in required if (table[column] == "").any()]
    if empty:
        raise ValueError(f"{label} contains empty review fields: {', '.join(empty)}")
    for value in table["reviewed_at"].unique():
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Review date must use YYYY-MM-DD: {value!r}") from error


def _require_rationale(table: pd.DataFrame, mask: pd.Series, label: str) -> None:
    if (table.loc[mask, "notes"] == "").any():
        raise ValueError(f"{label} require a non-empty rationale in notes")


def _runtime_identity() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "dependency_files": {
            name: sha256_file(project_root / name)
            for name in ("requirements.txt", "constraints-ci.txt")
        },
    }


def _nearest_existing_ancestor(path: Path) -> Path:
    current = Path(path).expanduser().absolute()
    while not current.exists() and not current.is_symlink():
        current = current.parent
    return current


def _qc_context(path: Path) -> tuple[dict, dict, dict]:
    root = Path(path).expanduser().absolute()
    provenance = verify_provenance(root)
    summary_path = root / "qc_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_sha256(summary) != provenance.get("summary_sha256"):
        raise ValueError("QC summary does not match its provenance")
    if summary.get("publication_allowed") is not False:
        raise ValueError("QC package does not have the controlled review boundary")
    identity = {
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
        "summary_sha256": sha256_file(summary_path),
        "channel_template_sha256": sha256_file(root / "channel_review_template.tsv"),
        "segment_template_sha256": sha256_file(root / "segment_review_template.tsv"),
    }
    return summary, provenance, identity


def _validate_channel_decisions(qc_root: Path, completed: pd.DataFrame) -> pd.DataFrame:
    expected, _ = _captured_tsv(qc_root / "channel_review_template.tsv")
    expected = _strip_columns(expected, CHANNEL_COLUMNS, "QC channel template")
    completed = _strip_columns(completed, CHANNEL_COLUMNS, "Channel decisions")
    if (
        expected["channel"].duplicated().any()
        or completed["channel"].duplicated().any()
    ):
        raise ValueError("Channel review contains duplicate channel rows")
    expected = expected.sort_values("channel", kind="stable").reset_index(drop=True)
    completed = completed.sort_values("channel", kind="stable").reset_index(drop=True)
    expected_rows = completed.loc[completed["channel"].isin(expected["channel"])].copy()
    expected_rows = expected_rows.sort_values("channel", kind="stable").reset_index(
        drop=True
    )
    if expected_rows["channel"].tolist() != expected["channel"].tolist():
        raise ValueError("Channel decisions must cover every QC candidate exactly once")
    if (
        expected_rows["candidate_reason"].tolist()
        != expected["candidate_reason"].tolist()
    ):
        raise ValueError("Channel candidate reasons must remain unchanged")
    additions = completed.loc[~completed["channel"].isin(expected["channel"])]
    if not additions.empty:
        known = set(
            pd.read_csv(qc_root / "line_noise_metrics.csv", usecols=["channel"])[
                "channel"
            ].astype(str)
        )
        if not set(additions["channel"]).issubset(known):
            raise ValueError("Manual channel additions must name a screened channel")
        if set(additions["candidate_reason"]) != {"manual_addition"}:
            raise ValueError("Manual channel additions require manual_addition reason")
    invalid = sorted(set(completed["decision"]) - {"keep", "interpolate"})
    if invalid:
        raise ValueError(f"Invalid channel decisions: {invalid}")
    _validate_review_metadata(completed, "Channel decisions")
    _require_rationale(
        completed,
        (completed["decision"] == "interpolate")
        | (completed["candidate_reason"] == "manual_addition"),
        "Interpolations and manual channel additions",
    )
    return completed


def _validate_segment_decisions(
    qc_root: Path, completed: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    expected, _ = _captured_tsv(qc_root / "segment_review_template.tsv")
    expected = _strip_columns(expected, SEGMENT_COLUMNS, "QC segment template")
    completed = _strip_columns(completed, SEGMENT_COLUMNS, "Segment decisions")
    if (
        expected["candidate_id"].duplicated().any()
        or completed["candidate_id"].duplicated().any()
    ):
        raise ValueError("Segment review contains duplicate candidate IDs")
    expected = expected.sort_values("candidate_id", kind="stable").reset_index(
        drop=True
    )
    completed = completed.sort_values("candidate_id", kind="stable").reset_index(
        drop=True
    )
    expected_rows = completed.loc[
        completed["candidate_id"].isin(expected["candidate_id"])
    ].copy()
    expected_rows = expected_rows.sort_values(
        "candidate_id", kind="stable"
    ).reset_index(drop=True)
    if expected_rows["candidate_id"].tolist() != expected["candidate_id"].tolist():
        raise ValueError("Segment decisions must cover every QC prompt exactly once")
    for column in ("channel", "candidate_reason"):
        if expected_rows[column].tolist() != expected[column].tolist():
            raise ValueError(f"Segment {column} values must remain unchanged")
    for column in ("prompt_onset_s", "prompt_duration_s"):
        try:
            observed = pd.to_numeric(expected_rows[column], errors="raise").to_numpy(
                float
            )
            reference = pd.to_numeric(expected[column], errors="raise").to_numpy(float)
        except ValueError as error:
            raise ValueError("Segment prompt timing must remain numeric") from error
        if not all(math.isfinite(value) for value in observed) or not all(
            math.isfinite(value) for value in reference
        ):
            raise ValueError("Segment prompt timing must remain finite")
        if not pd.Series(observed).round(9).equals(pd.Series(reference).round(9)):
            raise ValueError("Segment prompt timing must remain unchanged")

    additions = completed.loc[~completed["candidate_id"].isin(expected["candidate_id"])]
    if not additions.empty:
        if not additions["candidate_id"].str.startswith("manual-").all():
            raise ValueError("Manual segment IDs must start with manual-")
        if set(additions["candidate_reason"]) != {"manual_addition"}:
            raise ValueError("Manual segments require manual_addition reason")
        known = set(
            pd.read_csv(qc_root / "line_noise_metrics.csv", usecols=["channel"])[
                "channel"
            ].astype(str)
        ) | {"all"}
        if not set(additions["channel"]).issubset(known):
            raise ValueError("Manual segments must name a screened channel or all")
        if set(additions["decision"]) != {"exclude"}:
            raise ValueError("Manual segment additions must define an exclusion")

    invalid = sorted(set(completed["decision"]) - {"keep", "exclude"})
    if invalid:
        raise ValueError(f"Invalid segment decisions: {invalid}")
    _validate_review_metadata(completed, "Segment decisions")
    _require_rationale(
        completed,
        (completed["decision"] == "exclude")
        | (completed["candidate_reason"] == "manual_addition"),
        "Exclusions and manual segment additions",
    )
    qc_summary = json.loads((qc_root / "qc_summary.json").read_text(encoding="utf-8"))
    recording_duration = float(qc_summary["recording_duration_s"])
    if not math.isfinite(recording_duration) or recording_duration <= 0:
        raise ValueError("QC recording duration must be finite and positive")
    for row in completed.itertuples(index=False):
        if row.decision == "keep":
            if row.refined_onset_s or row.refined_duration_s or row.scope:
                raise ValueError("Kept segment prompts must not define an exclusion")
            continue
        if row.scope not in {"ica", "epochs", "both"}:
            raise ValueError("Excluded segments require scope: ica, epochs, or both")
        try:
            onset = float(row.refined_onset_s)
            duration = float(row.refined_duration_s)
        except ValueError as error:
            raise ValueError(
                "Excluded segments require numeric refined timing"
            ) from error
        if (
            not math.isfinite(onset)
            or not math.isfinite(duration)
            or onset < 0
            or duration <= 0
            or onset + duration > recording_duration + 1e-9
        ):
            raise ValueError("Excluded segment timing is outside the recording")
        if row.candidate_reason == "manual_addition":
            if row.prompt_onset_s or row.prompt_duration_s:
                raise ValueError("Manual segments must leave QC prompt timing empty")
            continue
        prompt_onset = float(row.prompt_onset_s)
        prompt_duration = float(row.prompt_duration_s)
        if (
            not math.isfinite(prompt_onset)
            or not math.isfinite(prompt_duration)
            or prompt_onset < 0
            or prompt_duration <= 0
            or prompt_onset + prompt_duration > recording_duration + 1e-9
        ):
            raise ValueError("Segment prompt timing is outside the recording")
        if min(onset + duration, prompt_onset + prompt_duration) <= max(
            onset, prompt_onset
        ):
            raise ValueError("Refined segment exclusion must overlap its prompt")
    return completed, recording_duration


def _merged_intervals(table: pd.DataFrame) -> list[dict]:
    rows = []
    excluded = table.loc[table["decision"] == "exclude"]
    for scope in ("ica", "epochs", "both"):
        selected = excluded.loc[excluded["scope"] == scope].copy()
        if selected.empty:
            continue
        selected["onset"] = selected["refined_onset_s"].astype(float)
        selected["stop"] = selected["onset"] + selected["refined_duration_s"].astype(
            float
        )
        selected = selected.sort_values(["onset", "stop"], kind="stable")
        current = None
        for row in selected.itertuples(index=False):
            if current is None or row.onset > current["stop_s"]:
                if current is not None:
                    rows.append(current)
                current = {
                    "onset_s": float(row.onset),
                    "stop_s": float(row.stop),
                    "scope": scope,
                    "candidate_ids": [row.candidate_id],
                }
            else:
                current["stop_s"] = max(current["stop_s"], float(row.stop))
                current["candidate_ids"].append(row.candidate_id)
        if current is not None:
            rows.append(current)
    for row in rows:
        row["duration_s"] = row["stop_s"] - row["onset_s"]
        row["candidate_ids"] = sorted(set(row["candidate_ids"]))
    return rows


def _effective_intervals(table: pd.DataFrame, target: str) -> list[dict]:
    """Merge exclusions that apply to one downstream target."""
    selected = table.loc[
        (table["decision"] == "exclude") & table["scope"].isin([target, "both"])
    ].copy()
    if selected.empty:
        return []
    selected["onset"] = selected["refined_onset_s"].astype(float)
    selected["stop"] = selected["onset"] + selected["refined_duration_s"].astype(float)
    selected = selected.sort_values(["onset", "stop"], kind="stable")
    merged = []
    current = None
    for row in selected.itertuples(index=False):
        if current is None or row.onset > current["stop_s"]:
            if current is not None:
                merged.append(current)
            current = {
                "onset_s": float(row.onset),
                "stop_s": float(row.stop),
                "scope": target,
                "candidate_ids": [row.candidate_id],
            }
        else:
            current["stop_s"] = max(current["stop_s"], float(row.stop))
            current["candidate_ids"].append(row.candidate_id)
    if current is not None:
        merged.append(current)
    for row in merged:
        row["duration_s"] = row["stop_s"] - row["onset_s"]
        row["candidate_ids"] = sorted(set(row["candidate_ids"]))
    return merged


def _participant_id(value: str) -> str:
    value = str(value)
    if not value or len(value) > 64 or not value.isascii() or not value.isalnum():
        raise ValueError("Participant ID must be 1-64 ASCII letters or digits")
    return value


def _decision_summary(
    channel_table: pd.DataFrame,
    segment_table: pd.DataFrame,
    participant_id: str,
    recording_duration: float,
) -> dict:
    """Build the canonical machine summary from validated review tables."""
    intervals = _merged_intervals(segment_table)
    ica_intervals = _effective_intervals(segment_table, "ica")
    epoch_intervals = _effective_intervals(segment_table, "epochs")
    return {
        "schema_version": "1",
        "status": "complete",
        "participant_id": participant_id,
        "channel_decisions": {
            "rows": len(channel_table),
            "qc_prompt_rows": int(
                (channel_table["candidate_reason"] != "manual_addition").sum()
            ),
            "manual_addition_rows": int(
                (channel_table["candidate_reason"] == "manual_addition").sum()
            ),
            "keep": int((channel_table["decision"] == "keep").sum()),
            "interpolate": sorted(
                channel_table.loc[
                    channel_table["decision"] == "interpolate", "channel"
                ].tolist()
            ),
        },
        "segment_decisions": {
            "rows": len(segment_table),
            "qc_prompt_rows": int(
                (segment_table["candidate_reason"] != "manual_addition").sum()
            ),
            "manual_addition_rows": int(
                (segment_table["candidate_reason"] == "manual_addition").sum()
            ),
            "keep": int((segment_table["decision"] == "keep").sum()),
            "exclude": int((segment_table["decision"] == "exclude").sum()),
            "merged_exclusion_intervals": intervals,
            "ica_exclusion_intervals": ica_intervals,
            "epoch_exclusion_intervals": epoch_intervals,
            "recording_duration_s": recording_duration,
            "exclusion_application": "global_recording_intervals_all_channels",
            "channel_field_role": "evidence_origin_not_exclusion_scope",
        },
        "reviewers": sorted(
            set(channel_table["reviewer"]) | set(segment_table["reviewer"])
        ),
        "automatic_decisions": 0,
        "classification": "controlled_derived",
        "publication_allowed": False,
    }


def finalize_review_decisions(
    inputs: BIDSEeglabInputs,
    qc_path: Path,
    channel_decisions_path: Path,
    segment_decisions_path: Path,
    participant_id: str,
    output: Path,
) -> Path:
    """Create an immutable exact-set bundle from completed human decisions."""
    qc_root = Path(qc_path).expanduser().absolute()
    channel_path = Path(channel_decisions_path).expanduser().absolute()
    segment_path = Path(segment_decisions_path).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Decision finalization requires a new output path")
    resolved_inputs = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved_inputs)
    dataset_root = resolved_inputs.dataset_root.resolve(strict=True)
    qc_resolved = qc_root.resolve(strict=True)
    nearest = _nearest_existing_ancestor(output.parent)
    nearest_resolved = nearest.resolve(strict=True)
    if nearest_resolved.is_relative_to(qc_resolved):
        raise ValueError("Decision output must remain outside the immutable QC package")
    if nearest_resolved.is_relative_to(dataset_root):
        raise ValueError("Decision output must remain outside the source dataset")
    participant_id = _participant_id(participant_id)
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    qc_summary, qc_provenance, qc_identity = _qc_context(qc_root)
    bound_participant = str(qc_summary.get("recording", {}).get("participant_id", ""))
    if participant_id != bound_participant:
        raise ValueError("Participant ID does not match the QC recording")
    if qc_provenance.get("controls", {}).get("source_manifest") != initial_source:
        raise ValueError("QC source context is stale for decision finalization")
    if qc_provenance.get("inputs") != initial_inputs:
        raise ValueError("QC package does not match the selected dataset inputs")
    if qc_provenance.get("runtime") != initial_runtime:
        raise ValueError("QC runtime context is incompatible with finalization")
    channel_table, channel_sha = _captured_tsv(channel_path)
    segment_table, segment_sha = _captured_tsv(segment_path)
    channel_table = _validate_channel_decisions(qc_root, channel_table)
    segment_table, recording_duration = _validate_segment_decisions(
        qc_root, segment_table
    )
    summary = _decision_summary(
        channel_table, segment_table, participant_id, recording_duration
    )
    decision_inputs = {
        "channel_decisions_sha256": channel_sha,
        "segment_decisions_sha256": segment_sha,
    }
    core = {
        "schema_version": "1",
        "workflow": "bids_eeglab_manual_review",
        "participant_id": participant_id,
        "qc": qc_identity,
        "inputs": initial_inputs,
        "decision_inputs": decision_inputs,
        "source_manifest": initial_source,
        "runtime": initial_runtime,
        "summary_sha256": canonical_sha256(summary),
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        channel_table.to_csv(temporary / "channel_decisions.tsv", sep="\t", index=False)
        segment_table.to_csv(temporary / "segment_decisions.tsv", sep="\t", index=False)
        (temporary / "decision_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if (
            sha256_file(channel_path) != channel_sha
            or sha256_file(segment_path) != segment_sha
        ):
            raise ValueError("Decision inputs changed during finalization")
        if source_manifest() != initial_source:
            raise ValueError("Executable source changed during finalization")
        if _runtime_identity() != initial_runtime:
            raise ValueError("Runtime context changed during finalization")
        if input_identities(resolve_inputs(inputs)) != initial_inputs:
            raise ValueError("Dataset inputs changed during finalization")
        final_qc_summary, final_qc_provenance, final_qc_identity = _qc_context(qc_root)
        if final_qc_identity != qc_identity or final_qc_provenance != qc_provenance:
            raise ValueError("QC package changed during decision finalization")
        if final_qc_summary != qc_summary:
            raise ValueError("QC summary changed during decision finalization")
        write_provenance(temporary, core)
        verify_provenance(temporary)
        temporary.replace(output)
        published = True
        verify_provenance(output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
