"""Fail-closed preparation of BIDS/EEGLAB data from verified human decisions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .bids_eeglab import (
    BIDSEeglabInputs,
    _captured_json_sha,
    _read_raw,
    input_identities,
    resolve_inputs,
)
from .bids_eeglab_review import (
    CHANNEL_COLUMNS,
    SEGMENT_COLUMNS,
    _decision_summary,
    _participant_id,
    _qc_context,
    _runtime_identity,
    _strip_columns,
    _validate_channel_decisions,
    _validate_segment_decisions,
)
from .provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
)


@dataclass(frozen=True)
class VerifiedReviewBundle:
    """Immutable decisions and identities authorized for downstream preparation."""

    participant_id: str
    bad_channels: tuple[str, ...]
    ica_intervals: tuple[dict, ...]
    epoch_intervals: tuple[dict, ...]
    channel_decisions: tuple[dict, ...]
    segment_decisions: tuple[dict, ...]
    identity: dict


@dataclass(frozen=True)
class ProcessingControls:
    """Profile-bound filtering and referencing decisions."""

    filter_hz: tuple[float, float]
    filter_method: str
    segment_guard: str
    average_reference: str


def _processing_controls(profile: dict) -> ProcessingControls:
    payload = profile.get("processing", {})
    required = {
        "filter_hz",
        "filter_method",
        "segment_guard",
        "average_reference",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Processing profile is missing fields: {missing}")
    cutoffs = tuple(float(value) for value in payload["filter_hz"])
    if (
        len(cutoffs) != 2
        or not all(math.isfinite(value) for value in cutoffs)
        or not 0 < cutoffs[0] < cutoffs[1]
    ):
        raise ValueError("Processing filter_hz must contain positive finite cutoffs")
    if payload["filter_method"] != "firwin_zero_phase_hamming":
        raise ValueError("Only the declared zero-phase FIR filter is supported")
    if payload["segment_guard"] != "half_filter_support":
        raise ValueError("Segment guard must use half_filter_support")
    expected_reference = "all_good_eeg_including_flat_online_reference"
    if payload["average_reference"] != expected_reference:
        raise ValueError("Processing requires the declared average-reference policy")
    return ProcessingControls(
        filter_hz=cutoffs,
        filter_method=payload["filter_method"],
        segment_guard=payload["segment_guard"],
        average_reference=payload["average_reference"],
    )


def _read_bundle_table(root: Path, name: str, columns: list[str]) -> pd.DataFrame:
    table = pd.read_csv(
        root / name,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    return _strip_columns(table, columns, name)


def load_verified_review_bundle(
    inputs: BIDSEeglabInputs,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
) -> VerifiedReviewBundle:
    """Revalidate a complete review bundle against its QC and source recording."""
    participant_id = _participant_id(participant_id)
    qc_root = Path(qc_path).expanduser().absolute()
    bundle_root = Path(bundle_path).expanduser().absolute()
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    resolved = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved)
    qc_summary, qc_provenance, qc_identity = _qc_context(qc_root)
    provenance = verify_provenance(bundle_root)
    if provenance.get("workflow") != "bids_eeglab_manual_review":
        raise ValueError("Decision bundle has the wrong workflow")
    if provenance.get("participant_id") != participant_id:
        raise ValueError("Decision bundle participant does not match the request")
    if provenance.get("inputs") != initial_inputs:
        raise ValueError("Decision bundle does not match the selected dataset inputs")
    if provenance.get("qc") != qc_identity:
        raise ValueError("Decision bundle does not match the selected QC package")
    if provenance.get("source_manifest") != initial_source:
        raise ValueError("Decision bundle source context is stale")
    if provenance.get("runtime") != initial_runtime:
        raise ValueError("Decision bundle runtime context is incompatible")
    if qc_provenance.get("inputs") != initial_inputs:
        raise ValueError("QC package does not match the selected dataset inputs")
    if qc_provenance.get("controls", {}).get("source_manifest") != initial_source:
        raise ValueError("QC package source context is stale")
    if qc_provenance.get("runtime") != initial_runtime:
        raise ValueError("QC package runtime context is incompatible")
    if qc_summary.get("recording", {}).get("participant_id") != participant_id:
        raise ValueError("QC participant does not match the request")

    channel_table = _read_bundle_table(
        bundle_root, "channel_decisions.tsv", CHANNEL_COLUMNS
    )
    segment_table = _read_bundle_table(
        bundle_root, "segment_decisions.tsv", SEGMENT_COLUMNS
    )
    channel_table = _validate_channel_decisions(qc_root, channel_table)
    segment_table, duration = _validate_segment_decisions(qc_root, segment_table)
    expected_summary = _decision_summary(
        channel_table, segment_table, participant_id, duration
    )
    summary_path = bundle_root / "decision_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary != expected_summary:
        raise ValueError("Decision summary does not match the validated review tables")
    if canonical_sha256(summary) != provenance.get("summary_sha256"):
        raise ValueError("Decision summary does not match bundle provenance")
    if summary.get("status") != "complete" or summary.get("automatic_decisions") != 0:
        raise ValueError("Decision bundle is not a complete manual review")
    if summary.get("publication_allowed") is not False:
        raise ValueError("Decision bundle does not retain the controlled boundary")

    final_qc_summary, final_qc_provenance, final_qc_identity = _qc_context(qc_root)
    if (
        final_qc_summary != qc_summary
        or final_qc_provenance != qc_provenance
        or final_qc_identity != qc_identity
    ):
        raise ValueError("QC package changed while loading the decision bundle")
    if verify_provenance(bundle_root) != provenance:
        raise ValueError("Decision bundle changed while it was loaded")
    if input_identities(resolve_inputs(inputs)) != initial_inputs:
        raise ValueError("Dataset inputs changed while loading the decision bundle")
    if source_manifest() != initial_source or _runtime_identity() != initial_runtime:
        raise ValueError("Source or runtime changed while loading the decision bundle")

    identity = {
        "provenance_sha256": sha256_file(bundle_root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
        "summary_sha256": sha256_file(summary_path),
    }
    return VerifiedReviewBundle(
        participant_id=participant_id,
        bad_channels=tuple(summary["channel_decisions"]["interpolate"]),
        ica_intervals=tuple(summary["segment_decisions"]["ica_exclusion_intervals"]),
        epoch_intervals=tuple(
            summary["segment_decisions"]["epoch_exclusion_intervals"]
        ),
        channel_decisions=tuple(channel_table.to_dict("records")),
        segment_decisions=tuple(segment_table.to_dict("records")),
        identity=identity,
    )


def _append_global_bad_intervals(
    raw: mne.io.BaseRaw,
    intervals: tuple[dict, ...],
    target: str,
    *,
    prefix: str = "BAD_review",
) -> None:
    if not intervals:
        return
    annotations = mne.Annotations(
        onset=[float(item["onset_s"]) for item in intervals],
        duration=[float(item["duration_s"]) for item in intervals],
        description=[f"{prefix}_{target}"] * len(intervals),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + annotations)


def _fir_guard_seconds(sfreq: float, controls: ProcessingControls) -> float:
    taps = mne.filter.create_filter(
        None,
        sfreq,
        controls.filter_hz[0],
        controls.filter_hz[1],
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        verbose="ERROR",
    )
    return float((len(taps) - 1) / (2 * sfreq))


def _guarded_intervals(
    intervals: tuple[dict, ...], duration: float, guard: float
) -> tuple[dict, ...]:
    expanded = []
    for item in intervals:
        start = max(0.0, float(item["onset_s"]) - guard)
        stop = min(duration, float(item["stop_s"]) + guard)
        if stop > start:
            expanded.append((start, stop, tuple(item["candidate_ids"])))
    merged: list[dict] = []
    for start, stop, candidate_ids in sorted(expanded):
        if not merged or start > merged[-1]["stop_s"]:
            merged.append(
                {
                    "onset_s": start,
                    "stop_s": stop,
                    "duration_s": stop - start,
                    "candidate_ids": list(candidate_ids),
                }
            )
        else:
            merged[-1]["stop_s"] = max(merged[-1]["stop_s"], stop)
            merged[-1]["duration_s"] = (
                merged[-1]["stop_s"] - merged[-1]["onset_s"]
            )
            merged[-1]["candidate_ids"] = sorted(
                set(merged[-1]["candidate_ids"]) | set(candidate_ids)
            )
    return tuple(merged)


def prepare_reviewed_raw(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
    target: str,
    *,
    expected_profile_sha256: str | None = None,
) -> tuple[mne.io.BaseRaw, VerifiedReviewBundle]:
    """Load one recording with verified bad channels and global BAD intervals."""
    if target not in {"ica", "epochs"}:
        raise ValueError("Preparation target must be ica or epochs")
    profile_path = Path(profile_path).expanduser().absolute()
    profile, profile_sha = _captured_json_sha(profile_path)
    if (
        expected_profile_sha256 is not None
        and profile_sha != expected_profile_sha256
    ):
        raise ValueError("Processing profile does not match its initial capture")
    review = load_verified_review_bundle(inputs, qc_path, bundle_path, participant_id)
    qc_provenance = verify_provenance(Path(qc_path).expanduser().absolute())
    if qc_provenance.get("controls", {}).get("profile_sha256") != profile_sha:
        raise ValueError("Processing profile does not match the reviewed QC profile")

    resolved = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved)
    raw = _read_raw(resolved.raw_set)
    duration = float(raw.n_times / raw.info["sfreq"])
    reviewed_duration = float(
        json.loads(
            (Path(bundle_path) / "decision_summary.json").read_text(encoding="utf-8")
        )["segment_decisions"]["recording_duration_s"]
    )
    if abs(duration - reviewed_duration) > 1.0 / float(raw.info["sfreq"]):
        raise ValueError("Reviewed recording duration does not match the loaded data")

    unknown = sorted(set(review.bad_channels) - set(raw.ch_names))
    if unknown:
        raise ValueError(
            f"Reviewed bad channels are absent from the recording: {unknown}"
        )
    non_eeg = [
        name
        for name in review.bad_channels
        if raw.get_channel_types(picks=[name])[0] != "eeg"
    ]
    if non_eeg:
        raise ValueError(f"Only EEG channels may be interpolated: {non_eeg}")
    geometry = profile.get("geometry_validation")
    if review.bad_channels and not geometry:
        raise ValueError("Interpolation requires a profile-validated standard montage")
    if geometry:
        montage_name = str(geometry["standard_montage"])
        raw.set_montage(
            mne.channels.make_standard_montage(montage_name),
            match_case=True,
            on_missing="raise",
            verbose="ERROR",
        )
    raw.info["bads"] = list(review.bad_channels)
    intervals = review.ica_intervals if target == "ica" else review.epoch_intervals
    _append_global_bad_intervals(raw, intervals, target)

    if sha256_file(profile_path) != profile_sha:
        raise ValueError("Processing profile changed while preparing the recording")
    if input_identities(resolve_inputs(inputs)) != initial_inputs:
        raise ValueError("Dataset inputs changed while preparing the recording")
    reloaded = load_verified_review_bundle(inputs, qc_path, bundle_path, participant_id)
    if reloaded.identity != review.identity:
        raise ValueError("Decision bundle changed while preparing the recording")
    return raw, review


def prepare_filtered_reviewed_raw(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    bundle_path: Path,
    participant_id: str,
    target: str,
    *,
    expected_profile_sha256: str | None = None,
) -> tuple[mne.io.BaseRaw, VerifiedReviewBundle, dict]:
    """Prepare a filtered average-referenced input with guarded review masks."""
    profile_path = Path(profile_path).expanduser().absolute()
    profile, profile_sha = _captured_json_sha(profile_path)
    if (
        expected_profile_sha256 is not None
        and profile_sha != expected_profile_sha256
    ):
        raise ValueError("Processing profile does not match its initial capture")
    controls = _processing_controls(profile)
    raw, review = prepare_reviewed_raw(
        inputs,
        profile_path,
        qc_path,
        bundle_path,
        participant_id,
        target,
        expected_profile_sha256=profile_sha,
    )
    low_hz, high_hz = controls.filter_hz
    sfreq = float(raw.info["sfreq"])
    if high_hz >= sfreq / 2:
        raise ValueError("Processing high cutoff must remain below Nyquist")
    reference = str(profile["reference_channel"]["name"])
    if reference not in raw.ch_names or reference in review.bad_channels:
        raise ValueError("The flat online-reference channel must remain available")
    if profile["reference_channel"].get("exclude_from_reference_average") is not False:
        raise ValueError("The flat online reference must be included in average reference")
    if profile["reference_channel"].get("post_reference_policy") != (
        "include_flat_online_reference_in_average_transform"
    ):
        raise ValueError("The reference reconstruction policy is unsupported")
    reference_before = raw.get_data(picks=[reference])
    if float(np.ptp(reference_before)) > np.finfo(float).eps:
        raise ValueError("The declared online-reference channel is not flat")

    exact_intervals = review.ica_intervals if target == "ica" else review.epoch_intervals
    duration = float(raw.n_times / sfreq)
    guard = _fir_guard_seconds(sfreq, controls)
    guarded = _guarded_intervals(exact_intervals, duration, guard)
    _append_global_bad_intervals(
        raw, guarded, target, prefix="BAD_filter_guard"
    )
    raw.filter(
        low_hz,
        high_hz,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        skip_by_annotation=(
            "edge",
            "bad_acq_skip",
            "BAD_review",
            "BAD_filter_guard",
        ),
        verbose="ERROR",
    )
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    reference_after = raw.get_data(picks=[reference])
    if np.allclose(reference_after, 0.0, rtol=0, atol=np.finfo(float).eps):
        raise RuntimeError("Average reference did not reconstruct the online reference")
    summary = {
        "target": target,
        "filter_hz": list(controls.filter_hz),
        "filter": controls.filter_method,
        "filter_guard_seconds": guard,
        "guard_policy": controls.segment_guard,
        "exact_review_intervals": list(exact_intervals),
        "guarded_filter_intervals": list(guarded),
        "average_reference": controls.average_reference,
        "online_reference_channel": reference,
        "bad_channels_excluded_from_reference": list(review.bad_channels),
        "review_bundle": review.identity,
        "automatic_decisions": 0,
    }
    if sha256_file(profile_path) != profile_sha:
        raise ValueError("Processing profile changed during preparation")
    reloaded = load_verified_review_bundle(
        inputs, qc_path, bundle_path, participant_id
    )
    if reloaded.identity != review.identity:
        raise ValueError("Decision bundle changed during filtered preparation")
    return raw, review, summary
