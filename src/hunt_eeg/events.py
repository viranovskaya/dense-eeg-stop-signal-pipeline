"""Marker normalization and stop-signal trial reconstruction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

from .config import EventCodebook, load_analysis_config, load_event_codebook


_STIMULUS_PATTERN = re.compile(
    r"(?:(?:Stimulus|\d+)/\s*)?S\s*(\d+)\s*", re.IGNORECASE
)
_TRIAL_COLUMNS = [
    "trial_index",
    "trial_type",
    "trial_class",
    "classification_status",
    "classification_confidence",
    "classification_reason",
    "marker_sequence",
    "source_marker_start_index",
    "source_marker_stop_index_exclusive",
    "closed_by_next_trial",
    "stimulus_code",
    "stimulus_onset_s",
    "stop_signal_onset_s",
    "outcome_code",
    "ambiguity_code",
    "stimulus_to_outcome_ms",
    "stop_signal_to_outcome_ms",
    "stimulus_to_stop_signal_ms",
]


def normalize_marker(description: str) -> str:
    """Convert BrainVision/MNE descriptions such as ``Stimulus/S  17`` to ``S17``."""
    match = _STIMULUS_PATTERN.fullmatch(str(description).strip())
    if match:
        return f"S{int(match.group(1))}"
    return str(description).strip()


def annotations_to_markers(annotations: Iterable) -> list[dict]:
    """Return time-ordered normalized stimulus markers from MNE annotations."""
    markers: list[dict] = []
    for annotation in annotations:
        marker = normalize_marker(annotation["description"])
        if re.fullmatch(r"S\d+", marker):
            markers.append(
                {
                    "onset_s": float(annotation["onset"]),
                    "duration_s": float(annotation["duration"]),
                    "marker": marker,
                }
            )
    return markers


def _trial_row(
    *,
    trial_index: int,
    trial_type: str,
    trial_class: str,
    status: str,
    confidence: str | None,
    reason: str,
    segment: list[dict],
    segment_start_index: int,
    stop_signal: dict | None = None,
    outcome: dict | None = None,
    ambiguity: dict | None = None,
    closed_by_next_trial: bool,
) -> dict:
    stimulus = segment[0]
    stimulus_to_outcome_ms = None
    stop_signal_to_outcome_ms = None
    stimulus_to_stop_signal_ms = None
    if outcome is not None:
        stimulus_to_outcome_ms = (
            outcome["onset_s"] - stimulus["onset_s"]
        ) * 1000
    if stop_signal is not None:
        stimulus_to_stop_signal_ms = (
            stop_signal["onset_s"] - stimulus["onset_s"]
        ) * 1000
        if outcome is not None:
            stop_signal_to_outcome_ms = (
                outcome["onset_s"] - stop_signal["onset_s"]
            ) * 1000
    return {
        "trial_index": trial_index,
        "trial_type": trial_type,
        "trial_class": trial_class,
        "classification_status": status,
        "classification_confidence": confidence,
        "classification_reason": reason,
        "marker_sequence": ">".join(item["marker"] for item in segment),
        "source_marker_start_index": segment_start_index,
        "source_marker_stop_index_exclusive": segment_start_index + len(segment),
        "closed_by_next_trial": closed_by_next_trial,
        "stimulus_code": stimulus["marker"],
        "stimulus_onset_s": stimulus["onset_s"],
        "stop_signal_onset_s": (
            stop_signal["onset_s"] if stop_signal is not None else None
        ),
        "outcome_code": outcome["marker"] if outcome is not None else None,
        "ambiguity_code": ambiguity["marker"] if ambiguity is not None else None,
        "stimulus_to_outcome_ms": stimulus_to_outcome_ms,
        "stop_signal_to_outcome_ms": stop_signal_to_outcome_ms,
        "stimulus_to_stop_signal_ms": stimulus_to_stop_signal_ms,
    }


def classify_trials(
    markers: list[dict], codebook: EventCodebook | None = None,
    integrity_limits_ms: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Reconstruct trials while preserving incomplete and ambiguous sequences.

    The marker definitions are loaded from the project codebook. The strongest
    direct evidence for the mapping is one preserved recording's condition files;
    the other recordings provide sequence-level checks rather than independent
    ground truth.
    """
    codebook = codebook or load_event_codebook()
    integrity_limits_ms = (
        integrity_limits_ms or load_analysis_config().event_integrity_limits_ms
    )
    onsets = [float(marker["onset_s"]) for marker in markers]
    if any(current < previous for previous, current in zip(onsets, onsets[1:])):
        raise ValueError("Markers must be ordered by onset")

    go_starts = codebook.markers(role="trial_start", trial_type="go")
    stop_starts = codebook.markers(role="trial_start", trial_type="stop")
    trial_starts = go_starts | stop_starts
    stop_signals = codebook.markers(role="stop_signal", trial_type="stop")
    ambiguous_markers = codebook.markers(role="ambiguous")
    go_outcome_classes = codebook.outcome_classes("go")
    stop_outcome_classes = codebook.outcome_classes("stop")
    inferred_stop_class, inferred_stop_confidence = codebook.inferred_outcome("stop")

    rows: list[dict] = []
    for index, current in enumerate(markers):
        marker = current["marker"]
        if marker not in trial_starts:
            continue

        next_start_index = next(
            (
                later_index
                for later_index in range(index + 1, len(markers))
                if markers[later_index]["marker"] in trial_starts
            ),
            len(markers),
        )
        segment = markers[index:next_start_index]
        closed_by_next_trial = next_start_index < len(markers)
        trial_interval_ms = (
            (markers[next_start_index]["onset_s"] - current["onset_s"]) * 1000
            if closed_by_next_trial
            else None
        )
        codes = [item["marker"] for item in segment]
        unknown = sorted(set(codes) - set(codebook.definitions))
        ambiguous = [item for item in segment if item["marker"] in ambiguous_markers]

        if marker in go_starts:
            outcomes = [
                item for item in segment if item["marker"] in go_outcome_classes
            ]
            incompatible = [
                item
                for item in segment
                if item["marker"] in stop_signals
                or item["marker"] in stop_outcome_classes
            ]
            outcome_delay = (
                (outcomes[0]["onset_s"] - current["onset_s"]) * 1000
                if len(outcomes) == 1 else None
            )
            timing_invalid = (
                outcome_delay is not None
                and outcome_delay > integrity_limits_ms["stimulus_to_outcome"]
            ) or (
                trial_interval_ms is not None
                and trial_interval_ms > integrity_limits_ms["trial_interval"]
            )
            if incompatible or len(outcomes) > 1 or timing_invalid:
                status = "invalid"
                reason = (
                    "event_interval_exceeds_integrity_limit"
                    if timing_invalid else "conflicting_or_multiple_go_outcomes"
                )
                trial_class = "go_unclassified"
                outcome = outcomes[0] if outcomes else incompatible[0]
                confidence = None
            elif unknown:
                status = "ambiguous"
                reason = "unknown_markers:" + ",".join(unknown)
                trial_class = "go_unclassified"
                outcome = outcomes[0] if outcomes else None
                confidence = None
            elif ambiguous:
                status = "ambiguous"
                reason = "ambiguous_marker:" + ambiguous[0]["marker"]
                trial_class = "go_unclassified"
                outcome = outcomes[0] if outcomes else None
                confidence = codebook.definitions[
                    ambiguous[0]["marker"]
                ].confidence
            elif not outcomes:
                status = "incomplete"
                reason = "missing_go_outcome"
                trial_class = "go_unclassified"
                outcome = None
                confidence = None
            else:
                status = "classified"
                reason = "go_outcome_observed"
                outcome = outcomes[0]
                trial_class = go_outcome_classes[outcome["marker"]]
                confidence = codebook.definitions[outcome["marker"]].confidence
            rows.append(
                _trial_row(
                    trial_index=len(rows) + 1,
                    trial_type="go",
                    trial_class=trial_class,
                    status=status,
                    confidence=confidence,
                    reason=reason,
                    segment=segment,
                    segment_start_index=index,
                    outcome=outcome,
                    ambiguity=ambiguous[0] if ambiguous else None,
                    closed_by_next_trial=closed_by_next_trial,
                )
            )
            continue

        signal_rows = [item for item in segment if item["marker"] in stop_signals]
        outcomes = [
            item for item in segment if item["marker"] in stop_outcome_classes
        ]
        incompatible = [
            item for item in segment if item["marker"] in go_outcome_classes
        ]
        signal = signal_rows[0] if signal_rows else None
        outcome = outcomes[0] if outcomes else None
        outcome_before_signal = bool(
            signal is not None
            and outcome is not None
            and outcome["onset_s"] < signal["onset_s"]
        )
        stimulus_to_signal = (
            (signal["onset_s"] - current["onset_s"]) * 1000
            if signal is not None else None
        )
        signal_to_outcome = (
            (outcome["onset_s"] - signal["onset_s"]) * 1000
            if signal is not None and outcome is not None else None
        )
        timing_invalid = any(
            (
                stimulus_to_signal is not None
                and stimulus_to_signal
                > integrity_limits_ms["stimulus_to_stop_signal"],
                signal_to_outcome is not None
                and signal_to_outcome
                > integrity_limits_ms["stop_signal_to_outcome"],
                trial_interval_ms is not None
                and trial_interval_ms > integrity_limits_ms["trial_interval"],
            )
        )

        if (
            incompatible
            or len(signal_rows) > 1
            or len(outcomes) > 1
            or outcome_before_signal
            or timing_invalid
        ):
            status = "invalid"
            reason = (
                "event_interval_exceeds_integrity_limit"
                if timing_invalid else "conflicting_stop_sequence"
            )
            trial_class = "stop_unclassified"
            confidence = None
        elif unknown:
            status = "ambiguous"
            reason = "unknown_markers:" + ",".join(unknown)
            trial_class = "stop_unclassified"
            confidence = None
        elif ambiguous:
            status = "ambiguous"
            reason = "ambiguous_marker:" + ambiguous[0]["marker"]
            trial_class = "stop_unclassified"
            confidence = codebook.definitions[ambiguous[0]["marker"]].confidence
        elif signal is None:
            status = "incomplete"
            reason = "missing_stop_signal"
            trial_class = "stop_unclassified"
            confidence = None
        elif outcome is not None:
            status = "classified"
            reason = "post_stop_response_observed"
            trial_class = stop_outcome_classes[outcome["marker"]]
            confidence = codebook.definitions[outcome["marker"]].confidence
        elif closed_by_next_trial:
            status = "inferred"
            reason = "response_absent_before_next_trial"
            trial_class = inferred_stop_class
            confidence = inferred_stop_confidence
        else:
            status = "incomplete"
            reason = "missing_trial_boundary_for_success"
            trial_class = "stop_unclassified"
            confidence = None

        rows.append(
            _trial_row(
                trial_index=len(rows) + 1,
                trial_type="stop",
                trial_class=trial_class,
                status=status,
                confidence=confidence,
                reason=reason,
                segment=segment,
                segment_start_index=index,
                stop_signal=signal,
                outcome=outcome,
                ambiguity=ambiguous[0] if ambiguous else None,
                closed_by_next_trial=closed_by_next_trial,
            )
        )

    return pd.DataFrame(rows, columns=_TRIAL_COLUMNS)


def reconcile_trials(
    markers: list[dict],
    trials: pd.DataFrame,
    codebook: EventCodebook | None = None,
) -> dict:
    """Count every trial start and verify that reconstruction is exhaustive."""
    codebook = codebook or load_event_codebook()
    trial_starts = codebook.markers(role="trial_start")
    detected_trial_starts = sum(
        marker["marker"] in trial_starts for marker in markers
    )
    if len(trials) != detected_trial_starts:
        raise RuntimeError(
            "Trial reconciliation failed: "
            f"{detected_trial_starts} starts but {len(trials)} rows"
        )
    status_counts = {
        str(status): int(count)
        for status, count in trials["classification_status"].value_counts().items()
    }
    class_counts = {
        str(trial_class): int(count)
        for trial_class, count in trials["trial_class"].value_counts().items()
    }
    covered_indices: set[int] = set()
    for trial in trials.itertuples(index=False):
        covered_indices.update(
            range(
                int(trial.source_marker_start_index),
                int(trial.source_marker_stop_index_exclusive),
            )
        )
    unassigned_indices = [
        index for index in range(len(markers)) if index not in covered_indices
    ]
    return {
        "markers_total": len(markers),
        "markers_assigned_to_trials": len(covered_indices),
        "unassigned_markers": [
            {"source_marker_index": index, "marker": markers[index]["marker"]}
            for index in unassigned_indices
        ],
        "marker_index_accounting_complete": len(covered_indices)
        + len(unassigned_indices)
        == len(markers),
        "detected_trial_starts": detected_trial_starts,
        "reconstructed_trial_rows": len(trials),
        "status_counts": status_counts,
        "class_counts": class_counts,
        "accounting_complete": sum(status_counts.values())
        == detected_trial_starts,
    }


def marker_counts(markers: list[dict]) -> pd.DataFrame:
    """Count normalized stimulus markers."""
    counts = Counter(marker["marker"] for marker in markers)
    return pd.DataFrame(
        [
            {"marker": marker, "count": count}
            for marker, count in sorted(
                counts.items(), key=lambda item: int(item[0].removeprefix("S"))
            )
        ],
        columns=["marker", "count"],
    )
