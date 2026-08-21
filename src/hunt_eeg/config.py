"""Validated project configuration and event definitions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "config" / "analysis.json"
DEFAULT_EVENT_CODEBOOK = PROJECT_ROOT / "config" / "event_codebook.csv"


@dataclass(frozen=True)
class QCConfig:
    """Parameters used only for conservative review candidates."""

    representative_window_count: int
    representative_window_seconds: float
    full_recording_window_seconds: float
    robust_z_threshold: float
    flat_fraction_threshold: float
    line_noise_ratio_db_threshold: float


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters that determine preprocessing and epoch construction."""

    line_frequency_hz: float
    filter_hz: tuple[float, float]
    ica_filter_hz: tuple[float, float]
    erp_filter_hz: tuple[float, float]
    montage: str
    reference: str
    channel_types: dict[str, str]
    epochs_seconds: dict[str, tuple[float, float]]
    event_integrity_limits_ms: dict[str, float]
    qc: QCConfig


@dataclass(frozen=True)
class EventDefinition:
    """Meaning assigned to one normalized acquisition marker."""

    marker: str
    role: str
    trial_type: str
    classification: str | None
    inferred_classification: str | None
    inference_confidence: str | None
    interpretation: str
    confidence: str


@dataclass(frozen=True)
class EventCodebook:
    """Validated marker definitions used by trial reconstruction."""

    definitions: dict[str, EventDefinition]

    def markers(self, *, role: str, trial_type: str | None = None) -> frozenset[str]:
        return frozenset(
            definition.marker
            for definition in self.definitions.values()
            if definition.role == role
            and (trial_type is None or definition.trial_type == trial_type)
        )

    def outcome_classes(self, trial_type: str) -> dict[str, str]:
        return {
            definition.marker: definition.classification
            for definition in self.definitions.values()
            if definition.role == "outcome"
            and definition.trial_type == trial_type
            and definition.classification is not None
        }

    def inferred_outcome(self, trial_type: str) -> tuple[str, str]:
        definitions = [
            definition
            for definition in self.definitions.values()
            if definition.trial_type == trial_type
            and definition.inferred_classification is not None
        ]
        if len(definitions) != 1:
            raise ValueError(
                f"Expected one inferred outcome definition for {trial_type!r}"
            )
        definition = definitions[0]
        return (
            str(definition.inferred_classification),
            str(definition.inference_confidence),
        )


def load_analysis_config(path: Path = DEFAULT_ANALYSIS_CONFIG) -> AnalysisConfig:
    """Load and validate preprocessing parameters."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "line_frequency_hz",
        "filter_hz",
        "ica_filter_hz",
        "erp_filter_hz",
        "montage",
        "reference",
        "channel_types",
        "epochs_seconds",
        "event_integrity_limits_ms",
        "qc",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Analysis config is missing required fields: {missing}")

    filters: dict[str, tuple[float, float]] = {}
    for name in ("filter_hz", "ica_filter_hz", "erp_filter_hz"):
        values = tuple(float(value) for value in payload[name])
        if (
            len(values) != 2
            or not all(math.isfinite(value) for value in values)
            or not 0 <= values[0] < values[1]
        ):
            raise ValueError(f"{name} must contain increasing low and high cutoffs")
        filters[name] = values

    epochs_seconds: dict[str, tuple[float, float]] = {}
    for condition in ("go", "stop"):
        if condition not in payload["epochs_seconds"]:
            raise ValueError(f"epochs_seconds is missing {condition!r}")
        window = tuple(float(value) for value in payload["epochs_seconds"][condition])
        if len(window) != 2 or not window[0] < 0 < window[1]:
            raise ValueError(
                f"epochs_seconds[{condition!r}] must span time zero"
            )
        epochs_seconds[condition] = window

    limit_names = {
        "stimulus_to_outcome",
        "stimulus_to_stop_signal",
        "stop_signal_to_outcome",
        "trial_interval",
    }
    limits = {
        str(name): float(value)
        for name, value in payload["event_integrity_limits_ms"].items()
    }
    if set(limits) != limit_names or not all(
        math.isfinite(value) and value > 0 for value in limits.values()
    ):
        raise ValueError(
            "event_integrity_limits_ms must define four positive finite "
            "corruption guards"
        )

    channel_types = {
        str(channel): str(kind)
        for channel, kind in payload["channel_types"].items()
    }
    if not channel_types:
        raise ValueError("channel_types must not be empty")
    reference = str(payload["reference"])
    if reference != "average EEG only":
        raise ValueError(
            "Only the recovered 'average EEG only' reference is currently supported"
        )
    qc_payload = payload["qc"]
    qc_required = {
        "representative_window_count",
        "representative_window_seconds",
        "full_recording_window_seconds",
        "robust_z_threshold",
        "flat_fraction_threshold",
        "line_noise_ratio_db_threshold",
    }
    qc_missing = sorted(qc_required - qc_payload.keys())
    if qc_missing:
        raise ValueError(f"qc config is missing required fields: {qc_missing}")
    qc = QCConfig(
        representative_window_count=int(qc_payload["representative_window_count"]),
        representative_window_seconds=float(
            qc_payload["representative_window_seconds"]
        ),
        full_recording_window_seconds=float(
            qc_payload["full_recording_window_seconds"]
        ),
        robust_z_threshold=float(qc_payload["robust_z_threshold"]),
        flat_fraction_threshold=float(qc_payload["flat_fraction_threshold"]),
        line_noise_ratio_db_threshold=float(
            qc_payload["line_noise_ratio_db_threshold"]
        ),
    )
    line_frequency_hz = float(payload["line_frequency_hz"])
    if not math.isfinite(line_frequency_hz) or line_frequency_hz <= 0:
        raise ValueError("line_frequency_hz must be a positive finite number")
    if qc.representative_window_count < 1:
        raise ValueError("representative_window_count must be at least 1")
    finite_qc_values = (
        qc.representative_window_seconds,
        qc.full_recording_window_seconds,
        qc.robust_z_threshold,
        qc.flat_fraction_threshold,
        qc.line_noise_ratio_db_threshold,
    )
    if not all(math.isfinite(value) for value in finite_qc_values):
        raise ValueError("QC thresholds and window length must be finite")
    if (
        qc.representative_window_seconds <= 0
        or qc.full_recording_window_seconds <= 0
        or qc.robust_z_threshold <= 0
    ):
        raise ValueError("QC window length and robust-z threshold must be positive")
    if not 0 <= qc.flat_fraction_threshold <= 1:
        raise ValueError("flat_fraction_threshold must be between 0 and 1")

    return AnalysisConfig(
        line_frequency_hz=line_frequency_hz,
        filter_hz=filters["filter_hz"],
        ica_filter_hz=filters["ica_filter_hz"],
        erp_filter_hz=filters["erp_filter_hz"],
        montage=str(payload["montage"]),
        reference=reference,
        channel_types=channel_types,
        epochs_seconds=epochs_seconds,
        event_integrity_limits_ms=limits,
        qc=qc,
    )


def load_event_codebook(path: Path = DEFAULT_EVENT_CODEBOOK) -> EventCodebook:
    """Load marker meanings and reject incomplete or contradictory definitions."""
    table = pd.read_csv(path, keep_default_na=False)
    expected_columns = [
        "marker",
        "role",
        "trial_type",
        "classification",
        "inferred_classification",
        "inference_confidence",
        "interpretation",
        "confidence",
    ]
    if table.columns.tolist() != expected_columns:
        raise ValueError(
            "Event codebook columns must be exactly " + ", ".join(expected_columns)
        )
    if table.empty:
        raise ValueError("Event codebook must not be empty")
    if table["marker"].duplicated().any():
        duplicates = sorted(table.loc[table["marker"].duplicated(), "marker"].unique())
        raise ValueError(f"Event codebook contains duplicate markers: {duplicates}")

    allowed_roles = {"trial_start", "stop_signal", "outcome", "ambiguous"}
    allowed_trial_types = {"go", "stop", "any"}
    allowed_confidence = {"low", "medium", "high"}
    definitions: dict[str, EventDefinition] = {}
    for row in table.itertuples(index=False):
        marker = str(row.marker).strip()
        role = str(row.role).strip()
        trial_type = str(row.trial_type).strip()
        classification = str(row.classification).strip() or None
        inferred_classification = str(row.inferred_classification).strip() or None
        inference_confidence = str(row.inference_confidence).strip() or None
        confidence = str(row.confidence).strip()
        interpretation = str(row.interpretation).strip()
        if not marker.startswith("S") or not marker[1:].isdigit():
            raise ValueError(f"Invalid normalized marker in codebook: {marker!r}")
        if role not in allowed_roles:
            raise ValueError(f"Invalid role for {marker}: {role!r}")
        if trial_type not in allowed_trial_types:
            raise ValueError(f"Invalid trial_type for {marker}: {trial_type!r}")
        if confidence not in allowed_confidence:
            raise ValueError(f"Invalid confidence for {marker}: {confidence!r}")
        if not interpretation:
            raise ValueError(f"Event codebook interpretation is empty for {marker}")
        if role != "ambiguous" and trial_type == "any":
            raise ValueError(f"Role {role!r} requires a go or stop trial type")
        if role == "stop_signal" and trial_type != "stop":
            raise ValueError("A stop-signal marker must use trial_type='stop'")
        if role == "outcome" and not classification:
            raise ValueError(f"Outcome marker {marker} needs a classification")
        if role == "outcome" and not classification.startswith(f"{trial_type}_"):
            raise ValueError(
                f"Outcome classification for {marker} must start with "
                f"{trial_type!r}"
            )
        if role != "outcome" and classification:
            raise ValueError(
                f"Only outcome markers may define a classification ({marker})"
            )
        if inferred_classification and role != "stop_signal":
            raise ValueError(
                "Only a stop-signal marker may define an inferred outcome "
                f"({marker})"
            )
        if inferred_classification and not inferred_classification.startswith(
            f"{trial_type}_"
        ):
            raise ValueError(
                f"Inferred classification for {marker} must start with "
                f"{trial_type!r}"
            )
        if bool(inferred_classification) != bool(inference_confidence):
            raise ValueError(
                f"Inferred classification and confidence must be paired ({marker})"
            )
        if inference_confidence and inference_confidence not in allowed_confidence:
            raise ValueError(
                f"Invalid inference confidence for {marker}: "
                f"{inference_confidence!r}"
            )
        definitions[marker] = EventDefinition(
            marker=marker,
            role=role,
            trial_type=trial_type,
            classification=classification,
            inferred_classification=inferred_classification,
            inference_confidence=inference_confidence,
            interpretation=interpretation,
            confidence=confidence,
        )

    codebook = EventCodebook(definitions=definitions)
    if not codebook.markers(role="trial_start", trial_type="go"):
        raise ValueError("Event codebook needs at least one go trial start")
    if not codebook.markers(role="trial_start", trial_type="stop"):
        raise ValueError("Event codebook needs at least one stop trial start")
    if not codebook.markers(role="stop_signal", trial_type="stop"):
        raise ValueError("Event codebook needs at least one stop-signal marker")
    if not codebook.outcome_classes("go"):
        raise ValueError("Event codebook needs at least one go outcome")
    if not codebook.outcome_classes("stop"):
        raise ValueError("Event codebook needs at least one failed-stop outcome")
    codebook.inferred_outcome("stop")
    return codebook
