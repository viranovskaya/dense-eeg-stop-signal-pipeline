"""Frozen stop-signal ERP endpoint for the fixed ten-recording study."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .provenance import canonical_sha256, sha256_file


RESPONSE_ABSENT_CLASS = "stop_successful"
RESPONSE_PRESENT_CLASS = "stop_failed"
CLASSIFIED_GO_RESPONSE_CLASSES = frozenset(
    {"go_correct", "go_incorrect_or_slow"}
)
ROI_CHANNELS = ("FC1", "FC2", "FCz")
PRE_GO_BASELINE_SECONDS = (-0.2, 0.0)
MEASUREMENT_SECONDS = (0.25, 0.45)
MINIMUM_EPOCHS_PER_CONDITION = 8
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_DRAWS = 10_000
MINIMUM_ELIGIBLE_PARTICIPANTS = 3
ENDPOINT_OUTPUT_FILES = {
    "endpoint_summary.json",
    "participant_endpoint.csv",
    "endpoint_mean_ci.png",
}


def _retained_mask(values: pd.Series) -> np.ndarray:
    """Return an explicit Boolean lineage mask without truthy-string coercion."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(
            "Stop lineage retained must contain only true or false; "
            f"found {unexpected}"
        )
    return normalized.eq("true").to_numpy(dtype=bool)


def verify_endpoint_package(path: Path) -> dict:
    """Verify the exact private endpoint package and its declared hashes."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Endpoint package must be a real directory")
    if any(candidate.is_symlink() for candidate in path.rglob("*")):
        raise ValueError("Endpoint package must not contain symlinks")
    provenance_path = path / "endpoint_provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    declared = {record["path"] for record in payload["outputs"]}
    if declared != ENDPOINT_OUTPUT_FILES:
        raise ValueError("Endpoint provenance does not declare the exact output set")
    actual = {
        candidate.relative_to(path).as_posix()
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate != provenance_path
    }
    if actual != declared:
        raise ValueError("Endpoint package contains missing or undeclared files")
    for record in payload["outputs"]:
        candidate = path / record["path"]
        if candidate.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Endpoint output size mismatch: {record['path']}")
        if sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"Endpoint output hash mismatch: {record['path']}")
    core = {key: value for key, value in payload.items() if key != "core_sha256"}
    if canonical_sha256(core) != payload["core_sha256"]:
        raise ValueError("Endpoint provenance core hash mismatch")
    return payload


def validate_stop_lineage(epochs: mne.Epochs, lineage: pd.DataFrame) -> None:
    """Require one exact lineage row for every retained stop epoch."""
    required = {
        "trial_index",
        "trial_class",
        "stimulus_to_stop_signal_ms",
        "retained",
        "output_epoch_index",
    }
    missing = sorted(required - set(lineage.columns))
    if missing:
        raise ValueError(f"Stop lineage is missing columns: {missing}")
    retained = lineage.loc[_retained_mask(lineage["retained"])].copy()
    output_indices = pd.to_numeric(
        retained["output_epoch_index"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(output_indices).all()
        or not np.equal(output_indices, np.floor(output_indices)).all()
    ):
        raise ValueError(
            "Retained stop lineage output_epoch_index values must be finite integers"
        )
    retained["output_epoch_index"] = output_indices.astype(int)
    retained = retained.sort_values("output_epoch_index")
    if retained["output_epoch_index"].tolist() != list(range(len(epochs))):
        raise ValueError("Stop lineage does not map one-to-one onto retained epochs")
    if epochs.metadata is None:
        raise ValueError("Stop epochs require trial metadata")
    for column in ("trial_index", "trial_class"):
        if column not in epochs.metadata:
            raise ValueError(f"Stop epoch metadata is missing {column!r}")
        if retained[column].astype(str).tolist() != epochs.metadata[column].astype(
            str
        ).tolist():
            raise ValueError(f"Stop lineage and epoch metadata disagree on {column}")
    retained_delays = pd.to_numeric(
        retained["stimulus_to_stop_signal_ms"], errors="coerce"
    ).to_numpy(dtype=float)
    epoch_delays = pd.to_numeric(
        epochs.metadata["stimulus_to_stop_signal_ms"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(retained_delays).all()
        or not np.isfinite(epoch_delays).all()
        or np.any(retained_delays <= 0)
        or np.any(epoch_delays <= 0)
    ):
        raise ValueError("Every endpoint epoch requires a positive finite stop delay")
    if not np.allclose(
        retained_delays,
        epoch_delays,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Stop lineage and epoch metadata disagree on stimulus_to_stop_signal_ms"
        )


def participant_endpoint(
    epochs: mne.Epochs,
    lineage: pd.DataFrame,
    *,
    participant_id: str,
) -> dict:
    """Measure the fixed ROI and latency-window contrast for one participant."""
    validate_stop_lineage(epochs, lineage)
    missing_channels = sorted(set(ROI_CHANNELS) - set(epochs.ch_names))
    if missing_channels:
        raise ValueError(f"ERP ROI channels are missing: {missing_channels}")
    if not np.isclose(float(epochs.info["highpass"]), 0.2) or not np.isclose(
        float(epochs.info["lowpass"]), 30.0
    ):
        raise ValueError("Stop epochs do not use the fixed 0.2-30 Hz ERP filter")

    prepared = epochs.copy().pick(list(ROI_CHANNELS))
    if any(channel_type != "eeg" for channel_type in prepared.get_channel_types()):
        raise ValueError("Every fixed endpoint ROI channel must be typed as EEG")
    metadata = prepared.metadata
    assert metadata is not None
    classes = metadata["trial_class"].astype(str).to_numpy()
    unexpected_classes = sorted(
        set(classes) - {RESPONSE_ABSENT_CLASS, RESPONSE_PRESENT_CLASS}
    )
    if unexpected_classes:
        raise ValueError(
            "Stop endpoint contains an unsupported trial class: "
            f"{unexpected_classes}"
        )
    delays_s = pd.to_numeric(
        metadata["stimulus_to_stop_signal_ms"], errors="coerce"
    ).to_numpy(dtype=float) / 1000.0
    if not np.isfinite(delays_s).all() or np.any(delays_s <= 0):
        raise ValueError("Every endpoint epoch requires a positive finite stop delay")
    time_mask = (prepared.times >= MEASUREMENT_SECONDS[0]) & (
        prepared.times <= MEASUREMENT_SECONDS[1]
    )
    expected_measurement_samples = (
        int(
            round(
                (MEASUREMENT_SECONDS[1] - MEASUREMENT_SECONDS[0])
                * float(prepared.info["sfreq"])
            )
        )
        + 1
    )
    if int(time_mask.sum()) != expected_measurement_samples:
        raise ValueError(
            "Stop epoch does not cover the complete 250-450 ms measurement window"
        )
    data = prepared.get_data(copy=True)
    baseline_corrected = np.empty_like(data)
    expected_baseline_samples = int(
        round(
            (PRE_GO_BASELINE_SECONDS[1] - PRE_GO_BASELINE_SECONDS[0])
            * float(prepared.info["sfreq"])
        )
    )
    if expected_baseline_samples < 1:
        raise ValueError("The sampling frequency cannot represent the pre-go baseline")
    sampling_period = 1.0 / float(prepared.info["sfreq"])
    alignment_tolerance = sampling_period / 2.0 + np.finfo(float).eps * 8
    for index, delay_s in enumerate(delays_s):
        baseline_start = -delay_s + PRE_GO_BASELINE_SECONDS[0]
        baseline_stop = -delay_s + PRE_GO_BASELINE_SECONDS[1]
        baseline_start_index = int(
            round(
                (baseline_start - float(prepared.times[0]))
                * float(prepared.info["sfreq"])
            )
        )
        baseline_stop_index = baseline_start_index + expected_baseline_samples
        baseline_is_complete = (
            baseline_start_index >= 0
            and baseline_stop_index <= len(prepared.times)
            and abs(
                float(prepared.times[baseline_start_index]) - baseline_start
            )
            <= alignment_tolerance
            and abs(
                float(prepared.times[baseline_stop_index - 1])
                + sampling_period
                - baseline_stop
            )
            <= alignment_tolerance
        )
        if not baseline_is_complete:
            raise ValueError(
                "Stop epoch does not cover the complete 200 ms pre-go baseline"
            )
        baseline_corrected[index] = data[index] - data[index][
            :, baseline_start_index:baseline_stop_index
        ].mean(axis=1, keepdims=True)
    epoch_values_uv = baseline_corrected[:, :, time_mask].mean(axis=(1, 2)) * 1e6
    absent = epoch_values_uv[classes == RESPONSE_ABSENT_CLASS]
    present = epoch_values_uv[classes == RESPONSE_PRESENT_CLASS]
    absent_count = int(len(absent))
    present_count = int(len(present))
    eligible = (
        absent_count >= MINIMUM_EPOCHS_PER_CONDITION
        and present_count >= MINIMUM_EPOCHS_PER_CONDITION
    )
    absent_mean = float(np.mean(absent)) if absent_count else None
    present_mean = float(np.mean(present)) if present_count else None
    return {
        "participant_id": str(participant_id),
        "response_absent_epochs": absent_count,
        "response_present_epochs": present_count,
        "eligible": bool(eligible),
        "exclusion_reason": (
            ""
            if eligible
            else "fewer_than_8_retained_epochs_in_one_or_both_conditions"
        ),
        "response_absent_mean_uv": absent_mean,
        "response_present_mean_uv": present_mean,
        "contrast_absent_minus_present_uv": (
            float(absent_mean - present_mean) if eligible else None
        ),
    }


def participant_behavioral_diagnostics(trials: pd.DataFrame) -> dict:
    """Describe stop-task behavior without estimating SSRT."""
    required = {
        "trial_type",
        "trial_class",
        "outcome_code",
        "stimulus_to_stop_signal_ms",
        "stimulus_to_outcome_ms",
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Reconstructed trials are missing columns: {missing}")

    trial_class = trials["trial_class"].astype(str)
    all_stop = trials.loc[trials["trial_type"].astype(str).eq("stop")]
    classified_stop = trials.loc[
        trial_class.isin({RESPONSE_ABSENT_CLASS, RESPONSE_PRESENT_CLASS})
    ].copy()
    if classified_stop.empty:
        raise ValueError("Behavioral diagnostics require classified stop trials")
    stop_classes = classified_stop["trial_class"].astype(str).to_numpy()
    stop_delays = pd.to_numeric(
        classified_stop["stimulus_to_stop_signal_ms"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(stop_delays).all() or np.any(stop_delays <= 0):
        raise ValueError("Every classified stop trial requires a positive finite SSD")

    absent_delays = stop_delays[stop_classes == RESPONSE_ABSENT_CLASS]
    present_delays = stop_delays[stop_classes == RESPONSE_PRESENT_CLASS]
    if not len(absent_delays) or not len(present_delays):
        raise ValueError("Behavioral diagnostics require both stop outcomes")

    failed = classified_stop.loc[
        classified_stop["trial_class"].astype(str).eq(RESPONSE_PRESENT_CLASS)
    ]
    failed_rt = pd.to_numeric(
        failed["stimulus_to_outcome_ms"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(failed_rt).all() or np.any(failed_rt <= 0):
        raise ValueError("Every response-present stop trial requires a positive RT")

    go_responses = trials.loc[
        trials["trial_type"].astype(str).eq("go")
        & trial_class.isin(CLASSIFIED_GO_RESPONSE_CLASSES)
        & trials["outcome_code"].astype(str).str.strip().ne("")
    ]
    go_rt = pd.to_numeric(
        go_responses["stimulus_to_outcome_ms"], errors="coerce"
    ).to_numpy(dtype=float)
    if not len(go_rt) or not np.isfinite(go_rt).all() or np.any(go_rt <= 0):
        raise ValueError("Behavioral diagnostics require positive finite go RTs")

    absent_count = int(len(absent_delays))
    present_count = int(len(present_delays))
    stop_total = int(len(all_stop))
    unclassified_count = stop_total - absent_count - present_count
    if unclassified_count < 0:
        raise RuntimeError("Classified stop-trial accounting exceeds stop trials")
    absent_ssd = float(absent_delays.mean())
    present_ssd = float(present_delays.mean())
    mean_failed_rt = float(failed_rt.mean())
    mean_go_rt = float(go_rt.mean())
    return {
        "stop_trials_total": stop_total,
        "classified_stop_trials": absent_count + present_count,
        "unclassified_stop_trials": unclassified_count,
        "classified_stop_fraction": (absent_count + present_count) / stop_total,
        "response_absent_trials": absent_count,
        "response_present_trials": present_count,
        "response_present_probability": present_count / (absent_count + present_count),
        "response_absent_mean_ssd_ms": absent_ssd,
        "response_present_mean_ssd_ms": present_ssd,
        "ssd_present_minus_absent_ms": present_ssd - absent_ssd,
        "mean_failed_stop_rt_from_go_ms": mean_failed_rt,
        "mean_go_response_rt_ms": mean_go_rt,
        "failed_stop_rt_shorter_than_go_rt": mean_failed_rt < mean_go_rt,
    }


def exact_sign_flip_p(contrasts: np.ndarray) -> float:
    """Return the exact two-sided sign-flip p value for participant contrasts."""
    values = np.asarray(contrasts, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Exact sign-flip inference requires finite contrasts")
    observed = abs(float(values.mean()))
    null = np.asarray(
        [
            np.mean(values * signs)
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    tolerance = np.finfo(float).eps * 8
    return float(np.mean(np.abs(null) >= observed - tolerance))


def summarize_endpoint(rows: list[dict]) -> dict:
    """Summarize eligible participants with deterministic group inference."""
    eligible = [row for row in rows if row["eligible"]]
    if len(eligible) < MINIMUM_ELIGIBLE_PARTICIPANTS:
        raise ValueError(
            "Fewer than three participants meet the fixed epoch-count criterion"
        )
    contrasts = np.asarray(
        [row["contrast_absent_minus_present_uv"] for row in eligible], dtype=float
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        contrasts,
        size=(BOOTSTRAP_DRAWS, len(contrasts)),
        replace=True,
    )
    lower, upper = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    response_probabilities = np.asarray(
        [row["response_present_probability"] for row in rows], dtype=float
    )
    ssd_differences = np.asarray(
        [row["ssd_present_minus_absent_ms"] for row in rows], dtype=float
    )
    race_checks = [
        bool(row["failed_stop_rt_shorter_than_go_rt"]) for row in rows
    ]
    classified_fractions = np.asarray(
        [row["classified_stop_fraction"] for row in rows], dtype=float
    )
    excluded = [row for row in rows if not row["eligible"]]
    exclusion_reasons = Counter(str(row["exclusion_reason"]) for row in excluded)
    return {
        "endpoint": {
            "lock": "observed S19 stop-signal onset",
            "conditions": {
                "response_absent": RESPONSE_ABSENT_CLASS,
                "response_present": RESPONSE_PRESENT_CLASS,
            },
            "interpretation": (
                "Operational response-absent versus response-present contrast; "
                "not a direct measure of an inhibition mechanism"
            ),
            "roi_channels": list(ROI_CHANNELS),
            "baseline": {
                "reference_event": "go stimulus onset",
                "seconds": list(PRE_GO_BASELINE_SECONDS),
                "implementation": "trial-specific within stop-signal-locked epochs",
            },
            "measurement_seconds": list(MEASUREMENT_SECONDS),
            "minimum_epochs_per_condition": MINIMUM_EPOCHS_PER_CONDITION,
        },
        "participants_total": len(rows),
        "participants_eligible": len(eligible),
        "participants_excluded": len(excluded),
        "participant_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "epoch_accounting": {
            "all_participants": {
                "response_absent": int(
                    sum(row["response_absent_epochs"] for row in rows)
                ),
                "response_present": int(
                    sum(row["response_present_epochs"] for row in rows)
                ),
            },
            "eligible_participants": {
                "response_absent": int(
                    sum(row["response_absent_epochs"] for row in eligible)
                ),
                "response_present": int(
                    sum(row["response_present_epochs"] for row in eligible)
                ),
            },
            "excluded_participants": {
                "response_absent": int(
                    sum(row["response_absent_epochs"] for row in excluded)
                ),
                "response_present": int(
                    sum(row["response_present_epochs"] for row in excluded)
                ),
            },
        },
        "mean_contrast_uv": float(contrasts.mean()),
        "participant_contrast_sd_uv": float(contrasts.std(ddof=1)),
        "median_contrast_uv": float(np.median(contrasts)),
        "participant_contrast_range_uv": [
            float(contrasts.min()),
            float(contrasts.max()),
        ],
        "bootstrap_95_ci_uv": [float(lower), float(upper)],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(contrasts),
        "participant_weighting": "equal weight per eligible participant",
        "behavioral_diagnostics": {
            "scope": (
                "Descriptive task-validity context from all classified trials; "
                "not an SSRT estimate and not an additional inferential endpoint"
            ),
            "participants": len(rows),
            "stop_trials_total": int(sum(row["stop_trials_total"] for row in rows)),
            "unclassified_stop_trials": int(
                sum(row["unclassified_stop_trials"] for row in rows)
            ),
            "mean_classified_stop_fraction": float(classified_fractions.mean()),
            "mean_response_present_probability": float(
                response_probabilities.mean()
            ),
            "mean_within_participant_ssd_present_minus_absent_ms": float(
                ssd_differences.mean()
            ),
            "failed_stop_rt_shorter_than_go_rt": {
                "passed": int(sum(race_checks)),
                "assessed": len(race_checks),
                "interpretation": (
                    "Necessary ordering diagnostic for the independent race model; "
                    "not proof that all model assumptions hold"
                ),
            },
        },
    }
