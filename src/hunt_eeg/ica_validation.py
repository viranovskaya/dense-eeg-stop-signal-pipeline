"""Bounded signal checks after reviewed ICA component removal."""

from __future__ import annotations

import numpy as np

from .benchmark import score_band_power, score_task_signal


def verify_validation_context(
    review: dict,
    processed: dict,
    *,
    current_source_manifest_sha256: str,
    analysis_config_sha256: str,
    ica_config_sha256: str,
) -> None:
    """Require validation metrics to use the exact reviewed processing context."""
    review_source = review["software"]["source_manifest"]["sha256"]
    processed_source = processed["software"]["source_manifest"]["sha256"]
    if {review_source, processed_source} != {current_source_manifest_sha256}:
        raise ValueError("Validation source does not match the reviewed processing run")
    review_analysis = review["configuration"]["analysis_sha256"]
    processed_analysis = processed["configuration"]["analysis_sha256"]
    if {review_analysis, processed_analysis} != {analysis_config_sha256}:
        raise ValueError("Validation analysis config does not match the reviewed run")
    if review["ica_configuration"]["config_sha256"] != ica_config_sha256:
        raise ValueError("Validation ICA config does not match the reviewed run")


def verified_fixture_seed(
    fixture: dict,
    fixture_summary: dict,
    requested_seed: int,
) -> int:
    """Bind a requested seed to an exact, provenance-verified fixture package."""
    if fixture.get("workflow") != "public_synthetic_ica_fixture":
        raise ValueError("Input is not a verified public synthetic ICA fixture")
    seed = int(fixture_summary["seed"])
    if seed != int(requested_seed):
        raise ValueError(
            f"Requested fixture seed {requested_seed} does not match verified seed {seed}"
        )
    return seed


def _absolute_correlations(
    eeg: np.ndarray,
    auxiliary: np.ndarray,
) -> np.ndarray:
    centered_eeg = eeg - eeg.mean(axis=1, keepdims=True)
    centered_auxiliary = auxiliary - auxiliary.mean()
    numerator = centered_eeg @ centered_auxiliary
    denominator = np.linalg.norm(centered_eeg, axis=1) * np.linalg.norm(
        centered_auxiliary
    )
    if np.any(denominator == 0):
        raise ValueError("Correlation inputs must have non-zero variance")
    return np.abs(numerator / denominator)


def auxiliary_correlation_summary(reference_raw, observed_raw) -> dict:
    """Compare EEG-to-EOG/ECG correlations before and after reviewed ICA."""
    eeg_channels = [
        channel
        for channel in reference_raw.copy().pick("eeg").ch_names
        if channel in observed_raw.ch_names
    ]
    if not eeg_channels:
        raise ValueError("No common EEG channels are available for validation")
    if reference_raw.n_times != observed_raw.n_times:
        raise ValueError("Validation recordings must have equal sample counts")

    reference_eeg = reference_raw.get_data(picks=eeg_channels)
    observed_eeg = observed_raw.get_data(picks=eeg_channels)
    summary = {}
    for channel_type in ("eog", "ecg"):
        auxiliaries = reference_raw.copy().pick(channel_type).ch_names
        if len(auxiliaries) != 1:
            raise ValueError(
                f"Expected one {channel_type.upper()} channel, found {len(auxiliaries)}"
            )
        auxiliary = reference_raw.get_data(picks=auxiliaries)[0]
        before = _absolute_correlations(reference_eeg, auxiliary)
        after = _absolute_correlations(observed_eeg, auxiliary)
        summary[channel_type] = {
            "channels_compared": len(eeg_channels),
            "maximum_absolute_correlation_before": float(before.max()),
            "maximum_absolute_correlation_after": float(after.max()),
            "median_absolute_correlation_before": float(np.median(before)),
            "median_absolute_correlation_after": float(np.median(after)),
        }
    return summary


def signal_preservation_summary(reference_raw, observed_raw) -> dict:
    """Summarize task-peak and channel-band changes after reviewed ICA."""
    task = score_task_signal(reference_raw, observed_raw)
    band = score_band_power(reference_raw, observed_raw)
    errors = band["absolute_log_ratio_db"]
    return {
        "task_signal": {
            "channels": task["channel"].tolist(),
            "maximum_peak_amplitude_change_uv": float(
                task["absolute_amplitude_error_uv"].max()
            ),
            "maximum_peak_latency_change_ms": float(
                task["absolute_latency_error_ms"].max()
            ),
        },
        "band_power": {
            "channel_band_values": len(errors),
            "median_absolute_change_db": float(errors.median()),
            "p95_absolute_change_db": float(errors.quantile(0.95)),
            "maximum_absolute_change_db": float(errors.max()),
        },
    }


def summarize_reviewed_ica(reference_raw, observed_raw) -> dict:
    """Return the bounded checks used for one reviewed ICA integration fixture."""
    return {
        "auxiliary_correlation": auxiliary_correlation_summary(
            reference_raw,
            observed_raw,
        ),
        "signal_preservation": signal_preservation_summary(
            reference_raw,
            observed_raw,
        ),
    }
