"""Bounded signal checks after reviewed ICA component removal."""

from __future__ import annotations

import numpy as np

from .benchmark import score_band_power, score_task_signal


def verify_validation_context(
    review: dict,
    processed: dict,
    control_processed: dict,
    *,
    current_source_manifest_sha256: str,
    analysis_config_sha256: str,
    ica_config_sha256: str,
) -> None:
    """Require validation metrics to use the exact reviewed processing context."""
    review_source = review["software"]["source_manifest"]["sha256"]
    processed_sources = {
        processed["software"]["source_manifest"]["sha256"],
        control_processed["software"]["source_manifest"]["sha256"],
    }
    if {review_source, *processed_sources} != {current_source_manifest_sha256}:
        raise ValueError("Validation source does not match the reviewed processing run")
    review_analysis = review["configuration"]["analysis_sha256"]
    processed_analyses = {
        processed["configuration"]["analysis_sha256"],
        control_processed["configuration"]["analysis_sha256"],
    }
    if {review_analysis, *processed_analyses} != {analysis_config_sha256}:
        raise ValueError("Validation analysis config does not match the reviewed run")
    if review["ica_configuration"]["config_sha256"] != ica_config_sha256:
        raise ValueError("Validation ICA config does not match the reviewed run")


def verify_matched_ica_processing(
    control: dict,
    reviewed: dict,
    control_summary: dict,
    reviewed_summary: dict,
) -> None:
    """Require two outputs that differ only in explicit ICA component decisions."""
    for field in (
        "participant_id",
        "inputs",
        "configuration",
        "bad_channel_decisions",
        "interval_review",
        "software",
    ):
        if control.get(field) != reviewed.get(field):
            raise ValueError(f"ICA validation processing mismatch: {field}")
    for field in ("solution_sha256", "review_provenance_sha256"):
        if control["ica_inputs"][field] != reviewed["ica_inputs"][field]:
            raise ValueError(f"ICA validation input mismatch: {field}")
    if (
        control["ica_inputs"]["decision_table_sha256"]
        == reviewed["ica_inputs"]["decision_table_sha256"]
    ):
        raise ValueError("ICA validation control and reviewed decisions must differ")
    if control["ica_inputs"].get("automatic_exclusion") or reviewed[
        "ica_inputs"
    ].get("automatic_exclusion"):
        raise ValueError("ICA validation requires explicit component decisions")
    control_ica = control_summary["ica"]
    reviewed_ica = reviewed_summary["ica"]
    for payload, summary in (
        (control, control_ica),
        (reviewed, reviewed_ica),
    ):
        if summary.get("solution_sha256") != payload["ica_inputs"]["solution_sha256"]:
            raise ValueError("ICA validation summary solution identity mismatch")
        if (
            summary.get("decision_table_sha256")
            != payload["ica_inputs"]["decision_table_sha256"]
        ):
            raise ValueError("ICA validation summary decision identity mismatch")
    if control_ica.get("excluded_components") != []:
        raise ValueError("ICA validation control must retain every component")
    if not reviewed_ica.get("excluded_components"):
        raise ValueError("ICA validation reviewed output must exclude a component")
    if control_ica.get("components") != reviewed_ica.get("components"):
        raise ValueError("ICA validation component counts do not match")
    if not control_ica.get("decision_record_complete") or not reviewed_ica.get(
        "decision_record_complete"
    ):
        raise ValueError("ICA validation decisions must be complete")
    for field in (
        "participant_id",
        "erp_filter_hz",
        "reference",
        "interpolated_bad_channels",
        "trial_reconciliation",
        "go_epoch_accounting",
        "stop_epoch_accounting",
        "interval_review",
    ):
        if control_summary.get(field) != reviewed_summary.get(field):
            raise ValueError(f"ICA validation output mismatch: {field}")


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


def auxiliary_correlation_summary(control_raw, reviewed_raw, auxiliary_raw) -> dict:
    """Compare matched-output EEG correlations before and after exclusions."""
    eeg_channels = [
        channel
        for channel in control_raw.copy().pick("eeg").ch_names
        if channel in reviewed_raw.ch_names
    ]
    if not eeg_channels:
        raise ValueError("No common EEG channels are available for validation")
    if len({control_raw.n_times, reviewed_raw.n_times, auxiliary_raw.n_times}) != 1:
        raise ValueError("Validation recordings must have equal sample counts")

    control_eeg = control_raw.get_data(picks=eeg_channels)
    reviewed_eeg = reviewed_raw.get_data(picks=eeg_channels)
    summary = {}
    for channel_type in ("eog", "ecg"):
        auxiliaries = auxiliary_raw.copy().pick(channel_type).ch_names
        if len(auxiliaries) != 1:
            raise ValueError(
                f"Expected one {channel_type.upper()} channel, found {len(auxiliaries)}"
            )
        auxiliary = auxiliary_raw.get_data(picks=auxiliaries)[0]
        control = _absolute_correlations(control_eeg, auxiliary)
        reviewed = _absolute_correlations(reviewed_eeg, auxiliary)
        summary[channel_type] = {
            "channels_compared": len(eeg_channels),
            "maximum_absolute_correlation_control": float(control.max()),
            "maximum_absolute_correlation_reviewed": float(reviewed.max()),
            "median_absolute_correlation_control": float(np.median(control)),
            "median_absolute_correlation_reviewed": float(np.median(reviewed)),
        }
    return summary


def signal_preservation_summary(control_raw, reviewed_raw) -> dict:
    """Summarize task-peak and channel-band changes after reviewed ICA."""
    task = score_task_signal(control_raw, reviewed_raw)
    band = score_band_power(control_raw, reviewed_raw)
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


def summarize_reviewed_ica(control_raw, reviewed_raw, auxiliary_raw) -> dict:
    """Return the bounded checks used for one reviewed ICA integration fixture."""
    if control_raw.ch_names != reviewed_raw.ch_names:
        raise ValueError("ICA validation outputs must have identical channel order")
    if control_raw.n_times != reviewed_raw.n_times:
        raise ValueError("ICA validation outputs must have equal sample counts")
    if control_raw.info["sfreq"] != reviewed_raw.info["sfreq"]:
        raise ValueError("ICA validation outputs must have equal sampling frequency")
    return {
        "auxiliary_correlation": auxiliary_correlation_summary(
            control_raw,
            reviewed_raw,
            auxiliary_raw,
        ),
        "signal_preservation": signal_preservation_summary(
            control_raw,
            reviewed_raw,
        ),
    }
