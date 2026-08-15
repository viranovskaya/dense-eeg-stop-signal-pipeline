"""Non-destructive raw EEG quality-control report generation."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.sparse import csr_matrix

from .config import QCConfig, load_analysis_config, load_event_codebook
from .events import (
    annotations_to_markers,
    classify_trials,
    marker_counts,
    reconcile_trials,
)
from .provenance import (
    build_provenance_core,
    verify_input_sources,
    write_provenance,
)


def _read_brainvision_compat(vhdr: Path) -> mne.io.BaseRaw:
    """Read BrainVision while tolerating legacy overlong measurement dates.

    Some marker files contain more than the 20 date digits accepted by modern
    MNE. The original files are never modified: on failure, temporary text-only
    header/marker copies are created with the measurement date removed. The
    binary EEG remains at its original path.
    """
    try:
        return mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    except ValueError as error:
        if "unconverted data remains" not in str(error):
            raise

    header_text = vhdr.read_text(encoding="utf-8-sig", errors="replace")
    data_match = re.search(r"^DataFile=(.+?)\r?$", header_text, flags=re.MULTILINE)
    marker_match = re.search(r"^MarkerFile=(.+?)\r?$", header_text, flags=re.MULTILINE)
    if not data_match or not marker_match:
        raise ValueError(f"Could not resolve DataFile/MarkerFile from {vhdr}")
    data_path = (vhdr.parent / data_match.group(1).strip()).resolve()
    marker_path = (vhdr.parent / marker_match.group(1).strip()).resolve()
    marker_text = marker_path.read_text(encoding="utf-8-sig", errors="replace")
    marker_text = re.sub(
        r"^(Mk\d+=New Segment,.*?,\d+,\d+,-?\d+),\d+\r?$",
        r"\1,00000000000000000000",
        marker_text,
        flags=re.MULTILINE,
    )

    with tempfile.TemporaryDirectory(prefix="brainvision-compat-") as temporary:
        temporary_path = Path(temporary)
        temporary_marker = temporary_path / marker_path.name
        temporary_header = temporary_path / vhdr.name
        temporary_marker.write_text(
            re.sub(
                r"^DataFile=.+?\r?$",
                f"DataFile={data_path}",
                marker_text,
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        temporary_header.write_text(
            re.sub(
                r"^MarkerFile=.+?\r?$",
                f"MarkerFile={temporary_marker.name}",
                re.sub(
                    r"^DataFile=.+?\r?$",
                    f"DataFile={data_path}",
                    header_text,
                    flags=re.MULTILINE,
                ),
                flags=re.MULTILINE,
            ),
            encoding="utf-8",
        )
        return mne.io.read_raw_brainvision(
            temporary_header, preload=False, verbose="ERROR"
        )


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.full_like(values, np.nan)
    return (values - median) / (1.4826 * mad)


def _window_qc_status(
    z_std: np.ndarray,
    z_range: np.ndarray,
) -> tuple[str, bool, bool]:
    """Report whether neither, one, or both window metrics are scorable."""
    std_scorable = not np.isnan(z_std).all()
    range_scorable = not np.isnan(z_range).all()
    if std_scorable and range_scorable:
        status = "scored"
    elif std_scorable or range_scorable:
        status = "partially_scorable_zero_mad"
    else:
        status = "unscorable_zero_mad"
    return status, std_scorable, range_scorable


def _representative_data(
    raw: mne.io.BaseRaw, picks: list[int], qc: QCConfig
) -> np.ndarray:
    """Read configured windows spread across the recording."""
    sfreq = float(raw.info["sfreq"])
    window_s = qc.representative_window_seconds
    duration_s = raw.n_times / sfreq
    if duration_s <= window_s:
        starts = np.array([0.0])
    else:
        edge = min(5.0, max(0.0, (duration_s - window_s) / 2.0))
        latest_start = max(edge, duration_s - window_s - edge)
        starts = np.linspace(edge, latest_start, qc.representative_window_count)
    start_samples = np.unique(np.round(starts * sfreq).astype(int))
    chunks = []
    for start in start_samples:
        stop = min(raw.n_times, start + round(window_s * sfreq))
        chunks.append(raw.get_data(picks=picks, start=start, stop=stop))
    return np.concatenate(chunks, axis=1)


def _channel_metrics(
    raw: mne.io.BaseRaw,
    filter_hz: tuple[float, float],
    qc: QCConfig,
    *,
    already_filtered: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    picks = mne.pick_types(raw.info, eeg=True, ecg=False, eog=False, exclude=[])
    data = _representative_data(raw, list(picks), qc)
    if already_filtered:
        filtered_data = data
    else:
        sos = butter(
            4,
            list(filter_hz),
            btype="bandpass",
            fs=float(raw.info["sfreq"]),
            output="sos",
        )
        filtered_data = sosfiltfilt(sos, data, axis=1)
    std_uv = np.std(data, axis=1) * 1e6
    low, high = np.percentile(data, [1, 99], axis=1)
    robust_range_uv = (high - low) * 1e6
    filtered_std_uv = np.std(filtered_data, axis=1) * 1e6
    filtered_low, filtered_high = np.percentile(filtered_data, [1, 99], axis=1)
    filtered_range_uv = (filtered_high - filtered_low) * 1e6
    flat_fraction = np.mean(np.abs(np.diff(data, axis=1)) < 1e-10, axis=1)
    z_std = _robust_z(np.log10(np.maximum(filtered_std_uv, 1e-12)))
    z_range = _robust_z(np.log10(np.maximum(filtered_range_uv, 1e-12)))
    candidate = (
        (np.abs(z_std) > qc.robust_z_threshold)
        | (np.abs(z_range) > qc.robust_z_threshold)
        | (flat_fraction > qc.flat_fraction_threshold)
    )

    metrics = pd.DataFrame(
        {
            "channel": [raw.ch_names[index] for index in picks],
            "std_uv": std_uv,
            "robust_range_uv": robust_range_uv,
            "filtered_std_uv": filtered_std_uv,
            "filtered_robust_range_uv": filtered_range_uv,
            "flat_fraction": flat_fraction,
            "robust_z_log_filtered_std": z_std,
            "robust_z_log_filtered_range": z_range,
            "candidate_bad": candidate,
        }
    )
    return metrics, picks, data, filtered_data


def _full_recording_window_metrics(
    raw: mne.io.BaseRaw,
    filtered_raw: mne.io.BaseRaw,
    qc: QCConfig,
) -> pd.DataFrame:
    """Screen every sample in non-overlapping windows for review candidates."""
    picks = mne.pick_types(raw.info, eeg=True, ecg=False, eog=False, exclude=[])
    sfreq = float(raw.info["sfreq"])
    window_samples = max(1, round(qc.full_recording_window_seconds * sfreq))
    starts = list(range(0, raw.n_times, window_samples))
    rows: list[dict] = []
    for window_index, start in enumerate(starts, start=1):
        stop = min(raw.n_times, start + window_samples)
        data = raw.get_data(picks=picks, start=start, stop=stop)
        filtered = filtered_raw.get_data(picks=picks, start=start, stop=stop)
        filtered_std_uv = np.std(filtered, axis=1) * 1e6
        low, high = np.percentile(filtered, [1, 99], axis=1)
        filtered_range_uv = (high - low) * 1e6
        flat_fraction = np.mean(np.abs(np.diff(data, axis=1)) < 1e-10, axis=1)
        z_std = _robust_z(np.log10(np.maximum(filtered_std_uv, 1e-12)))
        z_range = _robust_z(np.log10(np.maximum(filtered_range_uv, 1e-12)))
        window_flag = (np.abs(z_std) > qc.robust_z_threshold) | (
            np.abs(z_range) > qc.robust_z_threshold
        )
        flat_flag = flat_fraction > qc.flat_fraction_threshold
        raw_deviation = np.abs(data - np.median(data, axis=1, keepdims=True))
        filtered_deviation = np.abs(
            filtered - np.median(filtered, axis=1, keepdims=True)
        )
        raw_argmax = np.argmax(raw_deviation, axis=1)
        filtered_argmax = np.argmax(filtered_deviation, axis=1)
        qc_status, std_scorable, range_scorable = _window_qc_status(
            z_std,
            z_range,
        )
        for offset, pick in enumerate(picks):
            reasons = []
            if z_std[offset] > qc.robust_z_threshold:
                reasons.append("high_filtered_std")
            if z_std[offset] < -qc.robust_z_threshold:
                reasons.append("low_filtered_std")
            if z_range[offset] > qc.robust_z_threshold:
                reasons.append("high_filtered_range")
            if z_range[offset] < -qc.robust_z_threshold:
                reasons.append("low_filtered_range")
            if flat_flag[offset]:
                reasons.append("flat_segment")
            rows.append(
                {
                    "window_index": window_index,
                    "start_sample": start,
                    "stop_sample_exclusive": stop,
                    "start_s": start / sfreq,
                    "stop_s": stop / sfreq,
                    "duration_s": (stop - start) / sfreq,
                    "n_samples": stop - start,
                    "is_partial": (stop - start) < window_samples,
                    "channel": raw.ch_names[pick],
                    "filtered_std_uv": filtered_std_uv[offset],
                    "filtered_p01_p99_range_uv": filtered_range_uv[offset],
                    "flat_fraction": flat_fraction[offset],
                    "z_log_filtered_std": z_std[offset],
                    "z_log_filtered_range": z_range[offset],
                    "raw_max_abs_deviation_uv": float(
                        raw_deviation[offset, raw_argmax[offset]] * 1e6
                    ),
                    "raw_max_sample": int(start + raw_argmax[offset]),
                    "raw_max_time_s": float((start + raw_argmax[offset]) / sfreq),
                    "filtered_max_abs_deviation_uv": float(
                        filtered_deviation[offset, filtered_argmax[offset]] * 1e6
                    ),
                    "filtered_max_sample": int(start + filtered_argmax[offset]),
                    "filtered_max_time_s": float(
                        (start + filtered_argmax[offset]) / sfreq
                    ),
                    "window_flag": bool(window_flag[offset]),
                    "flat_flag": bool(flat_flag[offset]),
                    "flag_reason": ";".join(reasons),
                    "std_qc_status": "scored" if std_scorable else "zero_mad",
                    "range_qc_status": ("scored" if range_scorable else "zero_mad"),
                    "qc_status": qc_status,
                }
            )
    return pd.DataFrame(rows)


def _summarize_full_recording_windows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Reduce the full temporal screen without discarding window evidence."""
    rows = []
    total_duration = float(
        metrics[["window_index", "duration_s"]].drop_duplicates()["duration_s"].sum()
    )
    for channel, table in metrics.groupby("channel", sort=False):
        flagged = table["window_flag"] | table["flat_flag"]
        scored = table["qc_status"] != "unscorable_zero_mad"
        valid_std = table["z_log_filtered_std"].dropna()
        valid_range = table["z_log_filtered_range"].dropna()
        std_index = valid_std.abs().idxmax() if not valid_std.empty else None
        range_index = valid_range.abs().idxmax() if not valid_range.empty else None
        raw_index = table["raw_max_abs_deviation_uv"].idxmax()
        filtered_index = table["filtered_max_abs_deviation_uv"].idxmax()
        intervals = table.loc[flagged, ["start_s", "stop_s"]]
        rows.append(
            {
                "channel": channel,
                "windows_scanned": len(table),
                "scored_windows": int(scored.sum()),
                "flagged_window_count": int(flagged.sum()),
                "flagged_window_fraction": float(flagged.mean()),
                "flagged_duration_fraction": float(
                    table.loc[flagged, "duration_s"].sum() / total_duration
                ),
                "flat_window_count": int(table["flat_flag"].sum()),
                "maximum_abs_z_log_filtered_std": float(
                    abs(table.loc[std_index, "z_log_filtered_std"])
                    if std_index is not None
                    else np.nan
                ),
                "signed_z_at_maximum_abs_std": float(
                    table.loc[std_index, "z_log_filtered_std"]
                    if std_index is not None
                    else np.nan
                ),
                "maximum_abs_z_std_window_start_s": float(
                    table.loc[std_index, "start_s"] if std_index is not None else np.nan
                ),
                "maximum_abs_z_log_filtered_range": float(
                    abs(table.loc[range_index, "z_log_filtered_range"])
                    if range_index is not None
                    else np.nan
                ),
                "signed_z_at_maximum_abs_range": float(
                    table.loc[range_index, "z_log_filtered_range"]
                    if range_index is not None
                    else np.nan
                ),
                "maximum_abs_z_range_window_start_s": float(
                    table.loc[range_index, "start_s"]
                    if range_index is not None
                    else np.nan
                ),
                "maximum_raw_abs_deviation_uv": float(
                    table.loc[raw_index, "raw_max_abs_deviation_uv"]
                ),
                "maximum_raw_abs_deviation_time_s": float(
                    table.loc[raw_index, "raw_max_time_s"]
                ),
                "maximum_filtered_abs_deviation_uv": float(
                    table.loc[filtered_index, "filtered_max_abs_deviation_uv"]
                ),
                "maximum_filtered_abs_deviation_time_s": float(
                    table.loc[filtered_index, "filtered_max_time_s"]
                ),
                "flagged_intervals_s": ";".join(
                    f"{row.start_s:.3f}-{row.stop_s:.3f}"
                    for row in intervals.itertuples(index=False)
                ),
                "channel_requires_review": bool(flagged.any()),
            }
        )
    return pd.DataFrame(rows)


def _compute_psd(
    raw: mne.io.BaseRaw, line_frequency_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration_s = raw.n_times / float(raw.info["sfreq"])
    margin_s = min(5.0, max(0.0, duration_s / 10.0))
    tmin = margin_s
    tmax = min(duration_s - 1.0 / float(raw.info["sfreq"]), 305.0)
    if tmax <= tmin:
        tmin = 0.0
    psd_raw = raw.copy().crop(tmin=tmin, tmax=tmax)
    n_fft = min(4096, psd_raw.n_times)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"nperseg = .* is greater than input length",
            category=UserWarning,
        )
        spectrum = psd_raw.compute_psd(
            method="welch",
            fmin=0.5,
            fmax=min(100.0, float(raw.info["sfreq"]) / 2.0),
            picks="eeg",
            n_fft=n_fft,
            n_per_seg=n_fft,
            # BAD annotations can split the data into short clean fragments.
            # Zero overlap remains valid when scipy shortens a segment locally.
            n_overlap=0,
            verbose="ERROR",
        )
    psd, freqs = spectrum.get_data(return_freqs=True)
    psd_db_uv = 10.0 * np.log10(np.maximum(psd * 1e12, np.finfo(float).tiny))
    line_mask = (freqs >= line_frequency_hz - 1.0) & (freqs <= line_frequency_hz + 1.0)
    flank_mask = (
        (freqs >= line_frequency_hz - 5.0) & (freqs < line_frequency_hz - 2.0)
    ) | ((freqs > line_frequency_hz + 2.0) & (freqs <= line_frequency_hz + 5.0))
    if line_mask.any() and flank_mask.any():
        tiny = np.finfo(float).tiny
        line_power = np.maximum(np.mean(psd[:, line_mask], axis=1), tiny)
        flank_power = np.maximum(np.mean(psd[:, flank_mask], axis=1), tiny)
        line_ratio_db = 10.0 * np.log10(line_power / flank_power)
    else:
        line_ratio_db = np.full(psd.shape[0], np.nan)
    return freqs, psd_db_uv, line_ratio_db


def _save_plots(
    output: Path,
    raw: mne.io.BaseRaw,
    counts: pd.DataFrame,
    trials: pd.DataFrame,
    metrics: pd.DataFrame,
    picks: np.ndarray,
    representative_data: np.ndarray,
    filtered_representative_data: np.ndarray,
    freqs: np.ndarray,
    psd_db_uv: np.ndarray,
    line_frequency_hz: float,
    filter_hz: tuple[float, float],
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(counts["marker"], counts["count"], color="#386cb0")
    ax.set(title="BrainVision marker counts", xlabel="Marker", ylabel="Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "marker_counts.png", dpi=160)
    plt.close(fig)

    trial_counts = trials["trial_class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [
        "#4daf4a" if "correct" in label or "successful" in label else "#e41a1c"
        for label in trial_counts.index
    ]
    ax.bar(trial_counts.index, trial_counts.values, color=colors)
    ax.set(title="Reconstructed trial outcomes", ylabel="Trials")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "trial_outcomes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(metrics))
    colors = np.where(metrics["candidate_bad"], "#e41a1c", "#377eb8")
    ax.bar(x, metrics["filtered_std_uv"], color=colors, width=0.9)
    ax.set(
        title=(
            f"Representative {filter_hz[0]:g}–{filter_hz[1]:g} Hz "
            "EEG channel standard deviation"
        ),
        xlabel="EEG channel",
        ylabel="SD (µV)",
    )
    ax.set_xticks(x[::8], metrics["channel"].iloc[::8], rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "channel_amplitude.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    median = np.median(psd_db_uv, axis=0)
    low, high = np.percentile(psd_db_uv, [10, 90], axis=0)
    ax.fill_between(
        freqs, low, high, color="#9ecae1", alpha=0.65, label="10–90% channels"
    )
    ax.plot(freqs, median, color="#08519c", linewidth=1.5, label="median")
    ax.axvline(
        line_frequency_hz,
        color="#e41a1c",
        linestyle="--",
        linewidth=1.0,
        label=f"{line_frequency_hz:g} Hz",
    )
    ax.set(
        title="Raw EEG power spectral density",
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB µV²/Hz)",
        xlim=(0.5, 100.0),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "raw_psd.png", dpi=160)
    plt.close(fig)

    sfreq = float(raw.info["sfreq"])
    view = representative_data[:, : int(10 * sfreq) : 10] * 1e6
    view = view - np.median(view, axis=1, keepdims=True)
    clip = max(50.0, float(np.nanpercentile(np.abs(view), 98)))
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        view,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
        extent=[0, 10, len(picks), 0],
    )
    labels = [raw.ch_names[index] for index in picks]
    tick_positions = np.arange(0, len(labels), 10)
    ax.set_yticks(tick_positions + 0.5, [labels[index] for index in tick_positions])
    ax.set(
        title="Representative 10-second raw EEG window",
        xlabel="Time (s)",
        ylabel="EEG channel",
    )
    fig.colorbar(image, ax=ax, label="Amplitude (µV, channel median removed)")
    fig.tight_layout()
    fig.savefig(figures / "raw_overview.png", dpi=160)
    plt.close(fig)

    filtered_view = filtered_representative_data[:, : int(10 * sfreq) : 10] * 1e6
    filtered_view = filtered_view - np.median(filtered_view, axis=1, keepdims=True)
    filtered_clip = max(30.0, float(np.nanpercentile(np.abs(filtered_view), 98)))
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        filtered_view,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-filtered_clip,
        vmax=filtered_clip,
        extent=[0, 10, len(picks), 0],
    )
    ax.set_yticks(tick_positions + 0.5, [labels[index] for index in tick_positions])
    ax.set(
        title=(
            "Representative 10-second EEG window after "
            f"{filter_hz[0]:g}–{filter_hz[1]:g} Hz filter"
        ),
        xlabel="Time (s)",
        ylabel="EEG channel",
    )
    fig.colorbar(image, ax=ax, label="Amplitude (µV, channel median removed)")
    fig.tight_layout()
    fig.savefig(figures / "filtered_overview.png", dpi=160)
    plt.close(fig)


def _save_full_recording_scan_plot(
    output: Path,
    window_metrics: pd.DataFrame,
) -> None:
    """Plot window-level spatial outlier scores without showing EEG samples."""
    figures = output / "figures"
    channels = window_metrics["channel"].drop_duplicates().tolist()
    windows = sorted(window_metrics["window_index"].unique())
    score = window_metrics.assign(
        maximum_z=window_metrics[["z_log_filtered_std", "z_log_filtered_range"]]
        .abs()
        .max(axis=1)
    ).pivot(index="channel", columns="window_index", values="maximum_z")
    score = score.reindex(index=channels, columns=windows)
    flags = (
        window_metrics.assign(
            requires_review=(
                window_metrics["window_flag"] | window_metrics["flat_flag"]
            )
        )
        .pivot(
            index="channel",
            columns="window_index",
            values="requires_review",
        )
        .reindex(index=channels, columns=windows)
    )
    finite_scores = score.to_numpy()[np.isfinite(score.to_numpy())]
    color_maximum = (
        max(10.0, float(np.percentile(finite_scores, 99)))
        if finite_scores.size
        else 10.0
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        score.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=color_maximum,
    )
    flagged_y, flagged_x = np.where(flags.fillna(False).to_numpy())
    ax.scatter(flagged_x, flagged_y, s=9, facecolors="none", edgecolors="red")
    tick_positions = np.arange(0, len(channels), 10)
    ax.set_yticks(tick_positions, [channels[index] for index in tick_positions])
    ax.set(
        title="Full-recording channel-by-window QC screen",
        xlabel="Non-overlapping window from recording start",
        ylabel="EEG channel",
    )
    fig.colorbar(image, ax=ax, label="max |robust z| across amplitude metrics")
    fig.tight_layout()
    fig.savefig(figures / "full_recording_window_scan.png", dpi=160)
    plt.close(fig)


def _save_candidate_window_traces(
    output: Path,
    raw: mne.io.BaseRaw,
    filtered_raw: mne.io.BaseRaw,
    window_metrics: pd.DataFrame,
) -> None:
    """Save private raw/filtered evidence with montage-derived neighbors."""
    candidates = _select_candidate_review_windows(window_metrics)
    if candidates.empty:
        return
    review_dir = output / "figures" / "window_reviews"
    _save_window_trace_figures(review_dir, raw, filtered_raw, candidates)


def _save_window_trace_figures(
    review_dir: Path,
    raw: mne.io.BaseRaw,
    filtered_raw: mne.io.BaseRaw,
    candidates: pd.DataFrame,
    *,
    identifier_column: str | None = None,
    context_seconds: float = 0.0,
    mark_prompt: bool = False,
) -> list[Path]:
    """Save one trace panel for every supplied candidate row."""
    if candidates.empty:
        return []
    review_dir.mkdir(parents=True, exist_ok=True)
    eeg_indices = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[index] for index in eeg_indices]
    if len(eeg_names) < 4:
        adjacency_names = eeg_names
        adjacency = csr_matrix(~np.eye(len(eeg_names), dtype=bool))
    else:
        adjacency, adjacency_names = mne.channels.find_ch_adjacency(raw.info, "eeg")
        adjacency = csr_matrix(adjacency)
    name_to_index = {name: index for index, name in enumerate(adjacency_names)}
    sfreq = float(raw.info["sfreq"])
    written = []
    for row in candidates.itertuples(index=False):
        if row.channel not in name_to_index:
            raise ValueError(
                f"Candidate channel is absent from EEG adjacency: {row.channel}"
            )
        target_index = name_to_index[row.channel]
        neighbor_indices = adjacency.getrow(target_index).indices.tolist()
        neighbor_names = [
            adjacency_names[index]
            for index in neighbor_indices
            if adjacency_names[index] != row.channel
        ]
        names = [row.channel, *neighbor_names]
        prompt_start = int(row.start_sample)
        prompt_stop = int(row.stop_sample_exclusive)
        context_samples = max(0, round(context_seconds * sfreq))
        start = max(0, prompt_start - context_samples)
        stop = min(raw.n_times, prompt_stop + context_samples)
        time = np.arange(start, stop) / sfreq
        raw_data = raw.get_data(picks=names, start=start, stop=stop) * 1e6
        filtered_data = filtered_raw.get_data(picks=names, start=start, stop=stop) * 1e6
        raw_data -= np.median(raw_data, axis=1, keepdims=True)
        filtered_data -= np.median(filtered_data, axis=1, keepdims=True)
        limits = [
            max(float(np.max(np.abs(raw_data))), np.finfo(float).eps),
            max(float(np.max(np.abs(filtered_data))), np.finfo(float).eps),
        ]
        fig, axes = plt.subplots(
            len(names),
            2,
            figsize=(13, max(5, 1.8 * len(names))),
            sharex=True,
            squeeze=False,
        )
        for index, (name, raw_trace, filtered_trace) in enumerate(
            zip(names, raw_data, filtered_data, strict=True)
        ):
            for column, (trace, title) in enumerate(
                ((raw_trace, "Raw"), (filtered_trace, "Filtered"))
            ):
                axis = axes[index, column]
                axis.plot(time, trace, linewidth=0.65, color="#244f73")
                axis.set_ylim(-1.05 * limits[column], 1.05 * limits[column])
                axis.set_ylabel(f"{name}\nµV")
                axis.grid(alpha=0.15)
                if mark_prompt:
                    axis.axvline(float(row.start_s), color="#b33a3a", linestyle="--")
                    axis.axvline(float(row.stop_s), color="#b33a3a", linestyle="--")
                axis.text(
                    0.99,
                    0.88,
                    f"max |x − median(x)|={np.max(np.abs(trace)):.1f} µV",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
                if index == 0:
                    axis.set_title(f"{title} · shared scale across rows")
        axes[-1, 0].set_xlabel("Recording time (s)")
        axes[-1, 1].set_xlabel("Recording time (s)")
        fig.suptitle(
            (
                f"Review candidate {getattr(row, identifier_column)} · "
                if identifier_column is not None
                else "Review candidate "
            )
            + (
                f"{row.channel}: prompt {row.start_s:.3f}–{row.stop_s:.3f} s · "
                f"plotted {start / sfreq:.3f}–{stop / sfreq:.3f} s"
            )
        )
        fig.tight_layout()
        safe_channel = re.sub(r"[^A-Za-z0-9_-]", "_", row.channel)
        if identifier_column is None:
            filename = f"window-{int(row.window_index):03d}_{safe_channel}.png"
        else:
            identifier = re.sub(
                r"[^A-Za-z0-9_-]", "_", str(getattr(row, identifier_column))
            )
            filename = f"{identifier}_{safe_channel}.png"
        path = review_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def _select_candidate_review_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select at most one amplitude and one flat exemplar per channel."""
    amplitude = window_metrics.loc[window_metrics["window_flag"]].copy()
    if not amplitude.empty:
        amplitude["review_score"] = (
            amplitude[["z_log_filtered_std", "z_log_filtered_range"]].abs().max(axis=1)
        )
        amplitude = amplitude.loc[
            amplitude.groupby("channel", sort=False)["review_score"].idxmax()
        ]

    flat = window_metrics.loc[window_metrics["flat_flag"]].copy()
    if not flat.empty:
        flat["review_score"] = flat["flat_fraction"]
        flat = flat.loc[flat.groupby("channel", sort=False)["review_score"].idxmax()]

    selected = pd.concat([amplitude, flat], ignore_index=False)
    if selected.empty:
        return selected
    return selected.drop_duplicates(["channel", "window_index"], keep="first")


def _run_qc_into(vhdr: Path, output: Path, participant_id: str) -> dict:
    """Run the complete non-destructive QC pilot and return its summary."""
    vhdr = vhdr.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    analysis = load_analysis_config()
    codebook = load_event_codebook()

    raw = _read_brainvision_compat(vhdr)
    provenance_core = build_provenance_core(
        vhdr=vhdr,
        participant_id=participant_id,
        analysis=analysis,
        bad_channel_decisions=[],
    )
    nyquist = float(raw.info["sfreq"]) / 2
    if analysis.filter_hz[1] >= nyquist:
        raise ValueError(
            f"Filter high cutoff {analysis.filter_hz[1]:g} Hz must be below "
            f"Nyquist ({nyquist:g} Hz)"
        )
    if analysis.line_frequency_hz >= nyquist:
        raise ValueError(
            f"Line frequency {analysis.line_frequency_hz:g} Hz must be below "
            f"Nyquist ({nyquist:g} Hz)"
        )
    type_updates = {
        name: kind
        for name, kind in analysis.channel_types.items()
        if name in raw.ch_names
    }
    raw.set_channel_types(type_updates, verbose="ERROR")
    montage = mne.channels.make_standard_montage(analysis.montage)
    with warnings.catch_warnings(record=True) as montage_warnings:
        warnings.simplefilter("always")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")

    markers = annotations_to_markers(raw.annotations)
    counts = marker_counts(markers)
    trials = classify_trials(markers, codebook=codebook)
    reconciliation = reconcile_trials(markers, trials, codebook=codebook)
    metrics, picks, representative_data, filtered_representative_data = (
        _channel_metrics(raw, analysis.filter_hz, analysis.qc)
    )
    filtered_for_scan = raw.copy().load_data(verbose="ERROR")
    filtered_for_scan.filter(
        *analysis.filter_hz,
        picks=mne.pick_types(
            filtered_for_scan.info,
            eeg=True,
            eog=False,
            ecg=False,
            exclude=[],
        ),
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        verbose="ERROR",
    )
    window_metrics = _full_recording_window_metrics(
        raw,
        filtered_for_scan,
        analysis.qc,
    )
    temporal_summary = _summarize_full_recording_windows(window_metrics)
    freqs, psd_db_uv, line_ratio_db = _compute_psd(raw, analysis.line_frequency_hz)
    metrics["line_noise_ratio_db"] = line_ratio_db
    metrics["review_high_line_noise"] = (
        line_ratio_db > analysis.qc.line_noise_ratio_db_threshold
    )

    counts.to_csv(output / "marker_counts.csv", index=False)
    trials.to_csv(output / "reconstructed_trials.csv", index=False)
    metrics.to_csv(output / "channel_metrics.csv", index=False)
    window_metrics.to_csv(output / "channel_window_metrics.csv", index=False)
    temporal_summary.to_csv(output / "channel_window_summary.csv", index=False)
    window_metrics.loc[
        window_metrics["window_flag"] | window_metrics["flat_flag"]
    ].to_csv(output / "candidate_window_review.csv", index=False)

    trial_counts = trials["trial_class"].value_counts().sort_index().to_dict()
    representative_bads = metrics.loc[metrics["candidate_bad"], "channel"].tolist()
    temporal_bads = temporal_summary.loc[
        temporal_summary["channel_requires_review"],
        "channel",
    ].tolist()
    candidate_bads = temporal_bads
    recording_max_index = temporal_summary["maximum_raw_abs_deviation_uv"].idxmax()
    high_line_noise = metrics.loc[metrics["review_high_line_noise"], "channel"].tolist()
    summary = {
        "source_header": "withheld-private-source",
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "channels_total": len(raw.ch_names),
        "channels_eeg": len(picks),
        "channels_ecg": len(mne.pick_types(raw.info, ecg=True, eeg=False, eog=False)),
        "channels_eog": len(mne.pick_types(raw.info, eog=True, eeg=False, ecg=False)),
        "annotations_total": len(raw.annotations),
        "stimulus_markers_total": len(markers),
        "trial_counts": {str(key): int(value) for key, value in trial_counts.items()},
        "trial_status_counts": {
            str(key): int(value)
            for key, value in trials["classification_status"]
            .value_counts()
            .sort_index()
            .items()
        },
        "trial_reconciliation": reconciliation,
        "candidate_bad_channels": candidate_bads,
        "representative_candidate_bad_channels": representative_bads,
        "full_recording_candidate_bad_channels": temporal_bads,
        "full_recording_windows_scanned": int(window_metrics["window_index"].nunique()),
        "full_recording_window_seconds": analysis.qc.full_recording_window_seconds,
        "full_recording_coverage_fraction": float(
            window_metrics[["window_index", "n_samples"]]
            .drop_duplicates()["n_samples"]
            .sum()
            / raw.n_times
        ),
        "full_recording_scan_specification": {
            "window_seconds": analysis.qc.full_recording_window_seconds,
            "anchored_to_sample_zero": True,
            "non_overlapping": True,
            "last_partial_window_included": True,
            "filter": {
                "type": "fourth-order zero-phase Butterworth",
                "frequency_hz": list(analysis.filter_hz),
                "applied_before_windowing": True,
            },
            "robust_z_scope": "within each window across scalp EEG channels",
            "robust_z_threshold": analysis.qc.robust_z_threshold,
            "automatic_interpolation": False,
        },
        "full_recording_candidate_summary": temporal_summary.loc[
            temporal_summary["channel_requires_review"]
        ].to_dict(orient="records"),
        "recording_maximum_raw_abs_deviation_uv": float(
            temporal_summary.loc[
                recording_max_index,
                "maximum_raw_abs_deviation_uv",
            ]
        ),
        "recording_maximum_raw_abs_deviation_channel": str(
            temporal_summary.loc[recording_max_index, "channel"]
        ),
        "recording_maximum_raw_abs_deviation_time_s": float(
            temporal_summary.loc[
                recording_max_index,
                "maximum_raw_abs_deviation_time_s",
            ]
        ),
        "candidate_bad_channels_require_visual_confirmation": True,
        "channels_for_high_line_noise_review": high_line_noise,
        "line_frequency_hz": analysis.line_frequency_hz,
        "median_line_noise_ratio_db": float(np.median(line_ratio_db)),
        "montage_warnings": [str(item.message) for item in montage_warnings],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _save_plots(
        output,
        raw,
        counts,
        trials,
        metrics,
        picks,
        representative_data,
        filtered_representative_data,
        freqs,
        psd_db_uv,
        analysis.line_frequency_hz,
        analysis.filter_hz,
    )
    _save_full_recording_scan_plot(output, window_metrics)
    _save_candidate_window_traces(
        output,
        raw,
        filtered_for_scan,
        window_metrics,
    )

    trial_table = "\n".join(
        ["| Class | Trials |", "|---|---:|"]
        + [f"| {label} | {int(value)} |" for label, value in trial_counts.items()]
    )
    report = f"""# Raw EEG QC

> Private derived output. Review identifiers, counts, and figures before sharing.

## Recording

- Duration: {summary["duration_seconds"] / 60:.2f} minutes
- Sampling frequency: {summary["sampling_frequency_hz"]:.0f} Hz
- Channels: {summary["channels_total"]} total; {summary["channels_eeg"]} EEG; {summary["channels_ecg"]} ECG; {summary["channels_eog"]} EOG
- Stimulus markers: {summary["stimulus_markers_total"]}
- Median {summary["line_frequency_hz"]:g} Hz line-noise ratio: {summary["median_line_noise_ratio_db"]:.2f} dB

## Reconstructed trials

{trial_table}

## Candidate bad EEG channels

{", ".join(candidate_bads) if candidate_bads else "No channels crossed the conservative automatic thresholds."}

These are screening candidates only. Confirm them by inspecting the raw traces, spatial neighbors, spectra, and persistence across the recording before interpolation.

The candidate list comes from the complete window-by-window temporal screen.
The representative overview remains a compact visual aid. One flagged window
is enough to request review, but it is not enough to declare a channel bad
automatically.

## Channels with unusually concentrated line-frequency power

{", ".join(high_line_noise) if high_line_noise else "None."}

These channels are not automatically bad. They require review in the filtered trace before any interpolation decision.

## Figures

![Marker counts](figures/marker_counts.png)

![Trial outcomes](figures/trial_outcomes.png)

![Channel amplitude](figures/channel_amplitude.png)

![Raw PSD](figures/raw_psd.png)

![Raw overview](figures/raw_overview.png)

![Filtered overview](figures/filtered_overview.png)

![Full-recording window scan](figures/full_recording_window_scan.png)

## Files

- `marker_counts.csv`: normalized BrainVision marker counts
- `reconstructed_trials.csv`: one row per reconstructed go/stop trial
- `channel_metrics.csv`: channel amplitude, flatness, robust outlier scores, and line-noise ratio
- `channel_window_metrics.csv`: full-recording window-level channel evidence
- `channel_window_summary.csv`: persistence, maxima, and flagged intervals per channel
- `candidate_window_review.csv`: flagged channel-window pairs requiring review
- `figures/window_reviews/`: private raw/filtered trace figures with montage-derived neighbors; at most one amplitude exemplar and one flat-segment exemplar are retained per channel
- `summary.json`: machine-readable recording summary

No binary EEG recording was copied into this report directory. The private
figures contain derived trace excerpts and must be reviewed before sharing.
"""
    (output / "qc_report.md").write_text(report, encoding="utf-8")
    verify_input_sources(vhdr, provenance_core["inputs"])
    write_provenance(output, provenance_core)
    return summary


def run_qc(vhdr: Path, output: Path, participant_id: str = "qc") -> dict:
    """Create one private QC report atomically in a new directory."""
    vhdr = Path(vhdr).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("QC requires a new output path; reuse is disabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    temporary.rmdir()
    try:
        summary = _run_qc_into(vhdr, temporary, participant_id)
        temporary.replace(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
