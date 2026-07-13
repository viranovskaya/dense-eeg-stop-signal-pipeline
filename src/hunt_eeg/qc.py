"""Non-destructive raw EEG quality-control report generation."""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .events import annotations_to_markers, classify_trials, marker_counts


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
        return mne.io.read_raw_brainvision(temporary_header, preload=False, verbose="ERROR")


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def _representative_data(raw: mne.io.BaseRaw, picks: list[int]) -> np.ndarray:
    """Read five 20-second windows spread across the recording."""
    sfreq = float(raw.info["sfreq"])
    window_s = 20.0
    duration_s = raw.n_times / sfreq
    latest_start = max(5.0, duration_s - window_s - 5.0)
    starts = np.linspace(5.0, latest_start, 5)
    chunks = []
    for start_s in starts:
        start = int(round(start_s * sfreq))
        stop = min(raw.n_times, start + int(round(window_s * sfreq)))
        chunks.append(raw.get_data(picks=picks, start=start, stop=stop))
    return np.concatenate(chunks, axis=1)


def _channel_metrics(
    raw: mne.io.BaseRaw,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    picks = mne.pick_types(raw.info, eeg=True, ecg=False, eog=False, exclude=[])
    data = _representative_data(raw, list(picks))
    sos = butter(4, [1.0, 40.0], btype="bandpass", fs=float(raw.info["sfreq"]), output="sos")
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
    candidate = (np.abs(z_std) > 5.0) | (np.abs(z_range) > 5.0) | (flat_fraction > 0.10)

    metrics = pd.DataFrame(
        {
            "channel": [raw.ch_names[index] for index in picks],
            "std_uv": std_uv,
            "robust_range_uv": robust_range_uv,
            "filtered_1_40_std_uv": filtered_std_uv,
            "filtered_1_40_robust_range_uv": filtered_range_uv,
            "flat_fraction": flat_fraction,
            "robust_z_log_filtered_std": z_std,
            "robust_z_log_filtered_range": z_range,
            "candidate_bad": candidate,
        }
    )
    return metrics, picks, data, filtered_data


def _compute_psd(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration_s = raw.n_times / float(raw.info["sfreq"])
    psd_raw = raw.copy().crop(tmin=5.0, tmax=min(duration_s - 1.0, 305.0))
    spectrum = psd_raw.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=100.0,
        picks="eeg",
        n_fft=4096,
        n_per_seg=4096,
        n_overlap=2048,
        verbose="ERROR",
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    psd_db_uv = 10.0 * np.log10(np.maximum(psd * 1e12, np.finfo(float).tiny))
    line_mask = (freqs >= 49.0) & (freqs <= 51.0)
    flank_mask = ((freqs >= 45.0) & (freqs < 48.0)) | ((freqs > 52.0) & (freqs <= 55.0))
    line_ratio_db = 10.0 * np.log10(
        np.mean(psd[:, line_mask], axis=1) / np.mean(psd[:, flank_mask], axis=1)
    )
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
    colors = ["#4daf4a" if "correct" in label or "successful" in label else "#e41a1c" for label in trial_counts.index]
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
    ax.bar(x, metrics["filtered_1_40_std_uv"], color=colors, width=0.9)
    ax.set(title="Representative 1–40 Hz EEG channel standard deviation", xlabel="EEG channel", ylabel="SD (µV)")
    ax.set_xticks(x[::8], metrics["channel"].iloc[::8], rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "channel_amplitude.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    median = np.median(psd_db_uv, axis=0)
    low, high = np.percentile(psd_db_uv, [10, 90], axis=0)
    ax.fill_between(freqs, low, high, color="#9ecae1", alpha=0.65, label="10–90% channels")
    ax.plot(freqs, median, color="#08519c", linewidth=1.5, label="median")
    ax.axvline(50.0, color="#e41a1c", linestyle="--", linewidth=1.0, label="50 Hz")
    ax.set(title="Raw EEG power spectral density", xlabel="Frequency (Hz)", ylabel="PSD (dB µV²/Hz)", xlim=(0.5, 100.0))
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
    image = ax.imshow(view, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-clip, vmax=clip, extent=[0, 10, len(picks), 0])
    labels = [raw.ch_names[index] for index in picks]
    tick_positions = np.arange(0, len(labels), 10)
    ax.set_yticks(tick_positions + 0.5, [labels[index] for index in tick_positions])
    ax.set(title="Representative 10-second raw EEG window", xlabel="Time (s)", ylabel="EEG channel")
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
    ax.set(title="Representative 10-second EEG window after 1–40 Hz filter", xlabel="Time (s)", ylabel="EEG channel")
    fig.colorbar(image, ax=ax, label="Amplitude (µV, channel median removed)")
    fig.tight_layout()
    fig.savefig(figures / "filtered_1_40_overview.png", dpi=160)
    plt.close(fig)


def run_qc(vhdr: Path, output: Path) -> dict:
    """Run the complete non-destructive QC pilot and return its summary."""
    vhdr = vhdr.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = _read_brainvision_compat(vhdr)
    type_updates = {name: kind for name, kind in {"ECG": "ecg", "EOG": "eog"}.items() if name in raw.ch_names}
    raw.set_channel_types(type_updates, verbose="ERROR")
    montage = mne.channels.make_standard_montage("standard_1005")
    with warnings.catch_warnings(record=True) as montage_warnings:
        warnings.simplefilter("always")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")

    markers = annotations_to_markers(raw.annotations)
    counts = marker_counts(markers)
    trials = classify_trials(markers)
    metrics, picks, representative_data, filtered_representative_data = _channel_metrics(raw)
    freqs, psd_db_uv, line_ratio_db = _compute_psd(raw)
    metrics["line_noise_ratio_db"] = line_ratio_db
    metrics["review_high_line_noise"] = line_ratio_db > 20.0

    counts.to_csv(output / "marker_counts.csv", index=False)
    trials.to_csv(output / "reconstructed_trials.csv", index=False)
    metrics.to_csv(output / "channel_metrics.csv", index=False)

    trial_counts = trials["trial_class"].value_counts().sort_index().to_dict()
    candidate_bads = metrics.loc[metrics["candidate_bad"], "channel"].tolist()
    high_line_noise = metrics.loc[metrics["review_high_line_noise"], "channel"].tolist()
    summary = {
        "source_header": vhdr.name,
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "channels_total": len(raw.ch_names),
        "channels_eeg": int(len(picks)),
        "channels_ecg": int(len(mne.pick_types(raw.info, ecg=True, eeg=False, eog=False))),
        "channels_eog": int(len(mne.pick_types(raw.info, eog=True, eeg=False, ecg=False))),
        "annotations_total": len(raw.annotations),
        "stimulus_markers_total": len(markers),
        "trial_counts": {str(key): int(value) for key, value in trial_counts.items()},
        "candidate_bad_channels": candidate_bads,
        "candidate_bad_channels_require_visual_confirmation": True,
        "channels_for_high_line_noise_review": high_line_noise,
        "median_50hz_line_ratio_db": float(np.median(line_ratio_db)),
        "montage_warnings": [str(item.message) for item in montage_warnings],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

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
    )

    trial_table = "\n".join(
        ["| Class | Trials |", "|---|---:|"]
        + [f"| {label} | {int(value)} |" for label, value in trial_counts.items()]
    )
    report = f"""# Raw EEG QC: {vhdr.stem}

## Recording

- Duration: {summary['duration_seconds'] / 60:.2f} minutes
- Sampling frequency: {summary['sampling_frequency_hz']:.0f} Hz
- Channels: {summary['channels_total']} total; {summary['channels_eeg']} EEG; {summary['channels_ecg']} ECG; {summary['channels_eog']} EOG
- Stimulus markers: {summary['stimulus_markers_total']}
- Median 50 Hz line-noise ratio: {summary['median_50hz_line_ratio_db']:.2f} dB

## Reconstructed trials

{trial_table}

## Candidate bad EEG channels

{', '.join(candidate_bads) if candidate_bads else 'No channels crossed the conservative automatic thresholds.'}

These are screening candidates only. Confirm them by inspecting the raw traces, spatial neighbors, spectra, and persistence across the recording before interpolation.

## Channels with unusually concentrated 50 Hz power

{', '.join(high_line_noise) if high_line_noise else 'None.'}

These channels are not automatically bad: the planned 1–40 Hz filter removes 50 Hz. They require review in the filtered trace before any interpolation decision.

## Figures

![Marker counts](figures/marker_counts.png)

![Trial outcomes](figures/trial_outcomes.png)

![Channel amplitude](figures/channel_amplitude.png)

![Raw PSD](figures/raw_psd.png)

![Raw overview](figures/raw_overview.png)

![Filtered 1–40 Hz overview](figures/filtered_1_40_overview.png)

## Files

- `marker_counts.csv`: normalized BrainVision marker counts
- `reconstructed_trials.csv`: one row per reconstructed go/stop trial
- `channel_metrics.csv`: channel amplitude, flatness, robust outlier scores, and line-noise ratio
- `summary.json`: machine-readable recording summary

No raw or processed EEG samples were copied into this report directory.
"""
    (output / "qc_report.md").write_text(report, encoding="utf-8")
    return summary
