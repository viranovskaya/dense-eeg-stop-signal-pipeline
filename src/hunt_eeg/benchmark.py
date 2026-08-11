"""Synthetic corruption benchmark for the dense-EEG QC workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from .config import QCConfig
from .events import (
    annotations_to_markers,
    classify_trials,
    normalize_marker,
    reconcile_trials,
)
from .qc import _compute_psd, _full_recording_window_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_CONFIG = PROJECT_ROOT / "config" / "synthetic_benchmark.json"


def write_json(path: Path, payload: dict) -> None:
    """Write stable JSON while accepting NumPy scalar values."""

    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class Corruption:
    """One known channel-local corruption interval."""

    family: str
    benchmark_role: str
    channel_rank: int
    start_seconds: float
    stop_seconds: float
    amplitude_uv: float


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Validated parameters for one synthetic benchmark profile."""

    sampling_frequency_hz: float
    duration_seconds: float
    eeg_channel_count: int
    window_seconds: float
    random_seeds: dict[str, tuple[int, ...]]
    corruptions: tuple[Corruption, ...]


def load_benchmark_config(
    path: Path = DEFAULT_BENCHMARK_CONFIG,
) -> SyntheticBenchmarkConfig:
    """Load the benchmark profile and reject ambiguous truth definitions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "sampling_frequency_hz",
        "duration_seconds",
        "eeg_channel_count",
        "window_seconds",
        "random_seeds",
        "corruptions",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Benchmark config is missing required fields: {missing}")

    sfreq = float(payload["sampling_frequency_hz"])
    duration = float(payload["duration_seconds"])
    n_eeg = int(payload["eeg_channel_count"])
    window = float(payload["window_seconds"])
    if sfreq <= 2 * 50:
        raise ValueError("sampling_frequency_hz must resolve the 50 Hz test signal")
    if duration <= 0 or window <= 0:
        raise ValueError("Benchmark duration and window must be positive")
    if not 8 <= n_eeg <= 127:
        raise ValueError("eeg_channel_count must be between 8 and 127")

    seeds = {
        str(group): tuple(int(seed) for seed in values)
        for group, values in payload["random_seeds"].items()
    }
    if set(seeds) != {"calibration", "held_out", "stress"}:
        raise ValueError("random_seeds must define calibration, held_out and stress")
    all_seeds = [seed for values in seeds.values() for seed in values]
    if not all(values for values in seeds.values()) or len(all_seeds) != len(
        set(all_seeds)
    ):
        raise ValueError("Benchmark seeds must be non-empty and unique")

    corruptions = tuple(
        Corruption(
            family=str(item["family"]),
            benchmark_role=str(item["benchmark_role"]),
            channel_rank=int(item["channel_rank"]),
            start_seconds=float(item["start_seconds"]),
            stop_seconds=float(item["stop_seconds"]),
            amplitude_uv=float(item["amplitude_uv"]),
        )
        for item in payload["corruptions"]
    )
    allowed = {
        "persistent_noise",
        "intermittent_burst",
        "electrode_pop",
        "flat_segment",
        "line_noise",
    }
    if not corruptions:
        raise ValueError("At least one corruption must be defined")
    for item in corruptions:
        if item.family not in allowed:
            raise ValueError(f"Unknown corruption family: {item.family}")
        if item.benchmark_role not in {"primary", "stress"}:
            raise ValueError("benchmark_role must be primary or stress")
        if not 0 <= item.channel_rank < n_eeg:
            raise ValueError("Corruption channel_rank is outside the EEG montage")
        if not 0 <= item.start_seconds < item.stop_seconds <= duration:
            raise ValueError("Corruption intervals must lie inside the recording")
        if item.amplitude_uv < 0:
            raise ValueError("Corruption amplitude must not be negative")
    return SyntheticBenchmarkConfig(
        sampling_frequency_hz=sfreq,
        duration_seconds=duration,
        eeg_channel_count=n_eeg,
        window_seconds=window,
        random_seeds=seeds,
        corruptions=corruptions,
    )


def _eeg_names(count: int) -> list[str]:
    montage = mne.channels.make_standard_montage("standard_1005")
    excluded = {"Nz", "LPA", "RPA"}
    candidates = [name for name in montage.ch_names if name not in excluded]
    if len(candidates) < count:
        raise ValueError("The selected montage has too few EEG channels")
    selected = candidates[:count]
    for required in ("C3", "C4"):
        if required not in selected:
            selected[-1 if required == "C4" else -2] = required
    if len(selected) != len(set(selected)):
        raise RuntimeError("Synthetic montage selection contains duplicate channels")
    return selected


def _colored_noise(
    rng: np.random.Generator,
    channel_count: int,
    sample_count: int,
    sfreq: float,
) -> np.ndarray:
    white = rng.standard_normal((channel_count, sample_count))
    spectrum = np.fft.rfft(white, axis=1)
    frequencies = np.fft.rfftfreq(sample_count, 1.0 / sfreq)
    scaling = np.ones_like(frequencies)
    scaling[1:] = 1.0 / np.sqrt(frequencies[1:])
    colored = np.fft.irfft(spectrum * scaling, n=sample_count, axis=1)
    colored -= colored.mean(axis=1, keepdims=True)
    colored /= np.maximum(colored.std(axis=1, keepdims=True), 1e-12)
    return colored


def make_clean_recording(
    config: SyntheticBenchmarkConfig,
    seed: int,
) -> mne.io.RawArray:
    """Create public synthetic EEG with ocular, cardiac and task sources."""
    rng = np.random.default_rng(seed)
    sfreq = config.sampling_frequency_hz
    samples = int(round(config.duration_seconds * sfreq))
    time = np.arange(samples) / sfreq
    names = _eeg_names(config.eeg_channel_count)

    eeg = _colored_noise(rng, len(names), samples, sfreq) * 7e-6
    phase = rng.uniform(0, 2 * np.pi, (len(names), 1))
    alpha_weights = rng.uniform(1.5e-6, 5e-6, (len(names), 1))
    eeg += alpha_weights * np.sin(2 * np.pi * 10 * time + phase)
    eeg += rng.uniform(0.3e-6, 1.2e-6, (len(names), 1)) * np.sin(
        2 * np.pi * 20 * time + phase / 2
    )

    blink = np.zeros(samples)
    for onset in np.arange(12.0, config.duration_seconds, 31.0):
        blink += np.exp(-0.5 * ((time - onset) / 0.13) ** 2) * 150e-6
    frontal = np.linspace(0.35, 0.04, len(names))[:, np.newaxis]
    eeg += frontal * blink

    heart_rate_hz = 1.15
    cardiac = np.zeros(samples)
    for onset in np.arange(0.8, config.duration_seconds, 1 / heart_rate_hz):
        cardiac += np.exp(-0.5 * ((time - onset) / 0.025) ** 2) * 500e-6
    eeg += rng.uniform(0.005, 0.02, (len(names), 1)) * cardiac

    event_onsets = np.arange(10.0, config.duration_seconds - 3.0, 5.0)
    event_signal = np.zeros(samples)
    for onset in event_onsets:
        event_signal += -4e-6 * np.exp(-0.5 * ((time - onset - 0.32) / 0.07) ** 2)
    for channel, sign in (("C3", 1.0), ("C4", -1.0)):
        if channel in names:
            eeg[names.index(channel)] += sign * event_signal

    eog = blink + rng.normal(scale=4e-6, size=samples)
    ecg = cardiac + rng.normal(scale=8e-6, size=samples)
    data = np.vstack([eeg, eog, ecg])
    info = mne.create_info(
        names + ["EOG", "ECG"],
        sfreq,
        ["eeg"] * len(names) + ["eog", "ecg"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
    annotation_onsets = []
    descriptions = []
    for index, onset in enumerate(event_onsets):
        if index % 2 == 0:
            annotation_onsets.extend([onset, onset + 0.55])
            descriptions.extend(["Stimulus/S  17", "Stimulus/S   6"])
        else:
            annotation_onsets.extend([onset, onset + 0.25])
            descriptions.extend(["Stimulus/S   1", "Stimulus/S  19"])
            if index % 4 == 3 or index == len(event_onsets) - 1:
                annotation_onsets.append(onset + 0.45)
                descriptions.append("Stimulus/S   5")
    raw.set_annotations(
        mne.Annotations(
            onset=annotation_onsets,
            duration=np.zeros(len(annotation_onsets)),
            description=descriptions,
        )
    )
    return raw


def inject_corruptions(
    clean_raw: mne.io.BaseRaw,
    config: SyntheticBenchmarkConfig,
    seed: int,
) -> tuple[mne.io.RawArray, pd.DataFrame]:
    """Inject declared corruptions and return sample-level ground truth."""
    rng = np.random.default_rng(seed + 99173)
    corrupted = clean_raw.copy().load_data()
    data = corrupted.get_data()
    sfreq = float(corrupted.info["sfreq"])
    rows = []
    for item in config.corruptions:
        channel = corrupted.ch_names[item.channel_rank]
        start = int(round(item.start_seconds * sfreq))
        stop = int(round(item.stop_seconds * sfreq))
        interval_time = np.arange(stop - start) / sfreq
        amplitude = item.amplitude_uv * 1e-6
        if item.family == "persistent_noise":
            data[item.channel_rank, start:stop] += rng.normal(
                scale=amplitude, size=stop - start
            )
        elif item.family == "intermittent_burst":
            envelope = np.sin(np.linspace(0, np.pi, stop - start)) ** 2
            data[item.channel_rank, start:stop] += (
                amplitude * envelope * np.sin(2 * np.pi * 7 * interval_time)
            )
        elif item.family == "electrode_pop":
            data[item.channel_rank, start:stop] += amplitude * np.exp(
                -interval_time / 0.08
            )
        elif item.family == "flat_segment":
            data[item.channel_rank, start:stop] = data[item.channel_rank, start]
        elif item.family == "line_noise":
            data[item.channel_rank, start:stop] += amplitude * np.sin(
                2 * np.pi * 50 * interval_time
            )
        rows.append(
            {
                "seed": seed,
                "family": item.family,
                "benchmark_role": item.benchmark_role,
                "channel": channel,
                "channel_rank": item.channel_rank,
                "start_sample": start,
                "stop_sample_exclusive": stop,
                "start_s": item.start_seconds,
                "stop_s": item.stop_seconds,
                "amplitude_uv": item.amplitude_uv,
            }
        )
    corrupted._data[:] = data
    return corrupted, pd.DataFrame(rows)


def truth_by_window(
    truth: pd.DataFrame,
    channel_names: list[str],
    sample_count: int,
    sfreq: float,
    window_seconds: float,
) -> pd.DataFrame:
    """Map corruption intervals to the same channel-window grid used by QC."""
    window_samples = max(1, int(round(window_seconds * sfreq)))
    rows = []
    for window_index, start in enumerate(
        range(0, sample_count, window_samples), start=1
    ):
        stop = min(sample_count, start + window_samples)
        for channel in channel_names:
            overlaps = truth.loc[
                (truth["channel"] == channel)
                & (truth["start_sample"] < stop)
                & (truth["stop_sample_exclusive"] > start)
            ]
            rows.append(
                {
                    "window_index": window_index,
                    "channel": channel,
                    "truth_positive": not overlaps.empty,
                    "truth_families": ";".join(sorted(overlaps["family"].unique())),
                    "truth_roles": ";".join(
                        sorted(overlaps["benchmark_role"].unique())
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_detection(
    predicted_windows: pd.DataFrame,
    truth_windows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare review flags with known channel-window corruption labels."""
    predicted = predicted_windows[
        ["window_index", "channel", "window_flag", "flat_flag", "flag_reason"]
    ].copy()
    predicted["predicted_positive"] = predicted["window_flag"] | predicted["flat_flag"]
    comparison = truth_windows.merge(
        predicted,
        on=["window_index", "channel"],
        how="outer",
        validate="one_to_one",
    )
    if comparison.isna().any().any():
        raise ValueError("Prediction and truth grids do not match")
    return comparison, _binary_metrics(
        comparison["truth_positive"].astype(bool),
        comparison["predicted_positive"].astype(bool),
    )


def score_line_noise_detection(
    raw: mne.io.BaseRaw,
    truth: pd.DataFrame,
    line_frequency_hz: float,
    threshold_db: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Score the existing raw-PSD line-noise review prompt by channel."""
    eeg_names = raw.copy().pick("eeg").ch_names
    _, _, ratios = _compute_psd(raw, line_frequency_hz)
    truth_channels = set(
        truth.loc[truth["family"] == "line_noise", "channel"].astype(str)
    )
    comparison = pd.DataFrame(
        {
            "channel": eeg_names,
            "truth_positive": [name in truth_channels for name in eeg_names],
            "line_noise_ratio_db": ratios,
            "predicted_positive": ratios > threshold_db,
        }
    )
    return comparison, _binary_metrics(
        comparison["truth_positive"].astype(bool),
        comparison["predicted_positive"].astype(bool),
    )


def score_preservation(
    clean_raw: mne.io.BaseRaw,
    observed_raw: mne.io.BaseRaw,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify distortion separately for clean and corrupted samples."""
    if (
        clean_raw.ch_names != observed_raw.ch_names
        or clean_raw.n_times != observed_raw.n_times
    ):
        raise ValueError("Clean and observed recordings must share shape and channels")
    clean = clean_raw.get_data(picks="eeg")
    observed = observed_raw.get_data(picks="eeg")
    sfreq = float(clean_raw.info["sfreq"])
    rows = []
    for index, channel in enumerate(clean_raw.copy().pick("eeg").ch_names):
        corrupted_mask = np.zeros(clean.shape[1], dtype=bool)
        for item in truth.loc[truth["channel"] == channel].itertuples():
            corrupted_mask[item.start_sample : item.stop_sample_exclusive] = True
        for region, mask in (
            ("clean_samples", ~corrupted_mask),
            ("corrupted_samples", corrupted_mask),
        ):
            if not mask.any():
                continue
            reference = clean[index, mask]
            estimate = observed[index, mask]
            rmse_uv = float(np.sqrt(np.mean((estimate - reference) ** 2)) * 1e6)
            scale_uv = float(np.std(reference) * 1e6)
            correlation = (
                float(np.corrcoef(reference, estimate)[0, 1])
                if np.std(reference) > 0 and np.std(estimate) > 0
                else np.nan
            )
            rows.append(
                {
                    "channel": channel,
                    "region": region,
                    "sample_count": int(mask.sum()),
                    "duration_s": float(mask.sum() / sfreq),
                    "correlation": correlation,
                    "rmse_uv": rmse_uv,
                    "nrmse": rmse_uv / scale_uv if scale_uv > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def preprocess_with_known_decisions(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    filter_hz: tuple[float, float],
) -> mne.io.BaseRaw:
    """Apply the signal operations used by the methods workflow."""
    processed = raw.copy().load_data()
    signal_picks = mne.pick_types(
        processed.info,
        eeg=True,
        eog=True,
        ecg=False,
        exclude=[],
    )
    processed.filter(
        *filter_hz,
        picks=signal_picks,
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    processed.info["bads"] = list(bad_channels)
    processed.set_eeg_reference("average", projection=False, verbose="ERROR")
    if bad_channels:
        processed.interpolate_bads(
            reset_bads=True,
            method={"eeg": "spline"},
            verbose="ERROR",
        )
        processed.set_eeg_reference("average", projection=False, verbose="ERROR")
    return processed


def score_task_signal(
    reference_raw: mne.io.BaseRaw,
    observed_raw: mne.io.BaseRaw,
) -> pd.DataFrame:
    """Compare the known event-locked C3/C4 signal after processing."""
    sfreq = float(reference_raw.info["sfreq"])
    if not np.isclose(sfreq, float(observed_raw.info["sfreq"])):
        raise ValueError("Task-signal recordings must share sampling frequency")
    trial_start_markers = {"S1", "S2", "S17", "S18"}
    onsets = np.array(
        [
            annotation["onset"]
            for annotation in reference_raw.annotations
            if normalize_marker(annotation["description"]) in trial_start_markers
        ],
        dtype=float,
    )
    rows = []
    for channel in ("C3", "C4"):
        if (
            channel not in reference_raw.ch_names
            or channel not in observed_raw.ch_names
        ):
            continue
        channel_index = reference_raw.ch_names.index(channel)
        reference_epochs = []
        observed_epochs = []
        start_offset = int(round(-0.2 * sfreq))
        stop_offset = int(round(0.6 * sfreq))
        for onset in onsets:
            event_sample = int(round(onset * sfreq))
            start = event_sample + start_offset
            stop = event_sample + stop_offset
            if start < 0 or stop > reference_raw.n_times:
                continue
            reference_epochs.append(
                reference_raw.get_data(picks=[channel_index], start=start, stop=stop)[0]
            )
            observed_epochs.append(
                observed_raw.get_data(picks=[channel_index], start=start, stop=stop)[0]
            )
        reference_erp = np.mean(reference_epochs, axis=0)
        observed_erp = np.mean(observed_epochs, axis=0)
        baseline_stop = int(round(0.2 * sfreq))
        reference_erp -= reference_erp[:baseline_stop].mean()
        observed_erp -= observed_erp[:baseline_stop].mean()
        target_start = int(round(0.4 * sfreq))
        target_stop = int(round(0.65 * sfreq))
        reference_target = reference_erp[target_start:target_stop]
        observed_target = observed_erp[target_start:target_stop]
        if channel == "C3":
            reference_peak = int(np.argmin(reference_target))
            observed_peak = int(np.argmin(observed_target))
        else:
            reference_peak = int(np.argmax(reference_target))
            observed_peak = int(np.argmax(observed_target))
        reference_amplitude = float(reference_target[reference_peak] * 1e6)
        observed_amplitude = float(observed_target[observed_peak] * 1e6)
        reference_latency = (target_start + reference_peak) / sfreq - 0.2
        observed_latency = (target_start + observed_peak) / sfreq - 0.2
        rows.append(
            {
                "channel": channel,
                "epochs": len(reference_epochs),
                "reference_peak_amplitude_uv": reference_amplitude,
                "observed_peak_amplitude_uv": observed_amplitude,
                "absolute_amplitude_error_uv": abs(
                    observed_amplitude - reference_amplitude
                ),
                "reference_peak_latency_ms": reference_latency * 1000,
                "observed_peak_latency_ms": observed_latency * 1000,
                "absolute_latency_error_ms": abs(observed_latency - reference_latency)
                * 1000,
            }
        )
    return pd.DataFrame(rows)


def score_band_power(
    reference_raw: mne.io.BaseRaw,
    observed_raw: mne.io.BaseRaw,
) -> pd.DataFrame:
    """Compare channel-level band power after matched processing."""
    reference = reference_raw.get_data(picks="eeg")
    observed = observed_raw.get_data(picks="eeg")
    sfreq = float(reference_raw.info["sfreq"])
    nperseg = min(reference.shape[1], int(round(4 * sfreq)))
    frequencies, reference_psd = welch(
        reference,
        fs=sfreq,
        nperseg=nperseg,
        axis=1,
    )
    _, observed_psd = welch(
        observed,
        fs=sfreq,
        nperseg=nperseg,
        axis=1,
    )
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    names = reference_raw.copy().pick("eeg").ch_names
    rows = []
    tiny = np.finfo(float).tiny
    for band, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        reference_power = np.trapezoid(
            reference_psd[:, mask], frequencies[mask], axis=1
        )
        observed_power = np.trapezoid(observed_psd[:, mask], frequencies[mask], axis=1)
        for index, channel in enumerate(names):
            reference_value = max(float(reference_power[index]), tiny)
            observed_value = max(float(observed_power[index]), tiny)
            rows.append(
                {
                    "channel": channel,
                    "band": band,
                    "reference_power_uv2": reference_value * 1e12,
                    "observed_power_uv2": observed_value * 1e12,
                    "absolute_log_ratio_db": abs(
                        10 * np.log10(observed_value / reference_value)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_band_errors(
    table: pd.DataFrame,
    oracle_channels: set[str],
    corrupted_channels: set[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Label preservation rows and retain center plus upper-tail errors."""
    classified = table.copy()

    def channel_group(channel: str) -> str:
        if channel in oracle_channels:
            return "oracle_interpolated"
        if channel in corrupted_channels:
            return "non_oracle_corrupted"
        return "unaffected"

    classified["channel_group"] = classified["channel"].map(channel_group)

    def statistics(values: pd.Series) -> dict[str, float | int]:
        return {
            "values": int(len(values)),
            "median_absolute_error_db": float(values.median()),
            "p95_absolute_error_db": float(values.quantile(0.95)),
            "maximum_absolute_error_db": float(values.max()),
        }

    summaries = {
        "all_channel_band_values": statistics(classified["absolute_log_ratio_db"])
    }
    for group, group_table in classified.groupby("channel_group", sort=True):
        summaries[group] = statistics(group_table["absolute_log_ratio_db"])
    return classified, summaries


def summarize_evaluation_units(
    temporal_comparison: pd.DataFrame,
    primary_comparison: pd.DataFrame,
    stress_comparison: pd.DataFrame,
) -> dict[str, int]:
    """Count all temporal and primary-only channel-window units separately."""
    return {
        "temporal_channel_windows": int(len(temporal_comparison)),
        "primary_channel_windows": int(len(primary_comparison)),
        "primary_positive_channel_windows": int(
            primary_comparison["truth_positive"].astype(bool).sum()
        ),
        "stress_positive_channel_windows": int(
            stress_comparison["truth_positive"].astype(bool).sum()
        ),
    }


def verify_brainvision_round_trip(
    original: mne.io.BaseRaw,
    reread: mne.io.BaseRaw,
) -> dict[str, float | int]:
    """Verify channel, sample, signal and marker preservation after export."""
    if reread.ch_names != original.ch_names:
        raise ValueError("BrainVision round trip changed channel names")
    if reread.n_times != original.n_times:
        raise ValueError("BrainVision round trip changed sample count")
    maximum_signal_error_uv = float(
        np.max(np.abs(reread.get_data() - original.get_data())) * 1e6
    )
    if maximum_signal_error_uv > 1e-3:
        raise ValueError("BrainVision round trip changed signal values")
    original_markers = annotations_to_markers(original.annotations)
    round_trip_markers = annotations_to_markers(reread.annotations)
    original_codes = [marker["marker"] for marker in original_markers]
    round_trip_codes = [marker["marker"] for marker in round_trip_markers]
    if original_codes != round_trip_codes:
        raise ValueError("BrainVision round trip changed marker codes or order")
    maximum_onset_error_samples = max(
        (
            abs(original_marker["onset_s"] - round_trip_marker["onset_s"])
            * float(original.info["sfreq"])
            for original_marker, round_trip_marker in zip(
                original_markers, round_trip_markers
            )
        ),
        default=0.0,
    )
    if maximum_onset_error_samples > 1.0 + 1e-9:
        raise ValueError(
            "BrainVision round trip moved a marker by more than one sample"
        )
    return {
        "channels": len(reread.ch_names),
        "samples": reread.n_times,
        "markers": len(round_trip_markers),
        "maximum_signal_error_uv": maximum_signal_error_uv,
        "maximum_marker_onset_error_samples": maximum_onset_error_samples,
    }


def run_benchmark_seed(
    config: SyntheticBenchmarkConfig,
    qc: QCConfig,
    seed: int,
    filter_hz: tuple[float, float] = (1.0, 40.0),
) -> dict[str, object]:
    """Generate, corrupt and score one deterministic recording."""
    if not np.isclose(qc.full_recording_window_seconds, config.window_seconds):
        raise ValueError("Benchmark and QC window lengths must match")
    clean = make_clean_recording(config, seed)
    corrupted, truth = inject_corruptions(clean, config, seed)
    filtered = corrupted.copy().filter(
        1.0,
        40.0,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        verbose="ERROR",
    )
    predictions = _full_recording_window_metrics(corrupted, filtered, qc)
    eeg_names = corrupted.copy().pick("eeg").ch_names
    window_truth = truth_by_window(
        truth,
        eeg_names,
        corrupted.n_times,
        float(corrupted.info["sfreq"]),
        config.window_seconds,
    )
    comparison, detection = score_detection(predictions, window_truth)
    temporal_truth = comparison["truth_families"] != "line_noise"
    temporal_comparison = comparison.loc[temporal_truth].copy()
    temporal_detection = _binary_metrics(
        temporal_comparison["truth_positive"].astype(bool),
        temporal_comparison["predicted_positive"].astype(bool),
    )
    primary_comparison = temporal_comparison.loc[
        temporal_comparison["truth_roles"] != "stress"
    ].copy()
    primary_detection = _binary_metrics(
        primary_comparison["truth_positive"].astype(bool),
        primary_comparison["predicted_positive"].astype(bool),
    )
    stress_comparison = temporal_comparison.loc[
        temporal_comparison["truth_roles"] == "stress"
    ].copy()
    stress_detection_rate = float(stress_comparison["predicted_positive"].mean())
    line_comparison, line_detection = score_line_noise_detection(
        corrupted,
        truth,
        50.0,
        qc.line_noise_ratio_db_threshold,
    )
    corruption_impact = score_preservation(clean, corrupted, truth)
    persistent_families = {"persistent_noise", "flat_segment"}
    known_bad_channels = sorted(
        truth.loc[truth["family"].isin(persistent_families), "channel"].astype(str)
    )
    clean_processed = preprocess_with_known_decisions(clean, [], filter_hz)
    cleaned = preprocess_with_known_decisions(
        corrupted,
        known_bad_channels,
        filter_hz,
    )
    preservation = score_preservation(clean_processed, cleaned, truth)
    task_signal = score_task_signal(clean_processed, cleaned)
    band_power = score_band_power(clean_processed, cleaned)
    markers = annotations_to_markers(clean.annotations)
    trials = classify_trials(markers)
    trial_accounting = reconcile_trials(markers, trials)
    return {
        "clean": clean,
        "corrupted": corrupted,
        "truth": truth,
        "truth_windows": window_truth,
        "predictions": predictions,
        "comparison": comparison,
        "detection": detection,
        "temporal_comparison": temporal_comparison,
        "temporal_detection": temporal_detection,
        "primary_comparison": primary_comparison,
        "primary_detection": primary_detection,
        "stress_comparison": stress_comparison,
        "stress_detection_rate": stress_detection_rate,
        "line_noise_comparison": line_comparison,
        "line_noise_detection": line_detection,
        "known_bad_channels": known_bad_channels,
        "corruption_impact": corruption_impact,
        "preservation": preservation,
        "task_signal": task_signal,
        "band_power": band_power,
        "trials": trials,
        "trial_accounting": trial_accounting,
        "clean_processed": clean_processed,
        "cleaned": cleaned,
    }


def _binary_metrics(
    truth_values: pd.Series,
    predicted_values: pd.Series,
) -> dict[str, float | int]:
    """Return deterministic binary-classification counts and rates."""
    tp = int((truth_values & predicted_values).sum())
    fp = int((~truth_values & predicted_values).sum())
    fn = int((truth_values & ~predicted_values).sum())
    tn = int((~truth_values & ~predicted_values).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
