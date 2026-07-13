"""Reproducible continuous preprocessing and condition epoching."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from .events import annotations_to_markers, classify_trials
from .qc import _read_brainvision_compat


def _event_array(raw: mne.io.BaseRaw, trials, onset_column: str, classes: list[str]):
    event_id = {label: index + 1 for index, label in enumerate(classes)}
    rows = []
    for trial in trials.itertuples(index=False):
        if trial.trial_class not in event_id:
            continue
        onset = getattr(trial, onset_column)
        if onset is None or not np.isfinite(onset):
            continue
        sample = int(raw.time_as_index(float(onset), use_rounding=True)[0] + raw.first_samp)
        rows.append([sample, 0, event_id[trial.trial_class]])
    return np.asarray(rows, dtype=int), event_id


def _plot_c3_c4(epochs: mne.Epochs, output: Path, title: str) -> None:
    available = [channel for channel in ["C3", "C4"] if channel in epochs.ch_names]
    if not available:
        return
    prepared = epochs.copy().filter(None, 12.0, picks="eeg", verbose="ERROR")
    baseline_start = max(float(prepared.tmin), -1.0)
    prepared.apply_baseline((baseline_start, 0.0), verbose="ERROR")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"C3": "#377eb8", "C4": "#e41a1c"}
    styles = ["-", "--", ":", "-."]
    for condition_index, condition in enumerate(prepared.event_id):
        condition_epochs = prepared[condition]
        if len(condition_epochs) == 0:
            continue
        evoked = condition_epochs.average(picks=available)
        for channel_index, channel in enumerate(available):
            data = evoked.data[channel_index] * 1e6
            ax.plot(
                evoked.times,
                data,
                color=colors[channel],
                linestyle=styles[condition_index % len(styles)],
                label=f"{condition}: {channel} (n={len(condition_epochs)})",
            )
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.3)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude (µV), low-pass 12 Hz")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def preprocess_recording(
    vhdr: Path,
    output: Path,
    participant_id: str,
    bad_channels: list[str] | None = None,
    export_eeglab: bool = True,
) -> dict:
    """Filter, rereference, interpolate persistent bad EEG channels, and epoch."""
    bad_channels = bad_channels or []
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw = _read_brainvision_compat(vhdr.expanduser().resolve())
    type_updates = {
        name: kind
        for name, kind in {"ECG": "ecg", "EOG": "eog"}.items()
        if name in raw.ch_names
    }
    raw.set_channel_types(type_updates, verbose="ERROR")
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="warn",
        verbose="ERROR",
    )
    markers = annotations_to_markers(raw.annotations)
    trials = classify_trials(markers)

    if "ECG" in raw.ch_names:
        raw.drop_channels(["ECG"])
    unknown_bads = sorted(set(bad_channels) - set(raw.ch_names))
    if unknown_bads:
        raise ValueError(f"Bad-channel labels not present in recording: {unknown_bads}")
    non_eeg_bads = [
        name for name in bad_channels if raw.get_channel_types(picks=[name])[0] != "eeg"
    ]
    if non_eeg_bads:
        raise ValueError(f"Only scalp EEG channels may be interpolated: {non_eeg_bads}")

    raw.load_data(verbose="ERROR")
    signal_picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, exclude=[])
    raw.filter(1.0, 40.0, picks=signal_picks, method="fir", phase="zero", verbose="ERROR")

    raw.info["bads"] = list(bad_channels)
    # Marked bads are excluded from the reference estimate.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if bad_channels:
        raw.interpolate_bads(reset_bads=True, method={"eeg": "spline"}, verbose="ERROR")
        # Restore a true common-average reference after interpolation.
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    continuous_fif = output / f"sub-{participant_id}_continuous_1-40Hz_avgref_raw.fif"
    raw.save(continuous_fif, overwrite=True, fmt="single", verbose="ERROR")
    if export_eeglab:
        raw.export(
            output / f"sub-{participant_id}_continuous_1-40Hz_avgref.set",
            fmt="eeglab",
            overwrite=True,
            verbose="ERROR",
        )

    go_events, go_event_id = _event_array(
        raw,
        trials,
        "stimulus_onset_s",
        ["go_correct", "go_incorrect_or_slow"],
    )
    stop_events, stop_event_id = _event_array(
        raw,
        trials,
        "stop_signal_onset_s",
        ["stop_successful", "stop_failed"],
    )
    # Epoch exports should contain only the explicit reconstructed condition
    # event at time zero. Continuous BrainVision annotations remain preserved
    # in the continuous file, but copying them into every epoch creates invalid
    # out-of-window EEGLAB events.
    epoch_source = raw.copy()
    epoch_source.set_annotations(None)
    go_epochs = mne.Epochs(
        epoch_source,
        go_events,
        event_id=go_event_id,
        tmin=-1.5,
        tmax=3.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    stop_epochs = mne.Epochs(
        epoch_source,
        stop_events,
        event_id=stop_event_id,
        tmin=-2.0,
        tmax=2.0,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    go_epochs.save(
        output / f"sub-{participant_id}_go-epo.fif",
        overwrite=True,
        fmt="single",
        verbose="ERROR",
    )
    stop_epochs.save(
        output / f"sub-{participant_id}_stop-epo.fif",
        overwrite=True,
        fmt="single",
        verbose="ERROR",
    )
    if export_eeglab:
        go_epochs.export(
            output / f"sub-{participant_id}_go.set",
            fmt="eeglab",
            overwrite=True,
            verbose="ERROR",
        )
        stop_epochs.export(
            output / f"sub-{participant_id}_stop.set",
            fmt="eeglab",
            overwrite=True,
            verbose="ERROR",
        )

    _plot_c3_c4(
        go_epochs,
        figures / "c3_c4_go_lowpass12.png",
        "C3/C4: go-locked responses",
    )
    _plot_c3_c4(
        stop_epochs,
        figures / "c3_c4_stop_lowpass12.png",
        "C3/C4: stop-signal-locked responses",
    )

    summary = {
        "participant_id": participant_id,
        "source_header": vhdr.name,
        "filter_hz": [1.0, 40.0],
        "reference": "average EEG only",
        "ecg_removed": "ECG" in type_updates,
        "eog_retained_and_excluded_from_reference": "EOG" in type_updates,
        "interpolated_bad_channels": bad_channels,
        "continuous_channels": len(raw.ch_names),
        "go_epochs": {condition: len(go_epochs[condition]) for condition in go_event_id},
        "stop_epochs": {
            condition: len(stop_epochs[condition]) for condition in stop_event_id
        },
        "ica_status": "not fitted yet; component decisions must be explicit",
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trials.to_csv(output / "reconstructed_trials.csv", index=False)
    return summary
