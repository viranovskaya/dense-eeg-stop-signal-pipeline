"""Reproducible continuous preprocessing and condition epoching."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .config import AnalysisConfig, load_analysis_config, load_event_codebook
from .events import (
    annotations_to_markers,
    classify_trials,
    normalize_marker,
    reconcile_trials,
)
from .provenance import (
    build_provenance_core,
    verify_input_sources,
    write_provenance,
)
from .qc import _channel_metrics, _read_brainvision_compat


def _event_array(raw: mne.io.BaseRaw, trials, onset_column: str, classes: list[str]):
    event_id = {label: index + 1 for index, label in enumerate(classes)}
    rows = []
    metadata = []
    for trial in trials.itertuples(index=False):
        if trial.trial_class not in event_id:
            continue
        onset = getattr(trial, onset_column)
        if onset is None or not np.isfinite(onset):
            continue
        sample = int(raw.time_as_index(float(onset), use_rounding=True)[0] + raw.first_samp)
        rows.append([sample, 0, event_id[trial.trial_class]])
        metadata.append(
            {
                "trial_index": int(trial.trial_index),
                "trial_type": trial.trial_type,
                "trial_class": trial.trial_class,
                "classification_status": trial.classification_status,
                "classification_confidence": trial.classification_confidence,
                "classification_reason": trial.classification_reason,
                "marker_sequence": trial.marker_sequence,
            }
        )
    return (
        np.asarray(rows, dtype=int).reshape(-1, 3),
        event_id,
        pd.DataFrame(metadata),
    )


def _non_task_annotations(annotations: mne.Annotations) -> mne.Annotations:
    """Keep BAD and descriptive annotations while removing acquisition markers."""
    keep = [
        index
        for index, description in enumerate(annotations.description)
        if not re.fullmatch(r"S\d+", normalize_marker(description))
    ]
    return annotations[keep]


def _epoch_accounting(
    events: np.ndarray, epochs: mne.Epochs, metadata: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Reconcile proposed events with retained and dropped epochs."""
    proposed = len(events)
    retained = len(epochs)
    dropped = proposed - retained
    drop_reasons = Counter(
        reason
        for reasons in epochs.drop_log
        for reason in reasons
        if reason != "IGNORED"
    )
    dropped_rows = sum(
        bool([reason for reason in reasons if reason != "IGNORED"])
        for reasons in epochs.drop_log
    )
    if dropped < 0 or retained + dropped != proposed or dropped_rows != dropped:
        raise RuntimeError(
            "Epoch accounting failed: "
            f"{proposed} proposed, {retained} retained, {dropped_rows} dropped"
        )
    retained_sources = {int(source): index for index, source in enumerate(epochs.selection)}
    lineage = metadata.copy()
    lineage["proposed_event_index"] = np.arange(proposed)
    lineage["retained"] = [index in retained_sources for index in range(proposed)]
    lineage["output_epoch_index"] = [
        retained_sources.get(index) for index in range(proposed)
    ]
    lineage["drop_reasons"] = [
        ";".join(reason for reason in epochs.drop_log[index] if reason != "IGNORED")
        for index in range(proposed)
    ]
    accounting = {
        "proposed_events": proposed,
        "retained_epochs": retained,
        "dropped_epochs": dropped,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "accounting_complete": True,
    }
    return accounting, lineage


def _compact_qc(
    raw: mne.io.BaseRaw,
    analysis: AnalysisConfig,
    *,
    already_filtered: bool = False,
) -> dict:
    """Return only the metrics needed to inspect preprocessing changes."""
    metrics, _, _, _ = _channel_metrics(
        raw,
        analysis.filter_hz,
        analysis.qc,
        already_filtered=already_filtered,
    )
    return {
        "eeg_channels": len(metrics),
        "candidate_bad_channels": sorted(
            metrics.loc[metrics["candidate_bad"], "channel"].tolist()
        ),
        "median_filtered_std_uv": float(np.median(metrics["filtered_std_uv"])),
        "median_filtered_robust_range_uv": float(
            np.median(metrics["filtered_robust_range_uv"])
        ),
        "maximum_flat_fraction": float(np.max(metrics["flat_fraction"])),
    }


def _plot_c3_c4(
    epochs: mne.Epochs,
    output: Path,
    title: str,
    baseline: tuple[float, float] | None,
) -> None:
    available = [channel for channel in ["C3", "C4"] if channel in epochs.ch_names]
    if not available:
        return
    prepared = epochs.copy().filter(None, 12.0, picks="eeg", verbose="ERROR")
    if baseline is not None:
        prepared.apply_baseline(baseline, verbose="ERROR")

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
    baseline_label = "baseline corrected" if baseline is not None else "no baseline"
    ax.set(
        title=title,
        xlabel="Time (s)",
        ylabel=f"Amplitude (µV), low-pass 12 Hz, {baseline_label}",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _preprocess_recording_into(
    vhdr: Path,
    output: Path,
    participant_id: str,
    bad_channels: list[str] | None = None,
    bad_channel_decisions: list[dict] | None = None,
    export_eeglab: bool = True,
) -> dict:
    """Filter, rereference, interpolate persistent bad EEG channels, and epoch."""
    vhdr = Path(vhdr).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if (
        not participant_id
        or len(participant_id) > 64
        or not participant_id.isascii()
        or not participant_id.isalnum()
    ):
        raise ValueError("Participant ID must be 1-64 ASCII letters or digits")
    if bad_channels and bad_channel_decisions:
        raise ValueError(
            "Pass either bad_channels or bad_channel_decisions, not both"
        )
    if bad_channel_decisions is not None:
        bad_channels = sorted(
            {
                str(decision["channel"])
                for decision in bad_channel_decisions
                if decision.get("decision") == "interpolate"
            }
        )
    else:
        bad_channels = bad_channels or []
        bad_channel_decisions = [
            {
                "channel": channel,
                "decision": "interpolate",
                "reason": "",
                "reviewer": "",
                "reviewed_at": "",
                "evidence_windows": "",
                "source": "direct_argument",
            }
            for channel in bad_channels
        ]
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    analysis = load_analysis_config()
    codebook = load_event_codebook()
    provenance_core = build_provenance_core(
        vhdr=vhdr,
        participant_id=participant_id,
        analysis=analysis,
        bad_channel_decisions=bad_channel_decisions,
    )

    raw = _read_brainvision_compat(vhdr)
    type_updates = {
        name: kind
        for name, kind in analysis.channel_types.items()
        if name in raw.ch_names
    }
    raw.set_channel_types(type_updates, verbose="ERROR")
    raw.set_montage(
        mne.channels.make_standard_montage(analysis.montage),
        match_case=False,
        on_missing="warn",
        verbose="ERROR",
    )
    markers = annotations_to_markers(raw.annotations)
    trials = classify_trials(markers, codebook=codebook)
    reconciliation = reconcile_trials(markers, trials, codebook=codebook)
    if "ECG" in raw.ch_names:
        raw.drop_channels(["ECG"])
    decision_channels = {
        str(decision.get("channel", ""))
        for decision in bad_channel_decisions
        if decision.get("channel")
    }
    unknown_bads = sorted(decision_channels - set(raw.ch_names))
    if unknown_bads:
        raise ValueError(f"Bad-channel labels not present in recording: {unknown_bads}")
    non_eeg_bads = [
        name for name in bad_channels if raw.get_channel_types(picks=[name])[0] != "eeg"
    ]
    if non_eeg_bads:
        raise ValueError(f"Only scalp EEG channels may be interpolated: {non_eeg_bads}")

    raw.load_data(verbose="ERROR")
    signal_picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, exclude=[])
    low_hz, high_hz = analysis.filter_hz
    nyquist = float(raw.info["sfreq"]) / 2
    if high_hz >= nyquist:
        raise ValueError(
            f"Filter high cutoff {high_hz:g} Hz must be below Nyquist "
            f"({nyquist:g} Hz)"
        )
    raw.filter(
        low_hz,
        high_hz,
        picks=signal_picks,
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    # The comparison starts after the same FIR filter used for both states.
    before_qc = _compact_qc(raw, analysis, already_filtered=True)

    raw.info["bads"] = list(bad_channels)
    # Marked bads are excluded from the reference estimate.
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if bad_channels:
        raw.interpolate_bads(reset_bads=True, method={"eeg": "spline"}, verbose="ERROR")
        # Restore a true common-average reference after interpolation.
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    after_qc = _compact_qc(raw, analysis, already_filtered=True)

    go_classes = list(codebook.outcome_classes("go").values())
    inferred_stop_class, _ = codebook.inferred_outcome("stop")
    stop_classes = [
        inferred_stop_class,
        *codebook.outcome_classes("stop").values(),
    ]
    go_events, go_event_id, go_metadata = _event_array(
        raw,
        trials,
        "stimulus_onset_s",
        go_classes,
    )
    stop_events, stop_event_id, stop_metadata = _event_array(
        raw,
        trials,
        "stop_signal_onset_s",
        stop_classes,
    )
    if len(go_events) == 0 or len(stop_events) == 0:
        missing = [
            label
            for label, events in (("go", go_events), ("stop", stop_events))
            if len(events) == 0
        ]
        raise RuntimeError(
            "No eligible reconstructed events for: " + ", ".join(missing)
        )

    filter_label = f"{low_hz:g}-{high_hz:g}Hz"
    continuous_fif = (
        output / f"sub-{participant_id}_continuous_{filter_label}_avgref_raw.fif"
    )
    raw.save(continuous_fif, overwrite=True, fmt="single", verbose="ERROR")
    if export_eeglab:
        raw.export(
            output / f"sub-{participant_id}_continuous_{filter_label}_avgref.set",
            fmt="eeglab",
            overwrite=True,
            verbose="ERROR",
        )

    # Epoch exports contain one reconstructed condition event at time zero.
    # Acquisition markers are removed, but BAD and descriptive annotations are
    # kept so reject_by_annotation remains effective.
    epoch_source = raw.copy()
    epoch_source.set_annotations(_non_task_annotations(raw.annotations))
    go_window = analysis.epochs_seconds["go"]
    stop_window = analysis.epochs_seconds["stop"]
    go_epochs = mne.Epochs(
        epoch_source,
        go_events,
        event_id=go_event_id,
        tmin=go_window[0],
        tmax=go_window[1],
        baseline=None,
        metadata=go_metadata,
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
        verbose="ERROR",
    )
    stop_epochs = mne.Epochs(
        epoch_source,
        stop_events,
        event_id=stop_event_id,
        tmin=stop_window[0],
        tmax=stop_window[1],
        baseline=None,
        metadata=stop_metadata,
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
        verbose="ERROR",
    )
    go_epoch_accounting, go_lineage = _epoch_accounting(
        go_events, go_epochs, go_metadata
    )
    stop_epoch_accounting, stop_lineage = _epoch_accounting(
        stop_events, stop_epochs, stop_metadata
    )
    go_lineage.to_csv(output / "go_epoch_lineage.csv", index=False)
    stop_lineage.to_csv(output / "stop_epoch_lineage.csv", index=False)
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
        (max(float(go_epochs.tmin), -1.0), 0.0),
    )
    _plot_c3_c4(
        stop_epochs,
        figures / "c3_c4_stop_lowpass12.png",
        "C3/C4: stop-signal-locked responses",
        None,
    )

    summary = {
        "participant_id": participant_id,
        "source_header": "withheld-private-source",
        "filter_hz": list(analysis.filter_hz),
        "reference": analysis.reference,
        "ecg_removed": "ECG" in type_updates,
        "eog_retained_and_excluded_from_reference": "EOG" in type_updates,
        "interpolated_bad_channels": bad_channels,
        "bad_channel_decision_record_complete": provenance_core[
            "decision_record_complete"
        ],
        "continuous_channels": len(raw.ch_names),
        "go_epochs": {condition: len(go_epochs[condition]) for condition in go_event_id},
        "stop_epochs": {
            condition: len(stop_epochs[condition]) for condition in stop_event_id
        },
        "go_epoch_accounting": go_epoch_accounting,
        "stop_epoch_accounting": stop_epoch_accounting,
        "trial_reconciliation": reconciliation,
        "before_after_qc": {
            "before": before_qc,
            "after": after_qc,
            "scope": (
                "Targeted screening metrics only; not a global quality score "
                "or proof of artifact removal"
            ),
        },
        "ica_status": "not fitted yet; component decisions must be explicit",
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trials.to_csv(output / "reconstructed_trials.csv", index=False)
    verify_input_sources(vhdr, provenance_core["inputs"])
    write_provenance(output, provenance_core)
    return summary


def preprocess_recording(
    vhdr: Path,
    output: Path,
    participant_id: str,
    bad_channels: list[str] | None = None,
    bad_channel_decisions: list[dict] | None = None,
    export_eeglab: bool = True,
) -> dict:
    """Create one recording output atomically in a new directory."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "Auditable preprocessing requires a new output path; reuse is disabled"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    temporary.rmdir()
    try:
        summary = _preprocess_recording_into(
            vhdr=vhdr,
            output=temporary,
            participant_id=participant_id,
            bad_channels=bad_channels,
            bad_channel_decisions=bad_channel_decisions,
            export_eeglab=export_eeglab,
        )
        temporary.replace(output)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
