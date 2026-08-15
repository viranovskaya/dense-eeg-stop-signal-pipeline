"""Controlled channel and temporal QC for an inventoried BIDS/EEGLAB recording."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .bids_eeglab import (
    BIDSEeglabInputs,
    _read_raw,
    input_identities,
    resolve_inputs,
)
from .config import QCConfig
from .provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)
from .qc import (
    _compute_psd,
    _full_recording_window_metrics,
    _save_candidate_window_traces,
    _save_full_recording_scan_plot,
    _summarize_full_recording_windows,
)


@dataclass(frozen=True)
class BIDSEeglabQCControls:
    """Validated controls for one conservative QC screen."""

    filter_hz: tuple[float, float]
    qc: QCConfig


def _captured_json(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8-sig")), hashlib.sha256(data).hexdigest()


def _controls(profile: dict) -> BIDSEeglabQCControls:
    payload = profile.get("qc", {})
    required = {
        "screen_filter_hz",
        "full_recording_window_seconds",
        "robust_z_threshold",
        "flat_fraction_threshold",
        "line_noise_ratio_db_threshold",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"QC profile is missing fields: {missing}")
    filter_hz = tuple(float(value) for value in payload["screen_filter_hz"])
    if (
        len(filter_hz) != 2
        or not all(math.isfinite(value) for value in filter_hz)
        or not 0 <= filter_hz[0] < filter_hz[1]
    ):
        raise ValueError("screen_filter_hz must contain increasing finite cutoffs")
    qc = QCConfig(
        representative_window_count=1,
        representative_window_seconds=float(payload["full_recording_window_seconds"]),
        full_recording_window_seconds=float(payload["full_recording_window_seconds"]),
        robust_z_threshold=float(payload["robust_z_threshold"]),
        flat_fraction_threshold=float(payload["flat_fraction_threshold"]),
        line_noise_ratio_db_threshold=float(payload["line_noise_ratio_db_threshold"]),
    )
    numeric = (
        qc.full_recording_window_seconds,
        qc.robust_z_threshold,
        qc.flat_fraction_threshold,
        qc.line_noise_ratio_db_threshold,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("QC thresholds must be finite")
    if qc.full_recording_window_seconds <= 0 or qc.robust_z_threshold <= 0:
        raise ValueError("QC window and robust-z threshold must be positive")
    if not 0 <= qc.flat_fraction_threshold <= 1:
        raise ValueError("flat_fraction_threshold must be between zero and one")
    return BIDSEeglabQCControls(filter_hz=filter_hz, qc=qc)


def _verified_inventory(path: Path) -> tuple[dict, dict, dict[str, str]]:
    root = Path(path).expanduser().absolute()
    provenance = verify_provenance(root)
    audit_path = root / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if canonical_sha256(audit) != provenance.get("audit_sha256"):
        raise ValueError("Inventory audit hash does not match its provenance")
    if audit.get("status") != "pass" or audit.get("publication_allowed") is not False:
        raise ValueError("Inventory does not license controlled QC")
    identity = {
        "audit_sha256": sha256_file(audit_path),
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
    }
    return audit, provenance, identity


def _line_noise_table(
    raw: mne.io.BaseRaw,
    line_frequency_hz: float,
    threshold_db: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frequencies, psd_db_uv, ratios = _compute_psd(raw, line_frequency_hz)
    names = [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True)]
    table = pd.DataFrame(
        {
            "channel": names,
            "line_noise_ratio_db": ratios,
            "review_high_line_noise": ratios > threshold_db,
        }
    )
    return table, frequencies, psd_db_uv


def _segment_review_template(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "channel",
        "prompt_onset_s",
        "prompt_duration_s",
        "candidate_reason",
        "decision",
        "refined_onset_s",
        "refined_duration_s",
        "scope",
        "reviewer",
        "reviewed_at",
        "evidence",
        "notes",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(
        {
            "channel": candidates["channel"],
            "prompt_onset_s": candidates["start_s"],
            "prompt_duration_s": candidates["duration_s"],
            "candidate_reason": candidates["flag_reason"],
            "decision": "pending",
            "refined_onset_s": "",
            "refined_duration_s": "",
            "scope": "",
            "reviewer": "",
            "reviewed_at": "",
            "evidence": "",
            "notes": "",
        }
    )
    table = table.sort_values(["prompt_onset_s", "channel"], kind="stable").reset_index(
        drop=True
    )
    table.insert(
        0,
        "candidate_id",
        [f"segment-{index:04d}" for index in range(1, len(table) + 1)],
    )
    return table


def _channel_review_template(
    temporal_summary: pd.DataFrame,
    line_table: pd.DataFrame,
) -> pd.DataFrame:
    temporal = temporal_summary.set_index("channel")
    line = line_table.set_index("channel")
    channels = sorted(
        set(temporal.index[temporal["channel_requires_review"]])
        | set(line.index[line["review_high_line_noise"]])
    )
    rows = []
    for channel in channels:
        reasons = []
        if bool(temporal.loc[channel, "channel_requires_review"]):
            reasons.append("temporal_amplitude_or_flatness")
        if bool(line.loc[channel, "review_high_line_noise"]):
            reasons.append("line_noise_concentration")
        rows.append(
            {
                "channel": channel,
                "candidate_reason": ";".join(reasons),
                "decision": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "evidence": "",
                "notes": "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "channel",
            "candidate_reason",
            "decision",
            "reviewer",
            "reviewed_at",
            "evidence",
            "notes",
        ],
    )


def _runtime_identity() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "dependency_files": {
            name: sha256_file(Path(__file__).resolve().parents[2] / name)
            for name in ("requirements.txt", "constraints-ci.txt")
        },
    }


def _verify_publication_context(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    inventory_path: Path,
    output: Path,
) -> None:
    provenance = verify_provenance(output)
    if input_identities(resolve_inputs(inputs)) != provenance["inputs"]:
        raise ValueError("Dataset inputs changed before QC publication")
    if sha256_file(profile_path) != provenance["controls"]["profile_sha256"]:
        raise ValueError("QC profile changed before publication")
    if source_manifest() != provenance["controls"]["source_manifest"]:
        raise ValueError("Executable source changed before QC publication")
    if _runtime_identity() != provenance["runtime"]:
        raise ValueError("QC runtime context changed before publication")
    _, _, identity = _verified_inventory(inventory_path)
    if identity != provenance["inventory"]:
        raise ValueError("Inventory package changed before QC publication")


def _save_psd_plot(
    output: Path,
    frequencies: np.ndarray,
    psd_db_uv: np.ndarray,
    line_frequency_hz: float,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    median = np.nanmedian(psd_db_uv, axis=0)
    lower, upper = np.nanpercentile(psd_db_uv, [10, 90], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(frequencies, lower, upper, color="#8bb6d9", alpha=0.35)
    ax.plot(frequencies, median, color="#244f73", linewidth=1.2)
    ax.axvline(line_frequency_hz, color="#b33a3a", linestyle="--", linewidth=1)
    ax.set(
        title="Raw scalp-channel power spectrum",
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (dB µV²/Hz)",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "raw_psd.png", dpi=160)
    plt.close(fig)


def _save_line_noise_review_plot(
    output: Path,
    line_table: pd.DataFrame,
    line_frequency_hz: float,
    threshold_db: float,
) -> None:
    """Plot ranked line-noise concentration without assigning channel status."""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ranked = line_table.sort_values(
        ["line_noise_ratio_db", "channel"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    flagged = ranked["review_high_line_noise"].to_numpy(dtype=bool)
    colors = np.where(flagged, "#b84a4a", "#8ba6b8")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(ranked))
    ax.bar(positions, ranked["line_noise_ratio_db"], color=colors, width=0.86)
    ax.axhline(threshold_db, color="#222222", linestyle="--", linewidth=1)
    ax.set(
        title=f"Raw {line_frequency_hz:g} Hz concentration by channel",
        xlabel="Channels ranked by concentration",
        ylabel="Line-noise concentration (dB)",
        xlim=(-1, len(ranked)),
    )
    ax.set_xticks(positions[flagged])
    ax.set_xticklabels(ranked.loc[flagged, "channel"], rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.99,
        0.98,
        "Red bars request spectral review; they are not automatic bad channels.",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figures / "line_noise_review.png", dpi=160)
    plt.close(fig)


def _run_qc(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    inventory_path: Path,
    output: Path,
) -> dict:
    profile, profile_sha = _captured_json(profile_path)
    controls = _controls(profile)
    initial_manifest = source_manifest()
    inventory_audit, inventory_provenance, inventory_identity = _verified_inventory(
        inventory_path
    )
    if profile_sha != inventory_provenance.get("controls", {}).get("profile_sha256"):
        raise ValueError("QC profile does not match the inventoried profile")
    runtime_identity = _runtime_identity()
    if (
        inventory_provenance.get("controls", {}).get("source_manifest")
        != initial_manifest
    ):
        raise ValueError("Inventory source context is stale for the QC implementation")
    if inventory_provenance.get("runtime") != runtime_identity:
        raise ValueError("Inventory runtime context is incompatible with QC")
    resolved = resolve_inputs(inputs)
    initial_inputs = input_identities(resolved)
    if initial_inputs != inventory_provenance.get("inputs"):
        raise ValueError("Inventory and QC inputs do not match")

    raw = _read_raw(resolved.raw_set)
    reference = str(profile["reference_channel"]["name"])
    if reference not in raw.ch_names:
        raise ValueError("QC reference channel is absent")
    raw.set_channel_types({reference: "misc"}, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) != int(profile["expected"]["eeg_channel_count"]) - 1:
        raise ValueError("Unexpected number of scalp channels after QC exclusion")
    if controls.filter_hz[1] >= float(raw.info["sfreq"]) / 2:
        raise ValueError("QC filter high cutoff must be below Nyquist")

    filtered = raw.copy()
    filtered.filter(
        *controls.filter_hz,
        picks=picks,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        verbose="ERROR",
    )
    window_metrics = _full_recording_window_metrics(raw, filtered, controls.qc)
    channel_summary = _summarize_full_recording_windows(window_metrics)
    candidates = window_metrics.loc[
        window_metrics["window_flag"] | window_metrics["flat_flag"]
    ].copy()
    line_frequency = float(profile["expected"]["power_line_frequency_hz"])
    if line_frequency >= float(raw.info["sfreq"]) / 2:
        raise ValueError("Line frequency must be below Nyquist for QC")
    line_table, frequencies, psd_db_uv = _line_noise_table(
        raw,
        line_frequency,
        controls.qc.line_noise_ratio_db_threshold,
    )
    review_template = _segment_review_template(candidates)
    channel_review = _channel_review_template(channel_summary, line_table)
    final_window_seconds = float(
        window_metrics.loc[
            window_metrics["window_index"] == window_metrics["window_index"].max(),
            "duration_s",
        ].iloc[0]
    )

    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    window_metrics.to_csv(output / "channel_window_metrics.csv", index=False)
    channel_summary.to_csv(output / "channel_window_summary.csv", index=False)
    candidates.to_csv(output / "candidate_window_review.csv", index=False)
    line_table.to_csv(output / "line_noise_metrics.csv", index=False)
    review_template.to_csv(
        output / "segment_review_template.tsv", sep="\t", index=False
    )
    channel_review.to_csv(output / "channel_review_template.tsv", sep="\t", index=False)
    _save_full_recording_scan_plot(output, window_metrics)
    _save_candidate_window_traces(output, raw, filtered, window_metrics)
    _save_psd_plot(output, frequencies, psd_db_uv, line_frequency)
    _save_line_noise_review_plot(
        output,
        line_table,
        line_frequency,
        controls.qc.line_noise_ratio_db_threshold,
    )

    summary = {
        "schema_version": "1",
        "status": "review_required" if not channel_review.empty else "screen_complete",
        "scope": "controlled channel and temporal QC screening only",
        "dataset": inventory_audit["dataset"],
        "recording": inventory_audit["recording"],
        "task": inventory_audit["task"],
        "scalp_channels_screened": len(picks),
        "reference_channel_excluded_from_scoring": reference,
        "recording_duration_s": float(raw.n_times / raw.info["sfreq"]),
        "windows_scanned": int(window_metrics["window_index"].nunique()),
        "window_seconds": controls.qc.full_recording_window_seconds,
        "final_window_seconds": final_window_seconds,
        "channel_window_pairs": len(window_metrics),
        "flagged_channel_window_pairs": len(candidates),
        "channel_review_rows": len(channel_review),
        "segment_review_rows": len(review_template),
        "channels_requiring_temporal_review": channel_summary.loc[
            channel_summary["channel_requires_review"], "channel"
        ].tolist(),
        "channels_requiring_line_noise_review": line_table.loc[
            line_table["review_high_line_noise"], "channel"
        ].tolist(),
        "automatic_bad_channel_decisions": 0,
        "manual_review_status": "pending"
        if not channel_review.empty
        else "not_required",
        "screen_filter": {
            "type": "fourth-order zero-phase Butterworth",
            "frequency_hz": list(controls.filter_hz),
            "purpose": "relative temporal screening only",
        },
        "line_noise_screen": {
            "frequency_hz": line_frequency,
            "source": "raw PSD",
            "threshold_db": controls.qc.line_noise_ratio_db_threshold,
        },
        "limitations": [
            "Within-window spatial robust scores may miss common-mode artifacts.",
            f"Non-overlapping {controls.qc.full_recording_window_seconds:g}-second windows may dilute short transients.",
            "Moderate attenuation or bridging may remain below the review threshold.",
            "Montage-derived neighbours are visual aids; the coordinate frame is not independently validated.",
            "Candidate flags request review and do not establish bad channels.",
        ],
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    (output / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "qc_report.md").write_text(
        "\n".join(
            [
                "# BIDS/EEGLAB QC screen",
                "",
                "> Controlled derived output. Do not publish participant-level tables or figures.",
                "",
                f"- Scalp channels screened: {len(picks)}",
                f"- Windows: {summary['windows_scanned']} total; nominal {summary['window_seconds']:g} s; final {summary['final_window_seconds']:g} s",
                f"- Channel-window pairs: {len(window_metrics)}",
                f"- Flagged pairs awaiting review: {len(candidates)}",
                f"- Channels awaiting temporal review: {len(summary['channels_requiring_temporal_review'])}",
                f"- Channels awaiting {line_frequency:g} Hz review: {len(summary['channels_requiring_line_noise_review'])}",
                "- Automatic bad-channel decisions: 0",
                "",
                "The screen is a prompt for trace, montage-derived neighbour, persistence, and spectrum review. It does not authorize interpolation, rereferencing, ICA, epoch analysis, or publication.",
                "",
                "## What to review",
                "",
                "1. Open `channel_review_template.tsv`. For every row, record `keep` or `interpolate` and cite the evidence inspected (for example, trace, spectrum, persistence, and neighbours).",
                "2. Open `segment_review_template.tsv`. A window prompt is not an artifact boundary. Record `keep`, or enter a refined onset, duration, scope (`ica`, `epochs`, or `both`), evidence, and rationale. An exclusion becomes a global time mask across all channels; `channel` records where the evidence was first flagged.",
                "3. Treat line-noise concentration as a separate review cue. It does not by itself make a channel bad.",
                "4. Keep both ledgers controlled until every pending row has a reviewer and date.",
                "",
                "## Evidence files",
                "",
                "- `channel_window_metrics.csv`: all channel-window measurements",
                "- `channel_window_summary.csv`: per-channel persistence and maxima",
                "- `candidate_window_review.csv`: temporal prompts only",
                "- `line_noise_metrics.csv`: raw 60-Hz concentration by channel",
                "- `figures/full_recording_window_scan.png`: complete temporal heatmap",
                "- `figures/raw_psd.png`: raw scalp-channel PSD summary",
                "- `figures/line_noise_review.png`: ranked raw line-noise concentration; red bars are review prompts, not bad-channel decisions",
                "- `figures/window_reviews/`: controlled trace excerpts for review",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    final_inputs = input_identities(resolve_inputs(inputs))
    if final_inputs != initial_inputs:
        raise ValueError("Dataset inputs changed during QC")
    if (
        sha256_file(profile_path) != profile_sha
        or source_manifest() != initial_manifest
    ):
        raise ValueError("QC controls changed during QC")
    final_audit, final_inventory, final_identity = _verified_inventory(inventory_path)
    if (
        final_audit != inventory_audit
        or final_inventory != inventory_provenance
        or final_identity != inventory_identity
    ):
        raise ValueError("Inventory package changed during QC")
    core = {
        "schema_version": "1",
        "inputs": initial_inputs,
        "controls": {
            "profile_sha256": profile_sha,
            "source_manifest": initial_manifest,
            "qc_config_sha256": canonical_sha256(profile["qc"]),
        },
        "inventory": inventory_identity,
        "runtime": runtime_identity,
        "summary_sha256": canonical_sha256(summary),
        "classification": "controlled_derived",
        "publication_allowed": False,
    }
    core["core_sha256"] = canonical_sha256(core)
    write_provenance(output, core)
    verify_provenance(output)
    return summary


def publish_qc(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    inventory_path: Path,
    output: Path,
) -> Path:
    """Publish one exact-set-verified QC package atomically to a new directory."""
    output = Path(output).expanduser().absolute()
    dataset_root = (
        Path(inputs.dataset_root).expanduser().absolute().resolve(strict=True)
    )
    nearest = output.parent
    while not nearest.exists():
        nearest = nearest.parent
    if nearest.resolve(strict=True).is_relative_to(dataset_root):
        raise ValueError("Output must be outside the source dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    temporary.rmdir()
    try:
        _run_qc(inputs, profile_path, inventory_path, temporary)
        _verify_publication_context(
            inputs,
            profile_path,
            inventory_path,
            temporary,
        )
        temporary.replace(output)
        try:
            verify_provenance(output)
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
