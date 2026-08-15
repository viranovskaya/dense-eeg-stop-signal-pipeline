"""Build a complete controlled evidence pack for human BIDS/EEGLAB QC review."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .bids_eeglab import BIDSEeglabInputs, _read_raw, input_identities, resolve_inputs
from .bids_eeglab_qc import _controls, _runtime_identity
from .bids_eeglab_review import _nearest_existing_ancestor
from .provenance import (
    canonical_sha256,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)
from .qc import _compute_psd, _save_window_trace_figures

SEGMENT_CONTEXT_SECONDS = 2.0


def _qc_context(root: Path) -> tuple[dict, dict, dict]:
    root = Path(root).expanduser().absolute()
    provenance = verify_provenance(root)
    summary = json.loads((root / "qc_summary.json").read_text(encoding="utf-8"))
    if canonical_sha256(summary) != provenance.get("summary_sha256"):
        raise ValueError("QC summary does not match its provenance")
    if (
        summary.get("classification") != "controlled_derived"
        or summary.get("publication_allowed") is not False
    ):
        raise ValueError("QC package is not controlled review evidence")
    identity = {
        "provenance_sha256": sha256_file(root / "provenance.json"),
        "core_sha256": str(provenance["core_sha256"]),
        "summary_sha256": sha256_file(root / "qc_summary.json"),
    }
    return summary, provenance, identity


def _verified_output_bytes(root: Path, provenance: dict, relative: str) -> bytes:
    records = {record["path"]: record for record in provenance.get("outputs", [])}
    if relative not in records:
        raise ValueError(f"QC provenance does not declare {relative}")
    data = (root / relative).read_bytes()
    record = records[relative]
    if (
        len(data) != record["size_bytes"]
        or hashlib.sha256(data).hexdigest() != record["sha256"]
    ):
        raise ValueError(f"QC artifact changed before review-pack capture: {relative}")
    return data


def _captured_table(
    root: Path, provenance: dict, relative: str, *, sep: str = ","
) -> pd.DataFrame:
    return pd.read_csv(
        BytesIO(_verified_output_bytes(root, provenance, relative)),
        sep=sep,
        dtype=str,
        keep_default_na=False,
    )


def _candidate_key(
    channel: str, onset: str | float, duration: str | float, reason: str
) -> tuple[str, float, float, str]:
    return (
        str(channel),
        round(float(onset), 9),
        round(float(duration), 9),
        str(reason),
    )


def _bind_segment_candidates(
    candidates: pd.DataFrame, template: pd.DataFrame
) -> pd.DataFrame:
    candidate_columns = {
        "channel",
        "start_s",
        "duration_s",
        "flag_reason",
        "start_sample",
        "stop_sample_exclusive",
        "stop_s",
        "window_index",
    }
    if not candidate_columns.issubset(candidates.columns):
        raise ValueError("QC candidate table is missing trace fields")
    template_columns = {
        "candidate_id",
        "channel",
        "prompt_onset_s",
        "prompt_duration_s",
        "candidate_reason",
        "decision",
    }
    if not template_columns.issubset(template.columns):
        raise ValueError("QC segment template is missing binding fields")
    if template["candidate_id"].duplicated().any():
        raise ValueError("QC segment candidate IDs must be unique")
    if set(template["decision"]) not in (set(), {"pending"}):
        raise ValueError("Review pack requires the original pending segment template")

    candidate_rows: dict[tuple[str, float, float, str], dict] = {}
    for row in candidates.to_dict(orient="records"):
        key = _candidate_key(
            row["channel"], row["start_s"], row["duration_s"], row["flag_reason"]
        )
        if key in candidate_rows:
            raise ValueError("QC candidate prompts are not uniquely identifiable")
        candidate_rows[key] = row

    bound = []
    for row in template.to_dict(orient="records"):
        key = _candidate_key(
            row["channel"],
            row["prompt_onset_s"],
            row["prompt_duration_s"],
            row["candidate_reason"],
        )
        if key not in candidate_rows:
            raise ValueError(f"No QC trace prompt matches {row['candidate_id']}")
        bound.append({**candidate_rows.pop(key), "candidate_id": row["candidate_id"]})
    if candidate_rows:
        raise ValueError("QC candidate table contains prompts absent from the template")
    table = pd.DataFrame(bound)
    for column in (
        "window_index",
        "start_sample",
        "stop_sample_exclusive",
        "start_s",
        "stop_s",
        "duration_s",
    ):
        table[column] = pd.to_numeric(table[column], errors="raise")
    return table


def _render_index(
    channel_rows: pd.DataFrame,
    segment_rows: pd.DataFrame,
    summary: dict,
) -> str:
    channel_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row.channel)}</td>"
        f"<td>{html.escape(row.candidate_reason)}</td>"
        f'<td><a href="{html.escape(row.figure_path)}">evidence panel</a></td>'
        "<td>pending</td>"
        "</tr>"
        for row in channel_rows.itertuples(index=False)
    )
    cards = []
    for row in segment_rows.itertuples(index=False):
        cards.append(
            '<article class="card">'
            f"<h3>{html.escape(row.candidate_id)} · {html.escape(row.channel)}</h3>"
            f"<p>{float(row.prompt_onset_s):.3f}–"
            f"{float(row.prompt_onset_s) + float(row.prompt_duration_s):.3f} s · "
            f"{html.escape(row.candidate_reason)}</p>"
            f'<a href="{html.escape(row.figure_path)}">'
            f'<img src="{html.escape(row.figure_path)}" '
            f'alt="Trace evidence for {html.escape(row.candidate_id)}"></a>'
            "</article>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dense EEG controlled review pack</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;margin:0;background:#f4f7f9;color:#17222b}}
main{{max-width:1180px;margin:auto;padding:32px}} h1,h2{{color:#153f5b}}
.warning{{padding:16px;background:#fff2cc;border-left:5px solid #d49200}}
.summary{{display:flex;gap:12px;flex-wrap:wrap}}
.summary b{{background:white;padding:10px 14px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:8px;border-bottom:1px solid #dce4e9;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}}
.card{{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #0002}}
.card img{{width:100%;height:auto}} code{{background:#e8eef2;padding:2px 5px;border-radius:4px}}
</style>
</head>
<body><main>
<h1>Dense EEG controlled human-review pack</h1>
<p class="warning"><strong>Review evidence only.</strong> Every prompt is pending. This package makes no bad-channel or exclusion decision and is not approved for publication.</p>
<div class="summary"><b>{len(channel_rows)} channel prompts</b><b>{len(segment_rows)} segment prompts</b><b>0 automatic decisions</b></div>
<h2>How to use it</h2>
<ol><li>Review every channel prompt using the full-recording scan, ranked line-noise plot, raw PSD and temporal panels.</li>
<li>Make working copies of both QC decision templates outside the immutable QC and review-pack directories. Edit only those copies.</li>
<li>For each segment prompt, use its <code>candidate_id</code> to update the matching working-copy row.</li>
<li>Red dashed lines mark the prompt, with {SEGMENT_CONTEXT_SECONDS:g} s of context where the recording permits it. Confirm every refined onset and duration in a zoomable raw-data viewer.</li>
<li>Inspect the full trace before any final channel decision. A prompt is not an artifact boundary.</li></ol>
<p>Bound QC summary SHA-256: <code>{html.escape(summary['qc_summary_sha256'])}</code></p>
<h2>Global evidence</h2>
<p><a href="figures/full_recording_window_scan.png">Full-recording scan</a> · <a href="figures/line_noise_review.png">Ranked line-noise review</a> · <a href="figures/raw_psd.png">Raw PSD</a></p>
<h2>Channel prompts</h2>
<table><thead><tr><th>Channel</th><th>Reason</th><th>Evidence</th><th>Status</th></tr></thead><tbody>{channel_html}</tbody></table>
<h2>Segment prompts</h2><div class="grid">{''.join(cards)}</div>
</main></body></html>"""


def _minmax_envelope(
    data: np.ndarray, sfreq: float, maximum_bins: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a full trace without dropping the extrema in any time bin."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Trace envelope requires one finite data vector")
    bin_size = max(1, int(np.ceil(len(values) / maximum_bins)))
    indices = []
    envelope = []
    for start in range(0, len(values), bin_size):
        segment = values[start : start + bin_size]
        local = [int(np.argmin(segment)), int(np.argmax(segment))]
        for offset in sorted(set(local)):
            indices.append(start + offset)
            envelope.append(float(segment[offset]))
    return np.asarray(indices, dtype=float) / sfreq, np.asarray(envelope)


def _save_channel_review_figures(
    review_dir: Path,
    raw,
    filtered,
    channel_template: pd.DataFrame,
    line_frequency_hz: float,
) -> pd.DataFrame:
    """Save full-trace and spectral evidence for every channel prompt."""
    columns = [
        "channel",
        "candidate_reason",
        "figure_path",
        "decision_status",
    ]
    if channel_template.empty:
        return pd.DataFrame(columns=columns)
    review_dir.mkdir(parents=True, exist_ok=True)
    frequencies, psd_db_uv, ratios = _compute_psd(raw, line_frequency_hz)
    eeg_indices = mne.pick_types(raw.info, eeg=True, exclude=[])
    eeg_names = [raw.ch_names[index] for index in eeg_indices]
    name_to_index = {name: index for index, name in enumerate(eeg_names)}
    psd_median = np.nanmedian(psd_db_uv, axis=0)
    sfreq = float(raw.info["sfreq"])
    rows = []
    if channel_template["channel"].duplicated().any():
        raise ValueError("Channel review prompts must be unique")
    for ordinal, row in enumerate(channel_template.itertuples(index=False), start=1):
        if row.channel not in name_to_index:
            raise ValueError(f"Channel review prompt is absent from EEG data: {row.channel}")
        index = name_to_index[row.channel]
        raw_trace = raw.get_data(picks=[row.channel])[0] * 1e6
        filtered_trace = filtered.get_data(picks=[row.channel])[0] * 1e6
        raw_trace -= np.median(raw_trace)
        filtered_trace -= np.median(filtered_trace)
        raw_time, raw_envelope = _minmax_envelope(raw_trace, sfreq)
        filtered_time, filtered_envelope = _minmax_envelope(filtered_trace, sfreq)
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
        axes[0].plot(raw_time, raw_envelope, color="#244f73", linewidth=0.55)
        axes[0].set(
            title="Full raw trace min/max envelope, channel median removed",
            ylabel="µV",
        )
        axes[1].plot(
            filtered_time, filtered_envelope, color="#356f91", linewidth=0.55
        )
        axes[1].set(
            title=(
                "Full screening-filter trace min/max envelope, "
                "channel median removed"
            ),
            xlabel="Recording time (s)",
            ylabel="µV",
        )
        axes[2].plot(frequencies, psd_median, color="#a7b8c4", label="scalp median")
        axes[2].plot(
            frequencies,
            psd_db_uv[index],
            color="#244f73",
            linewidth=1.0,
            label=row.channel,
        )
        axes[2].axvline(line_frequency_hz, color="#b33a3a", linestyle="--")
        axes[2].set(
            title=f"Raw PSD · line-noise concentration {ratios[index]:.2f} dB",
            xlabel="Frequency (Hz)",
            ylabel="dB µV²/Hz",
        )
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.18)
        fig.suptitle(f"Channel review prompt · {row.channel} · {row.candidate_reason}")
        safe_channel = "".join(
            character if character.isalnum() or character in "_-" else "_"
            for character in str(row.channel)
        )
        filename = f"channel-{ordinal:04d}_{safe_channel}.png"
        relative = f"figures/channels/{filename}"
        fig.savefig(review_dir / filename, dpi=150)
        plt.close(fig)
        rows.append(
            {
                "channel": row.channel,
                "candidate_reason": row.candidate_reason,
                "figure_path": relative,
                "decision_status": "pending",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _verify_context(
    *,
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    profile_sha256: str,
    qc_path: Path,
    qc_identity: dict,
    initial_inputs: list[dict],
    initial_source: dict,
    initial_runtime: dict,
    output: Path | None = None,
) -> None:
    if input_identities(resolve_inputs(inputs)) != initial_inputs:
        raise ValueError("Dataset inputs changed during review-pack generation")
    if sha256_file(profile_path) != profile_sha256:
        raise ValueError("QC profile changed during review-pack generation")
    if source_manifest() != initial_source or _runtime_identity() != initial_runtime:
        raise ValueError("Source or runtime changed during review-pack generation")
    _, current_qc_provenance, current_qc_identity = _qc_context(qc_path)
    if current_qc_identity != qc_identity:
        raise ValueError("QC package changed during review-pack generation")
    if (
        current_qc_provenance.get("controls", {}).get("profile_sha256")
        != profile_sha256
        or current_qc_provenance.get("controls", {}).get("source_manifest")
        != initial_source
        or current_qc_provenance.get("runtime") != initial_runtime
    ):
        raise ValueError("QC generation context is stale for the review pack")
    if output is not None:
        verify_provenance(output)


def build_review_pack(
    inputs: BIDSEeglabInputs,
    profile_path: Path,
    qc_path: Path,
    output: Path,
) -> Path:
    """Atomically publish complete trace evidence for every pending QC prompt."""
    profile_path = Path(profile_path).expanduser().absolute()
    qc_root = Path(qc_path).expanduser().absolute()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Review-pack output must be a new path")
    resolved = resolve_inputs(inputs)
    protected_roots = [
        resolved.dataset_root.resolve(strict=True),
        qc_root.resolve(strict=True),
    ]
    nearest = _nearest_existing_ancestor(output.parent).resolve(strict=True)
    if any(nearest.is_relative_to(root) for root in protected_roots):
        raise ValueError("Review pack must remain outside the dataset and QC package")

    profile_bytes = profile_path.read_bytes()
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    profile = json.loads(profile_bytes.decode("utf-8-sig"))
    controls = _controls(profile)
    initial_inputs = input_identities(resolved)
    initial_source = source_manifest()
    initial_runtime = _runtime_identity()
    qc_summary, qc_provenance, qc_identity = _qc_context(qc_root)
    if qc_provenance.get("inputs") != initial_inputs:
        raise ValueError("QC package does not match the selected dataset inputs")
    if (
        qc_provenance.get("controls", {}).get("profile_sha256") != profile_sha256
        or qc_provenance.get("controls", {}).get("source_manifest") != initial_source
        or qc_provenance.get("runtime") != initial_runtime
    ):
        raise ValueError("QC generation context is stale for the review pack")
    if canonical_sha256(profile["qc"]) != qc_provenance.get("controls", {}).get(
        "qc_config_sha256"
    ):
        raise ValueError("Current profile does not reproduce the QC screen controls")
    if (
        str(profile["reference_channel"]["name"])
        != qc_summary.get("reference_channel_excluded_from_scoring")
        or list(controls.filter_hz)
        != qc_summary.get("screen_filter", {}).get("frequency_hz")
        or float(profile["expected"]["power_line_frequency_hz"])
        != qc_summary.get("line_noise_screen", {}).get("frequency_hz")
    ):
        raise ValueError("Current profile does not match the recorded QC screen")

    candidates = _captured_table(
        qc_root, qc_provenance, "candidate_window_review.csv"
    )
    segment_template = _captured_table(
        qc_root, qc_provenance, "segment_review_template.tsv", sep="\t"
    )
    channel_template = _captured_table(
        qc_root, qc_provenance, "channel_review_template.tsv", sep="\t"
    )
    bound = _bind_segment_candidates(candidates, segment_template)
    if set(channel_template["decision"]) not in (set(), {"pending"}):
        raise ValueError("Review pack requires the original pending channel template")

    raw = _read_raw(resolved.raw_set)
    reference = str(profile["reference_channel"]["name"])
    raw.set_channel_types({reference: "misc"}, verbose="ERROR")
    picks = [
        name
        for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
        if kind == "eeg"
    ]
    filtered = raw.copy()
    filtered.filter(
        *controls.filter_hz,
        picks=picks,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        verbose="ERROR",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        line_frequency = float(profile["expected"]["power_line_frequency_hz"])
        channel_index = _save_channel_review_figures(
            temporary / "figures" / "channels",
            raw,
            filtered,
            channel_template,
            line_frequency,
        )
        channel_index.to_csv(temporary / "channel_figure_index.csv", index=False)
        figures = temporary / "figures" / "segments"
        _save_window_trace_figures(
            figures,
            raw,
            filtered,
            bound,
            identifier_column="candidate_id",
            context_seconds=SEGMENT_CONTEXT_SECONDS,
            mark_prompt=True,
        )
        index_rows = []
        for row in bound.itertuples(index=False):
            safe_channel = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in str(row.channel)
            )
            figure_path = f"figures/segments/{row.candidate_id}_{safe_channel}.png"
            if not (temporary / figure_path).is_file():
                raise RuntimeError(f"Missing trace figure for {row.candidate_id}")
            index_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "channel": row.channel,
                    "prompt_onset_s": row.start_s,
                    "prompt_duration_s": row.duration_s,
                    "plotted_onset_s": max(
                        0.0, float(row.start_s) - SEGMENT_CONTEXT_SECONDS
                    ),
                    "plotted_duration_s": (
                        min(
                            float(raw.n_times / raw.info["sfreq"]),
                            float(row.stop_s) + SEGMENT_CONTEXT_SECONDS,
                        )
                        - max(0.0, float(row.start_s) - SEGMENT_CONTEXT_SECONDS)
                    ),
                    "candidate_reason": row.flag_reason,
                    "figure_path": figure_path,
                    "decision_status": "pending",
                }
            )
        segment_index = pd.DataFrame(index_rows)
        segment_index.to_csv(temporary / "segment_figure_index.csv", index=False)
        for relative in (
            "figures/full_recording_window_scan.png",
            "figures/line_noise_review.png",
            "figures/raw_psd.png",
        ):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                _verified_output_bytes(qc_root, qc_provenance, relative)
            )

        summary = {
            "schema_version": "1",
            "status": (
                "review_required"
                if len(channel_template) or len(bound)
                else "screen_complete"
            ),
            "scope": "complete controlled evidence index for human QC review",
            "channel_prompts": len(channel_template),
            "channel_figures": len(channel_index),
            "segment_prompts": len(bound),
            "segment_figures": len(segment_index),
            "segment_context_seconds": SEGMENT_CONTEXT_SECONDS,
            "automatic_decisions": 0,
            "decision_status": (
                "pending" if len(channel_template) or len(bound) else "not_required"
            ),
            "qc_summary_sha256": qc_identity["summary_sha256"],
            "classification": "controlled_derived",
            "publication_allowed": False,
        }
        (temporary / "review_pack_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "review_index.html").write_text(
            _render_index(channel_index, segment_index, summary), encoding="utf-8"
        )
        core = {
            "schema_version": "1",
            "workflow": "bids_eeglab_human_review_pack",
            "inputs": initial_inputs,
            "profile_sha256": profile_sha256,
            "qc_screen_config_sha256": qc_provenance["controls"][
                "qc_config_sha256"
            ],
            "qc": qc_identity,
            "source_manifest": initial_source,
            "runtime": initial_runtime,
            "summary_sha256": canonical_sha256(summary),
            "classification": "controlled_derived",
            "publication_allowed": False,
        }
        core["core_sha256"] = canonical_sha256(core)
        context = {
            "inputs": inputs,
            "profile_path": profile_path,
            "profile_sha256": profile_sha256,
            "qc_path": qc_root,
            "qc_identity": qc_identity,
            "initial_inputs": initial_inputs,
            "initial_source": initial_source,
            "initial_runtime": initial_runtime,
        }
        _verify_context(**context)
        write_provenance(temporary, core)
        verify_provenance(temporary)
        _verify_context(**context, output=temporary)
        temporary.replace(output)
        published = True
        _verify_context(**context, output=output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(output, ignore_errors=True)
        raise
