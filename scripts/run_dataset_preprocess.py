#!/usr/bin/env python3
"""Apply preprocessing to every BrainVision recording using an explicit manifest."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from hunt_eeg.dataset import build_dataset_aggregate
from hunt_eeg.decisions import load_bad_channel_manifest
from hunt_eeg.preprocess import preprocess_recording
from hunt_eeg.provenance import sha256_file, source_manifest, verify_provenance


def participant_id(path: Path) -> str:
    match = re.search(r"_(\d{3})_", path.name)
    if not match:
        raise ValueError(f"Could not extract participant code from {path.name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess all BrainVision recordings."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bad-channel-manifest", type=Path, required=True)
    parser.add_argument("--no-eeglab-export", action="store_true")
    parser.add_argument(
        "--ica-review-root",
        type=Path,
        help="Directory containing sub-<participant>/ica_solution.fif packages",
    )
    parser.add_argument(
        "--ica-decision-manifest",
        type=Path,
        help="Completed component decisions for every participant",
    )
    args = parser.parse_args()
    if (args.ica_review_root is None) != (args.ica_decision_manifest is None):
        parser.error(
            "--ica-review-root and --ica-decision-manifest must be used together"
        )

    manifest_path = args.bad_channel_manifest.expanduser().resolve()
    initial_manifest_sha256 = sha256_file(manifest_path)
    initial_source_manifest = source_manifest()
    manifest = load_bad_channel_manifest(manifest_path)
    decisions_by_participant = {
        participant: group.to_dict("records")
        for participant, group in manifest.groupby("participant_id")
    }

    headers = sorted(args.input_dir.expanduser().resolve().glob("*.vhdr"))
    if not headers:
        raise SystemExit("No BrainVision .vhdr files found in the input directory")
    participants = [participant_id(header) for header in headers]
    if len(participants) != len(set(participants)):
        raise SystemExit("Input contains more than one header for a participant")
    manifest_participants = set(decisions_by_participant)
    missing_reviews = sorted(set(participants) - manifest_participants)
    extra_reviews = sorted(manifest_participants - set(participants))
    if missing_reviews or extra_reviews:
        raise SystemExit(
            "The bad-channel manifest must match the input participants exactly; "
            f"missing reviews={missing_reviews}, extra reviews={extra_reviews}"
        )
    if args.ica_review_root is not None:
        ica_review_root = args.ica_review_root.expanduser().resolve()
        ica_decision_manifest = args.ica_decision_manifest.expanduser().resolve()
        initial_ica_decision_sha256 = sha256_file(ica_decision_manifest)
        ica_solutions = {
            participant: (ica_review_root / f"sub-{participant}" / "ica_solution.fif")
            for participant in participants
        }
        missing_ica = sorted(
            participant
            for participant, path in ica_solutions.items()
            if not path.is_file()
        )
        if missing_ica:
            raise SystemExit(
                "ICA review packages are missing for participants: "
                + ", ".join(missing_ica)
            )
        initial_ica_solution_hashes = {
            participant: sha256_file(path)
            for participant, path in sorted(ica_solutions.items())
        }
    else:
        ica_decision_manifest = None
        ica_solutions = {}
        initial_ica_decision_sha256 = None
        initial_ica_solution_hashes = {}
    requested_output = args.output.expanduser().resolve()
    if requested_output.exists():
        raise SystemExit(
            "Dataset preprocessing requires a new output directory; resume and "
            "in-place reuse are disabled"
        )
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_output.name}.tmp-",
            dir=requested_output.parent,
        )
    )
    cleanup = lambda: shutil.rmtree(output, ignore_errors=True)
    atexit.register(cleanup)
    rows = []
    summaries = {}
    for index, header in enumerate(headers, start=1):
        participant = participant_id(header)
        participant_output = output / f"sub-{participant}"
        decisions = decisions_by_participant.get(participant, [])
        bads = sorted(
            decision["channel"]
            for decision in decisions
            if decision["decision"] == "interpolate"
        )
        print(
            f"[{index}/{len(headers)}] sub-{participant}: "
            f"interpolate {bads if bads else 'none'}",
            flush=True,
        )
        summary = preprocess_recording(
            header,
            participant_output,
            participant_id=participant,
            bad_channel_decisions=decisions,
            ica_solution_path=ica_solutions.get(participant),
            ica_decision_path=ica_decision_manifest,
            export_eeglab=not args.no_eeglab_export,
        )
        summaries[participant] = summary
        reconciliation = summary["trial_reconciliation"]
        go_accounting = summary["go_epoch_accounting"]
        stop_accounting = summary["stop_epoch_accounting"]
        row = {
            "participant_id": participant,
            "decision_record_complete": summary["bad_channel_decision_record_complete"],
            "interpolated_channel_count": len(summary["interpolated_bad_channels"]),
            "detected_trial_starts": reconciliation["detected_trial_starts"],
            "go_events_proposed": go_accounting["proposed_events"],
            "go_epochs_retained": go_accounting["retained_epochs"],
            "go_epochs_dropped": go_accounting["dropped_epochs"],
            "stop_events_proposed": stop_accounting["proposed_events"],
            "stop_epochs_retained": stop_accounting["retained_epochs"],
            "stop_epochs_dropped": stop_accounting["dropped_epochs"],
            "before_candidate_bad_count": len(
                summary["before_after_qc"]["before"]["candidate_bad_channels"]
            ),
            "after_candidate_bad_count": len(
                summary["before_after_qc"]["after"]["candidate_bad_channels"]
            ),
            "ica_status": summary["ica"]["status"],
            "ica_excluded_component_count": len(
                summary["ica"].get("excluded_components", [])
            ),
        }
        for stage in ("before", "after"):
            qc = summary["before_after_qc"][stage]
            for metric in (
                "median_filtered_std_uv",
                "median_filtered_robust_range_uv",
                "maximum_flat_fraction",
            ):
                row[f"{stage}_{metric}"] = qc[metric]
        row.update(
            {
                f"trial_status_{status}": count
                for status, count in reconciliation["status_counts"].items()
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("participant_id")
    table.to_csv(output / "dataset_preprocessing_summary.csv", index=False)
    (output / "dataset_preprocessing_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregate = build_dataset_aggregate(table, summaries)
    (output / "dataset_preprocessing_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = range(len(table))
    axes[0].plot(
        x,
        table["before_median_filtered_std_uv"],
        marker="o",
        label="before",
    )
    axes[0].plot(
        x,
        table["after_median_filtered_std_uv"],
        marker="o",
        label="after",
    )
    axes[0].set(ylabel="Median EEG SD (µV)", title="Dataset before/after QC")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(x, table["before_candidate_bad_count"], marker="o", label="before")
    axes[1].plot(x, table["after_candidate_bad_count"], marker="o", label="after")
    axes[1].set(xlabel="Participant", ylabel="Screening candidates")
    axes[1].set_xticks(list(x), table["participant_id"], rotation=45, ha="right")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "dataset_before_after_qc.png", dpi=170)
    plt.close(figure)

    report = f"""# Dataset preprocessing report

- Recordings: {aggregate["recordings"]}
- Trial starts accounted for: {aggregate["detected_trial_starts_total"]}
- Interpolated channels: {aggregate["interpolated_channels_total"]}
- Complete decision records: {aggregate["all_decision_records_complete"]}
- ICA applied: {int(table["ica_status"].eq("applied reviewed decisions").sum())} / {len(table)} recordings
- ICA components excluded: {int(table["ica_excluded_component_count"].sum())}
- Go epochs: {aggregate["epochs"]["go"]["retained"]} retained / {aggregate["epochs"]["go"]["proposed"]} proposed
- Stop epochs: {aggregate["epochs"]["stop"]["retained"]} retained / {aggregate["epochs"]["stop"]["proposed"]} proposed

![Dataset before/after QC](dataset_before_after_qc.png)

The figure and aggregate JSON contain descriptive screening metrics and
accounting. They are not a global quality score or proof of artifact removal.
Participant-level values remain available in `dataset_preprocessing_summary.csv`.
"""
    (output / "dataset_preprocessing_report.md").write_text(
        report,
        encoding="utf-8",
    )
    participant_records = []
    for participant in sorted(summaries):
        path = output / f"sub-{participant}" / "provenance.json"
        payload = verify_provenance(output / f"sub-{participant}")
        if (
            payload["software"]["source_manifest"]["sha256"]
            != initial_source_manifest["sha256"]
        ):
            raise RuntimeError(
                f"Source changed during dataset run before sub-{participant}"
            )
        participant_records.append(
            {
                "participant_id": participant,
                "provenance_sha256": sha256_file(path),
                "core_sha256": payload["core_sha256"],
            }
        )
    dataset_provenance_path = output / "dataset_provenance.json"
    top_level_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path != dataset_provenance_path
    )
    dataset_provenance = {
        "schema_version": "1",
        "source_manifest": initial_source_manifest,
        "bad_channel_manifest_sha256": initial_manifest_sha256,
        "ica": {
            "decision_manifest_sha256": initial_ica_decision_sha256,
            "solution_sha256_by_participant": initial_ica_solution_hashes,
            "automatic_exclusion": False,
        },
        "participants": participant_records,
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in top_level_files
        ],
        "privacy": "Private derived output; review before sharing or publishing.",
    }
    dataset_provenance_path.write_text(
        json.dumps(dataset_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if sha256_file(manifest_path) != initial_manifest_sha256:
        raise RuntimeError("Bad-channel manifest changed during dataset run")
    if source_manifest()["sha256"] != initial_source_manifest["sha256"]:
        raise RuntimeError("Executable source changed during dataset run")
    if ica_decision_manifest is not None:
        if sha256_file(ica_decision_manifest) != initial_ica_decision_sha256:
            raise RuntimeError("ICA decision manifest changed during dataset run")
        final_ica_solution_hashes = {
            participant: sha256_file(path)
            for participant, path in sorted(ica_solutions.items())
        }
        if final_ica_solution_hashes != initial_ica_solution_hashes:
            raise RuntimeError("ICA solution changed during dataset run")
    output.replace(requested_output)
    atexit.unregister(cleanup)
    print(f"Dataset preprocessing complete: {requested_output}")


if __name__ == "__main__":
    main()
