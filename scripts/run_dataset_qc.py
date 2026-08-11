#!/usr/bin/env python3
"""Run the raw BrainVision QC pipeline for every recording in a directory."""

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

import pandas as pd

from hunt_eeg.qc import run_qc
from hunt_eeg.provenance import sha256_file, source_manifest, verify_provenance


def participant_id(path: Path) -> str:
    match = re.search(r"_(\d{3})_", path.name)
    if not match:
        raise ValueError(f"Could not extract a three-digit participant code from {path.name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QC for all BrainVision recordings in a directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .vhdr/.eeg/.vmrk triples")
    parser.add_argument("--output", type=Path, required=True, help="Dataset-level output directory")
    args = parser.parse_args()

    headers = sorted(args.input_dir.expanduser().resolve().glob("*.vhdr"))
    if not headers:
        raise SystemExit(f"No .vhdr files found in {args.input_dir}")
    participants = [participant_id(header) for header in headers]
    if len(participants) != len(set(participants)):
        raise SystemExit("Input contains more than one header for a participant")
    requested_output = args.output.expanduser().resolve()
    if requested_output.exists():
        raise SystemExit("Dataset QC requires a new output directory; reuse is disabled")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_output.name}.tmp-",
            dir=requested_output.parent,
        )
    )
    cleanup = lambda: shutil.rmtree(output, ignore_errors=True)
    atexit.register(cleanup)
    initial_source_manifest = source_manifest()

    rows = []
    summaries = {}
    for index, header in enumerate(headers, start=1):
        participant = participant_id(header)
        print(f"[{index}/{len(headers)}] QC participant {participant}: {header.name}", flush=True)
        summary = run_qc(
            header, output / f"sub-{participant}", participant_id=participant
        )
        summaries[participant] = summary
        row = {
            "participant_id": participant,
            "duration_seconds": summary["duration_seconds"],
            "eeg_channels": summary["channels_eeg"],
            "candidate_bad_count": len(summary["candidate_bad_channels"]),
            "candidate_bad_channels": ";".join(summary["candidate_bad_channels"]),
            "full_recording_windows_scanned": summary[
                "full_recording_windows_scanned"
            ],
            "full_recording_candidate_bad_count": len(
                summary["full_recording_candidate_bad_channels"]
            ),
            "full_recording_candidate_bad_channels": ";".join(
                summary["full_recording_candidate_bad_channels"]
            ),
            "recording_maximum_raw_abs_deviation_uv": summary[
                "recording_maximum_raw_abs_deviation_uv"
            ],
            "recording_maximum_raw_abs_deviation_channel": summary[
                "recording_maximum_raw_abs_deviation_channel"
            ],
            "recording_maximum_raw_abs_deviation_time_s": summary[
                "recording_maximum_raw_abs_deviation_time_s"
            ],
            "high_line_noise_review_count": len(summary["channels_for_high_line_noise_review"]),
            "high_line_noise_review_channels": ";".join(summary["channels_for_high_line_noise_review"]),
            "line_frequency_hz": summary["line_frequency_hz"],
            "median_line_noise_ratio_db": summary["median_line_noise_ratio_db"],
        }
        row.update(summary["trial_counts"])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("participant_id")
    table.to_csv(output / "dataset_summary.csv", index=False)
    (output / "dataset_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    participant_records = []
    for participant in sorted(summaries):
        payload = verify_provenance(output / f"sub-{participant}")
        if payload["software"]["source_manifest"]["sha256"] != initial_source_manifest["sha256"]:
            raise RuntimeError(f"Source changed during QC before sub-{participant}")
        path = output / f"sub-{participant}" / "provenance.json"
        participant_records.append(
            {
                "participant_id": participant,
                "provenance_sha256": sha256_file(path),
                "core_sha256": payload["core_sha256"],
            }
        )
    if source_manifest()["sha256"] != initial_source_manifest["sha256"]:
        raise RuntimeError("Executable source changed during dataset QC")
    provenance_path = output / "dataset_provenance.json"
    top_level = sorted(
        path for path in output.iterdir()
        if path.is_file() and path != provenance_path
    )
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_manifest": initial_source_manifest,
                "participants": participant_records,
                "outputs": [
                    {
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in top_level
                ],
                "privacy": (
                    "Private derived output; review before sharing or publishing."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output.replace(requested_output)
    atexit.unregister(cleanup)
    print(f"Dataset QC complete: {requested_output}")


if __name__ == "__main__":
    main()
