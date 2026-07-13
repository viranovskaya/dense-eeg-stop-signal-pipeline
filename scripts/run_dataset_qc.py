#!/usr/bin/env python3
"""Run the raw BrainVision QC pipeline for every recording in a directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = {}
    for index, header in enumerate(headers, start=1):
        participant = participant_id(header)
        print(f"[{index}/{len(headers)}] QC participant {participant}: {header.name}", flush=True)
        summary = run_qc(header, output / f"sub-{participant}")
        summaries[participant] = summary
        row = {
            "participant_id": participant,
            "duration_seconds": summary["duration_seconds"],
            "eeg_channels": summary["channels_eeg"],
            "candidate_bad_count": len(summary["candidate_bad_channels"]),
            "candidate_bad_channels": ";".join(summary["candidate_bad_channels"]),
            "high_line_noise_review_count": len(summary["channels_for_high_line_noise_review"]),
            "high_line_noise_review_channels": ";".join(summary["channels_for_high_line_noise_review"]),
            "median_50hz_line_ratio_db": summary["median_50hz_line_ratio_db"],
        }
        row.update(summary["trial_counts"])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("participant_id")
    table.to_csv(output / "dataset_summary.csv", index=False)
    (output / "dataset_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Dataset QC complete: {output}")


if __name__ == "__main__":
    main()

