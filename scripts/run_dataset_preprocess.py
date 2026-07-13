#!/usr/bin/env python3
"""Apply preprocessing to every BrainVision recording using an explicit manifest."""

from __future__ import annotations

import argparse
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

from hunt_eeg.preprocess import preprocess_recording


def participant_id(path: Path) -> str:
    match = re.search(r"_(\d{3})_", path.name)
    if not match:
        raise ValueError(f"Could not extract participant code from {path.name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess all BrainVision recordings.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bad-channel-manifest", type=Path, required=True)
    parser.add_argument("--no-eeglab-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.bad_channel_manifest, dtype={"participant_id": str})
    interpolate = manifest.loc[manifest["decision"] == "interpolate"].copy()
    bads_by_participant = (
        interpolate.groupby("participant_id")["channel"].apply(list).to_dict()
    )

    headers = sorted(args.input_dir.expanduser().resolve().glob("*.vhdr"))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index, header in enumerate(headers, start=1):
        participant = participant_id(header)
        participant_output = output / f"sub-{participant}"
        summary_file = participant_output / "preprocessing_summary.json"
        if summary_file.exists() and not args.overwrite:
            print(f"[{index}/{len(headers)}] sub-{participant}: already complete, skipping")
            continue
        bads = bads_by_participant.get(participant, [])
        print(
            f"[{index}/{len(headers)}] sub-{participant}: "
            f"interpolate {bads if bads else 'none'}",
            flush=True,
        )
        preprocess_recording(
            header,
            participant_output,
            participant_id=participant,
            bad_channels=bads,
            export_eeglab=not args.no_eeglab_export,
        )
    print(f"Dataset preprocessing complete: {output}")


if __name__ == "__main__":
    main()

