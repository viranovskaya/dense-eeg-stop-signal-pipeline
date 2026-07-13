#!/usr/bin/env python3
"""Preprocess one BrainVision recording and export FIF + EEGLAB datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from hunt_eeg.preprocess import preprocess_recording


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter, rereference, interpolate, and epoch one recording."
    )
    parser.add_argument("--vhdr", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bad-channels",
        default="",
        help="Comma-separated persistent bad EEG channels to interpolate",
    )
    parser.add_argument("--no-eeglab-export", action="store_true")
    args = parser.parse_args()
    bads = [item.strip() for item in args.bad_channels.split(",") if item.strip()]
    summary = preprocess_recording(
        args.vhdr,
        args.output,
        participant_id=args.participant_id,
        bad_channels=bads,
        export_eeglab=not args.no_eeglab_export,
    )
    print(f"Preprocessing complete: {args.output.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()

