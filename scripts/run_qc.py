#!/usr/bin/env python3
"""Command-line entry point for a raw BrainVision QC pilot."""

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

from hunt_eeg.qc import run_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate non-destructive QC for a BrainVision recording.")
    parser.add_argument("--vhdr", type=Path, required=True, help="Path to the BrainVision .vhdr file")
    parser.add_argument("--output", type=Path, required=True, help="Directory for QC outputs")
    args = parser.parse_args()
    summary = run_qc(args.vhdr, args.output)
    print(f"QC complete: {args.output.resolve()}")
    print(f"Trials: {summary['trial_counts']}")
    print(f"Candidate bad channels: {summary['candidate_bad_channels']}")


if __name__ == "__main__":
    main()

