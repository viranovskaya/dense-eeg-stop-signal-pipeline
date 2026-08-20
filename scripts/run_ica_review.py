#!/usr/bin/env python3
"""Create a private ICA component-review package for one recording."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ["MPLBACKEND"] = "Agg"
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from hunt_eeg.decisions import load_bad_channel_manifest
from hunt_eeg.fixed_study_intervals import load_fixed_study_interval_manifest
from hunt_eeg.ica import run_ica_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ICA and create a private component-review package."
    )
    parser.add_argument("--vhdr", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--bad-channel-manifest", type=Path, required=True)
    parser.add_argument("--interval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_bad_channel_manifest(args.bad_channel_manifest)
    decisions = manifest.loc[manifest["participant_id"] == args.participant].to_dict(
        "records"
    )
    if not decisions:
        raise SystemExit("No bad-channel review found for this participant")
    interval_manifest = load_fixed_study_interval_manifest(args.interval_manifest)
    interval_manifest.participant_rows(args.participant)
    summary = run_ica_review(
        vhdr=args.vhdr,
        output=args.output,
        participant_id=args.participant,
        bad_channel_decisions=decisions,
        interval_manifest=interval_manifest,
    )
    print(
        f"ICA review package complete: {args.output} "
        f"({summary['components']} components)"
    )


if __name__ == "__main__":
    main()
