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

from hunt_eeg.decisions import load_bad_channel_manifest
from hunt_eeg.fixed_study_intervals import load_fixed_study_interval_manifest
from hunt_eeg.preprocess import preprocess_recording


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter, rereference, interpolate, and epoch one recording."
    )
    parser.add_argument("--vhdr", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-manifest", type=Path, required=True)
    parser.add_argument(
        "--bad-channels",
        default="",
        help="Comma-separated persistent bad EEG channels to interpolate",
    )
    parser.add_argument(
        "--bad-channel-manifest",
        type=Path,
        help="Completed channel-review table; required when ICA is applied",
    )
    parser.add_argument("--no-eeglab-export", action="store_true")
    parser.add_argument(
        "--ica-solution",
        type=Path,
        help="Reviewed ICA solution created by run_ica_review.py",
    )
    parser.add_argument(
        "--ica-decisions",
        type=Path,
        help="Completed keep/exclude table bound to the ICA solution",
    )
    args = parser.parse_args()
    bads = [item.strip() for item in args.bad_channels.split(",") if item.strip()]
    if bads and args.bad_channel_manifest:
        parser.error("Use either --bad-channels or --bad-channel-manifest")
    if (args.ica_solution or args.ica_decisions) and not args.bad_channel_manifest:
        parser.error("ICA application requires --bad-channel-manifest")
    if args.bad_channel_manifest:
        manifest = load_bad_channel_manifest(args.bad_channel_manifest)
        bad_channel_decisions = manifest.loc[
            manifest["participant_id"] == args.participant_id
        ].to_dict("records")
        if not bad_channel_decisions:
            parser.error("No bad-channel review found for this participant")
    else:
        bad_channel_decisions = None
    interval_manifest = load_fixed_study_interval_manifest(args.interval_manifest)
    interval_manifest.participant_rows(args.participant_id)
    summary = preprocess_recording(
        args.vhdr,
        args.output,
        participant_id=args.participant_id,
        bad_channels=bads,
        bad_channel_decisions=bad_channel_decisions,
        ica_solution_path=args.ica_solution,
        ica_decision_path=args.ica_decisions,
        interval_manifest=interval_manifest,
        export_eeglab=not args.no_eeglab_export,
    )
    print(f"Preprocessing complete: {args.output.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
