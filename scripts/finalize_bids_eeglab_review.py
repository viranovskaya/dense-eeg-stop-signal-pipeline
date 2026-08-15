#!/usr/bin/env python3
"""Finalize completed BIDS/EEGLAB channel and segment decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs
from hunt_eeg.bids_eeglab_review import finalize_review_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--channel-decisions", type=Path, required=True)
    parser.add_argument("--segment-decisions", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = finalize_review_decisions(
        BIDSEeglabInputs(args.dataset_root, args.raw_set),
        args.qc,
        args.channel_decisions,
        args.segment_decisions,
        args.participant_id,
        args.output,
    )
    summary = json.loads((output / "decision_summary.json").read_text())
    print(json.dumps({"status": summary["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
