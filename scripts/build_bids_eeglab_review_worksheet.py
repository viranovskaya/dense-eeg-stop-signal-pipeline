#!/usr/bin/env python3
"""Build a local mutable decision worksheet from a verified review pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab_review_worksheet import build_review_worksheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--channel-decisions", type=Path, required=True)
    parser.add_argument("--segment-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session-id",
        required=True,
        help="Reviewer/session-specific local storage namespace",
    )
    args = parser.parse_args()
    output = build_review_worksheet(
        args.review_pack,
        args.qc,
        args.channel_decisions,
        args.segment_decisions,
        args.output,
        args.session_id,
    )
    print(json.dumps({"status": "mutable_unverified", "output": str(output)}))


if __name__ == "__main__":
    main()
