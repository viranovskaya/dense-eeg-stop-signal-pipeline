#!/usr/bin/env python3
"""Build a complete controlled human-review pack from a verified QC package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs
from hunt_eeg.bids_eeglab_review_pack import build_review_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_review_pack(
        BIDSEeglabInputs(dataset_root=args.dataset_root, raw_set=args.raw_set),
        args.profile,
        args.qc,
        args.output,
    )
    summary = json.loads(
        (output / "review_pack_summary.json").read_text(encoding="utf-8")
    )
    print(json.dumps({"status": summary["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
