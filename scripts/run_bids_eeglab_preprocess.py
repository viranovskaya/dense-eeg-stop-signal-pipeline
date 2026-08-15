#!/usr/bin/env python3
"""Run reviewed BIDS/EEGLAB preprocessing and epoch export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs
from hunt_eeg.bids_eeglab_preprocess import publish_reviewed_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--ica-review", type=Path, required=True)
    parser.add_argument("--ica-decisions", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = publish_reviewed_preprocessing(
        BIDSEeglabInputs(args.dataset_root, args.raw_set),
        args.profile,
        args.qc,
        args.review_bundle,
        args.ica_review,
        args.ica_decisions,
        args.participant_id,
        args.output,
    )
    summary = json.loads((output / "preprocessing_summary.json").read_text())
    print(json.dumps({"status": summary["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
