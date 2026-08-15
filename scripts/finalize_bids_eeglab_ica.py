#!/usr/bin/env python3
"""Finalize complete human decisions for a BIDS/EEGLAB ICA solution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs
from hunt_eeg.bids_eeglab_ica_decisions import finalize_ica_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--ica-review", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = finalize_ica_decisions(
        BIDSEeglabInputs(args.dataset_root, args.raw_set),
        args.qc,
        args.review_bundle,
        args.ica_review,
        args.decisions,
        args.participant_id,
        args.output,
    )
    summary = json.loads((output / "ica_decision_summary.json").read_text())
    print(json.dumps({"status": summary["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
