#!/usr/bin/env python3
"""Run controlled QC for one inventoried BIDS/EEGLAB recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs
from hunt_eeg.bids_eeglab_qc import publish_qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = publish_qc(
        BIDSEeglabInputs(dataset_root=args.dataset_root, raw_set=args.raw_set),
        args.profile,
        args.inventory,
        args.output,
    )
    summary = json.loads((output / "qc_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": summary["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
