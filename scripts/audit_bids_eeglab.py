#!/usr/bin/env python3
"""Validate one versioned BIDS EEGLAB recording before preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from hunt_eeg.bids_eeglab import BIDSEeglabInputs, publish_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fail-closed inventory for one BIDS EEGLAB recording."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--set", dest="raw_set", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = BIDSEeglabInputs(
        dataset_root=args.dataset_root,
        raw_set=args.raw_set,
    )
    output = publish_inventory(inputs, args.profile, args.output)
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": audit["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
