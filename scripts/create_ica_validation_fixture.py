#!/usr/bin/env python3
"""Create the public synthetic BrainVision fixture used for ICA review."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mne
import pandas as pd

from hunt_eeg.benchmark import (
    load_benchmark_config,
    make_clean_recording,
    verify_brainvision_round_trip,
    write_json,
)
from hunt_eeg.provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)

DEFAULT_FIXTURE_CONFIG = PROJECT_ROOT / "config" / "ica_validation.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic public BrainVision fixture for ICA review."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FIXTURE_CONFIG,
    )
    args = parser.parse_args()

    requested = args.output.expanduser().resolve()
    if requested.exists():
        raise FileExistsError(f"ICA fixture requires a new output path: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{requested.name}.tmp-", dir=requested.parent)
    )
    initial_source = source_manifest()
    fixture_config_hash = sha256_file(args.config)
    try:
        fixture_config = json.loads(args.config.read_text(encoding="utf-8"))
        seed = int(fixture_config["seed"])
        benchmark_config = load_benchmark_config()
        raw = make_clean_recording(benchmark_config, seed)
        vhdr = temporary / "synthetic_ica.vhdr"
        mne.export.export_raw(
            vhdr,
            raw,
            fmt="brainvision",
            overwrite=False,
            verbose="ERROR",
        )
        reread = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
        round_trip = verify_brainvision_round_trip(raw, reread)
        pd.DataFrame(
            [
                {
                    "participant_id": "synthetic",
                    "channel": "",
                    "decision": "none",
                    "reason": "generator declares no persistent bad channels",
                    "reviewer": "synthetic oracle",
                    "reviewed_at": "2026-08-12",
                    "evidence_windows": "0-180 s",
                }
            ]
        ).to_csv(temporary / "bad_channel_decisions.csv", index=False)
        write_json(
            temporary / "fixture_summary.json",
            {
                "schema_version": 1,
                "seed": seed,
                "scope": fixture_config["scope"],
                "eeg_channels": len(raw.copy().pick("eeg").ch_names),
                "sampling_frequency_hz": float(raw.info["sfreq"]),
                "recording_seconds": float(raw.n_times / raw.info["sfreq"]),
                "brainvision_round_trip": round_trip,
            },
        )
        if fixture_config_hash != sha256_file(args.config):
            raise RuntimeError("ICA fixture configuration changed during generation")
        if initial_source["sha256"] != source_manifest()["sha256"]:
            raise RuntimeError("Executable source changed during fixture generation")
        core = {
            "schema_version": 1,
            "workflow": "public_synthetic_ica_fixture",
            "fixture_config_sha256": fixture_config_hash,
            "benchmark_config_sha256": sha256_file(
                PROJECT_ROOT / "config" / "synthetic_benchmark.json"
            ),
            "software": {
                "python": platform.python_version(),
                "packages": package_versions(),
                "source_manifest": initial_source,
            },
            "scope": (
                "Public synthetic integration fixture; not participant evidence or "
                "validation of automatic ICA component classification."
            ),
        }
        core["core_sha256"] = canonical_sha256(core)
        write_provenance(temporary, core)
        verify_provenance(temporary)
        temporary.replace(requested)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Public ICA fixture: {requested}")


if __name__ == "__main__":
    main()
