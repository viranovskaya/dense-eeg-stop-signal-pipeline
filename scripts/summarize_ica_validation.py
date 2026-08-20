#!/usr/bin/env python3
"""Summarize one completed, reviewed ICA integration fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
(PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mne
import pandas as pd

from hunt_eeg.config import DEFAULT_ANALYSIS_CONFIG, load_analysis_config
from hunt_eeg.ica import (
    DEFAULT_ICA_CONFIG,
    load_ica_config,
    load_ica_decisions,
)
from hunt_eeg.ica_validation import (
    summarize_reviewed_ica,
    verified_fixture_seed,
    verify_matched_ica_processing,
    verify_validation_context,
)
from hunt_eeg.provenance import (
    sha256_file,
    source_manifest,
    verify_input_sources,
    verify_provenance,
)
from hunt_eeg.qc import _read_brainvision_compat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a completed public ICA integration fixture."
    )
    parser.add_argument("--vhdr", type=Path, required=True)
    parser.add_argument("--review-package", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--control-processed-output", type=Path, required=True)
    parser.add_argument("--processed-output", type=Path, required=True)
    parser.add_argument("--fixture-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Validation summary already exists: {args.output}")

    initial_source = source_manifest()
    fixture_root = args.vhdr.expanduser().resolve().parent
    fixture = verify_provenance(fixture_root)
    fixture_summary = json.loads(
        (fixture_root / "fixture_summary.json").read_text(encoding="utf-8")
    )
    fixture_seed = verified_fixture_seed(
        fixture,
        fixture_summary,
        args.fixture_seed,
    )
    if fixture["software"]["source_manifest"]["sha256"] != initial_source["sha256"]:
        raise ValueError("Fixture source does not match the validation source")
    review = verify_provenance(args.review_package)
    control_processed = verify_provenance(args.control_processed_output)
    processed = verify_provenance(args.processed_output)
    if review.get("workflow") != "ica_component_review":
        raise ValueError("Review package does not describe an ICA component review")
    if len(
        {
            review["participant_id"],
            control_processed["participant_id"],
            processed["participant_id"],
        }
    ) != 1:
        raise ValueError("Review and preprocessing participant IDs do not match")
    verify_input_sources(args.vhdr, review["inputs"])
    verify_validation_context(
        review,
        processed,
        control_processed,
        current_source_manifest_sha256=initial_source["sha256"],
        analysis_config_sha256=sha256_file(DEFAULT_ANALYSIS_CONFIG),
        ica_config_sha256=sha256_file(DEFAULT_ICA_CONFIG),
    )

    solution = args.review_package / "ica_solution.fif"
    review_summary = json.loads(
        (args.review_package / "ica_review_summary.json").read_text(encoding="utf-8")
    )
    if not review_summary.get("fit_stopping_rule_met"):
        raise ValueError("ICA review package did not meet its stopping rule")
    decisions = load_ica_decisions(
        args.decisions,
        participant_id=review["participant_id"],
        solution_sha256=sha256_file(solution),
        components=int(review_summary["components"]),
    )
    preprocessing_summary = json.loads(
        (args.processed_output / "preprocessing_summary.json").read_text(
            encoding="utf-8"
        )
    )
    control_preprocessing_summary = json.loads(
        (args.control_processed_output / "preprocessing_summary.json").read_text(
            encoding="utf-8"
        )
    )
    verify_matched_ica_processing(
        control_processed,
        processed,
        control_preprocessing_summary,
        preprocessing_summary,
    )
    ica_summary = preprocessing_summary["ica"]
    if ica_summary["solution_sha256"] != sha256_file(solution):
        raise ValueError("Processed output used a different ICA solution")
    if ica_summary["decision_table_sha256"] != sha256_file(args.decisions):
        raise ValueError("Processed output used a different ICA decision table")
    if ica_summary["automatic_exclusion"]:
        raise ValueError(
            "Validation requires an explicitly reviewed ICA decision table"
        )

    continuous = [
        path
        for path in args.processed_output.glob("*_continuous_*_raw.fif")
        if path.is_file()
    ]
    if len(continuous) != 1:
        raise ValueError("Expected exactly one processed continuous FIF file")
    control_continuous = [
        path
        for path in args.control_processed_output.glob("*_continuous_*_raw.fif")
        if path.is_file()
    ]
    if len(control_continuous) != 1:
        raise ValueError("Expected exactly one control continuous FIF file")
    reviewed_raw = mne.io.read_raw_fif(continuous[0], preload=True, verbose="ERROR")
    control_raw = mne.io.read_raw_fif(
        control_continuous[0], preload=True, verbose="ERROR"
    )
    analysis = load_analysis_config()
    auxiliary_raw = _read_brainvision_compat(args.vhdr)
    type_updates = {
        name: kind
        for name, kind in analysis.channel_types.items()
        if name in auxiliary_raw.ch_names
    }
    auxiliary_raw.set_channel_types(type_updates, verbose="ERROR")
    auxiliary_raw.load_data(verbose="ERROR")
    auxiliary_picks = mne.pick_types(
        auxiliary_raw.info,
        eeg=False,
        eog=True,
        ecg=True,
        exclude=[],
    )
    auxiliary_raw.filter(
        *analysis.erp_filter_hz,
        picks=auxiliary_picks,
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    checks = summarize_reviewed_ica(control_raw, reviewed_raw, auxiliary_raw)
    if source_manifest()["sha256"] != initial_source["sha256"]:
        raise RuntimeError("Executable source changed during ICA validation")

    diagnostics = pd.read_csv(args.review_package / "component_diagnostics.csv")
    excluded = decisions.loc[decisions["decision"] == "exclude", "component"].astype(
        int
    )
    selected = diagnostics.loc[diagnostics["component"].isin(excluded)].copy()
    selected = selected.sort_values("component")
    payload = {
        "schema_version": 2,
        "fixture": {
            "kind": "public synthetic integration fixture",
            "seed": fixture_seed,
            "recording_seconds": float(
                control_raw.times[-1] + 1 / control_raw.info["sfreq"]
            ),
            "sampling_frequency_hz": float(control_raw.info["sfreq"]),
            "eeg_channels": len(control_raw.copy().pick("eeg").ch_names),
        },
        "ica": {
            "method": load_ica_config().method,
            "components": int(ica_summary["components"]),
            "fit_iterations": int(review_summary["fit_iterations"]),
            "fit_stopping_rule_met": True,
            "fit_stopping_rule": review_summary["fit_stopping_rule"],
            "excluded_components": excluded.tolist(),
            "automatic_exclusion": False,
            "selected_component_cues": selected[
                ["component", "eog_correlation", "ecg_correlation"]
            ].to_dict("records"),
        },
        "checks": checks,
        "accounting": {
            "markers": preprocessing_summary["trial_reconciliation"]["markers_total"],
            "trial_starts": preprocessing_summary["trial_reconciliation"][
                "detected_trial_starts"
            ],
            "go_epochs_retained": preprocessing_summary["go_epoch_accounting"][
                "retained_epochs"
            ],
            "stop_epochs_retained": preprocessing_summary["stop_epoch_accounting"][
                "retained_epochs"
            ],
            "dropped_epochs": (
                preprocessing_summary["go_epoch_accounting"]["dropped_epochs"]
                + preprocessing_summary["stop_epoch_accounting"]["dropped_epochs"]
            ),
        },
        "provenance": {
            "fixture_provenance_sha256": sha256_file(fixture_root / "provenance.json"),
            "review_provenance_sha256": sha256_file(
                args.review_package / "provenance.json"
            ),
            "control_processed_provenance_sha256": sha256_file(
                args.control_processed_output / "provenance.json"
            ),
            "processed_provenance_sha256": sha256_file(
                args.processed_output / "provenance.json"
            ),
            "source_manifest_sha256": processed["software"]["source_manifest"][
                "sha256"
            ],
        },
        "runtime": {
            "python": processed["software"]["python"],
            "packages": processed["software"]["packages"],
        },
        "scope": (
            "One fixed public synthetic integration fixture. Component decisions "
            "were made explicitly after review. Signal checks compare the reviewed "
            "output with a matched output that retained all ICA components; all "
            "other processing decisions are identical. Results do not establish "
            "automatic artifact classification or generalization to participant EEG."
        ),
    }
    verify_input_sources(args.vhdr, review["inputs"])
    for path, expected in (
        (fixture_root, fixture),
        (args.review_package, review),
        (args.control_processed_output, control_processed),
        (args.processed_output, processed),
    ):
        if verify_provenance(path)["core_sha256"] != expected["core_sha256"]:
            raise RuntimeError("ICA validation package changed during summarization")
    if source_manifest()["sha256"] != initial_source["sha256"]:
        raise RuntimeError("Executable source changed during ICA validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ICA validation summary: {args.output}")


if __name__ == "__main__":
    main()
