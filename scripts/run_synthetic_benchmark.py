#!/usr/bin/env python3
"""Run the public synthetic corruption benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import atexit
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
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hunt_eeg.benchmark import (
    load_benchmark_config,
    run_benchmark_seed,
    summarize_band_errors,
    summarize_evaluation_units,
    verify_brainvision_round_trip,
    write_json,
)
from hunt_eeg.config import load_analysis_config
from hunt_eeg.provenance import (
    canonical_sha256,
    package_versions,
    sha256_file,
    source_manifest,
    verify_provenance,
    write_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a public dense-EEG corruption benchmark."
    )
    parser.add_argument(
        "--seed-group",
        choices=("calibration", "held_out", "stress"),
        default="held_out",
        help="Predeclared seed group to run (default: held_out)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory",
    )
    parser.add_argument(
        "--export-brainvision",
        action="store_true",
        help="Export and re-read the first corrupted recording",
    )
    args = parser.parse_args()

    requested_output = args.output.expanduser().resolve()
    if requested_output.exists():
        raise SystemExit("Synthetic benchmark requires a new output directory")
    requested_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix=f".{requested_output.name}.tmp-",
            dir=requested_output.parent,
        )
    )

    def cleanup() -> None:
        shutil.rmtree(output, ignore_errors=True)

    atexit.register(cleanup)

    benchmark = load_benchmark_config()
    analysis = load_analysis_config()
    initial_source = source_manifest()
    summaries = []
    task_tables = []
    band_tables = []
    oracle_channels: set[str] = set()
    corrupted_channels: set[str] = set()
    for index, seed in enumerate(benchmark.random_seeds[args.seed_group], start=1):
        print(f"[{index}] synthetic seed {seed}", flush=True)
        result = run_benchmark_seed(
            benchmark,
            analysis.qc,
            seed,
            filter_hz=analysis.filter_hz,
        )
        seed_output = output / f"seed-{seed}"
        seed_output.mkdir()
        result["truth"].to_csv(seed_output / "corruption_truth.csv", index=False)
        result["comparison"].to_csv(
            seed_output / "temporal_all_families.csv", index=False
        )
        result["temporal_comparison"].to_csv(
            seed_output / "temporal_detection.csv", index=False
        )
        result["primary_comparison"].to_csv(
            seed_output / "temporal_primary_detection.csv", index=False
        )
        result["stress_comparison"].to_csv(
            seed_output / "temporal_stress_cases.csv", index=False
        )
        result["line_noise_comparison"].to_csv(
            seed_output / "line_noise_detection.csv", index=False
        )
        result["corruption_impact"].to_csv(
            seed_output / "corruption_impact.csv", index=False
        )
        result["preservation"].to_csv(
            seed_output / "signal_preservation.csv", index=False
        )
        result["task_signal"].to_csv(
            seed_output / "task_signal_preservation.csv", index=False
        )
        seed_oracle_channels = set(result["known_bad_channels"])
        seed_corrupted_channels = set(result["truth"]["channel"].astype(str))
        classified_band_power, seed_band_summary = summarize_band_errors(
            result["band_power"],
            seed_oracle_channels,
            seed_corrupted_channels,
        )
        classified_band_power.to_csv(
            seed_output / "band_power_preservation.csv", index=False
        )
        result["trials"].to_csv(seed_output / "trial_reconstruction.csv", index=False)
        write_json(
            seed_output / "trial_accounting.json",
            result["trial_accounting"],
        )
        task_tables.append(result["task_signal"].assign(seed=seed))
        band_tables.append(classified_band_power.assign(seed=seed))
        oracle_channels.update(seed_oracle_channels)
        corrupted_channels.update(seed_corrupted_channels)
        evaluation_units = summarize_evaluation_units(
            result["temporal_comparison"],
            result["primary_comparison"],
            result["stress_comparison"],
        )
        metrics = {
            "seed": seed,
            "seed_group": args.seed_group,
            "temporal_detection": result["temporal_detection"],
            "temporal_primary_detection": result["primary_detection"],
            "temporal_stress_detection_rate": result["stress_detection_rate"],
            "line_noise_detection": result["line_noise_detection"],
            "known_bad_channels": result["known_bad_channels"],
            "maximum_task_amplitude_error_uv": float(
                result["task_signal"]["absolute_amplitude_error_uv"].max()
            ),
            "maximum_task_latency_error_ms": float(
                result["task_signal"]["absolute_latency_error_ms"].max()
            ),
            "band_power_preservation": seed_band_summary,
            "evaluation_units": evaluation_units,
            "trial_accounting": result["trial_accounting"],
        }
        write_json(seed_output / "metrics.json", metrics)

        if args.export_brainvision and index == 1:
            brainvision = seed_output / "brainvision"
            brainvision.mkdir()
            vhdr = brainvision / "synthetic_corrupted.vhdr"
            mne.export.export_raw(
                vhdr,
                result["corrupted"],
                fmt="brainvision",
                overwrite=False,
                verbose="ERROR",
            )
            reread = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            metrics["brainvision_round_trip"] = verify_brainvision_round_trip(
                result["corrupted"], reread
            )
            write_json(seed_output / "metrics.json", metrics)

        summaries.append(
            {
                "seed": seed,
                "seed_group": args.seed_group,
                "temporal_precision": result["temporal_detection"]["precision"],
                "temporal_recall": result["temporal_detection"]["recall"],
                "temporal_f1": result["temporal_detection"]["f1"],
                "temporal_true_positive": result["temporal_detection"]["true_positive"],
                "temporal_false_positive": result["temporal_detection"]["false_positive"],
                "temporal_false_negative": result["temporal_detection"]["false_negative"],
                "temporal_true_negative": result["temporal_detection"]["true_negative"],
                "temporal_primary_precision": result["primary_detection"]["precision"],
                "temporal_primary_recall": result["primary_detection"]["recall"],
                "temporal_primary_f1": result["primary_detection"]["f1"],
                "temporal_primary_true_positive": result["primary_detection"]["true_positive"],
                "temporal_primary_false_positive": result["primary_detection"]["false_positive"],
                "temporal_primary_false_negative": result["primary_detection"]["false_negative"],
                "temporal_primary_true_negative": result["primary_detection"]["true_negative"],
                "temporal_stress_detection_rate": result["stress_detection_rate"],
                "line_noise_precision": result["line_noise_detection"]["precision"],
                "line_noise_recall": result["line_noise_detection"]["recall"],
                "line_noise_f1": result["line_noise_detection"]["f1"],
                "maximum_task_amplitude_error_uv": float(
                    result["task_signal"]["absolute_amplitude_error_uv"].max()
                ),
                "maximum_task_latency_error_ms": float(
                    result["task_signal"]["absolute_latency_error_ms"].max()
                ),
                "median_absolute_band_power_error_db": float(
                    classified_band_power["absolute_log_ratio_db"].median()
                ),
                **evaluation_units,
            }
        )

    summary = pd.DataFrame(summaries).sort_values("seed")
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    task_table = pd.concat(task_tables, ignore_index=True)
    band_table = pd.concat(band_tables, ignore_index=True)
    task_table.to_csv(output / "task_signal_all_seeds.csv", index=False)
    band_table.to_csv(output / "band_power_all_seeds.csv", index=False)
    _, band_error_summary = summarize_band_errors(
        band_table,
        oracle_channels,
        corrupted_channels,
    )
    dependency_hashes = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in ("requirements.txt", "constraints-ci.txt")
    }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "dependency_file_sha256": dependency_hashes,
    }
    initial_source_hash = initial_source["sha256"]
    benchmark_config_hash = sha256_file(
        PROJECT_ROOT / "config" / "synthetic_benchmark.json"
    )
    analysis_config_hash = sha256_file(PROJECT_ROOT / "config" / "analysis.json")
    write_json(
        output / "benchmark_summary.json",
        {
            "schema_version": "1",
            "seed_group": args.seed_group,
            "holdout_scope": (
                "Reserved random seeds under the same fixed synthetic channels, "
                "intervals, amplitudes and corruption families."
            ),
            "recordings": len(summary),
            "evaluation_units_per_recording": {
                "temporal_channel_windows": int(
                    summary["temporal_channel_windows"].iloc[0]
                ),
                "primary_channel_windows": int(
                    summary["primary_channel_windows"].iloc[0]
                ),
                "primary_positive_channel_windows": int(
                    summary["primary_positive_channel_windows"].iloc[0]
                ),
                "stress_positive_channel_windows": int(
                    summary["stress_positive_channel_windows"].iloc[0]
                ),
            },
            "temporal_detection": {
                **{
                    column: int(summary[f"temporal_{column}"].sum())
                    for column in (
                        "true_positive", "false_positive", "false_negative", "true_negative"
                    )
                },
                **{
                    column: float(summary[f"temporal_{column}"].mean())
                    for column in ("precision", "recall", "f1")
                },
            },
            "temporal_primary_detection": {
                **{
                    column: int(summary[f"temporal_primary_{column}"].sum())
                    for column in (
                        "true_positive", "false_positive", "false_negative", "true_negative"
                    )
                },
                **{
                    column: float(summary[f"temporal_primary_{column}"].mean())
                    for column in ("precision", "recall", "f1")
                },
            },
            "temporal_stress_detection_rate": float(
                summary["temporal_stress_detection_rate"].mean()
            ),
            "line_noise_detection": {
                column: float(summary[f"line_noise_{column}"].mean())
                for column in ("precision", "recall", "f1")
            },
            "signal_preservation": {
                "maximum_amplitude_error_uv": float(
                    summary["maximum_task_amplitude_error_uv"].max()
                ),
                "maximum_latency_error_ms": float(
                    summary["maximum_task_latency_error_ms"].max()
                ),
                "band_power_absolute_error_db": band_error_summary,
            },
            "runtime": runtime,
            "source_manifest_sha256": initial_source_hash,
            "benchmark_config_sha256": benchmark_config_hash,
            "analysis_config_sha256": analysis_config_hash,
        },
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    detection_labels = [
        "Primary\ntemporal",
        "All temporal",
        "50 Hz",
        "Short-pop\nstress",
    ]
    detection_values = [
        summary["temporal_primary_f1"].mean(),
        summary["temporal_f1"].mean(),
        summary["line_noise_f1"].mean(),
        summary["temporal_stress_detection_rate"].mean(),
    ]
    axes[0].bar(
        detection_labels,
        detection_values,
        color=["#386cb0", "#7fc97f", "#fdc086", "#ef3b2c"],
    )
    axes[0].set(title="Detection", ylabel="F1 or detection rate", ylim=(0, 1.05))
    axes[0].grid(axis="y", alpha=0.2)

    for channel, table in task_table.groupby("channel", sort=False):
        axes[1].plot(
            table["seed"],
            table["absolute_amplitude_error_uv"],
            marker="o",
            label=channel,
        )
    seed_label = f"{args.seed_group.replace('_', ' ').title()} seed"
    axes[1].set(
        title="Task-signal preservation",
        xlabel=seed_label,
        ylabel="Peak amplitude error (µV)",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    group_order = [
        group
        for group in (
            "unaffected",
            "oracle_interpolated",
            "non_oracle_corrupted",
        )
        if group in band_error_summary
    ]
    group_labels = {
        "unaffected": "Unaffected",
        "oracle_interpolated": "Oracle\ninterpolated",
        "non_oracle_corrupted": "Other\ncorrupted",
    }
    for key, label, marker in (
        ("median_absolute_error_db", "Median", "o"),
        ("p95_absolute_error_db", "95th percentile", "s"),
        ("maximum_absolute_error_db", "Maximum", "^"),
    ):
        axes[2].plot(
            [group_labels[group] for group in group_order],
            [band_error_summary[group][key] for group in group_order],
            marker=marker,
            label=label,
        )
    axes[2].set(
        title="Band-power error by channel group",
        ylabel="Absolute error (dB, log scale)",
        yscale="log",
    )
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "benchmark_summary.png", dpi=180)
    plt.close(fig)
    if source_manifest()["sha256"] != initial_source["sha256"]:
        raise RuntimeError("Executable source changed during the benchmark")
    provenance_core = {
        "schema_version": "1",
        "benchmark_config_sha256": benchmark_config_hash,
        "analysis_config_sha256": analysis_config_hash,
        "source_manifest": initial_source,
        "runtime": runtime,
        "seed_group": args.seed_group,
        "scope": (
            "Public synthetic software benchmark; not participant evidence "
            "or validation of universal QC thresholds."
        ),
    }
    provenance_core["core_sha256"] = canonical_sha256(provenance_core)
    write_provenance(output, provenance_core)
    verify_provenance(output)
    output.replace(requested_output)
    atexit.unregister(cleanup)
    print(f"Synthetic benchmark complete: {requested_output}")


if __name__ == "__main__":
    main()
