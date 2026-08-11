"""Dataset-level accounting for completed preprocessing runs."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def build_dataset_aggregate(
    table: pd.DataFrame,
    summaries: dict[str, dict],
) -> dict:
    """Combine recording summaries without hiding participant-level variation."""
    if table.empty:
        raise ValueError("Dataset summary requires at least one completed recording")
    participants = table["participant_id"].astype(str).tolist()
    if set(participants) != set(summaries):
        raise ValueError("Dataset table and recording summaries do not match")

    trial_status_totals: Counter[str] = Counter()
    go_drop_reasons: Counter[str] = Counter()
    stop_drop_reasons: Counter[str] = Counter()
    for summary in summaries.values():
        trial_status_totals.update(
            summary["trial_reconciliation"]["status_counts"]
        )
        go_drop_reasons.update(
            summary["go_epoch_accounting"]["drop_reasons"]
        )
        stop_drop_reasons.update(
            summary["stop_epoch_accounting"]["drop_reasons"]
        )

    metric_columns = {
        "median_filtered_std_uv": (
            "before_median_filtered_std_uv",
            "after_median_filtered_std_uv",
        ),
        "median_filtered_robust_range_uv": (
            "before_median_filtered_robust_range_uv",
            "after_median_filtered_robust_range_uv",
        ),
        "maximum_flat_fraction": (
            "before_maximum_flat_fraction",
            "after_maximum_flat_fraction",
        ),
    }
    before_after = {
        metric: {
            "before_dataset_median": float(np.median(table[before_column])),
            "after_dataset_median": float(np.median(table[after_column])),
        }
        for metric, (before_column, after_column) in metric_columns.items()
    }
    return {
        "recordings": len(table),
        "participants": sorted(participants),
        "all_decision_records_complete": bool(
            table["decision_record_complete"].all()
        ),
        "interpolated_channels_total": int(
            table["interpolated_channel_count"].sum()
        ),
        "detected_trial_starts_total": int(table["detected_trial_starts"].sum()),
        "trial_status_totals": dict(sorted(trial_status_totals.items())),
        "epochs": {
            "go": {
                "proposed": int(table["go_events_proposed"].sum()),
                "retained": int(table["go_epochs_retained"].sum()),
                "dropped": int(table["go_epochs_dropped"].sum()),
                "drop_reasons": dict(sorted(go_drop_reasons.items())),
            },
            "stop": {
                "proposed": int(table["stop_events_proposed"].sum()),
                "retained": int(table["stop_epochs_retained"].sum()),
                "dropped": int(table["stop_epochs_dropped"].sum()),
                "drop_reasons": dict(sorted(stop_drop_reasons.items())),
            },
        },
        "before_after_qc": before_after,
        "interpretation": (
            "Descriptive dataset medians and accounting only; not a global "
            "quality score or evidence of artifact removal"
        ),
    }
