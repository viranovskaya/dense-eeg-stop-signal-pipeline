"""Validation for manual preprocessing decisions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


BAD_CHANNEL_COLUMNS = [
    "participant_id",
    "channel",
    "decision",
    "reason",
    "reviewer",
    "reviewed_at",
    "evidence_windows",
]


def load_bad_channel_manifest(path: Path) -> pd.DataFrame:
    """Load an explicit review table without silently filling missing evidence."""
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(set(BAD_CHANNEL_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(
            "Bad-channel manifest is missing required columns: "
            + ", ".join(missing)
        )
    manifest = manifest[BAD_CHANNEL_COLUMNS].copy()
    for column in BAD_CHANNEL_COLUMNS:
        manifest[column] = manifest[column].str.strip()
    empty_fields = [
        column
        for column in BAD_CHANNEL_COLUMNS
        if column != "channel" and (manifest[column] == "").any()
    ]
    if empty_fields:
        raise ValueError(
            "Bad-channel manifest contains empty required fields: "
            + ", ".join(empty_fields)
        )
    invalid_decisions = sorted(
        set(manifest["decision"]) - {"interpolate", "keep", "none"}
    )
    if invalid_decisions:
        raise ValueError(f"Invalid bad-channel decisions: {invalid_decisions}")
    none_rows = manifest["decision"] == "none"
    if (none_rows != (manifest["channel"] == "")).any():
        raise ValueError(
            "Use an empty channel only with decision='none'; channel-level "
            "decisions require a channel label"
        )
    participant_counts = manifest.groupby("participant_id")["decision"].agg(
        rows="size", none=lambda values: int((values == "none").sum())
    )
    if ((participant_counts["none"] > 0) & (participant_counts["rows"] > 1)).any():
        raise ValueError(
            "A participant-level 'none' decision cannot be combined with "
            "channel-level decisions"
        )
    if manifest.duplicated(["participant_id", "channel"]).any():
        raise ValueError("Bad-channel manifest contains duplicate participant/channel rows")
    for value in manifest["reviewed_at"].unique():
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"Bad-channel review date must use YYYY-MM-DD: {value!r}"
            ) from error
    return manifest
