"""Reviewed artifact intervals for the fixed ten-recording stop-signal study."""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mne
import pandas as pd

from .provenance import canonical_sha256, sha256_file


INTERVAL_COLUMNS = [
    "participant_id",
    "interval_id",
    "decision",
    "start_s",
    "stop_s",
    "scope",
    "reason",
    "reviewer",
    "reviewed_at",
    "evidence",
]
ALLOWED_SCOPES = {"ica", "epochs", "both"}
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_id(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or len(value) > 64 or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(
            f"{label} must contain 1-64 ASCII letters, digits, underscores or hyphens"
        )
    return value


@dataclass(frozen=True)
class FixedStudyIntervalManifest:
    """Captured and validated decisions for the fixed study."""

    path: Path
    sha256: str
    table: pd.DataFrame

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted(self.table["participant_id"].unique()))

    def verify_unchanged(self) -> None:
        if sha256_file(self.path) != self.sha256:
            raise ValueError("Fixed-study interval manifest changed during the run")

    def participant_rows(self, participant_id: str) -> list[dict]:
        participant_id = _safe_id(participant_id, "Participant ID")
        selected = self.table.loc[
            self.table["participant_id"] == participant_id
        ]
        if selected.empty:
            raise ValueError(
                f"No completed interval review for participant {participant_id}"
            )
        rows: list[dict] = []
        for row in selected.to_dict("records"):
            if row["decision"] == "none":
                row["start_s"] = None
                row["stop_s"] = None
            else:
                row["start_s"] = float(row["start_s"])
                row["stop_s"] = float(row["stop_s"])
            rows.append(row)
        return rows

    def identity(self, participant_id: str) -> dict:
        rows = self.participant_rows(participant_id)
        return {
            "manifest_sha256": self.sha256,
            "participant_decisions": rows,
            "participant_decisions_sha256": canonical_sha256(rows),
        }


def load_fixed_study_interval_manifest(
    path: Path,
    *,
    expected_participants: list[str] | tuple[str, ...] | None = None,
) -> FixedStudyIntervalManifest:
    """Capture and validate the complete interval review table."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Fixed-study interval manifest must be CSV or TSV")
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(
        io.StringIO(data.decode("utf-8")),
        sep=separator,
        dtype=str,
        keep_default_na=False,
    )
    missing = sorted(set(INTERVAL_COLUMNS) - set(table.columns))
    extra = sorted(set(table.columns) - set(INTERVAL_COLUMNS))
    if missing or extra:
        raise ValueError(
            "Fixed-study interval columns must match the declared schema; "
            f"missing={missing}, extra={extra}"
        )
    if table.empty:
        raise ValueError("Fixed-study interval manifest must not be empty")
    table = table[INTERVAL_COLUMNS].copy()
    for column in INTERVAL_COLUMNS:
        table[column] = table[column].str.strip()

    normalized: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for source_row, row in enumerate(table.to_dict("records"), start=2):
        participant = _safe_id(row["participant_id"], "Participant ID")
        interval_id = _safe_id(row["interval_id"], "Interval ID")
        key = (participant, interval_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate interval ID for participant: {key}")
        seen_keys.add(key)
        decision = row["decision"]
        if decision not in {"exclude", "none"}:
            raise ValueError(
                f"Row {source_row} decision must be 'exclude' or 'none'"
            )
        required_text = ("reason", "reviewer", "reviewed_at", "evidence")
        empty = [field for field in required_text if not row[field]]
        if empty:
            raise ValueError(
                f"Row {source_row} has empty review fields: {', '.join(empty)}"
            )
        try:
            date.fromisoformat(row["reviewed_at"])
        except ValueError as error:
            raise ValueError(
                f"Row {source_row} reviewed_at must use YYYY-MM-DD"
            ) from error

        if decision == "none":
            if interval_id != "none" or any(
                row[field] for field in ("start_s", "stop_s", "scope")
            ):
                raise ValueError(
                    "A no-interval completion row must use interval_id='none' "
                    "and blank start_s, stop_s and scope"
                )
            start_s: float | None = None
            stop_s: float | None = None
            scope = ""
        else:
            if interval_id == "none":
                raise ValueError("Excluded intervals cannot use interval_id='none'")
            scope = row["scope"]
            if scope not in ALLOWED_SCOPES:
                raise ValueError(
                    f"Row {source_row} scope must be one of {sorted(ALLOWED_SCOPES)}"
                )
            try:
                start_s = float(row["start_s"])
                stop_s = float(row["stop_s"])
            except ValueError as error:
                raise ValueError(
                    f"Row {source_row} interval bounds must be numeric"
                ) from error
            if (
                not math.isfinite(start_s)
                or not math.isfinite(stop_s)
                or start_s < 0
                or stop_s <= start_s
            ):
                raise ValueError(
                    f"Row {source_row} must have finite 0 <= start_s < stop_s"
                )
        normalized.append(
            {
                "participant_id": participant,
                "interval_id": interval_id,
                "decision": decision,
                "start_s": start_s,
                "stop_s": stop_s,
                "scope": scope,
                "reason": row["reason"],
                "reviewer": row["reviewer"],
                "reviewed_at": row["reviewed_at"],
                "evidence": row["evidence"],
            }
        )

    normalized_table = pd.DataFrame(normalized, columns=INTERVAL_COLUMNS).sort_values(
        ["participant_id", "interval_id"], kind="stable"
    )
    for participant, group in normalized_table.groupby("participant_id", sort=True):
        if (group["decision"] == "none").any() and len(group) != 1:
            raise ValueError(
                f"Participant {participant} cannot mix a no-interval row with exclusions"
            )
        intervals = group.loc[group["decision"] == "exclude"].sort_values(
            ["start_s", "stop_s", "interval_id"], kind="stable"
        )
        previous_stop: float | None = None
        for row in intervals.itertuples(index=False):
            if previous_stop is not None and float(row.start_s) < previous_stop:
                raise ValueError(
                    f"Reviewed intervals overlap for participant {participant}"
                )
            previous_stop = float(row.stop_s)

    if expected_participants is not None:
        expected = {_safe_id(value, "Participant ID") for value in expected_participants}
        observed = set(normalized_table["participant_id"])
        if observed != expected:
            raise ValueError(
                "Fixed-study interval manifest must match participants exactly; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
    return FixedStudyIntervalManifest(
        path=path,
        sha256=digest,
        table=normalized_table.reset_index(drop=True),
    )


def fir_half_support_seconds(
    sfreq: float, filter_hz: tuple[float, float]
) -> float:
    """Return the temporal half-support of the exact zero-phase FIR filter."""
    low_hz, high_hz = filter_hz
    kernel = mne.filter.create_filter(
        None,
        sfreq=float(sfreq),
        l_freq=float(low_hz),
        h_freq=float(high_hz),
        method="fir",
        phase="zero",
        verbose="ERROR",
    )
    return (len(kernel) - 1) / (2.0 * float(sfreq))


def reviewed_interval_application(
    manifest: FixedStudyIntervalManifest,
    *,
    participant_id: str,
    target: str,
    filter_hz: tuple[float, float],
    sfreq: float,
    duration_s: float,
) -> dict:
    """Describe the exact reviewed intervals and FIR-support guard to apply."""
    if target not in {"ica", "epochs"}:
        raise ValueError("Interval application target must be 'ica' or 'epochs'")
    rows = manifest.participant_rows(participant_id)
    guard_s = fir_half_support_seconds(float(sfreq), filter_hz)
    applied: list[dict] = []
    for row in rows:
        if row["decision"] != "exclude" or row["scope"] not in {target, "both"}:
            continue
        start_s = float(row["start_s"])
        stop_s = float(row["stop_s"])
        if stop_s > duration_s:
            raise ValueError(
                f"Interval {row['interval_id']} exceeds recording duration "
                f"({stop_s:g} > {duration_s:g} s)"
            )
        applied_start_s = max(0.0, start_s - guard_s)
        applied_stop_s = min(duration_s, stop_s + guard_s)
        description = f"BAD_fixed_{target}_{row['interval_id']}"
        applied.append(
            {
                "interval_id": row["interval_id"],
                "reviewed_start_s": start_s,
                "reviewed_stop_s": stop_s,
                "applied_start_s": applied_start_s,
                "applied_stop_s": applied_stop_s,
                "scope": row["scope"],
                "reason": row["reason"],
                "annotation": description,
            }
        )
    return {
        "identity": manifest.identity(participant_id),
        "application_target": target,
        "filter_hz": [float(value) for value in filter_hz],
        "fir_half_support_guard_s": guard_s,
        "applied_intervals": applied,
        "complete": True,
        "automatic_decisions": False,
    }


def apply_reviewed_intervals(
    raw: mne.io.BaseRaw,
    manifest: FixedStudyIntervalManifest,
    *,
    participant_id: str,
    target: str,
    filter_hz: tuple[float, float],
) -> dict:
    """Annotate the filter-contaminated support of reviewed exclusions."""
    application = reviewed_interval_application(
        manifest,
        participant_id=participant_id,
        target=target,
        filter_hz=filter_hz,
        sfreq=float(raw.info["sfreq"]),
        duration_s=float(raw.n_times) / float(raw.info["sfreq"]),
    )
    for interval in application["applied_intervals"]:
        raw.annotations.append(
            raw.first_time + interval["applied_start_s"],
            interval["applied_stop_s"] - interval["applied_start_s"],
            interval["annotation"],
        )
    return application
