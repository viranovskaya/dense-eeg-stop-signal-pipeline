"""Marker normalization and stop-signal trial reconstruction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd


_STIMULUS_PATTERN = re.compile(r"(?:Stimulus/)?S\s*(\d+)", re.IGNORECASE)


def normalize_marker(description: str) -> str:
    """Convert BrainVision/MNE descriptions such as ``Stimulus/S  17`` to ``S17``."""
    match = _STIMULUS_PATTERN.search(str(description))
    if match:
        return f"S{int(match.group(1))}"
    return str(description).strip()


def annotations_to_markers(annotations: Iterable) -> list[dict]:
    """Return time-ordered normalized stimulus markers from MNE annotations."""
    markers: list[dict] = []
    for annotation in annotations:
        marker = normalize_marker(annotation["description"])
        if re.fullmatch(r"S\d+", marker):
            markers.append(
                {
                    "onset_s": float(annotation["onset"]),
                    "duration_s": float(annotation["duration"]),
                    "marker": marker,
                }
            )
    return markers


def classify_trials(markers: list[dict]) -> pd.DataFrame:
    """Reconstruct go and stop outcomes from consecutive task markers.

    The mapping is validated against participant 003's saved condition datasets.
    Rare S7-containing sequences remain unclassified.
    """
    rows: list[dict] = []
    for index, current in enumerate(markers):
        marker = current["marker"]
        following = markers[index + 1] if index + 1 < len(markers) else None

        if marker in {"S17", "S18"}:
            outcome = following["marker"] if following else None
            if outcome == "S6":
                trial_class = "go_correct"
            elif outcome == "S4":
                trial_class = "go_incorrect_or_slow"
            else:
                trial_class = "go_unclassified"
            rows.append(
                {
                    "trial_index": len(rows) + 1,
                    "trial_type": "go",
                    "trial_class": trial_class,
                    "stimulus_code": marker,
                    "stimulus_onset_s": current["onset_s"],
                    "stop_signal_onset_s": None,
                    "outcome_code": outcome,
                    "response_time_ms": (
                        (following["onset_s"] - current["onset_s"]) * 1000
                        if outcome in {"S4", "S6"}
                        else None
                    ),
                    "stop_signal_delay_ms": None,
                }
            )

        elif marker in {"S1", "S2"}:
            if following and following["marker"] == "S19":
                # Inspect the complete post-stop sequence, stopping at the next
                # trial onset. S7 is unresolved in the recovered protocol and
                # must never be counted as successful inhibition.
                post_stop = []
                for later in markers[index + 2 :]:
                    if later["marker"] in {"S17", "S18", "S1", "S2"}:
                        break
                    post_stop.append(later)
                post_stop_codes = {item["marker"] for item in post_stop}
                response = next(
                    (item for item in post_stop if item["marker"] == "S5"),
                    None,
                )
                unresolved = "S7" in post_stop_codes
                if unresolved:
                    trial_class = "stop_unclassified"
                    outcome_code = "S7"
                elif response is not None:
                    trial_class = "stop_failed"
                    outcome_code = "S5"
                else:
                    trial_class = "stop_successful"
                    outcome_code = None
                rows.append(
                    {
                        "trial_index": len(rows) + 1,
                        "trial_type": "stop",
                        "trial_class": trial_class,
                        "stimulus_code": marker,
                        "stimulus_onset_s": current["onset_s"],
                        "stop_signal_onset_s": following["onset_s"],
                        "outcome_code": outcome_code,
                        "response_time_ms": (
                            (response["onset_s"] - current["onset_s"]) * 1000
                            if response is not None and not unresolved
                            else None
                        ),
                        "stop_signal_delay_ms": (
                            following["onset_s"] - current["onset_s"]
                        )
                        * 1000,
                    }
                )
            else:
                rows.append(
                    {
                        "trial_index": len(rows) + 1,
                        "trial_type": "stop",
                        "trial_class": "stop_unclassified",
                        "stimulus_code": marker,
                        "stimulus_onset_s": current["onset_s"],
                        "stop_signal_onset_s": None,
                        "outcome_code": following["marker"] if following else None,
                        "response_time_ms": None,
                        "stop_signal_delay_ms": None,
                    }
                )

    return pd.DataFrame(rows)


def marker_counts(markers: list[dict]) -> pd.DataFrame:
    """Count normalized stimulus markers."""
    counts = Counter(marker["marker"] for marker in markers)
    return pd.DataFrame(
        [
            {"marker": marker, "count": count}
            for marker, count in sorted(
                counts.items(), key=lambda item: int(item[0].removeprefix("S"))
            )
        ]
    )
