"""Utilities for the dense-EEG stop-signal dataset."""

from .events import classify_trials, normalize_marker, reconcile_trials

__all__ = ["classify_trials", "normalize_marker", "reconcile_trials"]
