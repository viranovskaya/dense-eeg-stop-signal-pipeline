"""Tests for explicit bad-channel review records."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunt_eeg.decisions import load_bad_channel_manifest


HEADER = (
    "participant_id,channel,decision,reason,reviewer,reviewed_at,"
    "evidence_windows\n"
)


class BadChannelDecisionTests(unittest.TestCase):
    def test_complete_manifest_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            path.write_text(
                HEADER
                + "003,E42,interpolate,persistent flat trace,reviewer-1,"
                "2026-08-11,0-20;120-140\n",
                encoding="utf-8",
            )

            manifest = load_bad_channel_manifest(path)

        self.assertEqual(manifest.iloc[0]["participant_id"], "003")
        self.assertEqual(manifest.iloc[0]["decision"], "interpolate")

    def test_empty_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            path.write_text(
                HEADER
                + "003,E42,interpolate,,reviewer-1,2026-08-11,0-20\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "empty required fields"):
                load_bad_channel_manifest(path)

    def test_invalid_review_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            path.write_text(
                HEADER
                + "003,E42,keep,transient only,reviewer-1,11-08-2026,0-20\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                load_bad_channel_manifest(path)

    def test_review_with_no_bad_channels_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            path.write_text(
                HEADER
                + "003,,none,no persistent bad channels,reviewer-1,"
                "2026-08-11,0-20;120-140\n",
                encoding="utf-8",
            )

            manifest = load_bad_channel_manifest(path)

        self.assertEqual(manifest.iloc[0]["decision"], "none")
        self.assertEqual(manifest.iloc[0]["channel"], "")

    def test_none_cannot_hide_channel_level_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            path.write_text(
                HEADER
                + "003,,none,no persistent bad channels,reviewer-1,"
                "2026-08-11,0-20;120-140\n"
                + "003,E42,keep,transient only,reviewer-1,"
                "2026-08-11,0-20;120-140\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                load_bad_channel_manifest(path)


if __name__ == "__main__":
    unittest.main()
