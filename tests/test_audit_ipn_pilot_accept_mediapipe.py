"""Metadata-only and synthetic tests; never open pilot video or run MediaPipe."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "tools/audit_ipn_pilot_accept_mediapipe.py"
SPEC = importlib.util.spec_from_file_location("ipn_accept_mediapipe", TOOL)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class IpnAcceptMediaPipeTests(unittest.TestCase):
    def test_all_18_decode_pass_and_exactly_seven_other_gesture(self):
        plan = audit.preflight()
        self.assertEqual(len(plan), 7)
        self.assertEqual(len({(p["video_id"], p["label"], p["start_1"], p["end_1"])
                              for p in plan}), 7)
        self.assertTrue(all(p["review"]["subtype_candidate"] == "OTHER_GESTURE" for p in plan))
        self.assertTrue(all(p["review"]["manual_decision"] == "ACCEPT" for p in plan))
        self.assertTrue(all(p["path"].is_file() for p in plan))

    def test_check_only_creates_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(["--check-only"]), 0)
        for path in (audit.REPORT, audit.FRAMES, audit.SUMMARY):
            self.assertFalse(path.exists())

    def test_wrong_accept_subtype_is_rejected_before_mediapipe(self):
        with audit.REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(Counter(row["manual_decision"] for row in rows),
                         {"ACCEPT": 7, "REVIEW": 7, "REJECT": 4})
        next(row for row in rows if row["manual_decision"] == "ACCEPT")["subtype_candidate"] = "UNKNOWN"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=audit.preview.REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(audit, "REVIEW_CSV", path):
                with self.assertRaisesRegex(ValueError, "OTHER_GESTURE"):
                    audit.preflight()

    def test_jz_sampler_and_continuity_policy_are_reused(self):
        from tests.test_wlasl_hand_identity_continuity import frame, hand

        frames = [frame(i, [hand(i * 0.1)]) for i in range(24)]
        result = audit.continuity.classify_frames(frames, prior_valid=24)
        self.assertEqual(result["continuity_decision"], "PASS")
        self.assertEqual(result["proposed_original_frame_indices"],
                         audit.continuity.sample_valid_frame_indices(list(range(24))))
        self.assertEqual(len(result["proposed_original_frame_indices"]), 16)


if __name__ == "__main__":
    unittest.main()
