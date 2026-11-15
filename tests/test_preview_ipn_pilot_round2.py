"""Metadata-only checks; these tests never open an AVI or create previews."""

import unittest
from unittest.mock import patch

from tools import preview_ipn_pilot_round2 as preview


class IpNPilotRoundTwoPreviewTests(unittest.TestCase):
    def test_exact_eighteen_annotations_and_d0x_priority(self):
        plan = preview.load_plan()
        self.assertEqual(len(plan), 18)
        self.assertEqual([item["label"] for item in plan[:4]], ["D0X"] * 4)
        self.assertEqual(sum(item["label"] == "D0X" for item in plan), 4)
        self.assertEqual(len({item["video_id"] for item in plan}), 6)
        self.assertEqual(len({(item["video_id"], item["label"], item["start_1"], item["end_1"])
                              for item in plan}), 18)

    def test_frame_indices_are_original_ordered_and_in_bounds(self):
        for item in preview.load_plan():
            indices = item["indices"]
            self.assertEqual(len(indices), 16)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual(indices[0], item["start_1"] - 1)
            self.assertEqual(indices[-1], item["end_1"] - 1)

    def test_review_schema_and_no_auto_decision(self):
        self.assertEqual(preview.pilot.REVIEW_FIELDS, (
            "video_id", "label", "segment_start_1based", "segment_end_1based",
            "preview_status", "manual_decision", "subtype_candidate", "jz_risk", "note"))
        for item in preview.load_plan():
            row = preview.review_row(item, "READY_FOR_MANUAL_REVIEW")
            self.assertEqual(tuple(row), preview.pilot.REVIEW_FIELDS)
            self.assertEqual(row["manual_decision"], "PENDING_USER_REVIEW")
            self.assertEqual(row["subtype_candidate"], "UNKNOWN")
            self.assertEqual(row["jz_risk"], "UNKNOWN")
        self.assertTrue(preview.OUTPUT.is_relative_to(preview.EXPERIMENT))

    def test_check_only_never_executes(self):
        with patch.object(preview, "execute", side_effect=AssertionError("execute called")):
            with patch("sys.argv", ["preview_ipn_pilot_round2.py", "--check-only"]):
                self.assertEqual(preview.main(), 0)


if __name__ == "__main__":
    unittest.main()
