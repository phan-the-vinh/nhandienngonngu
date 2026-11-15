"""Metadata-only tests; no pilot AVI is opened."""

import unittest
from unittest.mock import patch

from tools import preview_ipn_pilot_pass_segments as preview


class IpNPilotPreviewTests(unittest.TestCase):
    def test_only_eighteen_pass_segments_selected(self):
        plan = preview.load_plan()
        self.assertEqual(len(plan), 18)
        self.assertEqual({item["video_id"] for item in plan}, preview.EXPECTED_PASS_VIDEOS)
        self.assertEqual(len({(item["video_id"], item["label"], item["start_1"])
                              for item in plan}), 18)

    def test_selected_indices_are_ordered_and_inside_annotation(self):
        for item in preview.load_plan():
            indices = item["indices"]
            self.assertEqual(len(indices), 16)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertEqual(indices[0], item["start_0"])
            self.assertEqual(indices[-1], item["end_0"])

    def test_check_only_never_decodes_or_writes(self):
        with patch.object(preview, "execute", side_effect=AssertionError("execute called")):
            with patch("sys.argv", ["preview_ipn_pilot_pass_segments.py", "--check-only"]):
                self.assertEqual(preview.main(), 0)

    def test_review_schema_and_defaults(self):
        self.assertEqual(preview.REVIEW_FIELDS, (
            "video_id", "label", "segment_start_1based", "segment_end_1based",
            "preview_status", "manual_decision", "subtype_candidate", "jz_risk", "note"))
        self.assertTrue(preview.OUTPUT.is_relative_to(preview.EXPERIMENT))


if __name__ == "__main__":
    unittest.main()
