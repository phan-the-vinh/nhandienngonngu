"""Tests for explicit WLASL OTHER_GESTURE manual review application."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "apply_wlasl_other_gesture_review.py"
SPEC = importlib.util.spec_from_file_location("apply_wlasl_other_gesture_review", TOOL_PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ApplyWlaslReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "review_manifest_v1.csv"
        self.decisions = self.root / "manual_decisions_v1.csv"
        self.v2 = self.root / "review_manifest_v2.csv"
        self.summary = self.root / "review_decision_summary_v1.json"
        self.manifest_rows = [
            {"batch": "batch_001", "video_id": "a", "review_status": "REVIEW_REQUIRED", "review_reason": "preview", "gloss": "ONE"},
            {"batch": "batch_002", "video_id": "b", "review_status": "RANGE_REVIEW_REQUIRED", "review_reason": "long video", "gloss": "TWO"},
        ]
        self.write_csv(self.manifest, ["batch", "video_id", "review_status", "review_reason", "gloss"], self.manifest_rows)
        self.write_csv(self.decisions, review.DECISION_FIELDS, [
            {"batch": "batch_001", "video_id": "a", "decision": "ACCEPT_OTHER_GESTURE", "reason": "Human verified"},
        ])

    @staticmethod
    def write_csv(path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def validate(self):
        with patch.object(review, "EXPECTED_MANIFEST_COUNT", 2):
            return review.validate_inputs(self.manifest, self.decisions)

    def test_literal_decision_and_unlisted_row_preserved(self):
        header, source, decisions = self.validate()
        output, summary = review.build_result(source, decisions)
        self.assertEqual(len(header), 5)
        self.assertEqual(output[0]["review_status"], "ACCEPT_OTHER_GESTURE")
        self.assertEqual(output[0]["manual_reason"], "Human verified")
        self.assertEqual(output[0]["review_reason"], "preview")
        self.assertEqual(output[1]["review_status"], "RANGE_REVIEW_REQUIRED")
        self.assertEqual(output[1]["manual_decision"], "")
        self.assertEqual(summary["without_manual_decision"], 1)

    def test_all_three_manual_decisions_are_copied_verbatim(self):
        for decision in review.VALID_DECISIONS:
            with self.subTest(decision=decision):
                self.write_csv(self.decisions, review.DECISION_FIELDS, [
                    {"batch": "batch_001", "video_id": "a", "decision": decision, "reason": "Human decision"},
                ])
                _, source, decisions = self.validate()
                output, summary = review.build_result(source, decisions)
                self.assertEqual(output[0]["review_status"], decision)
                self.assertEqual(summary["decisions"][decision], 1)

    def test_rejects_unknown_duplicate_invalid_missing_and_wrong_batch(self):
        bad_rows = [
            [{"batch": "batch_001", "video_id": "missing", "decision": "REVIEW", "reason": "x"}],
            [{"batch": "batch_001", "video_id": "a", "decision": "REVIEW", "reason": "x"}] * 2,
            [{"batch": "batch_001", "video_id": "a", "decision": "ACCEPT", "reason": "x"}],
            [{"batch": "batch_001", "video_id": "a", "decision": "REJECT", "reason": ""}],
            [{"batch": "batch_002", "video_id": "a", "decision": "REVIEW", "reason": "x"}],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows):
                self.write_csv(self.decisions, review.DECISION_FIELDS, rows)
                with self.assertRaises(review.ReviewInputError):
                    self.validate()

    def test_rejects_duplicate_manifest_id_and_missing_decision_column(self):
        rows = [self.manifest_rows[0], dict(self.manifest_rows[0])]
        self.write_csv(self.manifest, list(rows[0]), rows)
        with self.assertRaisesRegex(review.ReviewInputError, "Duplicate video_id"):
            self.validate()
        self.write_csv(self.manifest, list(self.manifest_rows[0]), self.manifest_rows)
        self.write_csv(self.decisions, ["batch", "video_id", "decision"], [])
        with self.assertRaisesRegex(review.ReviewInputError, "header"):
            self.validate()

    def test_check_only_does_not_write_and_apply_requires_confirmation(self):
        patches = (
            patch.object(review, "ROOT", self.root),
            patch.object(review, "MANIFEST_V1", self.manifest),
            patch.object(review, "MANUAL_DECISIONS", self.decisions),
            patch.object(review, "MANIFEST_V2", self.v2),
            patch.object(review, "SUMMARY", self.summary),
            patch.object(review, "EXPECTED_MANIFEST_COUNT", 2),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            original_hash = review.sha256(self.manifest)
            with patch("sys.argv", ["tool", "--check-only"]):
                self.assertEqual(review.main(), 0)
            self.assertFalse(self.v2.exists())
            self.assertFalse(self.summary.exists())
            with patch("sys.argv", ["tool", "--apply"]), patch("builtins.input", return_value="n"):
                self.assertEqual(review.main(), 0)
            self.assertFalse(self.v2.exists())
            with patch("sys.argv", ["tool", "--apply"]), patch("builtins.input", return_value="y"):
                self.assertEqual(review.main(), 0)
            self.assertTrue(self.v2.exists())
            self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8"))["manual_decision_rows"], 1)
            self.assertEqual(review.sha256(self.manifest), original_hash)
            with patch("sys.argv", ["tool", "--apply"]), patch("builtins.input") as prompt:
                self.assertEqual(review.main(), 2)
                prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
