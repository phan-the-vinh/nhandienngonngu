import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
import finalize_preprocessing_minor_fixes as review


class SemanticReviewValidationTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((review.ROOT / "ai/preprocessing_semantic_observations.json").read_text(encoding="utf-8"))

    def test_current_review_matches_standard_asl_and_counts(self):
        review.validate_semantic_review(self.document)
        self.assertEqual(len(review.STATIC_CLASSES), 24)
        self.assertEqual(len(self.document["observations"]), 16)
        self.assertEqual(self.document["semantic_counts"]["NEEDS_MANUAL_SEMANTIC_REVIEW"], 12)

    def test_j_and_z_cannot_be_static_alternatives(self):
        for dynamic in ["J", "Z"]:
            with self.subTest(dynamic=dynamic):
                document = copy.deepcopy(self.document)
                document["observations"][0]["semantic_alternative_label"] = dynamic
                with self.assertRaises(AssertionError):
                    review.validate_semantic_review(document)

    def test_custom_mn_convention_is_rejected(self):
        self.document["alphabet_convention"] = "Custom M/N"
        with self.assertRaises(AssertionError):
            review.validate_semantic_review(self.document)

    def test_resolution_cannot_be_manufactured_without_evidence(self):
        self.document["observations"][0]["disposition"] = "REVIEW_RESOLVED"
        with self.assertRaises(AssertionError):
            review.validate_semantic_review(self.document)

    def test_inconsistent_status_counts_are_rejected(self):
        self.document["semantic_counts"]["SEMANTIC_LABEL_MISMATCH_CANDIDATE"] = 1
        with self.assertRaises(AssertionError):
            review.validate_semantic_review(self.document)

    def test_label_change_is_rejected(self):
        self.document["label_changes"] = ["K to T"]
        with self.assertRaises(AssertionError):
            review.validate_semantic_review(self.document)


if __name__ == "__main__":
    unittest.main()
