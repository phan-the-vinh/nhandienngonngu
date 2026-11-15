import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "letters"))
import finalize_static_dataset as freeze


def manual(status="LABEL_CORRECT", scope="WRONG_LABEL_CANDIDATE", alternative=""):
    return {"file": "sample.png", "original_label": "K", "semantic_status": status,
            "review_scope": scope, "semantic_alternative_label": alternative}


def geometry(decision="EXCLUDE_CORRESPONDENCE"):
    return {"file": "sample.png", "decision": decision, "evidence": "Saved overlay misses index tip",
            "saved_overlay": "saved.html", "roi_invalid_due_to_correspondence": True}


def map_row(name, group="same_group"):
    return {"file": name, "label": "K", "group_id": group, "group_type": "EXACT_VARIANT_GROUP",
            "group_size": "2", "filename_group_key": group, "filename_group_size": "2",
            "duplicate_status": "EXACT_VARIANT_GROUP", "nearest_distance": "0.0000123",
            "nearest_outside_filename_group_distance": "0.45"}


class FinalStaticDatasetTests(unittest.TestCase):
    def test_confirmed_label_kept_without_change(self):
        result = freeze.decide([manual()], [])[0]
        self.assertEqual((result["decision"], result["retained_label"], result["label_action"]),
                         ("KEEP", "K", "UNCHANGED"))

    def test_unresolved_manual_semantics_excluded_under_user_policy(self):
        result = freeze.decide([manual("NEEDS_MANUAL_SEMANTIC_REVIEW")], [])[0]
        self.assertEqual(result["decision"], "EXCLUDE_FROM_STATIC_CANONICAL")
        self.assertEqual(result["image_action"], "PRESERVE_ORIGINAL")

    def test_valid_manual_alternative_is_only_relabel_authority(self):
        result = freeze.decide([manual("LABEL_INCORRECT", alternative="T")], [])[0]
        self.assertEqual(result["label_action"], "MANUAL_RELABEL")
        self.assertEqual(result["retained_label"], "T")

    def test_missing_dynamic_or_same_alternative_rejected(self):
        for alternative in ["", "J", "Z", "K", "invalid"]:
            with self.subTest(alternative=alternative), self.assertRaises(ValueError):
                freeze.decide([manual("LABEL_INCORRECT", alternative=alternative)], [])

    def test_no_alternative_guess_for_correct_or_unresolved(self):
        for status in ["LABEL_CORRECT", "NEEDS_MANUAL_SEMANTIC_REVIEW"]:
            with self.subTest(status=status), self.assertRaises(ValueError):
                freeze.decide([manual(status, alternative="T")], [])

    def test_geometry_exclusion_overrides_confirmed_label(self):
        result = freeze.decide([manual()], [geometry()])[0]
        self.assertEqual(result["decision"], "EXCLUDE_FROM_STATIC_CANONICAL")
        self.assertEqual(result["original_label"], "K")
        self.assertEqual(result["label_action"], "UNCHANGED")

    def test_overlapping_reasons_produce_one_excluded_sample(self):
        result = freeze.decide([manual("NEEDS_MANUAL_SEMANTIC_REVIEW")], [geometry()])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["reason"].split("|")), 2)

    def test_missing_blur_requires_separate_disposition(self):
        with self.assertRaises(ValueError):
            freeze.decide([manual("NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE", "MISSING_HAND_BLUR_FOLLOWUP")], [])

    def test_failed_correspondence_resolves_blur_by_exclusion_not_imputation(self):
        result = freeze.decide([manual("NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE", "MISSING_HAND_BLUR_FOLLOWUP")], [geometry()])[0]
        self.assertEqual(result["decision"], "EXCLUDE_FROM_STATIC_CANONICAL")
        self.assertIn("HAND_BLUR_ROI_INVALID_DUE_TO_CORRESPONDENCE", result["reason"])

    def test_valid_correspondence_alone_does_not_pass_missing_blur(self):
        evidence = {**geometry("KEEP_CORRESPONDENCE"), "correspondence_valid": True}
        with self.assertRaises(ValueError):
            freeze.decide([manual("NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE", "MISSING_HAND_BLUR_FOLLOWUP")], [evidence])

    def test_valid_roi_and_policy_evidence_can_resolve_blur(self):
        evidence = {**geometry("KEEP_CORRESPONDENCE"), "correspondence_valid": True,
                    "blur_hand_usable": True, "blur_hand": 123.0, "blur_policy_pass": True}
        result = freeze.decide([manual("NOT_ASSESSED_IN_THIS_SEMANTIC_SCOPE", "MISSING_HAND_BLUR_FOLLOWUP")], [evidence])[0]
        self.assertEqual(result["decision"], "KEEP")

    def test_unknown_manual_status_and_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):
            freeze.decide([manual("SEMANTIC_ORIGINAL_LABEL_SUPPORTED")], [])
        with self.assertRaises(ValueError):
            freeze.decide([manual(), manual()], [])

    def test_group_ids_survive_filter_and_group_size_is_current(self):
        original = [map_row("a"), map_row("b")]
        before = copy.deepcopy(original)
        rows, sizes = freeze.make_group_map(original, {"a"}, {"a": "K"})
        self.assertEqual(original, before)
        self.assertEqual(rows[0]["group_id"], "same_group")
        self.assertEqual(rows[0]["group_size"], "1")
        self.assertEqual(rows[0]["filename_group_size"], "1")
        self.assertEqual(rows[0]["group_type"], "UNIQUE")
        self.assertEqual(sizes["same_group"], 1)

    def test_parent_distance_never_presented_as_current_recomputed_distance(self):
        rows, _ = freeze.make_group_map([map_row("a")], {"a"}, {"a": "K"})
        self.assertEqual(rows[0]["nearest_distance"], "")
        self.assertEqual(rows[0]["parent_nearest_distance"], "0.0000123")

    def test_label_change_cannot_silently_break_existing_group_keys(self):
        with self.assertRaises(ValueError):
            freeze.make_group_map([map_row("a"), map_row("b")], {"a", "b"}, {"a": "T", "b": "K"})

    def test_group_membership_must_be_one_to_one_and_complete(self):
        with self.assertRaises(ValueError):
            freeze.make_group_map([map_row("a"), map_row("a")], {"a"}, {"a": "K"})
        with self.assertRaises(ValueError):
            freeze.make_group_map([map_row("a")], {"a", "b"}, {"a": "K", "b": "K"})


if __name__ == "__main__":
    unittest.main()
