import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


AI_DIR = Path(__file__).resolve().parents[1] / "ai"
sys.path.insert(0, str(AI_DIR))

from phase2_review_promotion import (  # noqa: E402
    apply_quality_gate_to_decisions,
    build_splits,
    make_group_key,
    normalize_required_bool,
    normalize_filename_stem,
    wilson_lower_bound,
)
from regenerate_phase2_quality_gate import promotion_membership_changed  # noqa: E402


class GroupKeyTests(unittest.TestCase):
    def test_only_terminal_variants_are_removed(self):
        self.assertEqual(normalize_filename_stem("A0 (2).jpg"), "a0")
        self.assertEqual(normalize_filename_stem("A0 (9).PNG"), "a0")
        self.assertEqual(normalize_filename_stem("A0 - Copy.jpeg"), "a0")
        self.assertEqual(normalize_filename_stem("A0 (2) - Copy.jpg"), "a0")
        self.assertEqual(normalize_filename_stem("A0 (1).jpg"), "a0 (1)")
        self.assertEqual(normalize_filename_stem("A0 (10).jpg"), "a0 (10)")
        self.assertEqual(normalize_filename_stem("A0 (2) final.jpg"), "a0 (2) final")

    def test_parent_namespace_prevents_basename_collision(self):
        first = make_group_key("A", "internet", r"set_one\A0.jpg")
        variant = make_group_key("A", "internet", r"set_one\A0 (2).jpg")
        independent = make_group_key("A", "internet", r"set_two\A0.jpg")
        self.assertEqual(first, variant)
        self.assertNotEqual(first, independent)

    def test_label_is_part_of_group_key(self):
        self.assertNotEqual(
            make_group_key("A", "internet", r"set\sample.jpg"),
            make_group_key("B", "internet", r"set\sample.jpg"),
        )


class StatisticalTests(unittest.TestCase):
    def test_wilson_requires_real_support_at_perfect_precision(self):
        self.assertLess(wilson_lower_bound(100, 100), 0.99)
        self.assertGreaterEqual(wilson_lower_bound(381, 381), 0.99)
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)


class QualityGateTests(unittest.TestCase):
    def test_required_bool_rejects_missing_and_invalid_strings(self):
        with self.assertRaises(ValueError):
            normalize_required_bool(pd.Series([True, np.nan]), "flag")
        with self.assertRaises(ValueError):
            normalize_required_bool(pd.Series(["False", "not-a-bool"]), "flag")
        parsed = normalize_required_bool(
            pd.Series(["True", "False", True, False]), "flag"
        )
        self.assertEqual(parsed.tolist(), [True, False, True, False])

    def test_quality_flag_blocks_otherwise_valid_promotion(self):
        frame = pd.DataFrame(
            {
                "validated_policy_pass": [True, True],
                "model_label_agreement": [True, True],
                "bad_geometry": [False, False],
                "bad_stability": [False, True],
                "bad_image_quality": [False, False],
            }
        )
        result = apply_quality_gate_to_decisions(frame)
        self.assertEqual(result["quality_gate_pass"].tolist(), [True, False])
        self.assertEqual(
            result["promotion_decision"].tolist(), ["PROMOTED", "KEEP_REVIEW"]
        )

    def test_status_rename_is_not_a_semantic_decision_change(self):
        previous = pd.Series(["PROMOTED", "REVIEW", "PROMOTED"])
        current = pd.Series(["PROMOTED", "KEEP_REVIEW", "KEEP_REVIEW"])
        self.assertEqual(
            promotion_membership_changed(previous, current).tolist(),
            [False, False, True],
        )


class SplitTests(unittest.TestCase):
    def test_group_intersection_is_empty_for_every_fold(self):
        rows = []
        for label in ("A", "B"):
            for group_number in range(10):
                for variant in range(2):
                    rows.append(
                        {
                            "label": label,
                            "group_key": f"{label}-{group_number}",
                            "variant": variant,
                        }
                    )
        frame = pd.DataFrame(rows)
        splits, audit = build_splits(frame, seed=42)
        groups = frame["group_key"].to_numpy()
        self.assertTrue(audit["group_intersection_empty"].all())
        for train_idx, validation_idx in splits:
            self.assertFalse(
                set(groups[train_idx]).intersection(set(groups[validation_idx]))
            )
        validation_rows = np.concatenate([validation for _, validation in splits])
        self.assertEqual(sorted(validation_rows.tolist()), list(range(len(frame))))


if __name__ == "__main__":
    unittest.main()
