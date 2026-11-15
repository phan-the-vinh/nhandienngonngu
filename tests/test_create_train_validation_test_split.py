import unittest
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "letters"))
import create_train_validation_test_split as split


class TrainValidationTestSplitTests(unittest.TestCase):
    def test_target_matrix_preserves_rows_and_exact_global_targets(self):
        counts = {label: 100 + index for index, label in enumerate("ABCD")}
        matrix, global_targets = split.target_matrix(counts)
        self.assertEqual(sum(global_targets), sum(counts.values()))
        self.assertEqual(tuple(sum(matrix[label][i] for label in matrix) for i in range(3)), global_targets)
        for label in counts:
            self.assertEqual(sum(matrix[label]), counts[label])

    def test_group_assignment_never_splits_a_group(self):
        groups = {"A": [("a1", 5), ("a2", 3), ("a3", 2), ("a4", 1), ("a5", 1)],
                  "B": [("b1", 4), ("b2", 3), ("b3", 2), ("b4", 2), ("b5", 1)]}
        targets = {"A": (8, 2, 2), "B": (8, 2, 2)}
        assignments, actual = split.assign_groups(groups, targets, 42)
        self.assertEqual(set(assignments), {group for values in groups.values() for group, _ in values})
        self.assertEqual(sum(sum(values) for values in actual.values()), 24)
        self.assertTrue(all(value in split.SPLITS for value in assignments.values()))

    def test_assignment_is_reproducible(self):
        groups = {"A": [(f"g{i}", (i % 5) + 1) for i in range(100)]}
        targets = {"A": (210, 45, 45)}
        first = split.assign_groups(groups, targets, 42)
        second = split.assign_groups(groups, targets, 42)
        self.assertEqual(first, second)

    def test_seed_changes_membership_not_contract(self):
        groups = {"A": [(f"g{i}", 1) for i in range(30)]}
        targets = {"A": (21, 5, 4)}
        first, first_counts = split.assign_groups(groups, targets, 42)
        second, second_counts = split.assign_groups(groups, targets, 43)
        self.assertNotEqual(first, second)
        self.assertEqual(first_counts, second_counts)

    def test_allocator_can_hit_simple_exact_target(self):
        groups = {"A": [(f"g{i}", 1) for i in range(20)]}
        assignments, actual = split.assign_groups(groups, {"A": (14, 3, 3)}, 42)
        self.assertEqual(actual["A"], (14, 3, 3))
        self.assertEqual(Counter(assignments.values()), Counter({"TRAIN": 14, "VALIDATION": 3, "TEST": 3}))


if __name__ == "__main__":
    unittest.main()
