import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
import analyze_train_balance as analysis


class TrainBalanceAnalysisTests(unittest.TestCase):
    def test_distribution_uses_group_sizes(self):
        counts = {label: 3 for label in analysis.STATIC_CLASSES}
        groups = []
        for label in analysis.STATIC_CLASSES:
            groups.extend([{"split": "TRAIN", "label": label, "group_size": "2"},
                           {"split": "TRAIN", "label": label, "group_size": "1"}])
        rows = analysis.class_group_distribution(counts, groups)
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["train_groups"] == 2 for row in rows))
        self.assertTrue(all(row["samples_per_group"] == 1.5 for row in rows))

    def test_distribution_rejects_inconsistent_group_sizes(self):
        counts = {label: 3 for label in analysis.STATIC_CLASSES}
        groups = [{"split": "TRAIN", "label": label, "group_size": "2"}
                  for label in analysis.STATIC_CLASSES]
        with self.assertRaises(ValueError):
            analysis.class_group_distribution(counts, groups)

    def test_imbalance_assessment_detects_group_imbalance(self):
        rows = [{"label": "A", "train_samples": 100, "train_groups": 100},
                {"label": "B", "train_samples": 100, "train_groups": 10}]
        result = analysis.imbalance_summary(rows)
        self.assertEqual(result["assessment"], "HIGH_GROUP_IMBALANCE_AND_MATERIAL_SAMPLE_IMBALANCE")
        self.assertEqual(result["group_count"]["max_to_min_ratio"], 10)


if __name__ == "__main__":
    unittest.main()
