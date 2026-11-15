import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "letters"))
import train_compare_mlp_validation as training


class MLPValidationTrainingTests(unittest.TestCase):
    def test_feature_label_contract(self):
        self.assertEqual(training.FEATURES, [f"f{i}" for i in range(1, 64)])
        self.assertEqual(len(training.LABELS), 24)
        self.assertNotIn("J", training.LABELS)
        self.assertNotIn("Z", training.LABELS)

    def test_native_sample_weight_supported(self):
        self.assertIn("sample_weight", inspect.signature(MLPClassifier.fit).parameters)
        self.assertIn("sample_weight", inspect.signature(MLPClassifier.partial_fit).parameters)

    def test_model_uses_fixed_comparable_configuration(self):
        first, second = training.make_model(), training.make_model()
        self.assertEqual(first.get_params(), second.get_params())
        self.assertEqual(first.hidden_layer_sizes, (64, 32))
        self.assertFalse(first.early_stopping)
        self.assertEqual(first.random_state, 42)

    def test_weight_formula_equalizes_classes_and_groups(self):
        labels, groups, table_rows = [], [], []
        for label_id, label in enumerate(training.LABELS):
            labels.extend([label_id] * 4)
            groups.extend([f"{label}-large"] * 3 + [f"{label}-single"])
            table_rows.extend([(f"{label}-large", 3), (f"{label}-single", 1)])
        table = pd.DataFrame(table_rows, columns=["group_id", "group_size"]).set_index("group_id")
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        weights, _, _ = training.build_group_class_weights(labels, groups, table)
        self.assertAlmostEqual(weights.mean(), 1.0)
        class_totals = [weights[labels == label_id].sum() for label_id in range(24)]
        self.assertTrue(np.allclose(class_totals, class_totals[0]))
        for label in training.LABELS:
            self.assertAlmostEqual(weights[groups == f"{label}-large"].sum(),
                                   weights[groups == f"{label}-single"].sum())

    def test_selection_prefers_macro_f1_when_guard_passes(self):
        rows = [{"label": label, "recall": 0.9} for label in training.LABELS]
        evaluations = {
            "baseline": {"metrics": {"macro_f1": 0.90, "minimum_class_recall": 0.9,
                                      "accuracy": 0.91}, "per_class": rows},
            "weighted": {"metrics": {"macro_f1": 0.92, "minimum_class_recall": 0.9,
                                      "accuracy": 0.90}, "per_class": rows}}
        result = training.select_model(evaluations)
        self.assertEqual(result["selected_model"], "weighted")
        self.assertTrue(result["recall_guard_pass"])

    def test_selection_blocks_low_recall_macro_winner(self):
        baseline = [{"label": label, "recall": 0.95} for label in training.LABELS]
        weighted = [{"label": label, "recall": 0.95} for label in training.LABELS]
        weighted[0]["recall"] = 0.70
        evaluations = {
            "baseline": {"metrics": {"macro_f1": 0.90, "minimum_class_recall": 0.95,
                                      "accuracy": 0.90}, "per_class": baseline},
            "weighted": {"metrics": {"macro_f1": 0.91, "minimum_class_recall": 0.70,
                                      "accuracy": 0.91}, "per_class": weighted}}
        result = training.select_model(evaluations)
        self.assertIsNone(result["selected_model"])
        self.assertFalse(result["recall_guard_pass"])


if __name__ == "__main__":
    unittest.main()
