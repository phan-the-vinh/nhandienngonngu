import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "letters"))

import train_mlp_v2_validation as module  # noqa: E402


class TestTrainMLPV2(unittest.TestCase):
    def test_static_label_contract(self):
        self.assertEqual(len(module.LABELS), 24)
        self.assertNotIn("J", module.LABELS)
        self.assertNotIn("Z", module.LABELS)

    def test_feature_order(self):
        self.assertEqual(module.FEATURES, [f"f{i}" for i in range(1, 64)])

    def test_architecture_matches_v1(self):
        model = module.make_model()
        self.assertEqual(model.hidden_layer_sizes, (64, 32))
        self.assertFalse(model.early_stopping)
        self.assertEqual(model.random_state, 42)

    def test_required_variants(self):
        self.assertEqual(
            module.VARIANTS,
            {"v2_debug_5pct": "weight_debug_budget_5pct",
             "v2_debug_10pct": "weight_debug_budget_10pct_recommended"},
        )

    def test_confusion_lookup(self):
        matrix = np.arange(24 * 24).reshape(24, 24)
        self.assertEqual(
            module.confusion_value(matrix, "P", "Q"),
            int(matrix[module.LABEL_TO_ID["P"], module.LABEL_TO_ID["Q"]]),
        )


if __name__ == "__main__":
    unittest.main()
