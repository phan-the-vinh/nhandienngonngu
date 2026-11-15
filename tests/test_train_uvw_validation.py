import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "letters"))

import train_mlp_v2_validation as common  # noqa: E402
import train_uvw_validation as module  # noqa: E402


class TestTrainUVWValidation(unittest.TestCase):
    def test_variants_exact(self):
        self.assertEqual(module.VARIANTS, {
            "uvw_low": "weight_supplement_low",
            "uvw_medium": "weight_supplement_medium",
        })

    def test_feature_and_label_contract(self):
        self.assertEqual(module.FEATURES, [f"f{i}" for i in range(1, 64)])
        self.assertEqual(len(module.LABELS), 24)
        self.assertNotIn("J", module.LABELS)
        self.assertNotIn("Z", module.LABELS)

    def test_architecture_matches_v1(self):
        model = common.make_model()
        self.assertEqual(model.hidden_layer_sizes, (64, 32))
        self.assertFalse(model.early_stopping)
        self.assertEqual(model.random_state, 42)

    def test_early_stopping_contract(self):
        self.assertEqual(module.HYPERPARAMETERS["external_validation_patience"], 20)
        self.assertEqual(module.HYPERPARAMETERS["external_validation_min_delta_macro_f1"], 0.0001)
        self.assertEqual(module.HYPERPARAMETERS["max_epochs"], 250)

    def test_significant_recall_threshold(self):
        self.assertAlmostEqual(module.SIGNIFICANT_RECALL_DROP, 0.02)
        values = np.asarray([1.0, 0.98])
        self.assertAlmostEqual(float(values[0] - values[1]), module.SIGNIFICANT_RECALL_DROP)


if __name__ == "__main__":
    unittest.main()
