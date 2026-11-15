"""Unit tests for Stage 11 Temporal Deep Learning Models."""

import os
import tempfile
import unittest
import numpy as np
import torch

from app.inference.dynamic_jz.stage11_models import (
    TemporalConvNet,
    BiGRUNet,
    DeepLearningPredictor,
    DeepLearningInferenceResult,
    count_parameters,
    CLASS_NAMES,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)


class TestStage11Models(unittest.TestCase):
    """Test suite for 1D-TCN and Bi-GRU models."""

    def test_parameter_budget_under_50k(self):
        """Verify both 1D-TCN and Bi-GRU strictly adhere to the < 50,000 parameter constraint."""
        tcn = TemporalConvNet(input_dim=69, num_classes=2, hidden_channels=48)
        bigru = BiGRUNet(input_dim=69, hidden_size=48, num_classes=2)

        tcn_params = count_parameters(tcn)
        bigru_params = count_parameters(bigru)

        print(f"\n1D-TCN parameters: {tcn_params:,}")
        print(f"Bi-GRU parameters: {bigru_params:,}")

        self.assertLess(tcn_params, 50000, f"1D-TCN parameter count {tcn_params} exceeds 50,000 budget!")
        self.assertLess(bigru_params, 50000, f"Bi-GRU parameter count {bigru_params} exceeds 50,000 budget!")

    def test_tcn_forward_pass_shapes(self):
        """Test 1D-TCN output tensor dimensions with various batch sizes."""
        tcn = TemporalConvNet(input_dim=69, num_classes=2)
        tcn.eval()

        # Single sample batch
        x1 = torch.randn(1, 16, 69)
        out1 = tcn(x1)
        self.assertEqual(out1.shape, (1, 2))

        # Multi-sample batch
        x32 = torch.randn(32, 16, 69)
        out32 = tcn(x32)
        self.assertEqual(out32.shape, (32, 2))

    def test_bigru_forward_pass_shapes(self):
        """Test Bi-GRU output tensor dimensions with various batch sizes."""
        bigru = BiGRUNet(input_dim=69, hidden_size=48, num_classes=2)
        bigru.eval()

        # Single sample batch
        x1 = torch.randn(1, 16, 69)
        out1 = bigru(x1)
        self.assertEqual(out1.shape, (1, 2))

        # Multi-sample batch
        x32 = torch.randn(32, 16, 69)
        out32 = bigru(x32)
        self.assertEqual(out32.shape, (32, 2))

    def test_deep_learning_predictor_contract(self):
        """Verify DeepLearningPredictor returns properly structured inference results."""
        model = TemporalConvNet(input_dim=69, num_classes=2)
        predictor = DeepLearningPredictor(model)

        # Test with 2D array (16, 69)
        seq_2d = np.random.randn(16, 69).astype(np.float32)
        res_2d = predictor.predict(seq_2d)

        self.assertIsInstance(res_2d, DeepLearningInferenceResult)
        self.assertIn(res_2d.prediction, ["J", "Z"])
        self.assertTrue(0.0 <= res_2d.confidence <= 1.0)
        self.assertTrue(0.0 <= res_2d.prob_j <= 1.0)
        self.assertTrue(0.0 <= res_2d.prob_z <= 1.0)
        self.assertAlmostEqual(res_2d.prob_j + res_2d.prob_z, 1.0, places=5)
        self.assertAlmostEqual(res_2d.confidence, max(res_2d.prob_j, res_2d.prob_z), places=5)
        self.assertGreaterEqual(res_2d.margin, 0.0)
        self.assertGreaterEqual(res_2d.latency_ms, 0.0)

        # Test with 3D array (1, 16, 69)
        seq_3d = np.random.randn(1, 16, 69).astype(np.float32)
        res_3d = predictor.predict(seq_3d)
        self.assertIn(res_3d.prediction, ["J", "Z"])

    def test_checkpoint_save_and_reload(self):
        """Verify that saving model state and reloading through predictor reproduces identical logits."""
        model = BiGRUNet(input_dim=69, hidden_size=48, num_classes=2)
        model.eval()

        seq = np.random.randn(16, 69).astype(np.float32)
        pred_orig = DeepLearningPredictor(model).predict(seq)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save(model.state_dict(), tmp_path)
            new_model = BiGRUNet(input_dim=69, hidden_size=48, num_classes=2)
            pred_loaded = DeepLearningPredictor(new_model, checkpoint_path=tmp_path).predict(seq)

            self.assertEqual(pred_loaded.prediction, pred_orig.prediction)
            self.assertAlmostEqual(pred_loaded.confidence, pred_orig.confidence, places=5)
            self.assertAlmostEqual(pred_loaded.prob_j, pred_orig.prob_j, places=5)
            self.assertAlmostEqual(pred_loaded.prob_z, pred_orig.prob_z, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_reproducibility_with_seed(self):
        """Verify that fixed seed guarantees identical weights initialization and output."""
        torch.manual_seed(42)
        m1 = TemporalConvNet(input_dim=69, num_classes=2)
        x = torch.randn(2, 16, 69)

        torch.manual_seed(42)
        m2 = TemporalConvNet(input_dim=69, num_classes=2)

        m1.eval()
        m2.eval()
        out1 = m1(x)
        out2 = m2(x)
        self.assertTrue(torch.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()
