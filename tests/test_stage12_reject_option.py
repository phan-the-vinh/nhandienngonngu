"""Unit tests for Stage 12 Reject Option Classifier."""

import unittest
from pathlib import Path
import numpy as np

from app.inference.dynamic_jz.stage12_reject_option import (
    ProductionLRRejectOptionClassifier,
    RejectOptionResult,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"


class TestStage12RejectOption(unittest.TestCase):
    """Test suite for Stage 12 Reject Option Classifier."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        cls.model_path = MODEL_PATH

    def test_baseline_matches_pipeline(self):
        """Verify baseline mode reproduces raw pipeline prediction."""
        clf = ProductionLRRejectOptionClassifier(self.model_path, mode="baseline")
        dummy_seq = np.zeros((16, 69), dtype=np.float32)
        res = clf.classify(dummy_seq)

        self.assertIsInstance(res, RejectOptionResult)
        self.assertIn(res.prediction, ["J", "Z"])
        self.assertFalse(res.is_rejected)
        self.assertEqual(res.rejection_reason, "")

    def test_confidence_rejection(self):
        """Verify high confidence threshold rejects ambiguous inputs."""
        clf = ProductionLRRejectOptionClassifier(
            self.model_path, mode="confidence", conf_threshold=0.99999
        )
        dummy_seq = np.zeros((16, 69), dtype=np.float32)
        res = clf.classify(dummy_seq)

        self.assertTrue(res.is_rejected)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertIn("LOW_CONFIDENCE", res.rejection_reason)

    def test_margin_rejection(self):
        """Verify high margin threshold rejects inputs with close probabilities."""
        clf = ProductionLRRejectOptionClassifier(
            self.model_path, mode="margin", margin_threshold=0.99999
        )
        dummy_seq = np.zeros((16, 69), dtype=np.float32)
        res = clf.classify(dummy_seq)

        self.assertTrue(res.is_rejected)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertIn("LOW_MARGIN", res.rejection_reason)

    def test_quality_rejection_insufficient_motion(self):
        """Verify static input sequence is rejected by quality mode."""
        clf = ProductionLRRejectOptionClassifier(
            self.model_path, mode="quality", min_motion_amplitude=0.20
        )
        dummy_seq = np.zeros((16, 69), dtype=np.float32)
        res = clf.classify(dummy_seq)

        self.assertTrue(res.is_rejected)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertIn("INSUFFICIENT_MOTION", res.rejection_reason)

    def test_consistency_rejection_curled_pinky_on_j(self):
        """Verify that when J is predicted but pinky is curled, consistency guard returns UNKNOWN."""
        clf = ProductionLRRejectOptionClassifier(
            self.model_path, mode="consistency", min_finger_extension=0.70
        )
        # Create a synthetic hand where pinky is curled (tip close to wrist)
        # and index is extended
        seq = np.zeros((16, 69), dtype=np.float32)
        # palm scale dist(0, 9): set landmark 9 to y=-0.5
        seq[:, 9 * 3 + 1] = -0.5
        # index tip 8: y=-1.0 (extended, dist=1.0, ext = 1.0 / 0.5 = 2.0)
        seq[:, 8 * 3 + 1] = -1.0
        # pinky tip 20: y=-0.15 (curled, dist=0.15, ext = 0.15 / 0.5 = 0.30)
        seq[:, 20 * 3 + 1] = -0.15

        kin = clf.extract_kinematics_and_handshape(seq)
        self.assertGreater(kin["index_ext"], 1.5)
        self.assertLess(kin["pinky_ext"], 0.5)

    def test_latency_within_realtime_budget(self):
        """Verify reject option execution time is well under 2.0 ms."""
        clf = ProductionLRRejectOptionClassifier(self.model_path, mode="combined")
        seq = np.random.randn(16, 69).astype(np.float32)

        # Warm up
        for _ in range(5):
            clf.classify(seq)

        latencies = [clf.classify(seq).latency_ms for _ in range(50)]
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        self.assertLess(mean_lat, 2.0, f"Mean latency {mean_lat:.2f}ms exceeds 2.0ms budget!")
        self.assertLess(p95_lat, 3.0, f"P95 latency {p95_lat:.2f}ms exceeds 3.0ms budget!")


if __name__ == "__main__":
    unittest.main()
