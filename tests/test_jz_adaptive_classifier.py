"""Unit tests for Stage 8 Adaptive Motion Recognition Classifier."""

from __future__ import annotations

import unittest
import numpy as np

from app.inference.dynamic_jz.adaptive_classifier import (
    AdaptiveMotionJZClassifier,
    AdaptiveClassificationResult,
)
from tests.test_jz_fusion_classifier import _build_synthetic_sample


class TestAdaptiveMotionJZClassifier(unittest.TestCase):
    """Test suite verifying adaptive joint evidence scoring, anti-static guards, and robustness."""

    def setUp(self) -> None:
        self.classifier = AdaptiveMotionJZClassifier()

    def test_invalid_and_nan_inputs(self) -> None:
        """Classifier must gracefully handle malformed or non-finite inputs."""
        res_shape = self.classifier.classify(np.zeros((12, 69)))
        self.assertEqual(res_shape.prediction, "UNKNOWN")
        self.assertEqual(res_shape.decision_path, "INVALID_INPUT")

        nan_seq = np.zeros((16, 69))
        nan_seq[3, 8] = np.nan
        res_nan = self.classifier.classify(nan_seq)
        self.assertEqual(res_nan.prediction, "UNKNOWN")
        self.assertEqual(res_nan.decision_path, "INVALID_INPUT")

    def test_clean_j_adaptive(self) -> None:
        """Pinky handshape with downward hook trajectory must yield J."""
        seq = _build_synthetic_sample(handshape_type="J", trajectory_type="J_HOOK")
        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.decision_path, "ADAPTIVE_J")
        self.assertGreaterEqual(res.confidence, 0.45)
        self.assertIsNotNone(res.metrics)
        self.assertFalse(res.metrics.is_static_pose)

    def test_clean_z_adaptive(self) -> None:
        """Index handshape with 3-stroke zigzag trajectory must yield Z."""
        seq = _build_synthetic_sample(handshape_type="Z", trajectory_type="Z_ZIGZAG")
        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "Z")
        self.assertEqual(res.decision_path, "ADAPTIVE_Z")
        self.assertGreaterEqual(res.confidence, 0.45)
        self.assertIsNotNone(res.metrics)
        self.assertFalse(res.metrics.is_static_pose)

    def test_static_pose_rejected(self) -> None:
        """Holding a static hand pose (no motion) must be rejected as STATIC_POSE (not misclassified)."""
        seq_j_static = _build_synthetic_sample(handshape_type="J", trajectory_type="STATIC")
        res_j = self.classifier.classify(seq_j_static)
        self.assertEqual(res_j.prediction, "UNKNOWN")
        self.assertEqual(res_j.decision_path, "STATIC_POSE_REJECTED")
        self.assertTrue(res_j.metrics.is_static_pose)

        seq_z_static = _build_synthetic_sample(handshape_type="Z", trajectory_type="STATIC")
        res_z = self.classifier.classify(seq_z_static)
        self.assertEqual(res_z.prediction, "UNKNOWN")
        self.assertEqual(res_z.decision_path, "STATIC_POSE_REJECTED")
        self.assertTrue(res_z.metrics.is_static_pose)

    def test_handshape_conflict_rejected(self) -> None:
        """Conflicting handshape and trajectory must be rejected as UNKNOWN."""
        seq_z_j = _build_synthetic_sample(handshape_type="Z", trajectory_type="J_HOOK")
        res_z_j = self.classifier.classify(seq_z_j)
        self.assertEqual(res_z_j.prediction, "UNKNOWN")

        seq_j_z = _build_synthetic_sample(handshape_type="J", trajectory_type="Z_ZIGZAG")
        res_j_z = self.classifier.classify(seq_j_z)
        self.assertEqual(res_j_z.prediction, "UNKNOWN")

    def test_trajectory_without_valid_handshape(self) -> None:
        """Valid trajectory with curled fist or flat hand must be rejected as UNKNOWN."""
        seq_fist = _build_synthetic_sample(handshape_type="FIST", trajectory_type="J_HOOK")
        res_fist = self.classifier.classify(seq_fist)
        self.assertEqual(res_fist.prediction, "UNKNOWN")

        seq_flat = _build_synthetic_sample(handshape_type="FLAT", trajectory_type="Z_ZIGZAG")
        res_flat = self.classifier.classify(seq_flat)
        self.assertEqual(res_flat.prediction, "UNKNOWN")

    def test_ood_negative_controls_rejection(self) -> None:
        """Circular motions and straight drops without hook must be rejected as UNKNOWN."""
        seq_circle = _build_synthetic_sample(handshape_type="J", trajectory_type="CIRCLE")
        res_circle = self.classifier.classify(seq_circle)
        self.assertEqual(res_circle.prediction, "UNKNOWN")

        seq_down = _build_synthetic_sample(handshape_type="J", trajectory_type="LINE_DOWN")
        res_down = self.classifier.classify(seq_down)
        self.assertEqual(res_down.prediction, "UNKNOWN")

    def test_config_a_parameter_independence(self) -> None:
        """Config A classifier with z_diagonal_threshold=0.003 should maintain baseline default at 0.015."""
        cls_a = AdaptiveMotionJZClassifier(z_diagonal_threshold=0.003)
        self.assertEqual(cls_a.z_diag_thresh, 0.003)
        cls_base = AdaptiveMotionJZClassifier()
        self.assertEqual(cls_base.z_diag_thresh, 0.015)


if __name__ == "__main__":
    unittest.main()
