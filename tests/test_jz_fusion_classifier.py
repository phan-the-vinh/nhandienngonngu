"""Unit tests for Stage 7 Handshape + Trajectory Fusion Classifier."""

from __future__ import annotations

import unittest
import numpy as np

from app.inference.dynamic_jz.fusion_classifier import (
    HandshapeBranch,
    TrajectoryBranch,
    HandshapeTrajectoryFusionClassifier,
    FusionClassificationResult,
)


def _build_synthetic_sample(
    handshape_type: str,     # 'J', 'Z', 'FLAT', 'FIST'
    trajectory_type: str,    # 'J_HOOK', 'Z_ZIGZAG', 'STATIC', 'LINE_DOWN', 'CIRCLE'
) -> np.ndarray:
    """Build a synthetic (16, 69) sequence with controlled handshape and trajectory."""
    seq = np.zeros((16, 69), dtype=np.float64)
    lms = np.zeros((16, 21, 3), dtype=np.float64)

    # Base palm landmarks (0: wrist, 5: index mcp, 9: middle mcp, 17: pinky mcp)
    lms[:, 0, :] = [0.0, 0.0, 0.0]
    lms[:, 5, :] = [-0.10, 0.35, 0.0]
    lms[:, 9, :] = [0.00, 0.38, 0.0]
    lms[:, 17, :] = [0.12, 0.32, 0.0]
    palm_scale = 0.35

    # Configure handshape landmarks
    for t in range(16):
        if handshape_type == "J":
            # Pinky extended (tip 20 high/long), index, middle, ring curled
            lms[t, 20, :] = [0.15, 0.55, 0.0]   # pinky ext ~ 1.57 / palm
            lms[t, 8, :] = [-0.10, 0.38, 0.0]   # index curled ~ 1.08 / palm
            lms[t, 12, :] = [0.00, 0.39, 0.0]   # middle curled
            lms[t, 16, :] = [0.08, 0.36, 0.0]   # ring curled
            lms[t, 4, :] = [-0.12, 0.25, 0.0]   # thumb folded
        elif handshape_type == "Z":
            # Index extended (tip 8 high/long), pinky, middle, ring curled
            lms[t, 8, :] = [-0.10, 0.60, 0.0]   # index ext ~ 1.71 / palm
            lms[t, 20, :] = [0.12, 0.28, 0.0]   # pinky curled ~ 0.80 / palm
            lms[t, 12, :] = [0.00, 0.39, 0.0]   # middle curled
            lms[t, 16, :] = [0.08, 0.36, 0.0]   # ring curled
            lms[t, 4, :] = [-0.05, 0.28, 0.0]   # thumb folded
        elif handshape_type == "FLAT":
            # All fingers uniform/flat (degenerate / OOD)
            for idx in [4, 8, 12, 16, 20]:
                lms[t, idx, :] = [0.0, 0.40, 0.0]
        elif handshape_type == "FIST":
            # All fingers curled tightly
            for idx in [4, 8, 12, 16, 20]:
                lms[t, idx, :] = [0.0, 0.25, 0.0]

    seq[:, :63] = lms.reshape((16, 63))

    # Configure trajectories (cols 65:67 for index, 67:69 for pinky)
    if trajectory_type == "J_HOOK":
        # Downward drop from t=0 to 10, lateral hook from t=10 to 15
        for t in range(16):
            if t <= 10:
                y = 0.80 * (t / 10.0)
                x = 0.0
            else:
                y = 0.80 - 0.15 * ((t - 10) / 5.0)
                x = 0.35 * ((t - 10) / 5.0)
            seq[t, 67:69] = [x, y]
            seq[t, 65:67] = [x * 0.5, y * 0.5]

    elif trajectory_type == "Z_ZIGZAG":
        # Stroke 1: horizontal right (t=0..5)
        # Stroke 2: diagonal down-left (t=5..10)
        # Stroke 3: horizontal right (t=10..15)
        for t in range(16):
            if t <= 5:
                x = 0.50 * (t / 5.0)
                y = 0.0
            elif t <= 10:
                frac = (t - 5) / 5.0
                x = 0.50 - 0.50 * frac
                y = 0.45 * frac
            else:
                frac = (t - 10) / 5.0
                x = 0.50 * frac
                y = 0.45
            seq[t, 65:67] = [x, y]
            seq[t, 67:69] = [x * 0.5, y * 0.5]

    elif trajectory_type == "STATIC":
        # Tiny micro-jitter around zero (< 0.02)
        seq[:, 65:67] = np.zeros((16, 2))
        seq[:, 67:69] = np.zeros((16, 2))

    elif trajectory_type == "LINE_DOWN":
        # Straight vertical drop without hook
        for t in range(16):
            y = 0.80 * (t / 15.0)
            seq[t, 65:67] = [0.0, y]
            seq[t, 67:69] = [0.0, y]

    elif trajectory_type == "CIRCLE":
        # Circular motion loop
        for t in range(16):
            ang = 2 * np.pi * t / 15.0
            x = 0.35 * np.cos(ang)
            y = 0.35 * np.sin(ang)
            seq[t, 65:67] = [x, y]
            seq[t, 67:69] = [x, y]

    return seq


class TestHandshapeTrajectoryFusionClassifier(unittest.TestCase):
    """Test suite verifying correctness, robustness, and fusion logic of Stage 7."""

    def setUp(self) -> None:
        self.fusion = HandshapeTrajectoryFusionClassifier()

    def test_invalid_and_nan_inputs(self) -> None:
        """Classifier must gracefully reject wrong dimensions or non-finite inputs."""
        res_shape = self.fusion.classify(np.zeros((10, 69)))
        self.assertEqual(res_shape.prediction, "UNKNOWN")
        self.assertEqual(res_shape.decision_path, "INVALID_INPUT")

        nan_seq = np.zeros((16, 69))
        nan_seq[5, 10] = np.nan
        res_nan = self.fusion.classify(nan_seq)
        self.assertEqual(res_nan.prediction, "UNKNOWN")
        self.assertEqual(res_nan.decision_path, "INVALID_INPUT")

    def test_clean_j_fusion(self) -> None:
        """Handshape J + Trajectory J must cleanly fuse to J with high confidence."""
        seq = _build_synthetic_sample(handshape_type="J", trajectory_type="J_HOOK")
        res = self.fusion.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.decision_path, "AGREED_J")
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertEqual(res.primary_rejection_source, "NONE")
        self.assertEqual(res.handshape_result.prediction, "J")
        self.assertEqual(res.trajectory_result.prediction, "J")

    def test_clean_z_fusion(self) -> None:
        """Handshape Z + Trajectory Z must cleanly fuse to Z with high confidence."""
        seq = _build_synthetic_sample(handshape_type="Z", trajectory_type="Z_ZIGZAG")
        res = self.fusion.classify(seq)
        self.assertEqual(res.prediction, "Z")
        self.assertEqual(res.decision_path, "AGREED_Z")
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertEqual(res.primary_rejection_source, "NONE")
        self.assertEqual(res.handshape_result.prediction, "Z")
        self.assertEqual(res.trajectory_result.prediction, "Z")

    def test_conflict_rejected(self) -> None:
        """Handshape Z with Trajectory J (or vice-versa) must be rejected as UNKNOWN."""
        # Index handshape drawing J hook
        seq_z_j = _build_synthetic_sample(handshape_type="Z", trajectory_type="J_HOOK")
        res_z_j = self.fusion.classify(seq_z_j)
        self.assertEqual(res_z_j.prediction, "UNKNOWN")
        self.assertEqual(res_z_j.primary_rejection_source, "CONFLICT")
        self.assertEqual(res_z_j.decision_path, "CONFLICT_REJECTED")

        # Pinky handshape drawing Z zigzag
        seq_j_z = _build_synthetic_sample(handshape_type="J", trajectory_type="Z_ZIGZAG")
        res_j_z = self.fusion.classify(seq_j_z)
        self.assertEqual(res_j_z.prediction, "UNKNOWN")
        self.assertEqual(res_j_z.primary_rejection_source, "CONFLICT")
        self.assertEqual(res_j_z.decision_path, "CONFLICT_REJECTED")

    def test_static_pose_rejected(self) -> None:
        """Valid handshape with zero motion must be rejected as STATIC_POSE (not misclassified)."""
        seq_j_static = _build_synthetic_sample(handshape_type="J", trajectory_type="STATIC")
        res_j = self.fusion.classify(seq_j_static)
        self.assertEqual(res_j.prediction, "UNKNOWN")
        self.assertEqual(res_j.decision_path, "STATIC_POSE_REJECTED")
        self.assertEqual(res_j.primary_rejection_source, "TRAJECTORY")
        self.assertEqual(res_j.handshape_result.prediction, "J")
        self.assertEqual(res_j.trajectory_result.status, "STATIC_POSE")

        seq_z_static = _build_synthetic_sample(handshape_type="Z", trajectory_type="STATIC")
        res_z = self.fusion.classify(seq_z_static)
        self.assertEqual(res_z.prediction, "UNKNOWN")
        self.assertEqual(res_z.decision_path, "STATIC_POSE_REJECTED")
        self.assertEqual(res_z.primary_rejection_source, "TRAJECTORY")
        self.assertEqual(res_z.handshape_result.prediction, "Z")
        self.assertEqual(res_z.trajectory_result.status, "STATIC_POSE")

    def test_trajectory_without_valid_handshape(self) -> None:
        """Valid trajectory with degenerate/fist handshape must be rejected as HANDSHAPE_REJECTED."""
        seq_fist_j = _build_synthetic_sample(handshape_type="FIST", trajectory_type="J_HOOK")
        res_fist_j = self.fusion.classify(seq_fist_j)
        self.assertEqual(res_fist_j.prediction, "UNKNOWN")
        self.assertEqual(res_fist_j.decision_path, "HANDSHAPE_REJECTED")
        self.assertEqual(res_fist_j.primary_rejection_source, "HANDSHAPE")

        seq_flat_z = _build_synthetic_sample(handshape_type="FLAT", trajectory_type="Z_ZIGZAG")
        res_flat_z = self.fusion.classify(seq_flat_z)
        self.assertEqual(res_flat_z.prediction, "UNKNOWN")
        self.assertEqual(res_flat_z.primary_rejection_source, "HANDSHAPE")

    def test_negative_controls_rejection(self) -> None:
        """Circular motions and straight drops without hooks must be rejected as UNKNOWN."""
        seq_circle = _build_synthetic_sample(handshape_type="J", trajectory_type="CIRCLE")
        res_circle = self.fusion.classify(seq_circle)
        self.assertEqual(res_circle.prediction, "UNKNOWN")

        seq_down = _build_synthetic_sample(handshape_type="J", trajectory_type="LINE_DOWN")
        res_down = self.fusion.classify(seq_down)
        self.assertEqual(res_down.prediction, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
