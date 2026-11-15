"""Unit Tests for Target Finger Trajectory Recognizer (Stage 9).

Validates:
1. TargetPinkyJAnalyzer (plunge, hook, turn angle, anti-1D drop, anti-static, anti-circular).
2. TargetIndexZAnalyzer (3 strokes, 2 reversals, diagonal drop, anti-horizontal, anti-diagonal, anti-tremor).
3. TrajectoryOnlyJZRecognizer (trajectory morphology classification & margin gating).
4. HandshapeOnlyJZRecognizer (21-landmark handshape classification & OOD rejection).
5. TargetFingerTrajectoryRecognizer (fusion, soft handshape veto, static guard, OOD protection).
"""

from __future__ import annotations

import unittest
import numpy as np

from app.inference.dynamic_jz.target_trajectory_recognizer import (
    TargetPinkyJAnalyzer,
    TargetIndexZAnalyzer,
    TrajectoryOnlyJZRecognizer,
    HandshapeOnlyJZRecognizer,
    TargetFingerTrajectoryRecognizer,
)


class TestTargetFingerTrajectoryRecognizer(unittest.TestCase):
    """Test suite for Stage 9 Target Finger Trajectory Recognizer."""

    def setUp(self) -> None:
        self.pinky_analyzer = TargetPinkyJAnalyzer()
        self.index_analyzer = TargetIndexZAnalyzer()
        self.trajectory_only = TrajectoryOnlyJZRecognizer()
        self.handshape_only = HandshapeOnlyJZRecognizer()
        self.combined = TargetFingerTrajectoryRecognizer()

    def _make_clean_j_trajectory(self) -> np.ndarray:
        """Synthetic ideal J trajectory: plunge downward, then hook left and upward."""
        path = np.zeros((16, 2), dtype=np.float64)
        # Frames 0..10: downward plunge
        for t in range(11):
            path[t] = [0.0, 0.20 * t]  # y goes from 0.0 to 2.0
        # Frames 11..15: lateral hook (curve left and slightly upward)
        for t in range(11, 16):
            dt = t - 10
            path[t] = [-0.15 * dt, 2.0 - 0.05 * dt]  # hook left to -0.75, y to 1.75
        return path

    def _make_clean_z_trajectory(self) -> np.ndarray:
        """Synthetic ideal Z trajectory: stroke 1 (right) -> stroke 2 (down-left) -> stroke 3 (right)."""
        path = np.zeros((16, 2), dtype=np.float64)
        # Stroke 1: frames 0..4 (x goes 0 -> 1.0, y=0.0)
        for t in range(5):
            path[t] = [0.25 * t, 0.0]
        # Stroke 2: frames 5..10 (x goes 1.0 -> 0.0, y goes 0.0 -> 1.5)
        for t in range(5, 11):
            dt = t - 4
            path[t] = [1.0 - 0.1667 * dt, 0.25 * dt]
        # Stroke 3: frames 11..15 (x goes 0.0 -> 1.0, y=1.5)
        for t in range(11, 16):
            dt = t - 10
            path[t] = [0.20 * dt, 1.5]
        return path

    def _make_j_sequence(self) -> np.ndarray:
        """Generate a valid (16, 69) sequence with J handshape and J pinky trajectory."""
        seq = np.zeros((16, 69), dtype=np.float64)
        # Handshape: Pinky (lm 20) extended, others curled
        # Pinky tip is lm 20 (cols 60:63), Wrist is lm 0 (0,0,0)
        for t in range(16):
            # Pinky tip extended: distance from wrist = 1.15
            seq[t, 20 * 3 + 1] = -1.15
            # Index tip curled: distance from wrist = 0.40
            seq[t, 8 * 3 + 1] = -0.40
            # Middle tip curled
            seq[t, 12 * 3 + 1] = -0.40
            # Ring tip curled
            seq[t, 16 * 3 + 1] = -0.40
            # MCPs 5, 9, 17 to give palm scale ~ 0.8
            seq[t, 5 * 3] = 0.4; seq[t, 5 * 3 + 1] = -0.7
            seq[t, 9 * 3] = 0.0; seq[t, 9 * 3 + 1] = -0.8
            seq[t, 17 * 3] = -0.4; seq[t, 17 * 3 + 1] = -0.7
        # Trajectory: pinky (cols 67:69)
        seq[:, 67:69] = self._make_clean_j_trajectory()
        return seq

    def _make_z_sequence(self) -> np.ndarray:
        """Generate a valid (16, 69) sequence with Z handshape and Z index trajectory."""
        seq = np.zeros((16, 69), dtype=np.float64)
        # Handshape: Index (lm 8) extended, others curled
        for t in range(16):
            # Index tip extended: distance from wrist = 1.20
            seq[t, 8 * 3 + 1] = -1.20
            # Pinky tip curled: distance from wrist = 0.40
            seq[t, 20 * 3 + 1] = -0.40
            # Middle curled
            seq[t, 12 * 3 + 1] = -0.40
            # Ring curled
            seq[t, 16 * 3 + 1] = -0.40
            # MCPs 5, 9, 17
            seq[t, 5 * 3] = 0.4; seq[t, 5 * 3 + 1] = -0.7
            seq[t, 9 * 3] = 0.0; seq[t, 9 * 3 + 1] = -0.8
            seq[t, 17 * 3] = -0.4; seq[t, 17 * 3 + 1] = -0.7
        # Trajectory: index (cols 65:67)
        seq[:, 65:67] = self._make_clean_z_trajectory()
        return seq

    # --- Pinky Analyzer Tests ---

    def test_pinky_analyzer_clean_j(self) -> None:
        j_path = self._make_clean_j_trajectory()
        score, metrics = self.pinky_analyzer.analyze(j_path)
        self.assertGreater(score, 0.60)
        self.assertGreater(metrics.downward_drop, 1.0)
        self.assertGreater(metrics.hook_lateral, 0.4)
        self.assertEqual(metrics.hook_direction, "LEFT")
        self.assertFalse(metrics.is_1d_downward)
        self.assertFalse(metrics.is_static)
        self.assertFalse(metrics.is_circular_loop)

    def test_pinky_analyzer_anti_1d_drop(self) -> None:
        """Pure vertical downward drop without lateral hook must be vetoed."""
        path = np.zeros((16, 2), dtype=np.float64)
        for t in range(16):
            path[t] = [0.0, 0.15 * t]  # pure vertical
        score, metrics = self.pinky_analyzer.analyze(path)
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics.is_1d_downward)

    def test_pinky_analyzer_anti_static(self) -> None:
        """Static hand hold with tiny path length must be vetoed."""
        path = np.random.normal(0, 0.002, (16, 2))
        score, metrics = self.pinky_analyzer.analyze(path)
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics.is_static)

    def test_pinky_analyzer_anti_circular(self) -> None:
        """Closed circle loop must be vetoed."""
        path = np.zeros((16, 2), dtype=np.float64)
        r = 0.5
        for t in range(16):
            ang = 2 * np.pi * t / 15.0
            path[t] = [r * np.cos(ang), r * np.sin(ang)]
        score, metrics = self.pinky_analyzer.analyze(path)
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics.is_circular_loop)

    # --- Index Analyzer Tests ---

    def test_index_analyzer_clean_z(self) -> None:
        z_path = self._make_clean_z_trajectory()
        score, metrics = self.index_analyzer.analyze(z_path)
        self.assertGreater(score, 0.70)
        self.assertEqual(metrics.reversals_count, 2)
        self.assertTrue(metrics.reversals_valid)
        self.assertGreater(metrics.diagonal_downward, 0.5)
        self.assertFalse(metrics.is_flat_horizontal)
        self.assertFalse(metrics.is_single_diagonal)
        self.assertFalse(metrics.is_tremor)
        self.assertFalse(metrics.is_static)

    def test_index_analyzer_anti_flat_horizontal(self) -> None:
        """Horizontal line swipe without downward diagonal must be vetoed."""
        path = np.zeros((16, 2), dtype=np.float64)
        for t in range(16):
            path[t] = [0.10 * t, 0.0]
        score, metrics = self.index_analyzer.analyze(path)
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics.is_flat_horizontal)

    def test_index_analyzer_anti_single_diagonal(self) -> None:
        """Single diagonal slash without reversals must be vetoed."""
        path = np.zeros((16, 2), dtype=np.float64)
        for t in range(16):
            path[t] = [0.10 * t, 0.10 * t]
        score, metrics = self.index_analyzer.analyze(path)
        self.assertEqual(score, 0.0)
        self.assertTrue(metrics.is_single_diagonal)

    def test_index_analyzer_anti_tremor(self) -> None:
        """High frequency tremor with tiny stroke lengths must be vetoed."""
        path = np.zeros((16, 2), dtype=np.float64)
        for t in range(16):
            path[t] = [0.01 * ((-1) ** t), 0.005 * t]
        score, metrics = self.index_analyzer.analyze(path)
        self.assertEqual(score, 0.0)

    # --- Trajectory-Only Recognizer Tests ---

    def test_trajectory_only_j(self) -> None:
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, 67:69] = self._make_clean_j_trajectory()
        res = self.trajectory_only.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertGreater(res.confidence, 0.50)

    def test_trajectory_only_z(self) -> None:
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, 65:67] = self._make_clean_z_trajectory()
        res = self.trajectory_only.classify(seq)
        self.assertEqual(res.prediction, "Z")
        self.assertGreater(res.confidence, 0.60)

    def test_trajectory_only_static(self) -> None:
        seq = np.zeros((16, 69), dtype=np.float64)
        res = self.trajectory_only.classify(seq)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.decision_path, "STATIC_POSE_REJECTED")

    # --- Handshape-Only Recognizer Tests ---

    def test_handshape_only_j(self) -> None:
        seq = self._make_j_sequence()
        res = self.handshape_only.classify(seq)
        self.assertEqual(res.prediction, "J")

    def test_handshape_only_z(self) -> None:
        seq = self._make_z_sequence()
        res = self.handshape_only.classify(seq)
        self.assertEqual(res.prediction, "Z")

    def test_handshape_only_ood(self) -> None:
        """Uniform landmark values must be rejected as OOD."""
        seq = np.full((16, 69), 0.20, dtype=np.float64)
        res = self.handshape_only.classify(seq)
        self.assertEqual(res.prediction, "UNKNOWN")

    # --- Combined Target Finger Trajectory Recognizer Tests ---

    def test_combined_recognizer_j(self) -> None:
        seq = self._make_j_sequence()
        res = self.combined.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.decision_path, "TARGET_FINGER_J")
        self.assertGreater(res.confidence, 0.60)

    def test_combined_recognizer_z(self) -> None:
        seq = self._make_z_sequence()
        res = self.combined.classify(seq)
        self.assertEqual(res.prediction, "Z")
        self.assertEqual(res.decision_path, "TARGET_FINGER_Z")
        self.assertGreater(res.confidence, 0.60)

    def test_combined_recognizer_handshape_veto_z_on_j_pose(self) -> None:
        """If trajectory looks like Z, but handshape has pinky extended and index curled, veto to UNKNOWN."""
        seq = self._make_j_sequence()  # pinky extended
        seq[:, 67:69] = 0.0  # pinky has NO J motion
        seq[:, 65:67] = self._make_clean_z_trajectory()  # index has Z path
        res = self.combined.classify(seq)
        # Because index is curled (s_hs_z < 0.15), Z score is zeroed out
        # And pinky trajectory is 0, so J score is zeroed out
        self.assertEqual(res.prediction, "UNKNOWN")

    def test_combined_recognizer_static_rejected(self) -> None:
        """Static hand hold with valid handshape must be rejected by anti-static guard."""
        seq = self._make_j_sequence()
        seq[:, 63:69] = 0.0  # zero motion
        res = self.combined.classify(seq)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.decision_path, "STATIC_POSE_REJECTED")

    def test_combined_recognizer_ood_rejected(self) -> None:
        """Synthetic OOD non-hand input must be rejected."""
        seq = np.full((16, 69), 0.20, dtype=np.float64)
        seq[:, 65:67] = self._make_clean_z_trajectory()
        res = self.combined.classify(seq)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.decision_path, "OOD_REJECTED")


if __name__ == "__main__":
    unittest.main()
