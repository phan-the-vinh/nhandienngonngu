"""Unit tests for Ratio-Based J/Z Trajectory Classifier (Stage 6)."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from app.inference.dynamic_jz.ratio_classifier import (
    RatioBasedJZClassifier,
    RatioClassificationResult,
    RatioFeaturesJ,
    RatioFeaturesZ,
)


def _make_handshape(active_finger: str) -> np.ndarray:
    """Generate 16x21x3 landmarks with chosen active extended finger."""
    lms = np.full((16, 21, 3), 0.15, dtype=np.float64)
    # Palm scale reference landmarks (5, 9, 17) with realistic ~0.35 palm scale
    lms[:, 5, :] = [0.35, 0.0, 0.0]
    lms[:, 9, :] = [0.0, -0.35, 0.0]
    lms[:, 17, :] = [-0.35, 0.0, 0.0]

    if active_finger == "pinky":
        # Pinky tip extended (landmark 20)
        lms[:, 20, :] = [-0.25, -0.45, 0.0]  # norm ~ 0.515
        lms[:, 8, :] = [0.05, -0.05, 0.0]   # index curled
    elif active_finger == "index":
        # Index tip extended (landmark 8)
        lms[:, 8, :] = [0.20, -0.45, 0.0]   # norm ~ 0.492
        lms[:, 20, :] = [-0.05, -0.05, 0.0] # pinky curled
    return lms.reshape((16, 63))


class TestRatioBasedJZClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RatioBasedJZClassifier(
            min_confidence=0.45,
            min_margin=0.08,
            min_j_drop=0.25,
            min_j_hook=0.10,
            min_z_span=0.06,
            min_z_diag_drop=0.06,
            min_pinky_ext=0.50,
            min_index_ext=0.50,
        )

    def test_invalid_input_shape_and_nans(self) -> None:
        """Classifier safely returns UNKNOWN on invalid shapes and NaNs."""
        res = self.classifier.classify(np.zeros((10, 69)))
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "INVALID_INPUT")

        bad_arr = np.zeros((16, 69))
        bad_arr[5, 5] = np.nan
        res_nan = self.classifier.classify(bad_arr)
        self.assertEqual(res_nan.prediction, "UNKNOWN")
        self.assertEqual(res_nan.status, "INVALID_INPUT")

    def test_synthetic_clean_j_right_hook(self) -> None:
        """Clean synthetic J with rightward hook is recognized with high confidence."""
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, :63] = _make_handshape("pinky")

        # Pinky trajectory (cols 67:69): vertical drop to frame 11, then rightward hook
        for t in range(12):
            frac = t / 11.0
            seq[t, 67:69] = [0.0, 1.2 * frac]
        for t in range(12, 16):
            frac = (t - 11) / 4.0
            seq[t, 67:69] = [0.40 * frac, 1.2 - 0.15 * frac]

        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.status, "OK")
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertIsNotNone(res.features_j)
        assert res.features_j is not None
        self.assertEqual(res.features_j.hook_direction, "RIGHT")
        self.assertGreaterEqual(res.features_j.downward_drop, 1.0)
        self.assertGreaterEqual(res.features_j.hook_lateral, 0.35)

    def test_synthetic_clean_j_left_hook(self) -> None:
        """J variant with leftward hook is recognized cleanly."""
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, :63] = _make_handshape("pinky")

        for t in range(12):
            frac = t / 11.0
            seq[t, 67:69] = [0.0, 1.0 * frac]
        for t in range(12, 16):
            frac = (t - 11) / 4.0
            seq[t, 67:69] = [-0.35 * frac, 1.0 - 0.10 * frac]

        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.status, "OK")
        self.assertIsNotNone(res.features_j)
        assert res.features_j is not None
        self.assertEqual(res.features_j.hook_direction, "LEFT")

    def test_synthetic_clean_z(self) -> None:
        """Clean synthetic 3-stroke Z is recognized with valid reversals and corner angles."""
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, :63] = _make_handshape("index")

        # Stroke 1 (0..4): (0, 0) -> (0.8, 0)
        for t in range(5):
            frac = t / 4.0
            seq[t, 65:67] = [0.8 * frac, 0.0]
        # Stroke 2 (5..10): (0.8, 0) -> (0.0, 0.8)
        for t in range(5, 11):
            frac = (t - 4) / 6.0
            seq[t, 65:67] = [0.8 * (1.0 - frac), 0.8 * frac]
        # Stroke 3 (11..15): (0.0, 0.8) -> (0.8, 0.8)
        for t in range(11, 16):
            frac = (t - 10) / 5.0
            seq[t, 65:67] = [0.8 * frac, 0.8]

        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "Z")
        self.assertEqual(res.status, "OK")
        self.assertGreaterEqual(res.confidence, 0.50)
        self.assertIsNotNone(res.features_z)
        assert res.features_z is not None
        self.assertTrue(res.features_z.reversals_valid)
        self.assertEqual(res.features_z.reversals_count, 2)
        self.assertGreaterEqual(res.features_z.diagonal_downward, 0.70)

    def test_negative_controls_rejection(self) -> None:
        """Invalid motion patterns are strictly rejected as UNKNOWN."""
        # 1. Straight vertical drop
        seq_down = np.zeros((16, 69))
        seq_down[:, :63] = _make_handshape("pinky")
        for t in range(16):
            seq_down[t, 67:69] = [0.0, 1.2 * (t / 15.0)]
        self.assertEqual(self.classifier.classify(seq_down).prediction, "UNKNOWN")

        # 2. Straight horizontal swipe
        seq_horiz = np.zeros((16, 69))
        seq_horiz[:, :63] = _make_handshape("index")
        for t in range(16):
            seq_horiz[t, 65:67] = [1.0 * (t / 15.0), 0.0]
        self.assertEqual(self.classifier.classify(seq_horiz).prediction, "UNKNOWN")

        # 3. Circular wave
        seq_circ = np.zeros((16, 69))
        seq_circ[:, :63] = _make_handshape("pinky")
        for t in range(16):
            ang = 2 * np.pi * t / 15.0
            seq_circ[t, 67:69] = [0.4 * np.cos(ang), 0.4 * np.sin(ang)]
        self.assertEqual(self.classifier.classify(seq_circ).prediction, "UNKNOWN")

        # 4. Static jitter
        seq_jitter = np.zeros((16, 69))
        seq_jitter[:, :63] = _make_handshape("index")
        for t in range(16):
            seq_jitter[t, 65:67] = np.random.normal(0, 0.005, 2)
        self.assertEqual(self.classifier.classify(seq_jitter).prediction, "UNKNOWN")

    def test_handshape_mismatch_rejection(self) -> None:
        """A perfect trajectory is rejected if handshape does not match active finger."""
        # J motion with index extended (not pinky)
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, :63] = _make_handshape("index")  # wrong finger!
        for t in range(12):
            seq[t, 67:69] = [0.0, 1.2 * (t / 11.0)]
        for t in range(12, 16):
            seq[t, 67:69] = [0.40 * ((t - 11) / 4.0), 1.1]

        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertTrue("PINKY" in res.rejection_reason or "MISMATCH" in res.rejection_reason)

    def test_duplicate_frame_tolerance(self) -> None:
        """MediaPipe duplicate frames (~35%) do not break recognition of clean gestures."""
        seq = np.zeros((16, 69), dtype=np.float64)
        seq[:, :63] = _make_handshape("pinky")

        # J trajectory with freeze frames at t=2, 6, 10
        y_val = 0.0
        for t in range(12):
            if t not in (2, 6, 10):
                y_val += 0.15
            seq[t, 67:69] = [0.0, y_val]
        for t in range(12, 16):
            seq[t, 67:69] = [0.35 * ((t - 11) / 4.0), y_val]

        res = self.classifier.classify(seq)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.status, "OK")


if __name__ == "__main__":
    unittest.main()
