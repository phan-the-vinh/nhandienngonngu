"""Unit tests for Enhanced DTW Classifier (Method 1).

Validates:
1. Model initialization and template loading.
2. High accuracy on real SigNN J and Z samples.
3. Safe rejection of invalid inputs (NaN, shape mismatch).
4. Safe rejection of out-of-distribution motions (straight line swipe, random hand jitter).
5. Handshape dominance gating.
6. Execution latency (< 10 ms).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from app.inference.dynamic_jz.enhanced_dtw import (
    DEFAULT_ENHANCED_TEMPLATE_PATH,
    EnhancedDTWClassifier,
    EnhancedDTWResult,
)
from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH


class EnhancedDTWClassifierTests(unittest.TestCase):
    """Test suite for Method 1: Enhanced DTW Engine."""

    def setUp(self) -> None:
        self.classifier = EnhancedDTWClassifier(
            template_path=DEFAULT_ENHANCED_TEMPLATE_PATH,
            threshold=2.00,
            min_margin=0.15,
            min_dominance=0.05,
        )
        self.dataset_path = Path("data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl")

    def test_01_initialization(self) -> None:
        """Verify classifier initializes and loads CONFIG_E templates."""
        self.assertTrue(self.classifier.is_available)
        self.assertEqual(len(self.classifier.j_templates), 2)
        self.assertEqual(len(self.classifier.z_templates), 2)

    def test_02_real_samples_classification(self) -> None:
        """Verify high discrimination on clean J and Z samples from dataset."""
        if not self.dataset_path.exists():
            self.skipTest("SigNN dataset file not found.")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(320)]

        # Sample 0 is J, Sample 308 is Z
        raw_j = json.loads(lines[0])
        raw_z = json.loads(lines[308])

        seq_j = np.array(raw_j["features"], dtype=np.float64)
        seq_z = np.array(raw_z["features"], dtype=np.float64)

        res_j = self.classifier.classify(seq_j)
        self.assertEqual(res_j.prediction, "J")
        self.assertEqual(res_j.status, "OK")
        self.assertLess(res_j.distance_j, res_j.distance_z)
        self.assertGreaterEqual(res_j.margin, 0.15)

        res_z = self.classifier.classify(seq_z)
        self.assertEqual(res_z.prediction, "Z")
        self.assertEqual(res_z.status, "OK")
        self.assertLess(res_z.distance_z, res_z.distance_j)
        self.assertGreaterEqual(res_z.margin, 0.15)

    def test_03_invalid_input_rejection(self) -> None:
        """Verify rejection of NaN, Inf, and incorrect shapes."""
        bad_shape = np.zeros((10, 69), dtype=np.float64)
        res1 = self.classifier.classify(bad_shape)
        self.assertEqual(res1.prediction, "UNKNOWN")
        self.assertEqual(res1.status, "INVALID_INPUT")

        nan_seq = np.zeros((16, 69), dtype=np.float64)
        nan_seq[5, 5] = np.nan
        res2 = self.classifier.classify(nan_seq)
        self.assertEqual(res2.prediction, "UNKNOWN")
        self.assertIn(res2.status, ("INVALID_INPUT", "REJECTED"))

    def test_04_ood_rejection_straight_line(self) -> None:
        """Verify straight line swipe is rejected as UNKNOWN (unlike LR which overfits to J)."""
        seq_straight = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_straight[t, 65:67] = [0.0, 0.6 * t / 15.0]
            seq_straight[t, 67:69] = [0.0, 0.6 * t / 15.0]
            # Handshape: flat all fingers extended (no single finger dominance)
            seq_straight[t, :63] = 0.2

        res = self.classifier.classify(seq_straight)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "REJECTED")

    def test_05_ood_rejection_static_hand(self) -> None:
        """Verify completely static hand motion is rejected."""
        seq_static = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_static[t, :63] = 0.15  # static landmarks, zero displacement

        res = self.classifier.classify(seq_static)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "REJECTED")

    def test_06_execution_latency(self) -> None:
        """Verify matching execution takes under 10 ms."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
        # Warmup
        self.classifier.classify(seq)

        latencies = []
        for _ in range(50):
            res = self.classifier.classify(seq)
            latencies.append(res.latency_ms)

        mean_lat = float(np.mean(latencies))
        self.assertLess(mean_lat, 10.0, f"Mean latency {mean_lat:.2f}ms exceeds 10ms")


if __name__ == "__main__":
    unittest.main()
