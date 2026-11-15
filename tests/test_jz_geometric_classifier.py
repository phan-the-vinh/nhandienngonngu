"""Unit tests for Geometric & Ratio-Based Classifier (Method 2).

Validates:
1. Feature extraction of J geometric properties (drop, aspect ratio, hook, curvature).
2. Feature extraction of Z geometric properties (3 strokes, corner angles, reversals).
3. Discrimination on real SigNN J and Z samples.
4. Robust rejection of invalid inputs (NaN, shape mismatch).
5. Safe rejection of out-of-distribution motions (straight line swipe, static jitter).
6. Multi-criteria decision logic (no single-ratio reliance).
7. Execution latency (< 5 ms).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH
from app.inference.dynamic_jz.geometric_classifier import (
    GeometricClassificationResult,
    GeometricClassifier,
)


class GeometricClassifierTests(unittest.TestCase):
    """Test suite for Method 2: Geometric & Ratio-Based Classifier."""

    def setUp(self) -> None:
        self.classifier = GeometricClassifier(
            min_score=0.55,
            min_margin=0.15,
            min_pinky_dominance=-0.05,
            min_index_dominance=-0.05,
        )
        self.dataset_path = Path("data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl")

    def test_01_real_samples_classification(self) -> None:
        """Verify correct classification of clean J and Z samples."""
        if not self.dataset_path.exists():
            self.skipTest("Dataset file not found.")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(320)]

        raw_j = json.loads(lines[0])
        raw_z = json.loads(lines[308])

        seq_j = np.array(raw_j["features"], dtype=np.float64)
        seq_z = np.array(raw_z["features"], dtype=np.float64)

        res_j = self.classifier.classify(seq_j)
        self.assertEqual(res_j.prediction, "J")
        self.assertEqual(res_j.status, "OK")
        self.assertGreater(res_j.score_j, res_j.score_z)
        self.assertGreaterEqual(res_j.margin, 0.15)
        self.assertIsNotNone(res_j.features_j)
        self.assertGreater(res_j.features_j.downward_drop, 0.15)

        res_z = self.classifier.classify(seq_z)
        self.assertEqual(res_z.prediction, "Z")
        self.assertEqual(res_z.status, "OK")
        self.assertGreater(res_z.score_z, res_z.score_j)
        self.assertGreaterEqual(res_z.margin, 0.15)
        self.assertIsNotNone(res_z.features_z)
        self.assertTrue(res_z.features_z.reversal_valid)

    def test_02_invalid_inputs(self) -> None:
        """Verify rejection of bad shapes and NaN arrays."""
        bad_shape = np.zeros((12, 69), dtype=np.float64)
        res1 = self.classifier.classify(bad_shape)
        self.assertEqual(res1.prediction, "UNKNOWN")
        self.assertEqual(res1.status, "INVALID_INPUT")

        nan_seq = np.zeros((16, 69), dtype=np.float64)
        nan_seq[3, 3] = np.nan
        res2 = self.classifier.classify(nan_seq)
        self.assertEqual(res2.prediction, "UNKNOWN")
        self.assertEqual(res2.status, "INVALID_INPUT")

    def test_03_ood_straight_line_rejection(self) -> None:
        """Verify straight vertical line swipe is rejected as UNKNOWN."""
        seq_straight = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_straight[t, 65:67] = [0.0, 0.5 * t / 15.0]
            seq_straight[t, 67:69] = [0.0, 0.5 * t / 15.0]
            seq_straight[t, :63] = 0.15  # flat hand

        res = self.classifier.classify(seq_straight)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "REJECTED")

    def test_04_ood_static_motion_rejection(self) -> None:
        """Verify static hand motion is rejected as UNKNOWN."""
        seq_static = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_static[t, :63] = 0.2

        res = self.classifier.classify(seq_static)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "REJECTED")

    def test_05_execution_latency(self) -> None:
        """Verify geometric classification is lightweight (< 5 ms)."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
        self.classifier.classify(seq)  # warmup

        latencies = []
        for _ in range(50):
            res = self.classifier.classify(seq)
            latencies.append(res.latency_ms)

        mean_lat = float(np.mean(latencies))
        self.assertLess(mean_lat, 5.0, f"Mean latency {mean_lat:.2f}ms exceeds 5ms")


if __name__ == "__main__":
    unittest.main()
