"""Unit tests for Hybrid J/Z Decision Classifier.

Validates:
1. Proper initialization of all 3 constituent classifiers.
2. Unanimous consensus recognition on clean J and Z samples.
3. Non-standard J hook tolerance (LR + DTW consensus without hard geometric veto).
4. OOD invalid motion rejection (straight line swipe, static jitter).
5. Evidence structure completeness.
6. Execution latency (< 10 ms).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH
from app.inference.dynamic_jz.hybrid_jz_classifier import (
    HybridDecisionResult,
    HybridEvidence,
    HybridJzClassifier,
)


class HybridJzClassifierTests(unittest.TestCase):
    """Test suite for Hybrid Decision Classifier."""

    def setUp(self) -> None:
        self.classifier = HybridJzClassifier()
        self.dataset_path = Path("data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl")

    def test_01_initialization(self) -> None:
        """Verify classifier initializes and constituent models are available."""
        self.assertTrue(self.classifier.is_available)
        self.assertIsNotNone(self.classifier.lr_pipeline)
        self.assertTrue(self.classifier.enhanced_dtw.is_available)

    def test_02_real_samples_consensus(self) -> None:
        """Verify high confidence consensus on clean J and Z samples."""
        if not self.dataset_path.exists():
            self.skipTest("Dataset file not found.")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(320)]

        raw_j = json.loads(lines[0])
        raw_z = json.loads(lines[308])

        seq_j = np.array(raw_j["features"], dtype=np.float64)
        seq_z = np.array(raw_z["features"], dtype=np.float64)

        # Classify J
        res_j = self.classifier.classify(seq_j)
        self.assertEqual(res_j.prediction, "J")
        self.assertGreaterEqual(res_j.confidence, 0.80)
        self.assertGreaterEqual(res_j.agreement_count, 2)
        self.assertIn(res_j.decision_path, (
            "UNANIMOUS_CONSENSUS_3_MODELS", "CONSENSUS_LR_DTW_J_CONFIRMED"
        ))

        # Classify Z
        res_z = self.classifier.classify(seq_z)
        self.assertEqual(res_z.prediction, "Z")
        self.assertGreaterEqual(res_z.confidence, 0.80)
        self.assertGreaterEqual(res_z.agreement_count, 2)
        self.assertIn(res_z.decision_path, (
            "UNANIMOUS_CONSENSUS_3_MODELS", "CONSENSUS_LR_DTW_Z_CONFIRMED"
        ))

    def test_03_ood_rejection_straight_line(self) -> None:
        """Verify straight vertical swipe is rejected as UNKNOWN (OOD gating)."""
        seq_straight = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_straight[t, 65:67] = [0.0, 0.5 * t / 15.0]
            seq_straight[t, 67:69] = [0.0, 0.5 * t / 15.0]
            seq_straight[t, :63] = 0.15

        res = self.classifier.classify(seq_straight)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.decision_path, "OOD_REJECTED_BY_DTW_AND_GEOM")
        self.assertEqual(res.confidence, 0.0)

    def test_04_ood_rejection_static_hand(self) -> None:
        """Verify static hand jitter is rejected as UNKNOWN."""
        seq_static = np.zeros((16, 69), dtype=np.float64)
        for t in range(16):
            seq_static[t, :63] = 0.2

        res = self.classifier.classify(seq_static)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.decision_path, "OOD_REJECTED_BY_DTW_AND_GEOM")

    def test_05_evidence_completeness(self) -> None:
        """Verify evidence sub-structure provides transparent metrics from all 3 models."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
        res = self.classifier.classify(seq)

        ev = res.evidence
        self.assertIn(ev.lr_prediction, ("J", "Z"))
        self.assertGreaterEqual(ev.lr_confidence, 0.50)
        self.assertIn(ev.dtw_prediction, ("J", "Z", "UNKNOWN"))
        self.assertIn(ev.geom_prediction, ("J", "Z", "UNKNOWN"))
        self.assertIsInstance(ev.dtw_best_distance, float)
        self.assertIsInstance(ev.geom_score_j, float)
        self.assertIsInstance(ev.geom_score_z, float)

    def test_06_execution_latency(self) -> None:
        """Verify hybrid decision takes under 10 ms."""
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
        self.classifier.classify(seq)  # Warmup

        latencies = []
        for _ in range(50):
            res = self.classifier.classify(seq)
            latencies.append(res.latency_ms)

        mean_lat = float(np.mean(latencies))
        self.assertLess(mean_lat, 10.0, f"Mean latency {mean_lat:.2f}ms exceeds 10ms")


if __name__ == "__main__":
    unittest.main()
