"""Unit tests for Webcam Evaluation Harness and Metrics Calculator (Phase 1).

Covers 15 required mock validation scenarios, verifying that all calculations,
priority resolutions, safety metrics, and observability structures behave correctly
without touching live webcam hardware or modifying production code.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List
import numpy as np

from tools.calculate_webcam_metrics import calculate_metrics, compute_latency_stats
from tools.evaluate_webcam_full_system import (
    determine_primary_outcome,
    generate_planned_trials,
    EvaluationHarness,
)


class TestWebcamHarnessMock(unittest.TestCase):
    """15 Comprehensive Mock Evaluation Scenarios."""

    def test_01_all_trials_success(self):
        """Scenario 1: All trials succeed -> 100% accuracy, Macro-F1 = 1.0."""
        trials = [
            {
                "trial_id": f"TRIAL_{i:03d}",
                "branch": "ABC",
                "ground_truth": ch,
                "commit_text": ch,
                "commit_count": 1,
                "primary_outcome": "SUCCESS",
                "elapsed_time_ms": 3150.0,
            }
            for i, ch in enumerate(["A", "B", "C", "D"])
        ]
        metrics = calculate_metrics(trials)
        self.assertEqual(metrics["status"], "COMPUTED")
        self.assertEqual(metrics["total_trials"], 4)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["character_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["miss_rate"], 0.0)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["false_commit_rate"], 0.0)
        self.assertAlmostEqual(metrics["classification_metrics"]["macro_f1"], 1.0)

    def test_02_wrong_prediction_classification_error(self):
        """Scenario 2: Incorrect prediction committed -> CLASSIFICATION_ERROR and confusion matrix update."""
        outcome, flags = determine_primary_outcome(
            ground_truth="A",
            commits=[("B", 3.2)],
            landmarks_detected=True,
            validator_rejected=False,
            hold_aborted=False,
            raw_prediction="B",
            candidate="B",
            timeout_occurred=False,
        )
        self.assertEqual(outcome, "CLASSIFICATION_ERROR")
        self.assertIn("RAW_PREDICTION_MISMATCH", flags)

        trial = {
            "trial_id": "TRIAL_ERR_01",
            "branch": "ABC",
            "ground_truth": "A",
            "commit_text": "B",
            "commit_count": 1,
            "primary_outcome": outcome,
            "diagnostic_flags": flags,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["false_commit_rate"], 1.0)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["character_accuracy"], 0.0)
        self.assertEqual(metrics["classification_metrics"]["per_class"]["A"]["false_negatives"], 1)
        self.assertEqual(metrics["classification_metrics"]["per_class"]["B"]["false_positives"], 1)

    def test_03_landmark_detection_failure(self):
        """Scenario 3: Total landmark failure -> LANDMARK_DETECTION_FAILURE has top priority."""
        outcome, flags = determine_primary_outcome(
            ground_truth="A",
            commits=[],
            landmarks_detected=False,
            validator_rejected=False,
            hold_aborted=False,
            raw_prediction=None,
            candidate=None,
            timeout_occurred=True,
        )
        self.assertEqual(outcome, "LANDMARK_DETECTION_FAILURE")
        self.assertIn("NO_LANDMARKS_DETECTED", flags)

        trial = {
            "trial_id": "TRIAL_LDF_01",
            "branch": "ABC",
            "ground_truth": "A",
            "commit_text": "",
            "commit_count": 0,
            "primary_outcome": outcome,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["miss_rate"], 1.0)

    def test_04_validator_rejection(self):
        """Scenario 4: Gesture fails trajectory / margin check -> VALIDATOR_REJECTION."""
        outcome, flags = determine_primary_outcome(
            ground_truth="Z",
            commits=[],
            landmarks_detected=True,
            validator_rejected=True,
            hold_aborted=False,
            raw_prediction="Z",
            candidate="Z",
            timeout_occurred=False,
        )
        self.assertEqual(outcome, "VALIDATOR_REJECTION")
        self.assertIn("VALIDATOR_REJECTED", flags)

        trial = {
            "trial_id": "TRIAL_VAL_01",
            "branch": "JZ",
            "ground_truth": "Z",
            "commit_text": "",
            "commit_count": 0,
            "primary_outcome": outcome,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["false_rejection_rate"], 1.0)

    def test_05_hold_aborted(self):
        """Scenario 5: Hand moves violently during dwell -> HOLD_ABORTED."""
        outcome, flags = determine_primary_outcome(
            ground_truth="B",
            commits=[],
            landmarks_detected=True,
            validator_rejected=False,
            hold_aborted=True,
            raw_prediction="B",
            candidate="B",
            timeout_occurred=False,
        )
        self.assertEqual(outcome, "HOLD_ABORTED")
        self.assertIn("HOLD_STATE_ABORTED", flags)

        trial = {
            "trial_id": "TRIAL_HOLD_01",
            "branch": "ABC",
            "ground_truth": "B",
            "commit_text": "",
            "commit_count": 0,
            "primary_outcome": outcome,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["false_rejection_rate"], 1.0)

    def test_06_timeout_missed(self):
        """Scenario 6: Dwell time threshold not met within 5s window -> TIMEOUT_MISSED."""
        outcome, flags = determine_primary_outcome(
            ground_truth="C",
            commits=[],
            landmarks_detected=True,
            validator_rejected=False,
            hold_aborted=False,
            raw_prediction="C",
            candidate="C",
            timeout_occurred=True,
        )
        self.assertEqual(outcome, "TIMEOUT_MISSED")

        trial = {
            "trial_id": "TRIAL_TO_01",
            "branch": "ABC",
            "ground_truth": "C",
            "commit_text": "",
            "commit_count": 0,
            "primary_outcome": outcome,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["miss_rate"], 1.0)

    def test_07_duplicate_commit(self):
        """Scenario 7: Multiple commits fired in a single trial -> DUPLICATE_COMMIT."""
        outcome, flags = determine_primary_outcome(
            ground_truth="D",
            commits=[("D", 3.1), ("D", 4.2)],
            landmarks_detected=True,
            validator_rejected=False,
            hold_aborted=False,
            raw_prediction="D",
            candidate="D",
            timeout_occurred=False,
        )
        self.assertEqual(outcome, "DUPLICATE_COMMIT")
        self.assertIn("MULTIPLE_COMMITS_DETECTED", flags)

        trial = {
            "trial_id": "TRIAL_DUP_01",
            "branch": "ABC",
            "ground_truth": "D",
            "commit_text": "D",
            "commit_count": 2,
            "primary_outcome": outcome,
        }
        metrics = calculate_metrics([trial])
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["duplicate_commit_rate"], 1.0)

    def test_08_unknown_outcome(self):
        """Scenario 8: Model candidate is explicitly UNKNOWN -> UNKNOWN primary outcome."""
        outcome, flags = determine_primary_outcome(
            ground_truth="E",
            commits=[],
            landmarks_detected=True,
            validator_rejected=False,
            hold_aborted=False,
            raw_prediction="UNKNOWN",
            candidate="UNKNOWN",
            timeout_occurred=False,
        )
        self.assertEqual(outcome, "UNKNOWN")

    def test_09_multiple_diagnostic_flags(self):
        """Scenario 9: Multiple anomalous conditions are recorded together in diagnostic_flags."""
        outcome, flags = determine_primary_outcome(
            ground_truth="A",
            commits=[("B", 3.0), ("C", 4.0)],
            landmarks_detected=True,
            validator_rejected=True,
            hold_aborted=True,
            raw_prediction="B",
            candidate="B",
            timeout_occurred=False,
        )
        # Multiple commits has higher priority than validator rejection
        self.assertEqual(outcome, "DUPLICATE_COMMIT")
        self.assertIn("MULTIPLE_COMMITS_DETECTED", flags)
        self.assertIn("VALIDATOR_REJECTED", flags)
        self.assertIn("HOLD_STATE_ABORTED", flags)
        self.assertIn("RAW_PREDICTION_MISMATCH", flags)

    def test_10_macro_f1_strict_arithmetic_mean(self):
        """Scenario 10: Macro-F1 must be strictly the arithmetic mean of per-class F1."""
        trials = [
            # Class A: 2 TP -> F1 = 1.0
            {"trial_id": "T1", "ground_truth": "A", "commit_text": "A", "commit_count": 1, "primary_outcome": "SUCCESS"},
            {"trial_id": "T2", "ground_truth": "A", "commit_text": "A", "commit_count": 1, "primary_outcome": "SUCCESS"},
            # Class B: 0 TP, 1 FP (predicted as B when GT was C), 1 FN (GT is B, predicted as C) -> F1 = 0.0
            {"trial_id": "T3", "ground_truth": "C", "commit_text": "B", "commit_count": 1, "primary_outcome": "CLASSIFICATION_ERROR"},
            {"trial_id": "T4", "ground_truth": "B", "commit_text": "C", "commit_count": 1, "primary_outcome": "CLASSIFICATION_ERROR"},
            # Class C: 0 TP, 1 FP, 1 FN -> F1 = 0.0
        ]
        metrics = calculate_metrics(trials)
        per_class = metrics["classification_metrics"]["per_class"]
        f1_a = per_class["A"]["f1"]
        f1_b = per_class["B"]["f1"]
        f1_c = per_class["C"]["f1"]
        self.assertAlmostEqual(f1_a, 1.0)
        self.assertAlmostEqual(f1_b, 0.0)
        self.assertAlmostEqual(f1_c, 0.0)

        expected_macro_f1 = (1.0 + 0.0 + 0.0) / 3.0
        self.assertAlmostEqual(metrics["classification_metrics"]["macro_f1"], expected_macro_f1)

    def test_11_uncommitted_trials_in_denominator(self):
        """Scenario 11: Uncommitted trials must stay in the denominator of Character Accuracy."""
        trials = [
            {"trial_id": "T1", "ground_truth": "A", "commit_text": "A", "commit_count": 1, "primary_outcome": "SUCCESS"},
            {"trial_id": "T2", "ground_truth": "B", "commit_text": "B", "commit_count": 1, "primary_outcome": "SUCCESS"},
            {"trial_id": "T3", "ground_truth": "C", "commit_text": "", "commit_count": 0, "primary_outcome": "TIMEOUT_MISSED"},
            {"trial_id": "T4", "ground_truth": "D", "commit_text": "", "commit_count": 0, "primary_outcome": "VALIDATOR_REJECTION"},
            {"trial_id": "T5", "ground_truth": "E", "commit_text": "", "commit_count": 0, "primary_outcome": "LANDMARK_DETECTION_FAILURE"},
        ]
        metrics = calculate_metrics(trials)
        # Accuracy must be 2 / 5 = 0.40, NOT 2 / 2 = 1.00
        self.assertEqual(metrics["total_trials"], 5)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["character_accuracy"], 0.40)
        self.assertAlmostEqual(metrics["end_to_end_character_metrics"]["miss_rate"], 0.40)  # 2 missed (timeout + landmark)

    def test_12_far_and_frr_calculation(self):
        """Scenario 12: FAR and FRR are accurately computed for SPACE and J/Z branches."""
        trials = [
            # SPACE positive trials (2 total: 1 commit, 1 missed) -> FRR = 1/2 = 0.50
            {"trial_id": "S1", "branch": "SPACE", "ground_truth": "SPACE", "commit_text": "SPACE", "commit_count": 1, "primary_outcome": "SUCCESS"},
            {"trial_id": "S2", "branch": "SPACE", "ground_truth": "SPACE", "commit_text": "", "commit_count": 0, "primary_outcome": "TIMEOUT_MISSED"},
            # SPACE negative trials (2 total: 1 false commit, 1 quiet) -> FAR = 1/2 = 0.50
            {"trial_id": "S3", "branch": "SPACE", "ground_truth": "NEGATIVE", "commit_text": "SPACE", "commit_count": 1, "primary_outcome": "CLASSIFICATION_ERROR"},
            {"trial_id": "S4", "branch": "SPACE", "ground_truth": "NEGATIVE", "commit_text": "", "commit_count": 0, "primary_outcome": "SUCCESS"},
        ]
        metrics = calculate_metrics(trials)
        self.assertAlmostEqual(metrics["safety_metrics"]["far_space"], 0.50)
        self.assertAlmostEqual(metrics["safety_metrics"]["frr_space"], 0.50)

    def test_13_empty_data_returns_na_cleanly(self):
        """Scenario 13: Empty dataset returns N/A cleanly without zero division or exception."""
        metrics = calculate_metrics([])
        self.assertEqual(metrics["status"], "EMPTY_DATA")
        self.assertEqual(metrics["total_trials"], 0)
        self.assertEqual(metrics["classification_metrics"], "N/A")
        self.assertEqual(metrics["latency_metrics"], "N/A")

        lat_stats = compute_latency_stats([])
        self.assertEqual(lat_stats["mean_ms"], "N/A")
        self.assertEqual(lat_stats["p50_ms"], "N/A")

    def test_14_data_source_isolation(self):
        """Scenario 14: Mock mode enforces data_source='mock' and does not invoke live camera."""
        harness = EvaluationHarness(data_source="mock")
        self.assertEqual(harness.data_source, "mock")
        # Ensure simulated trial explicitly records data_source="mock"
        spec = {"trial_id": "T_MOCK", "branch": "ABC", "ground_truth": "A"}
        result = harness.run_trial_simulation(spec, scenario="success", simulated_frames=100)
        self.assertEqual(result["data_source"], "mock")
        self.assertEqual(result["primary_outcome"], "SUCCESS")

    def test_15_latency_statistics_units_ms(self):
        """Scenario 15: Latency stats correctly compute mean, p50, p95 in ms."""
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_stats(latencies)
        self.assertEqual(stats["sample_count"], 5)
        self.assertAlmostEqual(stats["mean_ms"], 30.0)
        self.assertAlmostEqual(stats["p50_ms"], 30.0)
        self.assertAlmostEqual(stats["min_ms"], 10.0)
        self.assertAlmostEqual(stats["max_ms"], 50.0)
        self.assertGreaterEqual(stats["p95_ms"], stats["p50_ms"])

    def test_16_planned_trials_queue_length(self):
        """Bonus Scenario 16: Standard planned trials queue has exactly 390 trials."""
        queue = generate_planned_trials()
        self.assertEqual(len(queue), 390)
        abc_count = sum(1 for t in queue if t["branch"] == "ABC")
        digit_count = sum(1 for t in queue if t["branch"] == "123")
        jz_count = sum(1 for t in queue if t["branch"] == "JZ")
        space_count = sum(1 for t in queue if t["branch"] == "SPACE")
        self.assertEqual(abc_count, 240)
        self.assertEqual(digit_count, 100)
        self.assertEqual(jz_count, 20)
        self.assertEqual(space_count, 30)


if __name__ == "__main__":
    unittest.main()
