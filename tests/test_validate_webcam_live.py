"""Unit tests for Stage 13.1 Live Webcam Validation Protocol (tools/validate_webcam_live.py)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features
from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
from tools.validate_webcam_live import (
    MODEL_PATH,
    compute_session_metrics,
    evaluate_attempt,
    main,
    parse_args,
)


def create_mock_hand(wrist_x: float = 0.5, wrist_y: float = 0.5, scale: float = 0.05):
    """Create a realistic MediaPipe-like 21-landmark list."""
    hand = []
    # 21 landmarks
    for i in range(21):
        # Place landmarks relative to wrist
        x = wrist_x + (i % 5) * (scale / 5.0)
        y = wrist_y + (i // 5) * (scale / 5.0)
        z = i * 0.001
        hand.append(SimpleNamespace(x=float(x), y=float(y), z=float(z)))
    return hand


class MockPipeline:
    def __init__(self, prob_j: float = 0.70, prob_z: float = 0.30):
        self.classes_ = np.asarray(["J", "Z"])
        self.prob_j = prob_j
        self.prob_z = prob_z
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = np.asarray(X, dtype=np.float64)
        n = len(X) if self.last_input.ndim > 1 else 1
        return np.tile([self.prob_j, self.prob_z], (n, 1))

    def predict(self, X):
        self.last_input = np.asarray(X, dtype=np.float64)
        n = len(X) if self.last_input.ndim > 1 else 1
        return np.repeat("J" if self.prob_j >= self.prob_z else "Z", n)


class TestValidateWebcamLive(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.trials_per_class, 20)
        self.assertEqual(args.camera_index, 0)
        self.assertAlmostEqual(args.margin_threshold, 0.10)
        self.assertFalse(args.preflight)
        self.assertFalse(args.dry_run)

    def test_parse_args_flags(self):
        args = parse_args(["--trials-per-class", "10", "--preflight", "--margin-threshold", "0.15"])
        self.assertEqual(args.trials_per_class, 10)
        self.assertTrue(args.preflight)
        self.assertAlmostEqual(args.margin_threshold, 0.15)
        self.assertFalse(args.dry_run)

    def test_build_sequence_features_call_and_shape(self):
        """Verify build_sequence_features succeeds without unexpected keyword arguments and returns (16, 69)."""
        hands = [create_mock_hand(wrist_x=0.4 + step * 0.005, wrist_y=0.4 + step * 0.005) for step in range(16)]
        sizes = [(640, 480)] * 16

        # Direct call must succeed without 'source_indices' error
        features, hand_scale_px = build_sequence_features(hands, sizes)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (16, 69))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(hand_scale_px, 0.0)

    def test_evaluate_attempt_successful_model_invocation(self):
        """Verify evaluate_attempt extracts features, sets model_invoked=True, and passes same features to both models."""
        pipe = MockPipeline(prob_j=0.85, prob_z=0.15)
        base = ProbabilityMarginModelWrapper(pipe, enable_reject_option=False, margin_threshold=0.10)
        rej = ProbabilityMarginModelWrapper(pipe, enable_reject_option=True, margin_threshold=0.10)

        # 20 valid frames (>= 16 minimum)
        valid_entries = [
            (i, create_mock_hand(wrist_x=0.4 + i * 0.005, wrist_y=0.4 + i * 0.005), (640, 480))
            for i in range(20)
        ]

        rec = evaluate_attempt(0, "J", valid_entries, 20, base, rej, 0.10)

        self.assertTrue(rec["model_invoked"])
        self.assertEqual(rec["baseline_prediction"], "J")
        self.assertEqual(rec["reject_option_prediction"], "J")
        self.assertAlmostEqual(rec["probability_J"], 0.85)
        self.assertAlmostEqual(rec["probability_Z"], 0.15)
        self.assertAlmostEqual(rec["margin"], 0.70)
        self.assertTrue(rec["baseline_committed"])
        self.assertTrue(rec["reject_option_committed"])
        self.assertGreater(rec["hand_scale_px"], 0.0)

    def test_evaluate_attempt_both_j_and_z_invoked(self):
        """Verify that both J and Z attempts invoke the model successfully."""
        pipe = MockPipeline(prob_j=0.20, prob_z=0.80)
        base = ProbabilityMarginModelWrapper(pipe, enable_reject_option=False, margin_threshold=0.10)
        rej = ProbabilityMarginModelWrapper(pipe, enable_reject_option=True, margin_threshold=0.10)

        valid_entries = [
            (i, create_mock_hand(wrist_x=0.5 - i * 0.004, wrist_y=0.5 + i * 0.003), (640, 480))
            for i in range(25)
        ]

        rec_z = evaluate_attempt(1, "Z", valid_entries, 25, base, rej, 0.10)
        self.assertTrue(rec_z["model_invoked"])
        self.assertEqual(rec_z["baseline_prediction"], "Z")
        self.assertEqual(rec_z["reject_option_prediction"], "Z")
        self.assertEqual(rec_z["rejection_reason"], "")

    def test_evaluate_attempt_insufficient_frames(self):
        """Verify attempt with < 16 frames is uninvoked and labeled INSUFFICIENT_VALID_FRAMES."""
        pipe = MockPipeline()
        base = ProbabilityMarginModelWrapper(pipe, enable_reject_option=False)
        rej = ProbabilityMarginModelWrapper(pipe, enable_reject_option=True)

        entries = [(i, create_mock_hand(), (640, 480)) for i in range(5)]
        res = evaluate_attempt(0, "J", entries, 10, base, rej, 0.10)
        self.assertFalse(res["model_invoked"])
        self.assertEqual(res["baseline_prediction"], "UNKNOWN")
        self.assertEqual(res["reject_option_prediction"], "UNKNOWN")
        self.assertIn("INSUFFICIENT_VALID_FRAMES", res["rejection_reason"])

    def test_evaluate_attempt_pipeline_error_handling(self):
        """Verify that any feature extraction error does NOT invoke model and records FEATURE_EXTRACTION_ERROR."""
        pipe = MockPipeline()
        base = ProbabilityMarginModelWrapper(pipe, enable_reject_option=False)
        rej = ProbabilityMarginModelWrapper(pipe, enable_reject_option=True)

        # 16 frames with invalid landmark count (< 21)
        invalid_entries = [(i, [SimpleNamespace(x=0, y=0, z=0)] * 5, (640, 480)) for i in range(16)]
        res = evaluate_attempt(0, "J", invalid_entries, 16, base, rej, 0.10)
        self.assertFalse(res["model_invoked"])
        self.assertEqual(res["baseline_prediction"], "UNKNOWN")
        self.assertEqual(res["reject_option_prediction"], "UNKNOWN")
        self.assertIn("FEATURE_EXTRACTION_ERROR", res["rejection_reason"])

    def test_compute_session_metrics_distinguishes_pipeline_errors(self):
        """Verify compute_session_metrics separates pipeline errors from model UNKNOWNs."""
        records = [
            # Successful invoked J
            {
                "trial_index": 1, "ground_truth": "J", "valid_frames": 25, "model_invoked": True,
                "baseline_prediction": "J", "reject_option_prediction": "J",
                "baseline_committed": True, "reject_option_committed": True,
            },
            # Successful invoked Z, margin < 0.10 -> rejected as UNKNOWN by model
            {
                "trial_index": 2, "ground_truth": "Z", "valid_frames": 25, "model_invoked": True,
                "baseline_prediction": "J", "reject_option_prediction": "UNKNOWN",
                "baseline_committed": False, "reject_option_committed": False,
            },
            # Pipeline error on J (valid_frames >= 16 but model_invoked == False)
            {
                "trial_index": 3, "ground_truth": "J", "valid_frames": 20, "model_invoked": False,
                "baseline_prediction": "UNKNOWN", "reject_option_prediction": "UNKNOWN",
                "baseline_committed": False, "reject_option_committed": False,
                "rejection_reason": "FEATURE_EXTRACTION_ERROR: SimError",
            },
            # Negative control, invoked
            {
                "trial_index": 4, "ground_truth": "NEGATIVE", "valid_frames": 18, "model_invoked": True,
                "baseline_prediction": "J", "reject_option_prediction": "UNKNOWN",
                "baseline_committed": False, "reject_option_committed": False,
            },
        ]
        metrics = compute_session_metrics(records)

        # Baseline & reject option diagnostics
        self.assertEqual(metrics["baseline"]["pipeline_error_positives_count"], 1)
        self.assertEqual(metrics["baseline"]["invoked_positives_count"], 2)
        self.assertEqual(metrics["baseline"]["total_positives_count"], 3)

        # Invoked accuracy must only be computed over the 2 invoked positives (1 correct J, 1 incorrect Z->J)
        self.assertAlmostEqual(metrics["baseline"]["invoked_accuracy"], 0.50)
        self.assertEqual(metrics["baseline"]["z_to_j_errors"], 1)

        # Reject option: 1 correct J, 1 model UNKNOWN
        self.assertEqual(metrics["reject_option"]["unknown_count"], 1)
        self.assertAlmostEqual(metrics["reject_option"]["unknown_rate"], 0.50)
        self.assertEqual(metrics["reject_option"]["z_to_j_errors"], 0)

    def test_dry_run_main(self):
        code = main(["--dry-run"])
        self.assertEqual(code, 0)

    def test_preflight_main(self):
        code = main(["--preflight", "--camera-index", "0"])
        self.assertIn(code, (0, 2))

    @patch("cv2.VideoCapture")
    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=ord('q'))
    @patch("cv2.destroyAllWindows")
    def test_interactive_early_quit(self, mock_destroy, mock_wait, mock_imshow, mock_cap):
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        cap_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.return_value = cap_instance

        code = main(["--trials-per-class", "1", "--output", "reports/data/webcam_mock_verification_results.json"])
        self.assertEqual(code, 0)
        cap_instance.release.assert_called()
        mock_destroy.assert_called()


if __name__ == "__main__":
    unittest.main()
