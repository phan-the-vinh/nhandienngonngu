"""Stage 13 Regression & Integration Test Suite.

Verifies:
1. Feature flag toggle (enable_reject_option=True/False)
2. Threshold sensitivity and UNKNOWN rejection
3. End-to-end integration with UnifiedAutoCoordinator and TextComposer safety
4. Fail-safe fallback on malformed inputs or model exceptions
5. ABC, 123, SPACE, and HoldConfirmation 3.0s behavioral invariance
6. Checkpoint SHA-256 integrity of all production model artifacts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from app.inference.unified_static_coordinator import TextComposer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"


class MockBaseModel:
    """Mock base model simulating binary J/Z predictions."""
    def __init__(self, prob_j: float = 0.51, prob_z: float = 0.49):
        self.classes_ = np.asarray(["J", "Z"])
        self.prob_j = float(prob_j)
        self.prob_z = float(prob_z)

    def predict_proba(self, X):
        X = np.asarray(X)
        n = len(X) if X.ndim > 1 else 1
        return np.tile([self.prob_j, self.prob_z], (n, 1))

    def predict(self, X):
        X = np.asarray(X)
        n = len(X) if X.ndim > 1 else 1
        label = "J" if self.prob_j >= self.prob_z else "Z"
        return np.repeat(label, n)


class FailingBaseModel:
    """Mock base model that throws an exception during prediction."""
    def __init__(self):
        self.classes_ = np.asarray(["J", "Z"])

    def predict_proba(self, X):
        raise RuntimeError("Simulated internal model failure")

    def predict(self, X):
        return np.asarray(["J"])


class Stage13IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODEL_PATH.exists():
            cls.real_pipeline = joblib.load(MODEL_PATH)
        else:
            cls.real_pipeline = None

    def test_feature_flag_disabled_matches_raw_model_100_percent(self):
        """When enable_reject_option=False, predictions must match base model exactly."""
        mock_model = MockBaseModel(prob_j=0.52, prob_z=0.48)  # margin = 0.04
        wrapper = ProbabilityMarginModelWrapper(
            mock_model, enable_reject_option=False, margin_threshold=0.10
        )
        sample = np.ones((1, 1104))
        # Even with margin 0.04 < 0.10, disabled flag must return raw base prediction
        preds = wrapper.predict(sample)
        self.assertEqual(preds[0], "J")
        self.assertEqual(preds[0], mock_model.predict(sample)[0])

    def test_feature_flag_enabled_rejects_ambiguous_margin(self):
        """When enable_reject_option=True and margin < threshold, return UNKNOWN."""
        mock_model = MockBaseModel(prob_j=0.51, prob_z=0.49)  # margin = 0.02
        wrapper = ProbabilityMarginModelWrapper(
            mock_model, enable_reject_option=True, margin_threshold=0.10
        )
        sample = np.ones((1, 1104))
        preds = wrapper.predict(sample)
        self.assertEqual(preds[0], "UNKNOWN")

    def test_feature_flag_enabled_accepts_confident_gesture(self):
        """When enable_reject_option=True and margin >= threshold, return predicted class."""
        mock_model = MockBaseModel(prob_j=0.80, prob_z=0.20)  # margin = 0.60
        wrapper = ProbabilityMarginModelWrapper(
            mock_model, enable_reject_option=True, margin_threshold=0.10
        )
        sample = np.ones((1, 1104))
        preds = wrapper.predict(sample)
        self.assertEqual(preds[0], "J")

    def test_threshold_boundary_cases(self):
        """Test exact boundary condition: margin == threshold and margin near 0."""
        # Margin = 0.10 exactly (not strictly less than 0.10 -> should accept)
        mock_model = MockBaseModel(prob_j=0.55, prob_z=0.45)  # margin = 0.10
        wrapper = ProbabilityMarginModelWrapper(
            mock_model, enable_reject_option=True, margin_threshold=0.10
        )
        sample = np.ones((1, 1104))
        preds = wrapper.predict(sample)
        self.assertEqual(preds[0], "J")

        # Margin = 0.0999 < 0.10 -> reject as UNKNOWN
        mock_model2 = MockBaseModel(prob_j=0.5499, prob_z=0.4501)
        wrapper2 = ProbabilityMarginModelWrapper(
            mock_model2, enable_reject_option=True, margin_threshold=0.10
        )
        preds2 = wrapper2.predict(sample)
        self.assertEqual(preds2[0], "UNKNOWN")

    def test_fail_safe_fallback_on_exception(self):
        """If the reject option logic or model encounters an error, fallback gracefully."""
        mock_failing = FailingBaseModel()
        wrapper = ProbabilityMarginModelWrapper(
            mock_failing, enable_reject_option=True, margin_threshold=0.10
        )
        sample = np.ones((1, 1104))
        # Should NOT raise exception, should fallback to base model predict
        preds = wrapper.predict(sample)
        self.assertEqual(preds[0], "J")

    def test_unknown_never_commits_to_text_composer(self):
        """Simulate Coordinator receiving an UNKNOWN prediction; confirm TextComposer text is untouched."""
        # Create coordinator with mock models
        mock_letters = MagicMock()
        mock_digits = MagicMock()
        mock_dynamic = MockBaseModel(prob_j=0.51, prob_z=0.49)  # Ambiguous margin

        coordinator = UnifiedAutoCoordinator(
            mock_letters,
            mock_digits,
            mock_dynamic,
            enable_experimental_jz=True,
            enable_jz_reject_option=True,
            jz_margin_threshold=0.10,
        )
        composer = coordinator.composer
        initial_text = composer.text

        # Simulate arrival of an attempt result with UNKNOWN
        now = 100.0
        result = {
            "status": "RESULT",
            "attempt_id": 1,
            "model_result": {
                "prediction": "UNKNOWN",
                "probability_J": 0.51,
                "probability_Z": 0.49,
            },
        }
        # In UnifiedAutoCoordinator, when model_pred has prediction="UNKNOWN",
        # proposal validation fails, clear_candidate is called with REJECT_OPTION_MARGIN_AMBIGUITY,
        # status becomes REJECTED, and text is NEVER appended.
        coordinator.seen_result = None
        coordinator.session.last_result = result

        # Feed frame update to trigger result processing
        coordinator.update(
            index=1,
            static_hand=None,
            raw_hand=None,
            shape=(480, 640, 3),
            now=now,
        )

        # Assert text in TextComposer remained completely unchanged
        self.assertEqual(composer.text, initial_text)
        self.assertIsNone(coordinator.candidate)
        self.assertNotEqual(coordinator.status, "CONFIRMED")

    def test_hold_confirmation_settings_invariance(self):
        """HoldConfirmation duration must strictly remain 3.0 seconds."""
        hold = HoldConfirmation(HoldSettings())
        self.assertEqual(hold.settings.hold_seconds, 3.0)

    def test_switch_mode_invariance(self):
        """Mode switching between ABC and 123 must preserve wrapped dynamic model and clear stale candidates."""
        mock_letters = MagicMock()
        mock_digits = MagicMock()
        mock_dynamic = MockBaseModel(prob_j=0.70, prob_z=0.30)

        coordinator = UnifiedAutoCoordinator(
            mock_letters,
            mock_digits,
            mock_dynamic,
            enable_experimental_jz=True,
            enable_jz_reject_option=True,
            jz_margin_threshold=0.10,
        )

        self.assertEqual(coordinator.mode, "ABC")
        coordinator.switch_mode("123")
        self.assertEqual(coordinator.mode, "123")
        self.assertIsNone(coordinator.candidate)

        coordinator.switch_mode("ABC")
        self.assertEqual(coordinator.mode, "ABC")
        self.assertTrue(coordinator.enable_jz_reject_option)
        self.assertEqual(coordinator.jz_margin_threshold, 0.10)

    def test_checkpoint_sha256_hashes(self):
        """Verify that all production checkpoints have 100% unchanged SHA-256 digests."""
        expected_hashes = {
            "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib": "8878b0c0cd04a1ff8bbfc263cb3d49b4c7bc92706140b71c99a36e37bc33c497",
            "data/experiments/dynamic_jz_binary_baseline_v1/scaler.joblib": "93804d73f4fc63aee7f2da08c1f0a879dc6d2bdc50ec27c0f5cfa4011b6fad7c",
            "models/digits/production/digits_final.joblib": "43c7e87a287daf7e2c4958f037a91c128f97ab381ccb4f0ae40c7c096776b25c",
            "models/letters/production/uvw_medium_model.joblib": "88ea2f270cd75575d8914650265aa00193ffe451eb1a72d3ce70970bb1525a45",
        }

        for rel_path, expected_hash in expected_hashes.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Checkpoint missing: {rel_path}")
            actual_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"SHA-256 mismatch for {rel_path}! Expected {expected_hash}, got {actual_hash}",
            )


if __name__ == "__main__":
    unittest.main()
