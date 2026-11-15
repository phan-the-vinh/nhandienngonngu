import unittest
from types import SimpleNamespace
import numpy as np

from app.inference.letter_inference import Prediction
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class ConfigurablePredictor:
    def __init__(self, label: str, confidence: float = 0.74, shape: float = 0.0):
        self.label = label
        self.confidence = confidence
        self.shape = shape
    def reset(self):
        pass
    def predict_landmarks(self, _hand):
        return Prediction(self.label, self.confidence, self.label, self.confidence,
                          (self.shape,) * 63, (self.confidence, 1.0 - self.confidence))


class DynamicModelProbStub:
    classes_ = np.asarray(["J", "Z"])
    def __init__(self, label="J", prob=0.7119):
        self.label = label
        self.prob = prob
    def predict_proba(self, x):
        p = self.prob
        return np.asarray([[p, 1.0 - p] if self.label == "J" else [1.0 - p, p]])


class JConfidenceDecouplingTests(unittest.TestCase):
    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.events = []

    def test_global_hold_settings_untouched(self):
        """Global HoldSettings.min_confidence must remain 0.75."""
        default_settings = HoldSettings()
        self.assertEqual(default_settings.min_confidence, 0.75)

    def test_dynamic_j_confidence_0_7119_passes_hold(self):
        """Dynamic J with confidence 0.7119 (Attempt 48) passes hold check under decoupled threshold 0.70."""
        model = DynamicModelProbStub(label="J", prob=0.7119)
        c = UnifiedAutoCoordinator(
            self.letters, self.digits, model,
            enable_experimental_jz=True,
            event_sink=self.events.append
        )
        self.assertEqual(c.min_dynamic_j_confidence, 0.70)
        self.assertEqual(c.min_dynamic_z_confidence, 0.75)

        hand = synthetic_hand()
        now = 10.0
        c.candidate_attempt_id = 48
        c.candidate_at = now
        c.candidate_created_at = now
        c.candidate_settled = True
        c.validation = {"status": "PASS", "j_trajectory_valid": True}
        c.candidate = {
            "prediction": "J",
            "probability_J": 0.7119,
            "prediction_probability": 0.7119,
            "source": "dynamic",
        }
        c.jz_hold = HoldConfirmation(HoldSettings(min_confidence=c.min_dynamic_j_confidence))
        c.endpoint_observation = c.raw_observation(hand, (480, 640, 3))
        c.proposal_observation = c.endpoint_observation
        c.status = "STATIC"

        # Update frame with matching hand observation
        c.update(index=1, static_hand=hand, raw_hand=hand, shape=(480, 640, 3), now=now + 0.1, raw_hand_count=1)

        # J candidate must remain alive and not rejected
        self.assertIsNotNone(c.candidate)
        self.assertNotEqual(c.candidate_cancel_reason, "JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD")

    def test_dynamic_j_confidence_below_0_70_rejected(self):
        """Dynamic J with confidence 0.690 is rejected below 0.70 threshold."""
        model = DynamicModelProbStub(label="J", prob=0.6900)
        c = UnifiedAutoCoordinator(
            self.letters, self.digits, model,
            enable_experimental_jz=True,
            event_sink=self.events.append
        )
        hand = synthetic_hand()
        now = 10.0
        c.candidate_attempt_id = 48
        c.candidate_at = now
        c.candidate_created_at = now
        c.candidate_settled = True
        c.validation = {"status": "PASS", "j_trajectory_valid": True}
        c.candidate = {
            "prediction": "J",
            "probability_J": 0.6900,
            "prediction_probability": 0.6900,
            "source": "dynamic",
        }
        c.jz_hold = HoldConfirmation(HoldSettings(min_confidence=c.min_dynamic_j_confidence))
        c.endpoint_observation = c.raw_observation(hand, (480, 640, 3))
        c.proposal_observation = c.endpoint_observation
        c.status = "STATIC"

        c.update(index=1, static_hand=hand, raw_hand=hand, shape=(480, 640, 3), now=now + 0.1, raw_hand_count=1)

        # Candidate must be cancelled
        self.assertIsNone(c.candidate)
        self.assertEqual(c.candidate_cancel_reason, "JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD")

    def test_dynamic_z_confidence_0_7119_rejected_under_0_75_threshold(self):
        """Dynamic Z with confidence 0.7119 must be rejected because Z retains 0.75 threshold."""
        model = DynamicModelProbStub(label="Z", prob=0.7119)
        c = UnifiedAutoCoordinator(
            self.letters, self.digits, model,
            enable_experimental_jz=True,
            event_sink=self.events.append
        )
        hand = synthetic_hand()
        now = 10.0
        c.candidate_attempt_id = 99
        c.candidate_at = now
        c.candidate_created_at = now
        c.candidate_settled = True
        c.validation = {"status": "PASS", "z_trajectory_valid": True}
        c.candidate = {
            "prediction": "Z",
            "probability_Z": 0.7119,
            "prediction_probability": 0.7119,
            "source": "dynamic",
        }
        c.jz_hold = HoldConfirmation(HoldSettings(min_confidence=c.min_dynamic_z_confidence))
        c.endpoint_observation = c.raw_observation(hand, (480, 640, 3))
        c.proposal_observation = c.endpoint_observation
        c.status = "STATIC"

        c.update(index=1, static_hand=hand, raw_hand=hand, shape=(480, 640, 3), now=now + 0.1, raw_hand_count=1)

        # Z candidate must be cancelled because 0.7119 < 0.75
        self.assertIsNone(c.candidate)
        self.assertEqual(c.candidate_cancel_reason, "JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD")

    def test_static_abc_confidence_0_74_fails_0_75_requirement(self):
        """Static ABC holds retain min_confidence=0.75: confidence 0.74 does NOT advance hold."""
        predictor = ConfigurablePredictor("B", confidence=0.74)
        model = DynamicModelProbStub(label="J", prob=0.99)
        c = UnifiedAutoCoordinator(predictor, self.digits, model, event_sink=self.events.append)
        self.assertEqual(c.static.hold_settings.min_confidence, 0.75)
        self.assertEqual(c.static.hold.settings.min_confidence, 0.75)

        hand = synthetic_hand()
        # Tick for 3.1 seconds with confidence 0.74
        for i in range(35):
            committed = c.update(
                index=i, static_hand=hand, raw_hand=hand,
                shape=(480, 640, 3), now=i * 0.1, raw_hand_count=1
            )[1]
            self.assertIsNone(committed, "Static B at 0.74 confidence must never commit!")

    def test_static_123_confidence_0_74_fails_0_75_requirement(self):
        """Static 123 holds retain min_confidence=0.75: confidence 0.74 does NOT advance hold."""
        predictor_digits = ConfigurablePredictor("5", confidence=0.74)
        model = DynamicModelProbStub(label="J", prob=0.99)
        c = UnifiedAutoCoordinator(self.letters, predictor_digits, model, event_sink=self.events.append)
        c.switch_mode("123")
        self.assertEqual(c.static.hold_settings.min_confidence, 0.75)

        hand = synthetic_hand()
        for i in range(35):
            committed = c.update(
                index=i, static_hand=hand, raw_hand=hand,
                shape=(480, 640, 3), now=i * 0.1, raw_hand_count=1
            )[1]
            self.assertIsNone(committed, "Static 5 at 0.74 confidence must never commit!")


if __name__ == "__main__":
    unittest.main()
