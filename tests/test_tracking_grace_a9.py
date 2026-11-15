import unittest
from types import SimpleNamespace
import numpy as np

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class DynamicModelStub:
    classes_ = np.asarray(["J", "Z"])
    def __init__(self, label="Z", prob=0.85):
        self.label = label
        self.prob = prob
    def predict_proba(self, x):
        p = self.prob
        return np.asarray([[1.0 - p, p] if self.label == "Z" else [p, 1.0 - p]])


class TrackingGraceTests(unittest.TestCase):
    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.model = DynamicModelStub(label="Z", prob=0.85)
        self.events = []
        self.c = UnifiedAutoCoordinator(
            self.letters, self.digits, self.model,
            enable_experimental_jz=True,
            event_sink=self.events.append
        )
        self.index = 0

    def _setup_settled_candidate(self, label="Z"):
        """Directly inject a valid candidate and settled endpoint observation for testing."""
        hand = synthetic_hand()
        now = 10.0
        self.c.candidate_attempt_id = 999
        self.c.candidate_at = now
        self.c.candidate_created_at = now
        self.c.candidate_settled = True
        self.c.validation = {"status": "PASS", "z_trajectory_valid": True}
        self.c.candidate = {
            "prediction": label,
            f"probability_{label}": 0.85,
            "prediction_probability": 0.85,
            "source": "dynamic",
        }
        pred_label = label
        min_conf = self.c.min_dynamic_j_confidence if pred_label == "J" else self.c.min_dynamic_z_confidence
        self.c.jz_hold = HoldConfirmation(HoldSettings(min_confidence=min_conf))
        self.c.endpoint_observation = self.c.raw_observation(hand, (480, 640, 3))
        self.c.proposal_observation = self.c.endpoint_observation
        self.c.status = "STATIC"

    def test_single_frame_tracking_grace_preserves_candidate(self):
        """A 1-frame image tracking loss when video is tracked grants grace and keeps candidate alive."""
        self._setup_settled_candidate("Z")
        hand = synthetic_hand()

        # Frame 1: Image hand temporarily lost (None), but video hand is present
        now = 10.1
        self.c.update(
            index=1,
            static_hand=hand,  # video hand present
            raw_hand=None,     # image hand dropped
            shape=(480, 640, 3),
            now=now,
            raw_hand_count=0
        )

        # Grace should have been granted
        self.assertTrue(self.c.tracking_grace_used)
        self.assertEqual(self.c.tracking_grace_frame_count, 1)
        self.assertIsNotNone(self.c.candidate, "Candidate must NOT be cancelled on single-frame drop")
        self.assertNotEqual(self.c.status, "REJECTED")
        self.assertEqual(self.c.frame_context.get("tracking_grace_used"), True)
        self.assertEqual(self.c.frame_context.get("tracking_grace_frame_count"), 1)
        self.assertEqual(self.c.frame_context.get("video_hand_detected"), True)
        self.assertEqual(self.c.frame_context.get("image_hand_detected"), False)

        # Frame 2: Image hand returns
        now = 10.2
        self.c.update(
            index=2,
            static_hand=hand,
            raw_hand=hand,
            shape=(480, 640, 3),
            now=now,
            raw_hand_count=1
        )

        # Candidate is still alive, grace streak counter resets to 0, tracking_grace_used remains True
        self.assertIsNotNone(self.c.candidate)
        self.assertEqual(self.c.tracking_grace_frame_count, 0)
        self.assertTrue(self.c.tracking_grace_used)

    def test_two_consecutive_drops_cancel_candidate(self):
        """>= 2 consecutive frames of image tracking loss exhaust grace and cancel candidate."""
        self._setup_settled_candidate("Z")
        hand = synthetic_hand()

        # Frame 1: drop 1 -> grace granted
        self.c.update(index=1, static_hand=hand, raw_hand=None, shape=(480, 640, 3), now=10.1, raw_hand_count=0)
        self.assertIsNotNone(self.c.candidate)
        self.assertEqual(self.c.tracking_grace_frame_count, 1)

        # Frame 2: drop 2 -> grace exhausted, candidate cancelled
        self.c.update(index=2, static_hand=hand, raw_hand=None, shape=(480, 640, 3), now=10.2, raw_hand_count=0)
        self.assertIsNone(self.c.candidate, "Candidate must be cancelled on 2nd consecutive drop")
        self.assertEqual(self.c.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
        self.assertEqual(self.c.tracking_grace_frame_count, 0)

    def test_simultaneous_video_and_image_loss_cancels_immediately(self):
        """If video also loses tracking, grace is NOT eligible and candidate is cancelled immediately."""
        self._setup_settled_candidate("Z")

        # Frame 1: Both image and video lost
        self.c.update(index=1, static_hand=None, raw_hand=None, shape=(480, 640, 3), now=10.1, raw_hand_count=0)
        self.assertIsNone(self.c.candidate, "Candidate must be cancelled immediately when video is also lost")
        self.assertEqual(self.c.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
        self.assertFalse(self.c.tracking_grace_used)

    def test_multiple_hands_cancel_immediately(self):
        """Ambiguous multiple hands (raw_hand_count > 1) cancel immediately without grace."""
        self._setup_settled_candidate("Z")
        hand = synthetic_hand()

        # Frame 1: 2 hands detected
        self.c.update(index=1, static_hand=hand, raw_hand=hand, shape=(480, 640, 3), now=10.1, raw_hand_count=2)
        self.assertIsNone(self.c.candidate, "Candidate must be cancelled immediately on ambiguous multiple hands")
        self.assertIn(self.c.candidate_cancel_reason, ("CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS", "MULTIPLE_HANDS_IDENTITY_UNVERIFIED"))


if __name__ == "__main__":
    unittest.main()
