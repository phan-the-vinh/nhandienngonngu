"""Unit tests for Static L Protection and Motion Lifecycle Integrity (Phase 3.10.8-A4)."""
import unittest
from types import SimpleNamespace
import numpy as np

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand, curved_hand


class DynamicZClassifier:
    classes_ = np.asarray(["J", "Z"])
    def __init__(self, label="Z", confidence=0.99):
        self.label = label
        self.confidence = confidence
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(x.copy())
        p_z = self.confidence if self.label == "Z" else (1.0 - self.confidence)
        return np.asarray([[1.0 - p_z, p_z]])


def static_l_hand(jitter_std: float = 0.0, rng: np.random.Generator | None = None):
    """Generate hand landmarks corresponding to ASL letter L (index extended, thumb out).

    Optional jitter_std adds Gaussian noise (in normalized coords) to simulate MediaPipe jitter.
    """
    hand = synthetic_hand()
    # Emulate L pose: index finger (landmarks 5-8) extended up
    hand[8].y -= 0.15
    hand[7].y -= 0.10
    hand[6].y -= 0.05
    # Thumb (landmarks 1-4) extended out
    hand[4].x -= 0.10
    hand[3].x -= 0.06

    if jitter_std > 0:
        generator = rng or np.random.default_rng(42)
        noise = generator.normal(0.0, jitter_std, size=(21, 3))
        for idx, pt in enumerate(hand):
            pt.x += float(noise[idx, 0])
            pt.y += float(noise[idx, 1])
            pt.z += float(noise[idx, 2])
    return hand


class StaticLProtectionTests(unittest.TestCase):
    """Group A: Static L & Jitter Rejection tests."""

    def setUp(self):
        self.letters = FakePredictor("L", 0.6)
        self.digits = FakePredictor("0", 0.5)
        self.dynamic_model = DynamicZClassifier(label="Z", confidence=0.99)
        self.events = []
        self.coordinator = UnifiedAutoCoordinator(
            self.letters, self.digits, self.dynamic_model,
            enable_experimental_jz=True,
            event_sink=self.events.append,
        )

    def test_static_l_held_2s_without_z_activation(self):
        """Holding static L for 2.0s does not activate Z or form a confirmed Z candidate."""
        now = 0.0
        for i in range(60):  # 2.0 seconds at 30 fps
            hand = static_l_hand()
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertNotIn("Z", self.coordinator.composer.text)
        self.assertNotEqual(self.coordinator.display_label, "Z")

    def test_static_l_with_realistic_jitter_does_not_commit_z(self):
        """Static L with realistic MediaPipe jitter (2-8px ~= 0.003-0.012 norm) never commits Z."""
        now = 0.0
        rng = np.random.default_rng(12345)
        for i in range(100):  # 3.33 seconds
            hand = static_l_hand(jitter_std=0.008, rng=rng)
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertNotIn("Z", self.coordinator.composer.text)

    def test_fingertip_jitter_with_stable_wrist_does_not_commit_z(self):
        """Index fingertip tremor on static L does not cause Z to commit."""
        now = 0.0
        for i in range(100):
            hand = static_l_hand()
            # Add tremor specifically to landmark 8 (index tip)
            hand[8].x += 0.015 * np.sin(i * 0.8)
            hand[8].y += 0.015 * np.cos(i * 0.8)
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertNotIn("Z", self.coordinator.composer.text)

    def test_static_l_commits_successfully_when_held(self):
        """Static L commits successfully to TextComposer when held for required hold time."""
        now = 0.0
        for i in range(95):  # 3.16 seconds
            hand = static_l_hand()
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertEqual(self.coordinator.composer.text, "L")


class StateLifecycleIntegrityTests(unittest.TestCase):
    """Group D: State & Lifecycle Integrity tests."""

    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("0", 0.5)
        self.dynamic_model = DynamicZClassifier(label="Z", confidence=0.99)
        self.events = []
        self.coordinator = UnifiedAutoCoordinator(
            self.letters, self.digits, self.dynamic_model,
            enable_experimental_jz=True,
            event_sink=self.events.append,
        )

    def test_rejected_z_bout_resets_segmenter_state(self):
        """When a non-Z bout is rejected by trajectory validation, segmenter state resets."""
        now = 0.0
        # Produce 65 frames of articulated hand
        for i in range(65):
            hand = articulated_hand(i)
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        # Endpoint frames causing rejection
        for i in range(65, 100):
            hand = articulated_hand(64)
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        # Verify candidate cleared and segmenter ready/not actively locked
        self.assertIsNone(self.coordinator.candidate)
        self.assertIn(self.coordinator.session.last_detector_state, ("READY", "IDLE"))

    def test_dynamic_lock_released_after_rejection(self):
        """Unverified CAPTURING never blocks static ABC; verified Z candidate locks then releases on tracking loss."""
        now = 0.0
        for i in range(20):
            hand = articulated_hand(i)
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertFalse(self.coordinator.blocked)

        # Reset and drive a trajectory-verified Z candidate to lock, then lose tracking
        self.setUp()
        now = 0.0
        for i in range(65):
            hand = curved_hand(min(int(i * 1.8), 60), "Z")
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0
        self.assertTrue(self.coordinator.blocked)

        # Release hand / gap to cancel candidate
        for i in range(65, 75):
            self.coordinator.update(i, None, None, (480, 640, 3), now, raw_hand_count=0)
            now += 1.0 / 30.0

        self.assertFalse(self.coordinator.blocked)

    def test_no_double_commit_on_subsequent_hold(self):
        """Holding hand after commit does not produce duplicate commits without physical re-arm."""
        now = 0.0
        # Confirm static A
        self.letters.label, self.letters.shape = "A", 0.0
        for i in range(95):  # 3.16 seconds
            hand = synthetic_hand()
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertEqual(self.coordinator.composer.text, "A")

        # Continue holding same hand without release for another 2 seconds
        for i in range(95, 155):
            hand = synthetic_hand()
            self.coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        # Text must remain "A", not "AA"
        self.assertEqual(self.coordinator.composer.text, "A")


if __name__ == "__main__":
    unittest.main()
