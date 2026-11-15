"""Unit tests for ASL Letter Z Trajectory Validation and Coordinator Integration (Phase 3.10.8-A4)."""
import unittest
from types import SimpleNamespace
import numpy as np

from app.inference.integrated_jz_validation import (
    ZTrajectorySettings,
    validate_z_trajectory,
    validate_proposal,
)
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand, curved_hand


def make_synthetic_z_features(
    excursion: float = 1.0,
    dx_legs: tuple[float, float, float] = (0.35, -0.35, 0.35),
    dy_legs: tuple[float, float, float] = (0.02, 0.40, 0.02),
) -> np.ndarray:
    """Construct synthetic (16, 69) feature array with custom index-fingertip trajectory."""
    features = np.zeros((16, 69), dtype=np.float64)
    # 16 frames split into 3 segments: 0-5, 5-10, 10-15
    path = np.zeros((16, 2), dtype=np.float64)
    # Leg 1: frames 0 -> 5
    for i in range(1, 6):
        u = i / 5.0
        path[i] = [dx_legs[0] * u, dy_legs[0] * u]
    # Leg 2: frames 5 -> 10
    p5 = path[5].copy()
    for i in range(6, 11):
        u = (i - 5) / 5.0
        path[i] = p5 + [dx_legs[1] * u, dy_legs[1] * u]
    # Leg 3: frames 10 -> 15
    p10 = path[10].copy()
    for i in range(11, 16):
        u = (i - 10) / 5.0
        path[i] = p10 + [dx_legs[2] * u, dy_legs[2] * u]

    features[:, 65:67] = path
    return features


class DynamicZModel:
    classes_ = np.asarray(["J", "Z"])
    def __init__(self, label="Z", confidence=0.98):
        self.label = label
        self.confidence = confidence
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(x.copy())
        p_z = self.confidence if self.label == "Z" else (1.0 - self.confidence)
        return np.asarray([[1.0 - p_z, p_z]])


class ZTrajectoryValidationTests(unittest.TestCase):
    """Group B: Z Trajectory Validation unit tests."""

    def test_valid_z_trajectory_accepted(self):
        features = make_synthetic_z_features(
            excursion=0.8,
            dx_legs=(0.30, -0.30, 0.30),
            dy_legs=(0.02, 0.30, 0.02),
        )
        res = validate_z_trajectory(features)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")
        self.assertGreaterEqual(res["excursion"], 0.35)

    def test_insufficient_excursion_rejected(self):
        features = make_synthetic_z_features(
            dx_legs=(0.05, -0.05, 0.05),
            dy_legs=(0.01, 0.05, 0.01),
        )
        res = validate_z_trajectory(features)
        self.assertFalse(res["valid"])
        self.assertIn("INSUFFICIENT_Z_EXCURSION", res["reason"])

    def test_missing_first_direction_reversal_rejected(self):
        # Top stroke and diagonal stroke move in the same horizontal direction
        features = make_synthetic_z_features(
            dx_legs=(0.30, 0.30, -0.30),
            dy_legs=(0.02, 0.30, 0.02),
        )
        res = validate_z_trajectory(features)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "FIRST_DIRECTION_REVERSAL_MISSING")

    def test_missing_second_direction_reversal_rejected(self):
        # Diagonal and bottom stroke move in the same horizontal direction
        features = make_synthetic_z_features(
            dx_legs=(0.30, -0.30, -0.30),
            dy_legs=(0.02, 0.30, 0.02),
        )
        res = validate_z_trajectory(features)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "SECOND_DIRECTION_REVERSAL_MISSING")

    def test_diagonal_not_downward_rejected(self):
        # Diagonal stroke moves upward in image coordinates (negative dy)
        features = make_synthetic_z_features(
            dx_legs=(0.40, -0.40, 0.40),
            dy_legs=(0.02, -0.20, 0.02),
        )
        res = validate_z_trajectory(features)
        self.assertFalse(res["valid"])
        self.assertIn("DIAGONAL_NOT_DOWNWARD", res["reason"])

    def test_top_stroke_not_horizontal_rejected(self):
        # Top stroke has steep slope |dy/dx| > 0.70
        features = make_synthetic_z_features(
            dx_legs=(0.20, -0.30, 0.30),
            dy_legs=(0.25, 0.30, 0.02),
        )
        res = validate_z_trajectory(features)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "TOP_STROKE_NOT_HORIZONTAL")

    def test_high_confidence_z_proposal_without_trajectory_blocked_from_commit(self):
        """A classifier predicting Z with 99% confidence on non-Z motion cannot commit."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicZModel(label="Z", confidence=0.99)
        events = []
        coordinator = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            event_sink=events.append,
        )

        # Move 65 frames with articulated hand (pinky moving only, no index Z trajectory)
        index = 0
        for i in range(65):
            hand = articulated_hand(i)
            coordinator.update(index, hand, hand, (480, 640, 3), index / 30.0, raw_hand_count=1)
            index += 1

        # Feed quiet endpoint frames
        for i in range(65, 110):
            hand = articulated_hand(64)
            coordinator.update(index, hand, hand, (480, 640, 3), index / 30.0, raw_hand_count=1)
            index += 1

        # Assert Z was NOT committed to text composer
        self.assertNotIn("Z", coordinator.composer.text)
        # Verify rejection event emitted
        rejections = [e for e in events if e.get("event") == "JZ_PROPOSAL_REJECTED_TRAJECTORY"]
        self.assertGreaterEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["reason"], "Z_TRAJECTORY_VALIDATION_FAILED")


class CompatibilityTests(unittest.TestCase):
    """Group C: Compatibility and Non-Regression tests."""

    def test_dynamic_j_unaffected_by_z_validation(self):
        """Dynamic J retains normal functionality without being gated by Z trajectory validation."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicZModel(label="J", confidence=0.98)
        events = []
        coordinator = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            event_sink=events.append,
        )

        index = 0
        for i in range(65):
            hand = curved_hand(i, label="J")
            coordinator.update(index, hand, hand, (480, 640, 3), index / 30.0, raw_hand_count=1)
            index += 1

        for i in range(65, 120):
            hand = curved_hand(64, label="J")
            coordinator.update(index, hand, hand, (480, 640, 3), index / 30.0, raw_hand_count=1)
            index += 1

        # J was able to be processed without Z trajectory rejection
        rejections = [e for e in events if e.get("event") == "JZ_PROPOSAL_REJECTED_TRAJECTORY"]
        self.assertEqual(len(rejections), 0)

    def test_static_abc_letters_commit_normally(self):
        """Static letters A, P, L commit normally without interference."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicZModel(label="Z")
        coordinator = UnifiedAutoCoordinator(letters, digits, model)

        now = 0.0
        # Hold A for 3.1 seconds (shape 0.0)
        letters.label, letters.shape = "A", 0.0
        for i in range(95):
            hand = synthetic_hand()
            coordinator.update(i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertEqual(coordinator.composer.text, "A")

        # Gap
        for i in range(10):
            coordinator.update(95 + i, None, None, (480, 640, 3), now, raw_hand_count=0)
            now += 1.0 / 30.0

        # Hold L for 3.1 seconds (shape 0.6)
        letters.label, letters.shape = "L", 0.6
        for i in range(95):
            hand = synthetic_hand()
            coordinator.update(105 + i, hand, hand, (480, 640, 3), now, raw_hand_count=1)
            now += 1.0 / 30.0

        self.assertEqual(coordinator.composer.text, "AL")

    def test_mode_switch_abc_to_123_and_back(self):
        """Switching between ABC and 123 modes operates cleanly."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("1", 0.5)
        model = DynamicZModel(label="Z")
        coordinator = UnifiedAutoCoordinator(letters, digits, model)

        coordinator.composer.text = "HELLO"
        coordinator.switch_mode("123")
        self.assertEqual(coordinator.mode, "123")
        self.assertEqual(coordinator.composer.text, "HELLO")

        coordinator.switch_mode("ABC")
        self.assertEqual(coordinator.mode, "ABC")
        self.assertEqual(coordinator.composer.text, "HELLO")


class ZDynamicPositiveFixturesTests(unittest.TestCase):
    """Group D: 10 Positive Z Gesture Variations."""

    def test_positive_1_standard_uniform(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:6] = np.linspace([0, 0], [0.35, 0.01], 6)
        p[5:11] = np.linspace([0.35, 0.01], [0.0, 0.40], 6)
        p[10:16] = np.linspace([0.0, 0.40], [0.35, 0.41], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")
        self.assertGreaterEqual(res["corner1_index"], 1)
        self.assertGreater(res["corner2_index"], res["corner1_index"])
        self.assertGreater(res["endpoint_index"], res["corner2_index"])

    def test_positive_2_long_top_short_mid(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:8] = np.linspace([0, 0], [0.45, 0.02], 8)
        p[7:11] = np.linspace([0.45, 0.02], [0.15, 0.30], 4)
        p[10:16] = np.linspace([0.15, 0.30], [0.45, 0.32], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_3_short_top_long_mid(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:3] = np.linspace([0, 0], [0.18, 0.01], 3)
        p[2:11] = np.linspace([0.18, 0.01], [-0.15, 0.45], 9)
        p[10:16] = np.linspace([-0.15, 0.45], [0.15, 0.46], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_4_early_corner1(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:3] = np.linspace([0, 0], [0.25, 0.01], 3)
        p[2:8] = np.linspace([0.25, 0.01], [-0.05, 0.35], 6)
        p[7:16] = np.linspace([-0.05, 0.35], [0.30, 0.37], 9)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_5_late_corner2(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:7] = np.linspace([0, 0], [0.35, 0.01], 7)
        p[6:14] = np.linspace([0.35, 0.01], [0.05, 0.40], 8)
        p[13:16] = np.linspace([0.05, 0.40], [0.25, 0.41], 3)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_6_non_uniform_speed(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        t_s0 = np.linspace(0, 1, 6)**2
        for i, t in enumerate(t_s0):
            p[i] = [0.35 * t, 0.02 * t]
        t_s1 = 1 - (1 - np.linspace(0, 1, 6))**2
        p5 = p[5].copy()
        for i, t in enumerate(t_s1):
            p[5+i] = p5 + [-0.35 * t, 0.38 * t]
        p10 = p[10].copy()
        for i in range(1, 6):
            p[10+i] = p10 + [0.30 * (i/5.0), 0.01 * (i/5.0)]
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_7_pause_between_strokes(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:5] = np.linspace([0, 0], [0.35, 0.01], 5)
        p[4] = [0.35, 0.01]
        p[5] = [0.35, 0.01]
        p[5:10] = np.linspace([0.35, 0.01], [0.0, 0.38], 5)
        p[9:11] = [0.0, 0.38]
        p[10:16] = np.linspace([0.0, 0.38], [0.30, 0.39], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_8_terminal_hold(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:4] = np.linspace([0, 0], [0.30, 0.01], 4)
        p[3:7] = np.linspace([0.30, 0.01], [0.02, 0.35], 4)
        p[6:10] = np.linspace([0.02, 0.35], [0.32, 0.36], 4)
        np.random.seed(42)
        for i in range(10, 16):
            p[i] = p[9] + np.random.uniform(-0.01, 0.01, 2)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_9_asymmetrical_strokes(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:5] = np.linspace([0, 0], [0.42, 0.03], 5)
        p[4:10] = np.linspace([0.42, 0.03], [0.18, 0.35], 6)
        p[9:16] = np.linspace([0.18, 0.35], [0.35, 0.36], 7)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")

    def test_positive_10_diagonal_angle_variation(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:5] = np.linspace([0, 0], [0.30, 0.01], 5)
        p[4:11] = np.linspace([0.30, 0.01], [0.12, 0.52], 7)
        p[10:16] = np.linspace([0.12, 0.52], [0.40, 0.53], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_Z_TRAJECTORY")


class ZNegativeFixturesTests(unittest.TestCase):
    """Group E: 10 Negative Fixtures."""

    def test_negative_1_static_l(self):
        f = np.zeros((16, 69))
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])
        self.assertIn("INSUFFICIENT_Z_EXCURSION", res["reason"])

    def test_negative_2_static_l_jitter(self):
        f = np.zeros((16, 69))
        np.random.seed(123)
        f[:, 65:67] = np.random.uniform(-0.05, 0.05, (16, 2))
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])
        self.assertIn("INSUFFICIENT_Z_EXCURSION", res["reason"])

    def test_negative_3_pure_horizontal(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[:, 0] = np.linspace(0, 0.50, 16)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_4_pure_diagonal(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[:, 0] = np.linspace(0, -0.40, 16)
        p[:, 1] = np.linspace(0, 0.40, 16)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_5_missing_leg1(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:8] = np.linspace([0, 0], [-0.35, 0.40], 8)
        p[7:16] = np.linspace([-0.35, 0.40], [0.05, 0.41], 9)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_6_missing_leg2(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:8] = np.linspace([0, 0], [0.35, 0.01], 8)
        p[7:16] = np.linspace([0.35, 0.01], [0.05, 0.02], 9)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_7_missing_leg3(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:8] = np.linspace([0, 0], [0.35, 0.01], 8)
        p[7:16] = np.linspace([0.35, 0.01], [0.05, 0.40], 9)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_8_reversed_order(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:6] = np.linspace([0, 0], [0.35, 0.01], 6)
        p[5:11] = np.linspace([0.35, 0.01], [0.0, -0.40], 6)
        p[10:16] = np.linspace([0.0, -0.40], [0.35, -0.39], 6)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_9_random_motion(self):
        f = np.zeros((16, 69))
        np.random.seed(999)
        steps = np.random.randn(16, 2) * 0.15
        p = np.cumsum(steps, axis=0)
        p -= p[0]
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])

    def test_negative_10_single_spike(self):
        f = np.zeros((16, 69))
        p = np.zeros((16, 2))
        p[0:8] = np.linspace([0, 0], [0.45, 0.02], 8)
        p[7:16] = np.linspace([0.45, 0.02], [0.0, 0.0], 9)
        f[:, 65:67] = p
        res = validate_z_trajectory(f)
        self.assertFalse(res["valid"])


if __name__ == "__main__":
    unittest.main()
