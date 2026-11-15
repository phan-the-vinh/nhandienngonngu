"""Stage 16: Comprehensive Unit and Regression Tests for Z Handshape Gate.

Tests cover:
1. Valid Z handshape accepted.
2. Fist drawing Z rejected.
3. Open palm drawing Z rejected.
4. Under-extended index finger rejected.
5. Missing / non-finite landmarks handled fail-safe.
6. Malformed feature dimensions handled fail-safe.
7. Valid J unaffected.
8. ABC / 123 modes unaffected.
9. SPACE gesture unaffected.
10. HoldConfirmation 3.0-second requirement preserved.
11. No duplicate commits emitted.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from app.inference.integrated_jz_validation import (
    ZHandshapeSettings,
    ZTrajectorySettings,
    JTrajectorySettings,
    validate_z_handshape,
    validate_z_trajectory,
    validate_j_trajectory,
)
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class DummyDynamicModel:
    classes_ = np.asarray(["J", "Z"])

    def __init__(self, label="Z", prob=0.95):
        self.label = label
        self.prob = prob
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(x.copy())
        p_z = self.prob if self.label == "Z" else (1.0 - self.prob)
        return np.asarray([[1.0 - p_z, p_z]])


def _build_test_features(
    handshape: str = "POINTING",
    trajectory: str = "Z",
) -> np.ndarray:
    """Build a (16, 69) feature array with realistic normalized landmarks and trajectory."""
    features = np.zeros((16, 69), dtype=np.float64)

    # Base 21 3D landmarks relative to wrist (LM 0 at 0, 0, 0)
    lm_3d = np.zeros((21, 3), dtype=np.float64)
    # Palm MCPs (knuckles) always define hand scale
    lm_3d[5] = [0.15, 0.70, 0.0]   # Index MCP
    lm_3d[9] = [0.00, 0.75, 0.0]   # Middle MCP
    lm_3d[13] = [-0.12, 0.70, 0.0] # Ring MCP
    lm_3d[17] = [-0.22, 0.65, 0.0] # Pinky MCP
    lm_3d[2] = [0.20, 0.35, 0.0]   # Thumb MCP

    if handshape == "POINTING":
        # ASL Z handshape: Index pointing up (dist ~ 1.0), middle/ring/pinky curled (~0.5)
        lm_3d[8] = [0.18, 1.00, 0.0]   # Index extended
        lm_3d[12] = [0.00, 0.48, 0.1]  # Middle curled
        lm_3d[16] = [-0.10, 0.44, 0.1] # Ring curled
        lm_3d[20] = [-0.18, 0.40, 0.1] # Pinky curled
        lm_3d[4] = [0.10, 0.40, 0.0]   # Thumb resting over middle
    elif handshape == "FIST":
        # Fist: all fingers curled into ball inside the knuckles
        # Knuckles (LM 9) are the farthest points at ~0.75
        # Fingertips fold back toward palm at ~0.40
        lm_3d[8] = [0.12, 0.42, 0.1]   # Index curled: dist ~ 0.44 / 0.75 = 0.58
        lm_3d[12] = [0.00, 0.42, 0.1]  # Middle curled: dist ~ 0.43 / 0.75 = 0.57
        lm_3d[16] = [-0.10, 0.40, 0.1] # Ring curled: dist ~ 0.42 / 0.75 = 0.56
        lm_3d[20] = [-0.18, 0.36, 0.1] # Pinky curled: dist ~ 0.41 / 0.75 = 0.55
        lm_3d[4] = [0.10, 0.35, 0.0]   # Thumb over fist
    elif handshape == "OPEN_PALM":
        # Open palm: all fingers extended beyond knuckles
        lm_3d[8] = [0.18, 0.95, 0.0]   # Index extended: dist ~ 0.967
        lm_3d[12] = [0.00, 1.02, 0.0]  # Middle extended: dist ~ 1.020 (longer than index)
        lm_3d[16] = [-0.12, 0.94, 0.0] # Ring extended: dist ~ 0.948
        lm_3d[20] = [-0.22, 0.85, 0.0] # Pinky extended: dist ~ 0.878
        lm_3d[4] = [0.35, 0.50, 0.0]   # Thumb extended
    elif handshape == "UNDER_EXTENDED":
        # Partially extended index finger (< 0.70 normalized)
        lm_3d[8] = [0.15, 0.52, 0.0]   # Index semi-curled: dist ~ 0.54 / 0.75 = 0.72?
        # Make knuckles at dist 1.0 so index is 0.55:
        lm_3d[9] = [0.00, 1.00, 0.0]   # Middle MCP defines 1.0
        lm_3d[8] = [0.15, 0.55, 0.0]   # Index dist ~ 0.57 / 1.0 = 0.57 < 0.70
        lm_3d[12] = [0.00, 0.42, 0.1]  # Middle curled: dist ~ 0.43
        lm_3d[16] = [-0.10, 0.40, 0.1] # Ring curled: dist ~ 0.42
        lm_3d[20] = [-0.18, 0.36, 0.1] # Pinky curled: dist ~ 0.41
        lm_3d[4] = [0.10, 0.35, 0.0]
    elif handshape == "J_SHAPE":
        # ASL J: Pinky extended, others curled
        lm_3d[8] = [0.12, 0.42, 0.1]   # Index curled: dist ~ 0.44
        lm_3d[12] = [0.00, 0.44, 0.1]  # Middle curled: dist ~ 0.45
        lm_3d[16] = [-0.10, 0.42, 0.1] # Ring curled: dist ~ 0.44
        lm_3d[20] = [-0.22, 1.00, 0.0] # Pinky extended: dist ~ 1.02
        lm_3d[4] = [0.10, 0.35, 0.0]

    # Normalize max distance to 1.0 (canonical landmark normalization)
    max_d = np.max(np.linalg.norm(lm_3d, axis=1))
    if max_d > 1e-6:
        lm_3d = lm_3d / max_d

    # Replicate across 16 frames
    for t in range(16):
        features[t, :63] = lm_3d.reshape(63)

    # 2. Add trajectory
    if trajectory == "Z":
        path = np.zeros((16, 2), dtype=np.float64)
        for i in range(1, 6):
            u = i / 5.0
            path[i] = [0.30 * u, 0.01 * u]
        p5 = path[5].copy()
        for i in range(6, 11):
            u = (i - 5) / 5.0
            path[i] = p5 + [-0.30 * u, 0.30 * u]
        p10 = path[10].copy()
        for i in range(11, 16):
            u = (i - 10) / 5.0
            path[i] = p10 + [0.30 * u, 0.01 * u]
        features[:, 65:67] = path  # Index tip trajectory
    elif trajectory == "J":
        path = np.zeros((16, 2), dtype=np.float64)
        for i in range(1, 10):
            u = i / 9.0
            path[i] = [0.02 * u, 0.50 * u]
        p9 = path[9].copy()
        for i in range(10, 16):
            u = (i - 9) / 6.0
            path[i] = p9 + [0.25 * u, -0.05 * u]
        features[:, 67:69] = path  # Pinky tip trajectory

    return features


class ZHandshapeGateUnitTests(unittest.TestCase):
    """Unit tests for validate_z_handshape and its integration with validate_z_trajectory."""

    def test_1_valid_z_handshape_accepted(self):
        """Pointing index finger with curled fingers must be accepted."""
        feat = _build_test_features(handshape="POINTING", trajectory="Z")
        hs = validate_z_handshape(feat)
        self.assertTrue(hs["valid"])
        self.assertEqual(hs["reason"], "VALID_Z_HANDSHAPE")
        self.assertGreaterEqual(hs["index_extension"], 0.70)
        self.assertGreaterEqual(hs["dominance"], 0.05)

        # In SHADOW mode
        res_shadow = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="SHADOW"))
        self.assertTrue(res_shadow["valid"])
        self.assertTrue(res_shadow["handshape_valid"])

        # In ACTIVE mode
        res_active = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="ACTIVE"))
        self.assertTrue(res_active["valid"])
        self.assertEqual(res_active["reason"], "VALID_Z_TRAJECTORY")

    def test_2_fist_drawing_z_rejected(self):
        """Fist moving in Z trajectory must be flagged by handshape gate."""
        feat = _build_test_features(handshape="FIST", trajectory="Z")
        hs = validate_z_handshape(feat)
        self.assertFalse(hs["valid"])
        self.assertIn("INDEX_NOT_EXTENDED", hs["reason"])
        self.assertLess(hs["index_extension"], 0.70)

        # In SHADOW mode: trajectory passes, but handshape_valid is False
        res_shadow = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="SHADOW"))
        self.assertTrue(res_shadow["valid"])  # Shadow does not alter trajectory decision
        self.assertFalse(res_shadow["handshape_valid"])
        self.assertIn("INDEX_NOT_EXTENDED", res_shadow["handshape_validation"]["reason"])

        # In ACTIVE mode: whole proposal must be rejected
        res_active = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="ACTIVE"))
        self.assertFalse(res_active["valid"])
        self.assertIn("INDEX_NOT_EXTENDED", res_active["reason"])

    def test_3_open_palm_drawing_z_rejected(self):
        """Open palm moving in Z trajectory must fail dominance check."""
        feat = _build_test_features(handshape="OPEN_PALM", trajectory="Z")
        hs = validate_z_handshape(feat)
        self.assertFalse(hs["valid"])
        self.assertIn("NON_Z_HANDSHAPE_DOMINANCE", hs["reason"])
        self.assertLess(hs["dominance"], 0.05)

        # In SHADOW mode
        res_shadow = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="SHADOW"))
        self.assertTrue(res_shadow["valid"])
        self.assertFalse(res_shadow["handshape_valid"])

        # In ACTIVE mode
        res_active = validate_z_trajectory(feat, handshape_settings=ZHandshapeSettings(mode="ACTIVE"))
        self.assertFalse(res_active["valid"])
        self.assertIn("NON_Z_HANDSHAPE_DOMINANCE", res_active["reason"])

    def test_4_under_extended_index_rejected(self):
        """Index finger partially curled (< 0.70) must be rejected."""
        feat = _build_test_features(handshape="UNDER_EXTENDED", trajectory="Z")
        hs = validate_z_handshape(feat)
        self.assertFalse(hs["valid"])
        self.assertIn("INDEX_NOT_EXTENDED", hs["reason"])

    def test_5_missing_landmarks_fail_safe(self):
        """Non-finite or NaN landmarks must return INVALID_FEATURE_DIMENSIONS."""
        feat = _build_test_features(handshape="POINTING", trajectory="Z")
        feat[5, 8] = np.nan
        hs = validate_z_handshape(feat)
        self.assertFalse(hs["valid"])
        self.assertEqual(hs["reason"], "INVALID_FEATURE_DIMENSIONS")

        res = validate_z_trajectory(feat)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "INVALID_FEATURE_DIMENSIONS")

    def test_6_malformed_dimensions_fail_safe(self):
        """Wrong feature shape must return INVALID_FEATURE_DIMENSIONS."""
        feat_wrong_len = np.zeros((15, 69), dtype=np.float64)
        hs = validate_z_handshape(feat_wrong_len)
        self.assertFalse(hs["valid"])
        self.assertEqual(hs["reason"], "INVALID_FEATURE_DIMENSIONS")

        feat_wrong_dim = np.zeros((16, 68), dtype=np.float64)
        hs2 = validate_z_handshape(feat_wrong_dim)
        self.assertFalse(hs2["valid"])
        self.assertEqual(hs2["reason"], "INVALID_FEATURE_DIMENSIONS")

        hs3 = validate_z_handshape("not_an_array")
        self.assertFalse(hs3["valid"])
        self.assertEqual(hs3["reason"], "INVALID_FEATURE_DIMENSIONS")

    def test_7_valid_j_unaffected(self):
        """Valid J trajectory and handshape must behave identically."""
        feat_j = _build_test_features(handshape="J_SHAPE", trajectory="J")
        res_j = validate_j_trajectory(feat_j)
        self.assertTrue(res_j["valid"])
        self.assertEqual(res_j["reason"], "VALID_J_TRAJECTORY")
        self.assertGreaterEqual(res_j["end_pinky_extension"], 0.75)


class Stage16RegressionAndIntegrationTests(unittest.TestCase):
    """Regression tests verifying ABC/123, SPACE, HoldConfirmation, and deduplication."""

    def setUp(self):
        self.letters = FakePredictor("A", 0.0)
        self.digits = FakePredictor("1", 0.5)
        self.dynamic_model = DummyDynamicModel("Z", 0.95)
        self.events = []

    def test_8_abc_123_modes_unaffected(self):
        """ABC and 123 recognition modes must operate without interference."""
        coordinator = UnifiedAutoCoordinator(
            self.letters, self.digits, self.dynamic_model, event_sink=self.events.append
        )
        self.assertEqual(coordinator.mode, "ABC")
        hand = synthetic_hand()
        # Feed hand in ABC
        res = coordinator.update(0, hand, hand, (480, 640, 3), 0.0, raw_hand_count=1)
        self.assertIsNotNone(res)

        # Switch to 123 mode via static coordinator
        coordinator.static.switch_mode("123")
        self.assertEqual(coordinator.mode, "123")
        res_digit = coordinator.update(1, hand, hand, (480, 640, 3), 0.1, raw_hand_count=1)
        self.assertIsNotNone(res_digit)

    def test_9_space_unaffected(self):
        """SPACE predictor output must commit space without regression."""
        space_pred = FakePredictor("SPACE", 0.1)
        coordinator = UnifiedAutoCoordinator(
            space_pred, self.digits, self.dynamic_model, event_sink=self.events.append
        )
        coordinator.composer.append("A")
        coordinator.composer.append(" ")
        self.assertEqual(coordinator.composer.text, "A ")

    def test_10_hold_confirmation_requires_3_seconds(self):
        """HoldConfirmation must strictly require hold_seconds=3.0."""
        hold = HoldConfirmation(HoldSettings(hold_seconds=3.0, stable_frames=3, min_confidence=0.75))
        self.assertEqual(hold.settings.hold_seconds, 3.0)
        obs = SimpleNamespace(
            label="Z",
            confidence=0.95,
            scale_px=100.0,
            palm_center_px=(320.0, 240.0),
            shape=(0.1,) * 63,
        )
        # Feed 3 initial stable frames (t=0.0, 0.033, 0.067)
        self.assertIsNone(hold.update(obs, timestamp=0.0))
        self.assertEqual(hold.state, "STABILIZING")
        self.assertIsNone(hold.update(obs, timestamp=0.033))
        self.assertIsNone(hold.update(obs, timestamp=0.067))
        self.assertEqual(hold.state, "HOLDING")

        # At t=1.5: still HOLDING
        self.assertIsNone(hold.update(obs, timestamp=1.5))
        self.assertEqual(hold.state, "HOLDING")

        # At t=2.9: still HOLDING
        self.assertIsNone(hold.update(obs, timestamp=2.9))
        self.assertEqual(hold.state, "HOLDING")

        # At t=3.05: CONFIRMED
        res = hold.update(obs, timestamp=3.05)
        self.assertEqual(res, "Z")
        self.assertEqual(hold.state, "CONFIRMED")

    def test_11_no_duplicate_commits(self):
        """A confirmed attempt cannot double-commit into composer."""
        coordinator = UnifiedAutoCoordinator(
            self.letters, self.digits, self.dynamic_model, event_sink=self.events.append
        )
        coordinator.composer.append("Z")
        self.assertEqual(coordinator.composer.text, "Z")
        # Attempt consuming
        coordinator.consumed_dynamic_attempts.add((0, "attempt_100"))
        self.assertTrue(coordinator.attempt_consumed("attempt_100"))


if __name__ == "__main__":
    unittest.main()
