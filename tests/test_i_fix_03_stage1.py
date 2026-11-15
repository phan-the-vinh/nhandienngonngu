"""PHASE I-FIX-03 Stage 1 controlled implementation verification suite.

Covers:
1. I-protection scenarios (8 scenarios: static I >=5s, pinky jitter, 1-frame Y/D blip,
   fake motion onset, J candidate overlap without switching I->J and allowing I commit,
   1-frame IMAGE loss, >=2 IMAGE loss, simultaneous VIDEO+IMAGE loss).
2. J-STEP-1 isolated 1-frame spike recovery scenarios (5 scenarios: raw valid J unperturbed,
   isolated 1-frame spike recovered with exact index & non-mutation, multi-frame spike rejected,
   step/teleport jump rejected, invalid trajectory rejected).
3. Z-STEP-1 pending_dynamic & candidate_settled tracking grace + telemetry scenarios (5 scenarios:
   1-frame IMAGE drop preserves candidate when VIDEO has 1 hand, 2 consecutive IMAGE drops cancel,
   simultaneous VIDEO+IMAGE loss cancels immediately, multiple hands cancel immediately, zero stale
   landmarks and zero hold time added on grace frame).
"""
from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features
from app.inference.integrated_jz_validation import (
    JTrajectorySettings,
    validate_j_trajectory,
    validate_z_trajectory,
)
from app.inference.letter_inference import Prediction
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_dynamic_jz_motion_gate import articulated_hand, curved_hand
from tests.test_static_l_protection import static_l_hand
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


def _make_j_features(path_xy: np.ndarray) -> np.ndarray:
    """Build (16, 69) feature array with ASL J/I-like pinky extension and given pinky path."""
    feat = np.zeros((16, 69), dtype=np.float64)
    lm = np.zeros((16, 21, 3), dtype=np.float64)
    lm[:, 0, :2] = [0.0, 0.0]
    for idx in (8, 12, 16):
        lm[:, idx, :2] = [0.05, -0.35]
    lm[:, 20, :2] = [-0.20, -1.00]
    feat[:, :63] = lm.reshape(16, 63)
    feat[:, 67:69] = np.asarray(path_xy, dtype=np.float64)
    return feat


def _smooth_j_path() -> np.ndarray:
    """Smooth downward-then-upward-hook J pinky path passing all raw validators."""
    pts = [
        [0.00, 0.00],
        [0.00, 0.14],
        [0.00, 0.28],
        [0.00, 0.44],
        [0.00, 0.60],
        [0.00, 0.76],
        [0.00, 0.90],
        [0.00, 1.00],
        [-0.08, 1.04],
        [-0.20, 1.00],
        [-0.32, 0.90],
        [-0.44, 0.78],
        [-0.54, 0.66],
        [-0.60, 0.56],
        [-0.63, 0.49],
        [-0.65, 0.45],
    ]
    return np.asarray(pts, dtype=np.float64)


class FakeBlipLettersPredictor(FakePredictor):
    """Fake static predictor supporting deterministic frame-indexed instant/stable blips."""

    def __init__(self, label: str = "I", shape: float = 0.0):
        super().__init__(label, shape)
        self.call_idx = 0
        self.blip_labels: dict[int, tuple[str, str, float]] = {}

    def predict_landmarks(self, landmarks):
        instant_l, stable_l, cur_shape = self.blip_labels.get(
            self.call_idx, (self.label, self.label, self.shape)
        )
        self.call_idx += 1
        return Prediction(
            instant_l,
            0.95,
            stable_l,
            0.95,
            (cur_shape,) * 63,
            (0.95, 0.05),
        )


class GroupIProtectionStage1Tests(unittest.TestCase):
    def _coord(self, letters=None, dyn_label: str = "J"):
        letters = letters or FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel(dyn_label)
        events = []
        coord = UnifiedAutoCoordinator(
            letters,
            digits,
            model,
            enable_experimental_jz=True,
            event_sink=events.append,
        )
        return coord, model, events

    def test_i1_static_i_held_over_5_seconds_uninterrupted(self):
        coord, model, _ = self._coord()
        commits = []
        base = curved_hand(0, "J")
        for k in range(180):  # 6.0s at 30fps
            raw = deepcopy(base)
            static_h = mirrored(raw)
            _, commit = coord.update(k, static_h, raw, (480, 640, 3), k / 30.0, raw_hand_count=1)
            if commit:
                commits.append(commit)
        self.assertEqual(commits, ["I"])
        self.assertEqual(coord.composer.text, "I")
        self.assertEqual(len(model.inputs), 0)
        self.assertFalse(coord.blocked)

    def test_i2_pinky_jitter_does_not_trigger_j_or_interrupt_i(self):
        coord, model, _ = self._coord()
        commits = []
        base = curved_hand(0, "J")
        for k in range(150):  # 5.0s with sub-threshold pinky noise
            raw = deepcopy(base)
            raw[20].x += 0.002 * np.sin(k * 0.7)
            raw[20].y += 0.002 * np.cos(k * 0.9)
            static_h = mirrored(raw)
            _, commit = coord.update(k, static_h, raw, (480, 640, 3), k / 30.0, raw_hand_count=1)
            if commit:
                commits.append(commit)
        self.assertEqual(commits, ["I"])
        self.assertEqual(len(model.inputs), 0)
        self.assertIsNone(coord.pending_dynamic)
        self.assertIsNone(coord.candidate)

    def test_i3_single_frame_y_or_d_blip_preserves_static_i_hold(self):
        letters = FakeBlipLettersPredictor("I", 0.0)
        # Single-frame Y and D instant blips (and stable blip when blip_grace_frames=2) during 3.5s I hold
        letters.blip_labels[45] = ("Y", "I", 0.0)
        letters.blip_labels[65] = ("D", "I", 0.0)
        coord, _, _ = self._coord(letters=letters)
        commits = []
        base = curved_hand(0, "J")
        for k in range(105):  # 3.5s
            raw = deepcopy(base)
            _, commit = coord.update(k, mirrored(raw), raw, (480, 640, 3), k / 30.0, raw_hand_count=1)
            if commit:
                commits.append(commit)
        self.assertEqual(commits, ["I"])
        self.assertEqual(coord.composer.text, "I")

    def test_i4_fake_motion_onset_never_commits_j_and_allows_i_commit(self):
        coord, _, _ = self._coord(dyn_label="J")
        commits = []
        idx = 0
        # Phase 1: small fake drift (not valid J)
        for k in range(40):
            raw = articulated_hand(k)
            _, c = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            if c:
                commits.append(c)
            idx += 1
        # Phase 2: settle on static I for 3.6s
        still = deepcopy(curved_hand(0, "J"))
        for _ in range(110):
            raw = deepcopy(still)
            _, c = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            if c:
                commits.append(c)
            idx += 1
        self.assertEqual(commits, ["I"])
        self.assertNotIn("J", coord.composer.text)
        self.assertNotIn("Z", coord.composer.text)

    def test_i5_cancelled_j_candidate_overlap_never_switches_to_j_and_i_commits(self):
        coord, _, events = self._coord(dyn_label="J")
        commits = []
        idx = 0
        for k in range(60):
            raw = curved_hand(k, "J")
            _, c = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            if c:
                commits.append(c)
            idx += 1
        self.assertIsNotNone(coord.pending_dynamic)
        self.assertNotEqual(getattr(coord.static_prediction, "instant_label", None), "J")
        # 2 consecutive image drops cancel overlapping J candidate
        h0 = curved_hand(60, "J")
        coord.update(idx, mirrored(h0), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
        idx += 1
        coord.update(idx, mirrored(h0), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
        idx += 1
        self.assertIsNone(coord.pending_dynamic)
        self.assertIsNone(coord.candidate)
        self.assertEqual(coord.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
        # User holds I for >3s and commits I cleanly
        for _ in range(110):
            raw = deepcopy(h0)
            _, c = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            if c:
                commits.append(c)
            idx += 1
        self.assertEqual(commits, ["I"])
        self.assertTrue(any(e["event"] == "JZ_CANDIDATE_CANCELLED" for e in events))

    def test_i6_single_frame_image_loss_keeps_static_i_holding(self):
        coord, _, _ = self._coord()
        commits = []
        base = curved_hand(0, "J")
        for k in range(105):  # 3.5s
            raw = None if k == 45 else deepcopy(base)
            rc = 0 if k == 45 else 1
            _, c = coord.update(k, mirrored(base), raw, (480, 640, 3), k / 30.0, raw_hand_count=rc)
            if c:
                commits.append(c)
        self.assertEqual(commits, ["I"])

    def test_i7_multi_frame_image_loss_cancels_candidate_or_lets_video_commit_i(self):
        coord, _, _ = self._coord()
        commits = []
        base = curved_hand(0, "J")
        for k in range(110):
            raw = None if k in (30, 31, 32) else deepcopy(base)
            rc = 0 if raw is None else 1
            _, c = coord.update(k, mirrored(base), raw, (480, 640, 3), k / 30.0, raw_hand_count=rc)
            if c:
                commits.append(c)
        self.assertEqual(commits, ["I"])
        self.assertEqual(coord.composer.text, "I")

    def test_i8_simultaneous_video_and_image_loss_cancels_candidate_and_never_phantom_commits(self):
        coord, _, _ = self._coord(dyn_label="J")
        idx = 0
        for k in range(60):
            raw = curved_hand(k, "J")
            coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            idx += 1
        self.assertIsNotNone(coord.pending_dynamic)
        # Simultaneous VIDEO + IMAGE loss
        _, commit = coord.update(idx, None, None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
        self.assertIsNone(commit)
        self.assertIsNone(coord.pending_dynamic)
        self.assertIsNone(coord.candidate)
        self.assertEqual(coord.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
        self.assertEqual(coord.composer.text, "")


class GroupJSpikeRecoveryStage1Tests(unittest.TestCase):
    def test_j1_raw_valid_j_unperturbed_with_audit_metadata(self):
        feat = _make_j_features(_smooth_j_path())
        res = validate_j_trajectory(feat)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_J_TRAJECTORY")
        self.assertFalse(res["spike_recovery_applied"])
        self.assertEqual(res["spike_repaired_frame_count"], 0)
        self.assertEqual(res["spike_repaired_frame_indices"], [])
        self.assertEqual(res["spike_recovery_reason"], "NONE")
        self.assertTrue(res["pre_recovery_valid"])
        self.assertTrue(res["post_recovery_valid"])

    def test_j2_isolated_one_frame_spike_recovered_without_mutating_input(self):
        feat = _make_j_features(_smooth_j_path())
        feat[5, 67:69] = [0.40, 4.20]  # single isolated outlier frame at index 5
        snapshot = feat.copy()
        res = validate_j_trajectory(feat)
        np.testing.assert_array_equal(feat, snapshot)  # Never mutate caller array
        self.assertTrue(res["valid"])
        self.assertEqual(res["reason"], "VALID_J_TRAJECTORY")
        self.assertFalse(res["pre_recovery_valid"])
        self.assertTrue(
            res["pre_recovery_reason"].startswith(
                ("SINGLE_FRAME_TRAJECTORY_SPIKE", "SINGLE_FRAME_DOWNWARD_COLLAPSE")
            )
        )
        self.assertTrue(res["post_recovery_valid"])
        self.assertTrue(res["spike_recovery_applied"])
        self.assertEqual(res["spike_repaired_frame_count"], 1)
        self.assertEqual(res["spike_repaired_frame_indices"], [5])

    def test_j3_multi_frame_spikes_rejected_without_recovery(self):
        # 2 adjacent asymmetric spike frames where raw triggers SINGLE_FRAME_TRAJECTORY_SPIKE
        feat_adj = _make_j_features(_smooth_j_path())
        feat_adj[4, 67:69] = [0.10, 4.40]
        feat_adj[5, 67:69] = [0.15, 2.40]
        res_adj = validate_j_trajectory(feat_adj)
        self.assertFalse(res_adj["pre_recovery_valid"])
        self.assertFalse(res_adj["valid"])
        self.assertFalse(res_adj["spike_recovery_applied"])
        self.assertEqual(res_adj["spike_repaired_frame_count"], 0)

        # 2 separated large spikes at frames 4 and 10
        feat_sep = _make_j_features(_smooth_j_path())
        feat_sep[4, 67:69] = [0.50, 3.80]
        feat_sep[10, 67:69] = [-2.50, 0.80]
        res_sep = validate_j_trajectory(feat_sep)
        self.assertFalse(res_sep["valid"])
        self.assertFalse(res_sep["spike_recovery_applied"])
        self.assertEqual(res_sep["spike_repaired_frame_count"], 0)

    def test_j4_step_teleport_jump_stays_rejected(self):
        p_teleport = np.zeros((16, 2), dtype=np.float64)
        p_teleport[7:, 1] = 1.0  # instantaneous step edge, not isolated glitch
        p_teleport[-1, 0] = -0.35
        p_teleport[-1, 1] = 0.80
        res = validate_j_trajectory(_make_j_features(p_teleport))
        self.assertFalse(res["valid"])
        self.assertFalse(res["spike_recovery_applied"])
        self.assertEqual(res["spike_repaired_frame_count"], 0)

    def test_j5_invalid_trajectory_and_static_l_rejected(self):
        feat_l, _ = build_sequence_features([static_l_hand()] * 16, [(640, 480)] * 16)
        res_l = validate_j_trajectory(feat_l)
        self.assertFalse(res_l["valid"])
        self.assertFalse(res_l["spike_recovery_applied"])

        p_no_hook = np.zeros((16, 2), dtype=np.float64)
        p_no_hook[:, 1] = np.linspace(0.0, 1.0, 16)
        res_nh = validate_j_trajectory(_make_j_features(p_no_hook))
        self.assertFalse(res_nh["valid"])
        self.assertFalse(res_nh["spike_recovery_applied"])


class GroupZTrackingGraceStage1Tests(unittest.TestCase):
    def _drive_to_z_pending(self):
        model = DynamicModel("Z")
        events = []
        coord = UnifiedAutoCoordinator(
            FakePredictor("A", 0.0),
            FakePredictor("0", 0.5),
            model,
            enable_experimental_jz=True,
            event_sink=events.append,
        )
        idx = 0
        for k in range(36):
            u_hand = curved_hand(min(int(k * 1.8), 60), "Z")
            coord.update(idx, mirrored(u_hand), u_hand, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            idx += 1
            if coord.pending_dynamic is not None:
                break
        self.assertIsNotNone(coord.pending_dynamic)
        self.assertFalse(coord.candidate_settled)
        return coord, events, idx

    def _drive_to_z_settled(self):
        coord, events, idx = self._drive_to_z_pending()
        end_hand = curved_hand(60, "Z")
        for _ in range(15):
            coord.update(
                idx,
                mirrored(deepcopy(end_hand)),
                deepcopy(end_hand),
                (480, 640, 3),
                idx / 30.0,
                raw_hand_count=1,
            )
            idx += 1
            if coord.candidate_settled:
                break
        self.assertTrue(coord.candidate_settled)
        self.assertIsNotNone(coord.candidate)
        return coord, events, idx

    def test_z1_one_frame_image_drop_during_pending_and_settled_preserves_and_commits_z(self):
        # Part A: pending_dynamic 1-frame IMAGE drop while VIDEO sees 1 hand
        coord, events, idx = self._drive_to_z_pending()
        end_hand = curved_hand(60, "Z")
        coord.update(idx, mirrored(deepcopy(end_hand)), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
        self.assertIsNotNone(coord.pending_dynamic)
        self.assertTrue(coord.frame_context["video_hand_detected"])
        self.assertFalse(coord.frame_context["image_hand_detected"])
        self.assertTrue(coord.frame_context["pending_dynamic"])
        self.assertFalse(coord.frame_context["candidate_settled"])
        self.assertEqual(coord.frame_context["image_drop_streak"], 1)
        self.assertTrue(coord.frame_context["tracking_grace_used"])
        self.assertIsNone(coord.frame_context["tracking_cancel_reason"])
        idx += 1
        # Recover & complete hold -> settles and commits Z
        committed = None
        for _ in range(110):
            _, c = coord.update(
                idx,
                mirrored(deepcopy(end_hand)),
                deepcopy(end_hand),
                (480, 640, 3),
                idx / 30.0,
                raw_hand_count=1,
            )
            idx += 1
            if c:
                committed = c
                break
        self.assertEqual(committed, "Z")
        self.assertEqual(coord.composer.text, "Z")

        # Part B: candidate_settled 1-frame IMAGE drop while VIDEO sees 1 hand
        coord2, _, idx2 = self._drive_to_z_settled()
        coord2.update(idx2, mirrored(deepcopy(end_hand)), None, (480, 640, 3), idx2 / 30.0, raw_hand_count=0)
        self.assertTrue(coord2.candidate_settled)
        self.assertIsNotNone(coord2.candidate)
        self.assertEqual(coord2.frame_context["image_drop_streak"], 1)
        self.assertTrue(coord2.frame_context["tracking_grace_used"])
        idx2 += 1
        committed2 = None
        for _ in range(110):
            _, c = coord2.update(
                idx2,
                mirrored(deepcopy(end_hand)),
                deepcopy(end_hand),
                (480, 640, 3),
                idx2 / 30.0,
                raw_hand_count=1,
            )
            idx2 += 1
            if c:
                committed2 = c
                break
        self.assertEqual(committed2, "Z")

    def test_z2_two_consecutive_image_drops_cancel_in_pending_and_settled(self):
        end_hand = curved_hand(60, "Z")
        for builder in (self._drive_to_z_pending, self._drive_to_z_settled):
            coord, _, idx = builder()
            coord.update(idx, mirrored(deepcopy(end_hand)), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
            self.assertEqual(coord.frame_context["image_drop_streak"], 1)
            idx += 1
            coord.update(idx, mirrored(deepcopy(end_hand)), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
            self.assertIsNone(coord.pending_dynamic)
            self.assertIsNone(coord.candidate)
            self.assertEqual(coord.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
            self.assertEqual(coord.frame_context["tracking_cancel_reason"], "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")

    def test_z3_simultaneous_video_and_image_loss_cancels_immediately_in_pending_and_settled(self):
        for builder in (self._drive_to_z_pending, self._drive_to_z_settled):
            coord, _, idx = builder()
            coord.update(idx, None, None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
            self.assertIsNone(coord.pending_dynamic)
            self.assertIsNone(coord.candidate)
            self.assertEqual(coord.candidate_cancel_reason, "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
            self.assertFalse(coord.frame_context["video_hand_detected"])
            self.assertFalse(coord.frame_context["image_hand_detected"])

    def test_z4_multiple_hands_cancel_immediately_in_pending_and_settled(self):
        end_hand = curved_hand(60, "Z")
        for builder in (self._drive_to_z_pending, self._drive_to_z_settled):
            coord, _, idx = builder()
            coord.update(
                idx,
                mirrored(deepcopy(end_hand)),
                deepcopy(end_hand),
                (480, 640, 3),
                idx / 30.0,
                raw_hand_count=2,
            )
            self.assertIsNone(coord.pending_dynamic)
            self.assertIsNone(coord.candidate)
            self.assertEqual(coord.candidate_cancel_reason, "MULTIPLE_HANDS_IDENTITY_UNVERIFIED")
            self.assertEqual(coord.frame_context["tracking_cancel_reason"], "MULTIPLE_HANDS_IDENTITY_UNVERIFIED")

    def test_z5_grace_frame_adds_zero_stale_landmarks_and_zero_hold_time(self):
        end_hand = curved_hand(60, "Z")
        coord, _, idx = self._drive_to_z_settled()
        # Advance hold for a few settled frames first
        for _ in range(5):
            coord.update(
                idx,
                mirrored(deepcopy(end_hand)),
                deepcopy(end_hand),
                (480, 640, 3),
                idx / 30.0,
                raw_hand_count=1,
            )
            idx += 1
        snap_obs = deepcopy(coord.endpoint_observation)
        snap_progress = coord.jz_hold.progress
        snap_started_at = coord.jz_hold.hold_started_at
        snap_state = coord.jz_hold.state
        snap_last_ts = coord.jz_hold._last_timestamp

        # Execute 1-frame IMAGE-only drop (grace frame)
        coord.update(idx, mirrored(deepcopy(end_hand)), None, (480, 640, 3), idx / 30.0, raw_hand_count=0)
        self.assertEqual(coord.jz_hold.progress, snap_progress)
        self.assertEqual(coord.jz_hold.hold_started_at, snap_started_at)
        self.assertEqual(coord.jz_hold.state, snap_state)
        self.assertEqual(coord.jz_hold._last_timestamp, snap_last_ts)
        self.assertEqual(coord.endpoint_observation, snap_obs)
