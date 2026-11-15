"""Regression & safety tests for PHASE I-FIX-01:
- Static 'I' protection against false J/Z hijacking and unverified CAPTURING suppression
- J trajectory validation without weakening A4/A9 Z validation
- Bounded HoldConfirmation blip grace with strict identity/tracking/multi-hand/no-false-commit guards
"""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features
from app.inference.integrated_jz_validation import (
    JTrajectorySettings,
    ZTrajectorySettings,
    validate_j_trajectory,
    validate_z_trajectory,
)
from app.inference.letter_inference import Prediction
from app.inference.push_confirmation_camera_test import observe_hand
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from src.model import load_dynamic
from tests.test_dynamic_jz_motion_gate import articulated_hand, curved_hand
from tests.test_static_l_protection import static_l_hand
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand

ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = ROOT / "data" / "experiments" / "dynamic_jz_v1" / "diagnostics"


def _static_i_hand(jitter_std: float = 0.0, offset_x: float = 0.0, offset_y: float = 0.0,
                   rng: np.random.Generator | None = None):
    """Realistic static 'I' landmark pose (wrist->pinky extended, index/middle/ring curled)."""
    hand = synthetic_hand()
    # Curl index/middle/ring closer to MCP/palm, keep pinky [20] strongly extended up
    for tip, pip in ((8, 6), (12, 10), (16, 14)):
        hand[tip].x = hand[0].x
        hand[tip].y = hand[0].y - 0.05
        hand[pip].y = hand[0].y - 0.06
    hand[17].y = hand[0].y - 0.08
    hand[18].y = hand[0].y - 0.14
    hand[19].y = hand[0].y - 0.19
    hand[20].x = hand[17].x
    hand[20].y = hand[0].y - 0.25

    noise = (rng.normal(0.0, jitter_std, size=(21, 3)) if jitter_std > 0 else np.zeros((21, 3)))
    for idx, pt in enumerate(hand):
        pt.x += offset_x + float(noise[idx, 0])
        pt.y += offset_y + float(noise[idx, 1])
        pt.z += float(noise[idx, 2])
    return hand


def _make_pinky_path_features(path_xy: np.ndarray) -> np.ndarray:
    feat = np.zeros((16, 69), dtype=np.float64)
    feat[:, 67:69] = path_xy
    return feat


class GroupIFalseJPreventionTests(unittest.TestCase):
    def test_static_i_held_over_10_seconds_continuous_commits_and_never_triggers_j(self):
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")
        c = UnifiedAutoCoordinator(letters, digits, model, enable_experimental_jz=True)
        rng = np.random.default_rng(2026)

        saw_jz_display = False
        saw_blocked = False
        for i in range(360):  # 12.0s at 30 fps with jitter + slow drift
            drift_x = 0.004 * np.sin(i / 45.0)
            drift_y = 0.003 * np.cos(i / 60.0)
            hand = _static_i_hand(jitter_std=0.003, offset_x=drift_x, offset_y=drift_y, rng=rng)
            c.update(i, mirrored(hand), hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
            if c.display_label in ("J", "Z"):
                saw_jz_display = True
            if c.blocked:
                saw_blocked = True

        self.assertEqual(c.composer.text, "I")
        self.assertFalse(saw_jz_display)
        self.assertFalse(saw_blocked)
        self.assertEqual(c.display_label, "I")

    def test_hand_entry_then_hold_i_unblocked_and_commits_i(self):
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")
        events = []
        c = UnifiedAutoCoordinator(letters, digits, model, enable_experimental_jz=True, event_sink=events.append)

        # Simulate hand sweeping into 'I' then holding steady
        for i in range(150):
            u = min(i / 18.0, 1.0)
            hand = _static_i_hand(offset_x=0.10 * (1.0 - u), offset_y=0.06 * (1.0 - u))
            c.update(i, mirrored(hand), hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
            self.assertFalse(c.blocked)
            self.assertNotIn(c.display_label, ("J", "Z"))

        self.assertEqual(c.composer.text, "I")

    def test_unverified_articulated_hand_never_blocks_or_resets_static_i_hold(self):
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")
        c = UnifiedAutoCoordinator(letters, digits, model, enable_experimental_jz=True)

        # Build static I hold to 1.5s (HOLDING at progress ~0.5)
        for i in range(45):
            hand = _static_i_hand()
            c.update(i, hand, hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
        self.assertEqual(c.static.hold.state, "HOLDING")
        self.assertGreater(c.static.hold.progress, 0.4)

        # Feed unverified articulated_hand on raw IMAGE while VIDEO sees static I
        for k in range(60):
            raw = articulated_hand(k)
            static_h = _static_i_hand()
            c.update(45 + k, static_h, raw, (480, 640, 3), (45 + k) / 30.0, raw_hand_count=1)
            self.assertFalse(c.blocked)
            self.assertNotEqual(c.display_label, "J")

        self.assertEqual(c.composer.text, "I")
        self.assertNotIn("J", c.composer.text)

    def test_diagnostic_mode_false_never_suppresses_or_overrides_display_on_invalid_j(self):
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")
        c = UnifiedAutoCoordinator(letters, digits, model, enable_experimental_jz=False)

        for k in range(110):
            raw = articulated_hand(min(k, 65))
            static_h = _static_i_hand()
            c.update(k, static_h, raw, (480, 640, 3), k / 30.0, raw_hand_count=1)
            self.assertFalse(c.blocked)
            self.assertIsNone(c.candidate)
            self.assertNotEqual(c.display_label, "J")

        self.assertEqual(c.composer.text, "I")

    def test_diagnostic_replay_false_j_attempts_all_rejected(self):
        if not DIAG_ROOT.is_dir():
            self.skipTest("Diagnostic folder missing")
        dyn_model, _ = load_dynamic()
        classes = list(dyn_model.classes_)
        j_pred_attempts = []
        for sess in sorted(DIAG_ROOT.iterdir()):
            if not sess.is_dir():
                continue
            for att in sorted(sess.iterdir()):
                npz = att / "replay_data.npz"
                if not npz.is_file():
                    continue
                data = np.load(npz, allow_pickle=True)
                if "features" not in data.files or data["features"].shape != (16, 69):
                    continue
                feat = np.asarray(data["features"], dtype=np.float64)
                probs = dyn_model.predict_proba(feat.reshape(1, -1))[0]
                if classes[int(np.argmax(probs))] == "J":
                    j_pred_attempts.append((f"{sess.name}/{att.name}", feat))
                    res = validate_j_trajectory(feat)
                    self.assertFalse(
                        res["valid"],
                        f"False-J diagnostic attempt {sess.name}/{att.name} unexpectedly passed: {res}",
                    )
        self.assertGreaterEqual(len(j_pred_attempts), 10)

    def test_replay_coordinator_with_real_diag_i_features_no_hijack(self):
        if not DIAG_ROOT.is_dir():
            self.skipTest("Diagnostic folder missing")
        target = DIAG_ROOT / "20261014T053707008198Z_708b76da" / "generation_000_attempt_00018" / "replay_data.npz"
        if not target.is_file():
            self.skipTest("Target diagnostic NPZ missing")
        data = np.load(target, allow_pickle=True)
        feat = np.asarray(data["features"], dtype=np.float64)
        dyn_model, _ = load_dynamic()
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        c = UnifiedAutoCoordinator(letters, digits, dyn_model, enable_experimental_jz=True)
        # Replay the normalized 21 landmarks shifted slightly by pink/wrist trace, then hold I cleanly
        lm_seq = feat[:, :63].reshape(16, 21, 3)
        for i in range(16):
            hand = [SimpleNamespace(x=0.5 + 0.1 * float(lm_seq[i, j, 0]),
                                    y=0.5 + 0.1 * float(lm_seq[i, j, 1]),
                                    z=0.1 * float(lm_seq[i, j, 2])) for j in range(21)]
            c.update(i, hand, hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
            self.assertNotIn(c.display_label, ("J", "Z"))
            self.assertFalse(c.blocked)
        for i in range(16, 130):
            hand = _static_i_hand()
            c.update(i, hand, hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
            self.assertNotIn(c.display_label, ("J", "Z"))
        self.assertEqual(c.composer.text, "I")


class GroupJZTrajectoryValidationTests(unittest.TestCase):
    def test_valid_curved_j_and_z_accepted_and_committed(self):
        for label in ("J", "Z"):
            with self.subTest(label=label):
                letters = FakePredictor("I", 0.0)
                digits = FakePredictor("0", 0.5)
                model = DynamicModel(label)
                c = UnifiedAutoCoordinator(letters, digits, model, enable_experimental_jz=True)
                step = 1.8 if label == "Z" else 1.0
                for i in range(250):
                    hand = curved_hand(min(int(i * step), 60), label)
                    c.update(i, mirrored(hand), hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
                self.assertEqual(c.composer.text, label)

    def test_incomplete_static_and_spike_j_rejected(self):
        # 1. Pure downward drop with no lateral hook
        p_no_hook = np.zeros((16, 2), dtype=np.float64)
        p_no_hook[:, 1] = np.linspace(0.0, 1.2, 16)
        res1 = validate_j_trajectory(_make_pinky_path_features(p_no_hook))
        self.assertFalse(res1["valid"])
        self.assertIn("INSUFFICIENT_J_HOOK_LATERAL", res1["reason"])

        # 2. Upward drift before drop
        p_up = np.zeros((16, 2), dtype=np.float64)
        p_up[:6, 1] = np.linspace(0.0, -0.8, 6)
        p_up[6:11, 1] = np.linspace(-0.8, 0.5, 5)
        p_up[10:, 0] = np.linspace(0.0, 0.5, 6)
        p_up[10:, 1] = np.linspace(0.5, 0.2, 6)
        res2 = validate_j_trajectory(_make_pinky_path_features(p_up))
        self.assertFalse(res2["valid"])
        self.assertIn("UPWARD_MOTION_BEFORE_DROP", res2["reason"])

        # 3. Single-frame teleport spike
        p_spike = np.zeros((16, 2), dtype=np.float64)
        p_spike[7:, 1] = 1.0  # 1-frame jump from frame 6 -> 7
        p_spike[12:, 0] = 0.4
        p_spike[12:, 1] = 0.8
        res3 = validate_j_trajectory(_make_pinky_path_features(p_spike))
        self.assertFalse(res3["valid"])
        self.assertIn("SINGLE_FRAME", res3["reason"])

        # 4. Horizontal articulated_hand & static_l_hand
        frames_a = [articulated_hand(i) for i in range(65)]
        fa, _ = build_sequence_features([frames_a[int(k)] for k in np.linspace(0, 64, 16)], [(640, 480)] * 16)
        self.assertFalse(validate_j_trajectory(fa)["valid"])

        frames_l = [static_l_hand(jitter_std=0.005) for _ in range(16)]
        fl, _ = build_sequence_features(frames_l, [(640, 480)] * 16)
        self.assertFalse(validate_j_trajectory(fl)["valid"])

    def test_z_validator_thresholds_preserved(self):
        c = UnifiedAutoCoordinator(FakePredictor("A", 0.0), FakePredictor("0", 0.5), DynamicModel("Z"))
        self.assertEqual(c.min_dynamic_z_confidence, 0.75)
        self.assertEqual(ZTrajectorySettings().min_excursion, 0.35)
        # Incomplete synthetic Z still rejected
        p = np.zeros((16, 69), dtype=np.float64)
        p[:, 65] = np.linspace(0.0, 0.4, 16)
        self.assertFalse(validate_z_trajectory(p)["valid"])


class GroupHoldConfirmationBlipGraceTests(unittest.TestCase):
    def _obs(self, label="I", confidence=0.95, x=320.0, y=240.0, scale=100.0,
             shape_val=0.0, raw_stable="I"):
        pred = SimpleNamespace(
            instant_label="T" if label is None else label,
            stable_label=raw_stable,
            stable_confidence=confidence,
            features=(shape_val,) * 63,
        )
        # Construct HandObservation cleanly without relying on exact pixel projection
        from app.inference.push_confirmation_camera_test import HandObservation
        return HandObservation(
            label=label,
            confidence=confidence,
            scale_px=scale,
            width_px=scale,
            wrist_px=(x, y),
            shape=(shape_val,) * 63,
            palm_center_px=(x, y),
            instant_label=pred.instant_label,
            raw_stable_label=raw_stable,
        )

    def test_single_frame_instant_mismatch_via_observe_hand_preserves_hold(self):
        m = HoldConfirmation(HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15))
        hand = _static_i_hand()
        # 3 frames valid 'I' -> HOLDING
        for idx, t in enumerate((0.0, 0.05, 0.10)):
            pred = SimpleNamespace(instant_label="I", stable_label="I", stable_confidence=0.95, features=(0.0,) * 63)
            self.assertIsNone(m.update(observe_hand(hand, pred, (480, 640, 3)), timestamp=t))
        self.assertEqual(m.state, "HOLDING")

        # 1 frame instant mismatch ("T" != "I") -> observe_hand sets label=None, raw_stable_label="I"
        blip_pred = SimpleNamespace(instant_label="T", stable_label="I", stable_confidence=0.92, features=(0.0,) * 63)
        blip_obs = observe_hand(hand, blip_pred, (480, 640, 3))
        self.assertIsNone(blip_obs.label)
        self.assertEqual(blip_obs.raw_stable_label, "I")
        self.assertIsNone(m.update(blip_obs, timestamp=0.13))
        self.assertEqual(m.state, "HOLDING")

        # Valid resumes immediately at 0.16s: must still complete full 3.0s valid non-blip elapsed (~3.06s total)
        pred_ok = SimpleNamespace(instant_label="I", stable_label="I", stable_confidence=0.95, features=(0.0,) * 63)
        self.assertIsNone(m.update(observe_hand(hand, pred_ok, (480, 640, 3)), timestamp=0.16))
        self.assertIsNone(m.update(observe_hand(hand, pred_ok, (480, 640, 3)), timestamp=3.00))
        self.assertEqual(m.update(observe_hand(hand, pred_ok, (480, 640, 3)), timestamp=3.07), "I")

    def test_three_consecutive_blips_reset_to_idle(self):
        m = HoldConfirmation(HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15))
        for t in (0.0, 0.05, 0.10):
            m.update(self._obs("I"), timestamp=t)
        self.assertEqual(m.state, "HOLDING")
        # 2 blips allowed
        m.update(self._obs(label=None, raw_stable="I"), timestamp=0.13)
        self.assertEqual(m.state, "HOLDING")
        m.update(self._obs(label=None, raw_stable="I"), timestamp=0.16)
        self.assertEqual(m.state, "HOLDING")
        # 3rd consecutive blip exceeds blip_grace_frames=2 -> resets immediately
        m.update(self._obs(label=None, raw_stable="I"), timestamp=0.19)
        self.assertEqual(m.state, "IDLE")

    def test_blip_exceeding_grace_seconds_or_tracking_loss_resets(self):
        m = HoldConfirmation(HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15))
        for t in (0.0, 0.05, 0.10):
            m.update(self._obs("I"), timestamp=t)
        m.update(self._obs(label=None, raw_stable="I"), timestamp=0.30)  # 0.20s > 0.15s
        self.assertEqual(m.state, "IDLE")

        for t in (1.0, 1.05, 1.10):
            m.update(self._obs("I"), timestamp=t)
        self.assertEqual(m.state, "HOLDING")
        m.update(None, timestamp=1.13)  # tracking loss always resets
        self.assertEqual(m.state, "IDLE")

    def test_hand_identity_or_pose_jump_resets_immediately(self):
        m = HoldConfirmation(HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15))
        for t in (0.0, 0.05, 0.10):
            m.update(self._obs("I"), timestamp=t)
        self.assertEqual(m.state, "HOLDING")
        # Palm teleports 60 px (> 0.25 * 100 scale) during blip
        m.update(self._obs(label=None, raw_stable="I", x=380.0), timestamp=0.13)
        self.assertEqual(m.state, "IDLE")

    def test_multi_hand_resets_static_hold_in_coordinator(self):
        letters = FakePredictor("I", 0.0)
        digits = FakePredictor("0", 0.5)
        c = UnifiedAutoCoordinator(letters, digits, DynamicModel("J"), enable_experimental_jz=True)
        hand = _static_i_hand()
        for i in range(40):
            c.update(i, hand, hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
        self.assertEqual(c.static.hold.state, "HOLDING")
        # 2 hands detected -> static hold must reset
        c.update(40, hand, hand, (480, 640, 3), 40 / 30.0, raw_hand_count=2)
        self.assertEqual(c.static.hold.state, "IDLE")
        self.assertEqual(c.composer.text, "")

    def test_no_false_commit_on_grace_frame_or_stale_label(self):
        m = HoldConfirmation(HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15))
        for t in (0.0, 0.05, 0.10, 2.95):
            self.assertIsNone(m.update(self._obs("I"), timestamp=t))
        self.assertEqual(m.state, "HOLDING")

        # Even though now=3.02 >= 3.00, a grace frame MUST NEVER trigger a commit
        self.assertIsNone(m.update(self._obs(label=None, raw_stable="I"), timestamp=3.02))
        self.assertEqual(m.text, "")

        # And if smoother switches raw_stable to "T", grace MUST NOT keep "I" alive
        self.assertIsNone(m.update(self._obs(label=None, raw_stable="T"), timestamp=3.05))
        self.assertEqual(m.state, "IDLE")
        self.assertEqual(m.text, "")


if __name__ == "__main__":
    unittest.main()
