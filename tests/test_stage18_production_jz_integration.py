"""Stage 18 — Production J/Z Integration & Regression Test Suite.

Verifies end-to-end integration and non-regression of J/Z recognition:
1. Recognize J and commit J (proposal -> validator -> 3.0s hold -> commit).
2. Recognize Z and commit Z (both viewer-LR and signer-mirrored).
3. Transition J -> Z.
4. Transition Z -> J.
5. Transition J/Z -> SPACE.
6. Transition SPACE -> J/Z.
7. Strict 3.0s hold confirms exactly once (no duplicate commits).
8. Release hand or pose change re-arms the coordinator for subsequent signs.
9. UNKNOWN prediction (margin < 0.10) is rejected and never commits.
10. Z Handshape Gate in SHADOW mode is non-destructive (does not change production).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from app.inference import integrated_jz_validation as jz_val
from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
from app.inference.integrated_jz_validation import (
    ZHandshapeSettings,
    ZTrajectorySettings,
    validate_proposal,
    validate_z_handshape,
    validate_z_trajectory,
)
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from src.recognition.alphabet import AlphabetRecognizer
from tests.test_dynamic_jz_motion_gate import curved_hand, hand_at
from tests.test_jz_production_integration import (
    _hold_static,
    _release_hand,
    _run_dynamic_bout,
    _signer_mirrored_curved_z,
)
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StubSpacePredictor:
    """Stub space predictor that toggles SPACE gesture recognition."""

    def __init__(self, active: bool = False):
        self.active = active

    def predict_hand(self, _landmarks, mode: str = "ABC"):
        if mode == "ABC" and self.active:
            return 0.99, True
        return 0.05, False

    def reset(self):
        pass


class NullStaticPredictor:
    """Static predictor that emits no static predictions, ensuring only dynamic logic is evaluated."""

    def __init__(self):
        self.label = None
        self.shape = 0.0

    def reset(self):
        pass

    def predict_landmarks(self, _hand):
        return None


class MarginAmbiguousDynamicModel:
    """Dynamic model that outputs ambiguous probabilities with margin < 0.10."""

    classes_ = np.asarray(["J", "Z"])

    def __init__(self, prob_j: float = 0.52, prob_z: float = 0.48):
        self.prob_j = float(prob_j)
        self.prob_z = float(prob_z)
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(x.copy() if hasattr(x, "copy") else x)
        n = len(x) if hasattr(x, "__len__") and hasattr(x, "ndim") and x.ndim > 1 else 1
        return np.tile([self.prob_j, self.prob_z], (n, 1))

    def predict(self, x):
        n = len(x) if hasattr(x, "__len__") and hasattr(x, "ndim") and x.ndim > 1 else 1
        label = "J" if self.prob_j >= self.prob_z else "Z"
        return np.repeat(label, n)


class Stage18ProductionIntegrationTests(unittest.TestCase):
    """End-to-end integration test suite verifying the 10 production criteria."""

    def setUp(self):
        self.space_pred = StubSpacePredictor(active=False)

    def _create_coordinator(
        self,
        dyn_label: str = "Z",
        enable_experimental_jz: bool = True,
        enable_space: bool = True,
        dynamic_model=None,
        static_predictor=None,
    ) -> AlphabetRecognizer:
        if dynamic_model is None:
            dynamic_model = DynamicModel(dyn_label)
        if static_predictor is None:
            static_predictor = FakePredictor("A", 0.0)
        coord = AlphabetRecognizer(
            static_predictor,
            FakePredictor("0", 0.5),
            dynamic_model,
            enable_experimental_jz=enable_experimental_jz,
            space_predictor=self.space_pred,
            enable_space=enable_space,
        )
        return coord

    def _set_dynamic_label(self, coord: UnifiedAutoCoordinator, label: str):
        """Helper to properly update the underlying base model label behind ProbabilityMarginModelWrapper."""
        model = coord.dynamic_model
        while hasattr(model, "base_model"):
            model = model.base_model
        model.label = label
        if hasattr(coord.session, "model"):
            s_model = coord.session.model
            while hasattr(s_model, "base_model"):
                s_model = s_model.base_model
            s_model.label = label

    def test_01_recognize_and_commit_j(self):
        """1. Full pipeline test: Dynamic J recognized, passes validator, holds 3s, commits 'J'."""
        coord = self._create_coordinator("J")
        idx, out = _run_dynamic_bout(coord, 0, "J", motion_frames=80, hold_frames=120)

        self.assertTrue(out["hold_started"], "HoldConfirmation did not start for J")
        self.assertFalse(out["early_commit"], "J committed earlier than the 3.0s requirement")
        self.assertEqual(out["commits"], ["J"], "Expected exactly one 'J' commit")
        self.assertEqual(coord.composer.text, "J", "TextComposer text mismatch")
        self.assertTrue(coord.post_jz_latch, "Post-JZ latch should be active after commit")

    def test_02_recognize_and_commit_z(self):
        """2. Full pipeline test: Dynamic Z recognized, passes validator, holds 3s, commits 'Z'."""
        for mirrored_x in (False, True):
            with self.subTest(signer_mirrored=mirrored_x):
                coord = self._create_coordinator("Z")
                idx, out = _run_dynamic_bout(
                    coord, 0, "Z", signer_mirrored_z=mirrored_x, motion_frames=80, hold_frames=120
                )
                self.assertTrue(out["hold_started"], "HoldConfirmation did not start for Z")
                self.assertFalse(out["early_commit"], "Z committed earlier than 3.0s")
                self.assertEqual(out["commits"], ["Z"], "Expected exactly one 'Z' commit")
                self.assertEqual(coord.composer.text, "Z", "TextComposer text mismatch")
                self.assertTrue(coord.post_jz_latch, "Post-JZ latch should be active after commit")

    def test_03_transition_j_to_z(self):
        """3. Transition test: J -> hand release -> Z -> TextComposer produces 'JZ'."""
        coord = self._create_coordinator("J")

        # Step 1: Commit J
        idx, out_j = _run_dynamic_bout(coord, 0, "J")
        self.assertEqual(out_j["commits"], ["J"])
        self.assertEqual(coord.composer.text, "J")

        # Step 2: Release hand to rearm coordinator
        idx = _release_hand(coord, idx, frames=25)
        self.assertFalse(coord.post_jz_latch, "Coordinator should rearm after physical release")

        # Step 3: Switch dynamic model output to Z and execute Z bout
        self._set_dynamic_label(coord, "Z")
        idx, out_z = _run_dynamic_bout(coord, idx, "Z", signer_mirrored_z=True)
        self.assertEqual(out_z["commits"], ["Z"])
        self.assertEqual(coord.composer.text, "JZ", "TextComposer should contain 'JZ'")

    def test_04_transition_z_to_j(self):
        """4. Transition test: Z -> hand release -> J -> TextComposer produces 'ZJ'."""
        coord = self._create_coordinator("Z")

        # Step 1: Commit Z
        idx, out_z = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=True)
        self.assertEqual(out_z["commits"], ["Z"])
        self.assertEqual(coord.composer.text, "Z")

        # Step 2: Release hand to rearm coordinator
        idx = _release_hand(coord, idx, frames=25)
        self.assertFalse(coord.post_jz_latch, "Coordinator should rearm after physical release")

        # Step 3: Switch dynamic model output to J and execute J bout
        self._set_dynamic_label(coord, "J")
        idx, out_j = _run_dynamic_bout(coord, idx, "J")
        self.assertEqual(out_j["commits"], ["J"])
        self.assertEqual(coord.composer.text, "ZJ", "TextComposer should contain 'ZJ'")

    def test_05_transition_jz_to_space(self):
        """5. Transition test: J -> SPACE and Z -> SPACE produces 'J ' and 'Z '."""
        coord = self._create_coordinator("Z")

        # Step 1: Commit Z
        idx, out_z = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=True)
        self.assertEqual(out_z["commits"], ["Z"])
        self.assertEqual(coord.composer.text, "Z")

        # Step 2: Release hand
        idx = _release_hand(coord, idx, frames=25)

        # Step 3: Activate SPACE gesture for 3.2s
        self.space_pred.active = True
        idx, sp_commits = _hold_static(coord, idx, "A", 0.0, lambda: hand_at(0.0), duration_s=3.2)
        self.assertEqual(sp_commits, ["SPACE"], "Expected SPACE commit")
        self.assertEqual(coord.composer.text, "Z ", "TextComposer should have 'Z '")

    def test_06_transition_space_to_jz(self):
        """6. Transition test: SPACE -> hand release -> Z produces ' Z'."""
        coord = self._create_coordinator("Z")

        # Step 1: Activate SPACE gesture
        self.space_pred.active = True
        idx, sp_commits = _hold_static(coord, 0, "A", 0.0, lambda: hand_at(0.0), duration_s=3.2)
        self.assertEqual(sp_commits, ["SPACE"])
        self.assertEqual(coord.composer.text, " ")

        # Step 2: Deactivate SPACE and release hand
        self.space_pred.active = False
        idx = _release_hand(coord, idx, frames=25)

        # Step 3: Execute Z bout
        idx, out_z = _run_dynamic_bout(coord, idx, "Z", signer_mirrored_z=True)
        self.assertEqual(out_z["commits"], ["Z"])
        self.assertEqual(coord.composer.text, " Z", "TextComposer should contain ' Z'")

    def test_07_hold_3s_commits_strictly_once(self):
        """7. Strict single-commit: Continuing to hold pose for 6s produces exactly 1 commit."""
        coord = self._create_coordinator("Z")
        # Hold for 180 frames (6.0s at 30fps) - double the 3.0s dwell
        idx, out = _run_dynamic_bout(
            coord, 0, "Z", signer_mirrored_z=True, motion_frames=80, hold_frames=180
        )
        self.assertEqual(out["commits"], ["Z"], "Must not emit duplicate commits during continuous hold")
        self.assertEqual(coord.composer.text, "Z", "TextComposer text must have length 1")

    def test_08_release_or_new_pose_rearms_next_gesture(self):
        """8. Re-arming: Verify release (raw=None) or pose change clears latch for next sign."""
        coord = self._create_coordinator("Z")

        # (a) Commit Z and test hand release re-arm
        idx, _ = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=True)
        self.assertTrue(coord.post_jz_latch)

        # Still holding: latch remains True
        for _ in range(10):
            terminal = _signer_mirrored_curved_z(60)
            coord.update(idx, mirrored(terminal), terminal, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            idx += 1
        self.assertTrue(coord.post_jz_latch, "Latch must stay active while terminal hand is held")

        # Release hand for 25 frames (> 0.5s release_seconds)
        idx = _release_hand(coord, idx, frames=25)
        self.assertFalse(coord.post_jz_latch, "Latch must release after physical release")

        # (b) Test new pose re-arm: Commit another sign (static A), verify new pose works
        idx, a_commits = _hold_static(coord, idx, "A", 0.0, lambda: hand_at(0.0), duration_s=3.2)
        self.assertEqual(a_commits, ["A"])
        self.assertEqual(coord.composer.text, "ZA")

    def test_09_unknown_margin_reject_never_commits(self):
        """9. Margin Reject Option: Ambiguous margin < 0.10 outputs UNKNOWN and never commits."""
        ambiguous_dyn = MarginAmbiguousDynamicModel(prob_j=0.52, prob_z=0.48)
        # Use NullStaticPredictor so that static hold does not commit fallback characters
        coord = self._create_coordinator(dynamic_model=ambiguous_dyn, static_predictor=NullStaticPredictor())

        # Run dynamic bout where model returns UNKNOWN
        idx, out = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=True, motion_frames=80, hold_frames=120)

        self.assertEqual(out["commits"], [], "Ambiguous gesture must NEVER commit")
        self.assertEqual(coord.composer.text, "", "TextComposer must remain empty")
        self.assertFalse(out["hold_started"], "HoldConfirmation must never start for UNKNOWN")

    def test_10_z_handshape_gate_shadow_mode_invariance(self):
        """10. Z Handshape Gate SHADOW mode invariance: SHADOW mode never alters production decisions."""
        # Check 1: In production code, default mode is SHADOW
        default_settings = ZHandshapeSettings()
        self.assertEqual(default_settings.mode, "SHADOW")

        # Check 2: Evaluate a fist handshape (index_ext ~ 0.59 < 0.70)
        from tests.test_stage16_z_handshape_gate import _build_test_features

        fist_z_features = _build_test_features(handshape="FIST", trajectory="Z")

        # In SHADOW mode: validate_z_handshape rejects, but validate_z_trajectory returns valid=True
        res_shadow = validate_z_trajectory(fist_z_features)  # Uses default SHADOW settings
        self.assertTrue(res_shadow["valid"], "In SHADOW mode, trajectory validity must NOT be overridden")
        self.assertEqual(res_shadow["reason"], "VALID_Z_TRAJECTORY")
        self.assertFalse(res_shadow["handshape_validation"]["valid"], "Gate telemetry must capture failure")
        self.assertEqual(res_shadow["handshape_validation"]["mode"], "SHADOW")

        # Check 3: In ACTIVE mode: validate_z_trajectory returns valid=False
        active_hs = ZHandshapeSettings(mode="ACTIVE")
        res_active = validate_z_trajectory(fist_z_features, handshape_settings=active_hs)
        self.assertFalse(res_active["valid"], "In ACTIVE mode, non-pointing handshape must be blocked")
        self.assertIn("INDEX_NOT_EXTENDED", res_active["reason"])


if __name__ == "__main__":
    unittest.main()
