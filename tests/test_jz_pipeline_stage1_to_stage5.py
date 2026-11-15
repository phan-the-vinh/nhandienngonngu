"""Stage 1-5 regressions for webcam J/Z duplicate handling, duration window, continuity, and unsettled pose tolerance."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from app.inference.automatic_jz_experiment import MotionSegmenter, SegmentSettings
from app.inference.first16_jz_segmenter import First16Segmenter, MotionGateSettings
from app.inference.integrated_jz_validation import validate_proposal
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_dynamic_jz_motion_gate import curved_hand
from tests.test_unified_abc123_jz_auto import DynamicModel, FakePredictor, mirrored


def _mirrored_raw_set(hand):
    return mirrored(hand), hand


class Stage1To5JzPipelineRegressionTests(unittest.TestCase):
    def test_duplicate_polls_are_excluded_from_selected_entries_and_preserve_run_continuity(self):
        seg = First16Segmenter(
            replace(SegmentSettings(), max_missing_frames=6, max_frame_gap_seconds=0.4),
        )
        completed = None
        metrics = None
        for idx in range(70):
            # Emulate 15fps camera polled at 30fps (alternating duplicate poses)
            src = min(idx // 2, 60)
            hand = curved_hand(min(int(src * 1.8), 60), "Z")
            state, _reason, selected = seg.update(idx, hand, (640, 480), idx / 30.0)
            if state == "COMPLETE":
                completed = selected
                metrics = dict(seg.last_diagnostics)
                break
        self.assertIsNotNone(completed)
        self.assertGreaterEqual(len(completed), 16)
        self.assertGreater(metrics["duplicate_frames"], 0)
        self.assertGreater(metrics["duplicate_ratio"], 0.0)
        self.assertEqual(len(completed), metrics["unique_valid_frames"])
        # Confirm adjacent selected landmarks have no exact duplicate frame pairs
        for a, b in zip(completed, completed[1:]):
            arr_a = np.array([[p.x, p.y, p.z] for p in a[1]])
            arr_b = np.array([[p.x, p.y, p.z] for p in b[1]])
            self.assertGreater(float(np.max(np.abs(arr_a - arr_b))), 1e-7)

    def test_slow_z_extends_past_1s_to_capture_stroke3_before_completing(self):
        model = DynamicModel("Z")
        coord = UnifiedAutoCoordinator(
            FakePredictor("I", 0.0),
            FakePredictor("0", 0.0),
            model,
            enable_experimental_jz=True,
        )
        # Slow Z spanning ~55 frames (~1.83s): at t=1.0s stroke 3 has not yet swept horizontally
        for k in range(60):
            u_hand = curved_hand(min(int(k * 1.05), 60), "Z")
            coord.update(k, mirrored(u_hand), u_hand, (480, 640, 3), k / 30.0, raw_hand_count=1)
            if coord.pending_dynamic is not None:
                break
        self.assertIsNotNone(coord.pending_dynamic)
        self.assertEqual(coord.pending_dynamic["prediction"], "Z")
        self.assertTrue(coord.validation["z_trajectory_valid"])

    def test_wrist_supination_scale_shape_change_with_continuous_palm_is_accepted(self):
        seg = MotionSegmenter()
        base = curved_hand(0, "J")
        seg.update(0, base, (640, 480), 0.0)
        raw = np.array([[p.x, p.y, p.z] for p in base])
        wrist = raw[0].copy()
        # Simulate side-on J-hook rotation shrinking 2D palm projection ~55% with continuous wrist center
        rot = np.array([[np.cos(0.45), -np.sin(0.45)], [np.sin(0.45), np.cos(0.45)]])
        raw[:, :2] = (raw[:, :2] - wrist[:2]) @ rot.T * 0.45 + wrist[:2]
        rotated = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in raw]
        state, reason, _ = seg.update(1, rotated, (640, 480), 1 / 30.0)
        self.assertNotEqual(reason, "SCALE_AND_SHAPE_DISCONTINUITY")
        self.assertNotEqual(state, "REJECTED")

    def test_unsettled_scale_settling_under_045_survives_before_quiet_endpoint_while_hold_stays_strict(self):
        model = DynamicModel("J")
        coord = UnifiedAutoCoordinator(
            FakePredictor("I", 0.0),
            FakePredictor("0", 0.0),
            model,
            enable_experimental_jz=True,
        )
        coord.pending_dynamic = {"prediction": "J", "probability_J": 0.96, "probability_Z": 0.04}
        coord.proposal_observation = SimpleNamespace(scale_px=100.0, shape=np.zeros(63))
        # 39% projected scale recovery (palm turning back toward camera) passes unsettled proposal check
        self.assertTrue(coord.proposal_pose_matches(SimpleNamespace(scale_px=139.0, shape=np.zeros(63))))
        # 50% scale jump or huge shape change still fails
        self.assertFalse(coord.proposal_pose_matches(SimpleNamespace(scale_px=150.0, shape=np.zeros(63))))
        self.assertFalse(coord.proposal_pose_matches(SimpleNamespace(scale_px=100.0, shape=np.ones(63))))
        # Hold duration requirement stays strictly 3.0s
        self.assertEqual(coord.jz_hold.settings.hold_seconds, 3.0)


if __name__ == "__main__":
    unittest.main()
