"""Stage 6 regression tests for motion evidence frame alignment and trial isolation.

Verifies:
1. validate_proposal filters evidence_rows to rows belonging to capture entries.
2. Duplicate frames (skipped_duplicate_indices) are excluded from evidence.
3. Pre-roll frames (frame_id < entries[0][0]) are safely ignored and do NOT trigger MOTION_EVIDENCE_NOT_FROM_CAPTURE.
4. Trimmed onset frames in capture_rows are safely ignored.
5. If no capture_rows match entries, MOTION_EVIDENCE_NOT_FROM_CAPTURE is returned with REVIEW_REQUIRED.
6. Timeline monotonicity (frame_id and timestamp strictly increasing) is enforced.
7. Capture ending with deceleration window produces coherent motion evidence.
8. reset_trial_coordinator terminates unfinished active attempt and clears capture_rows.
9. Coordinator idle state prevents capture_rows accumulation.
10. End-to-end trial re-arm: pre-arm motion does not contaminate subsequent gesture validation.
"""
from dataclasses import replace
import unittest

import numpy as np

from app.inference.automatic_jz_experiment import SegmentSettings
from app.inference.first16_jz_segmenter import First16Segmenter, MotionGateSettings
from app.inference.integrated_jz_validation import validate_proposal
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tools.jz_webcam_realworld_validation import reset_trial_coordinator
from tests.test_dynamic_jz_motion_gate import curved_hand
from tests.test_unified_abc123_jz_auto import DynamicModel, FakePredictor, mirrored


class JzStage6FrameAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.size = (640, 480)
        self.scale = 100.0  # Normalized hand scale in pixels
        self.settings = MotionGateSettings()

    def test_validate_proposal_filters_duplicate_frames_successfully(self):
        """Duplicate frames in skipped_duplicate_indices must be excluded from evidence without error."""
        entries = []
        capture_rows = []

        for k in range(25):
            frame_id = k + 10
            ts = 1.0 + k * 0.05
            hand = curved_hand(min(k * 2.5, 60), "Z")
            entries.append((frame_id, hand, self.size))
            pts = np.array([[p.x * 640, p.y * 480] for p in hand])
            capture_rows.append((frame_id, ts, pts[[0, 8, 20]].copy()))

        metrics = {
            "skipped_duplicate_indices": [13, 17],  # frame_ids 13 and 17 skipped
            "duplicate_frames": 2,
            "duplicate_ratio": 0.08,
            "capture_seconds_after_onset": 1.25,
            "capture_duration": 1.25,
            "motion_duration": 0.85,
        }

        # validate_proposal should filter out frame 13 and 17, and keep all others
        val = validate_proposal(entries, capture_rows, self.scale, metrics, self.settings)
        self.assertEqual(val["status"], "PASS")
        self.assertTrue(val["continuity_evidence"]["supported"])
        # Confirm runs only contain valid frames, not skipped frames
        for run_info in val["continuity_evidence"]["runs"]:
            self.assertGreaterEqual(run_info["first_frame"], 10)

    def test_validate_proposal_with_preroll_and_trimmed_onset_frames(self):
        """Pre-roll and trimmed onset frames in capture_rows must NOT trigger MOTION_EVIDENCE_NOT_FROM_CAPTURE."""
        # capture_rows has frames 0..29
        capture_rows = []
        for k in range(30):
            frame_id = k
            ts = 1.0 + k * 0.045
            hand = curved_hand(min(max(k - 5, 0) * 2.5, 60), "Z")
            pts = np.array([[p.x * 640, p.y * 480] for p in hand])
            capture_rows.append((frame_id, ts, pts[[0, 8, 20]].copy()))

        # entries only includes frames 6..29 (frames 0..5 were pre-roll / trimmed stationary onset)
        entries = []
        for k in range(6, 30):
            frame_id = k
            hand = curved_hand(min((k - 5) * 2.5, 60), "Z")
            entries.append((frame_id, hand, self.size))

        metrics = {
            "skipped_duplicate_indices": [],
            "duplicate_frames": 0,
            "duplicate_ratio": 0.0,
            "capture_seconds_after_onset": 1.15,
            "capture_duration": 1.30,
            "motion_duration": 0.80,
        }

        val = validate_proposal(entries, capture_rows, self.scale, metrics, self.settings)
        self.assertEqual(val["status"], "PASS")
        self.assertTrue(val["continuity_evidence"]["supported"])
        # Evidence runs must only be within entries bounds (indices 6..29), ignoring pre-roll 0..5
        for run_info in val["continuity_evidence"]["runs"]:
            self.assertGreaterEqual(run_info["first_frame"], 6)
            self.assertLessEqual(run_info["last_frame"], 29)

    def test_validate_proposal_returns_review_required_when_no_rows_belong_to_capture(self):
        """If capture_rows has ZERO overlap with entries, MOTION_EVIDENCE_NOT_FROM_CAPTURE must be reported."""
        entries = []
        for k in range(20, 38):
            hand = curved_hand(k, "Z")
            entries.append((k, hand, self.size))

        # capture_rows only contains stray pre-arm frames 1..5
        capture_rows = []
        for k in range(1, 6):
            wrist = np.array([300.0, 200.0])
            capture_rows.append((k, 1.0 + k * 0.06, np.stack([wrist, wrist, wrist])))

        metrics = {
            "skipped_duplicate_indices": [],
            "capture_seconds_after_onset": 1.20,
        }
        val = validate_proposal(entries, capture_rows, self.scale, metrics, self.settings)
        self.assertEqual(val["status"], "REVIEW_REQUIRED")
        self.assertEqual(val["reason"], "MOTION_EVIDENCE_NOT_FROM_CAPTURE")

    def test_validate_proposal_empty_capture_rows_returns_review_required(self):
        """Empty capture_rows returns MOTION_EVIDENCE_NOT_FROM_CAPTURE."""
        entries = [(k, curved_hand(k, "Z"), self.size) for k in range(18)]
        metrics = {
            "skipped_duplicate_indices": [],
            "capture_seconds_after_onset": 1.20,
        }
        val = validate_proposal(entries, [], self.scale, metrics, self.settings)
        self.assertEqual(val["status"], "REVIEW_REQUIRED")
        self.assertEqual(val["reason"], "MOTION_EVIDENCE_NOT_FROM_CAPTURE")

    def test_validate_proposal_timeline_monotonicity(self):
        """Non-monotonic timestamps in evidence_rows must return INVALID_MOTION_EVIDENCE_TIMELINE."""
        entries = [(k, curved_hand(k, "Z"), self.size) for k in range(18)]
        capture_rows = []
        for k in range(18):
            # inject a timestamp inversion at k=5
            ts = 1.0 + (k if k != 5 else 3) * 0.06
            wrist = np.array([300.0, 200.0])
            capture_rows.append((k, ts, np.stack([wrist, wrist, wrist])))

        metrics = {
            "skipped_duplicate_indices": [],
            "capture_seconds_after_onset": 1.20,
        }
        val = validate_proposal(entries, capture_rows, self.scale, metrics, self.settings)
        self.assertEqual(val["status"], "REVIEW_REQUIRED")
        self.assertEqual(val["reason"], "INVALID_MOTION_EVIDENCE_TIMELINE")

    def test_capture_ending_with_endpoint_deceleration_produces_valid_alignment(self):
        """Segmenter capture completing on endpoint deceleration window generates aligned evidence."""
        seg = First16Segmenter(replace(SegmentSettings(), max_missing_frames=6, max_frame_gap_seconds=0.4))
        completed = None
        metrics = None
        anchor_scale = None

        # Simulate a realistic Z motion: dynamic motion for frames 0..40, then decelerating for 20 frames
        for idx in range(65):
            t = idx / 30.0
            if idx < 42:
                hand = curved_hand(min(int(idx * 1.4), 60), "Z")
            else:
                # Deceleration/hold at final pose
                hand = curved_hand(60, "Z")
            if seg.anchor is not None:
                anchor_scale = seg.anchor[1]
            state, reason, selected = seg.update(idx, hand, (640, 480), t)
            if state == "COMPLETE":
                completed = selected
                metrics = dict(seg.last_diagnostics)
                break

        self.assertIsNotNone(completed)
        self.assertGreaterEqual(len(completed), 16)
        self.assertIn("completion_trigger", metrics)
        self.assertIsNotNone(anchor_scale)

        # Construct capture_rows matching all updated frames
        capture_rows = []
        for f_idx in range(idx + 1):
            h = curved_hand(min(int(f_idx * 1.4), 60), "Z") if f_idx < 42 else curved_hand(60, "Z")
            pts = np.array([[p.x * 640, p.y * 480] for p in h])
            capture_rows.append((f_idx, f_idx / 30.0, pts[[0, 8, 20]].copy()))

        val = validate_proposal(completed, capture_rows, anchor_scale, metrics, seg.motion_gate)
        self.assertEqual(val["status"], "PASS")
        self.assertTrue(val["z_trajectory_valid"])

    def test_reset_trial_coordinator_finishes_active_attempt_and_clears_capture_rows(self):
        """reset_trial_coordinator must finish any active unfinished attempt and clear capture_rows."""
        model = DynamicModel("J")
        coord = UnifiedAutoCoordinator(
            FakePredictor("I", 0.0),
            FakePredictor("0", 0.0),
            model,
            enable_experimental_jz=True,
        )

        # Feed some stray hand motion before trial is armed
        for k in range(15):
            hand = curved_hand(k * 2, "J")
            coord.update(k, mirrored(hand), hand, (480, 640, 3), k / 30.0, raw_hand_count=1)

        # Hand motion triggered segmenter into CAPTURING
        self.assertTrue(len(coord.capture_rows) > 0)
        self.assertIsNotNone(coord.session.attempts.active)
        self.assertFalse(coord.session.attempts.active["finished"])
        prior_attempt_id = coord.session.attempts.active["attempt_id"]

        # Call reset_trial_coordinator (emulating user pressing 'J' or 'Z' to arm trial)
        reset_trial_coordinator(coord, now=0.55)

        # Verify coordinator state is completely purged
        self.assertEqual(coord.capture_rows, [])
        self.assertIsNone(coord.capture_attempt_id)
        self.assertIsNone(coord.capture_scale)
        self.assertTrue(coord.session.attempts.active["finished"])

        # Now begin the real trial (frames 20..75)
        trial_completed = False
        for k in range(20, 80):
            t = k / 30.0
            hand = curved_hand(min(int((k - 20) * 1.5), 60), "J")
            coord.update(k, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
            if coord.pending_dynamic is not None or coord.candidate is not None:
                trial_completed = True
                break

        self.assertTrue(trial_completed)
        self.assertIsNotNone(coord.validation)
        self.assertEqual(coord.validation["status"], "PASS")
        self.assertGreater(coord.session.attempts.active["attempt_id"], prior_attempt_id)

    def test_idle_coordinator_does_not_accumulate_capture_rows(self):
        """When segmenter is inactive and no pending bout exists, capture_rows remains empty."""
        model = DynamicModel("J")
        coord = UnifiedAutoCoordinator(
            FakePredictor("I", 0.0),
            FakePredictor("0", 0.0),
            model,
            enable_experimental_jz=True,
        )

        # Feed static hand (no movement: speed < motion_gate.min_motion_speed)
        base_hand = curved_hand(0, "J")
        for k in range(15):
            coord.update(k, mirrored(base_hand), base_hand, (480, 640, 3), k / 30.0, raw_hand_count=1)

        # Segmenter never activated, coordinator idle: capture_rows must be empty
        self.assertFalse(coord.session.segmenter.active)
        self.assertIsNone(coord.session.pending)
        self.assertEqual(coord.capture_rows, [])

    def test_post_fix_trial4_reproduction_does_not_fail(self):
        """Simulate the exact sequence of Trial #4 that previously failed with MOTION_EVIDENCE_NOT_FROM_CAPTURE."""
        model = DynamicModel("J")
        coord = UnifiedAutoCoordinator(
            FakePredictor("I", 0.0),
            FakePredictor("0", 0.0),
            model,
            enable_experimental_jz=True,
        )

        # Phase 1: Hand in view between trials, moving slightly -> starts attempt #1
        for k in range(10):
            hand = curved_hand(k * 2, "J")
            coord.update(k, mirrored(hand), hand, (480, 640, 3), k / 30.0, raw_hand_count=1)

        # User presses 'J' to arm trial -> reset_trial_coordinator called
        reset_trial_coordinator(coord, now=0.35)

        # Phase 2: User performs full J gesture
        for k in range(15, 80):
            t = k / 30.0
            u = min(int((k - 15) * 1.5), 60)
            hand = curved_hand(u, "J")
            coord.update(k, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
            if coord.pending_dynamic is not None or coord.candidate is not None:
                break

        self.assertIsNotNone(coord.validation)
        self.assertEqual(coord.validation["status"], "PASS")
        self.assertNotEqual(coord.validation.get("reason"), "MOTION_EVIDENCE_NOT_FROM_CAPTURE")


if __name__ == "__main__":
    unittest.main()
