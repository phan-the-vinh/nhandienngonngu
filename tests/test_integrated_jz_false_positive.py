"""Integrated arbitration regressions; synthetic hands/models, no camera or files.

These tests cannot establish real ASL recall or actual-model OOD behaviour.
"""
import unittest
from unittest.mock import patch

import numpy as np

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand
from app.inference.first16_jz_segmenter import measure_motion, MotionGateSettings
from app.inference.integrated_jz_validation import coherent_motion_evidence


class FalsePositiveTests(unittest.TestCase):
    def make(self, label="A", enabled=False):
        self.events = []
        self.model = DynamicModel("Z")
        self.c = UnifiedAutoCoordinator(FakePredictor(label, 0.), FakePredictor("0", .5),
            self.model, enable_experimental_jz=enabled, event_sink=self.events.append)

    def tick(self, i, hand):
        return self.c.update(i, mirrored(hand) if hand else None, hand, (480,640,3),
            i/30, raw_hand_count=1 if hand else 0)

    def test_unvalidated_binary_z_cannot_cover_any_static_letter_or_commit(self):
        for label in "ABCDEFGHIKLMNOPQRSTUVWXY":
            with self.subTest(label=label):
                self.make(label, True)
                self.c.session.last_result = {"status":"RESULT", "model_result":{
                    "prediction":"Z", "probability_J":.01, "probability_Z":.99},
                    "attempt_id":17, "reason":"injected raw result"}
                self.tick(0, synthetic_hand())
                self.assertIsNone(self.c.candidate)
                self.assertEqual(self.c.display_label, label)
                self.assertEqual(self.c.composer.text, "")
                self.assertFalse(self.c.blocked)

    def test_raw_prediction_waits_for_endpoint_without_covering_static(self):
        self.make("B")
        for i in range(65):
            self.tick(i, articulated_hand(i))
            if self.c.raw_dynamic_prediction:
                break
        self.assertIsNotNone(self.c.raw_dynamic_prediction)
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.display_label, "B")
        self.assertIsNone(self.c.jz_hold.hold_started_at)

    def test_changed_pose_before_quiet_cannot_supply_old_prediction_endpoint(self):
        self.make("L", True)
        for i in range(65):
            self.tick(i, articulated_hand(i))
            if self.c.raw_dynamic_prediction:
                break
        self.assertIsNotNone(self.c.pending_dynamic)
        # Synthetic shape distance beyond existing HOLD tolerance; not a
        # real-letter model claim. Translation alone must not cause this test.
        with patch.object(self.c, "proposal_pose_matches", return_value=False):
            self.tick(i+1, articulated_hand(i+1))
        self.assertIsNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.candidate)
        self.assertFalse(self.c.blocked)
        self.assertEqual(self.c.display_label, "L")
        self.assertEqual(self.c.composer.text, "")

    def test_post_confirmation_latch_does_not_cover_current_static_prediction(self):
        self.make("V")
        self.c.post_jz_latch = True
        self.c.confirmed_raw_shape = np.asarray([0.]*63)
        self.c.jz_hold.last_confirmed = "Z"
        self.tick(0, synthetic_hand())
        self.assertEqual(self.c.display_label, "V")
        self.assertEqual(self.c.composer.text, "")

    def test_static_startup_fast_slow_straight_and_diagonal_entry_many_labels(self):
        for label in "ABCDILPVWY":
            for mode in ("startup", "fast", "slow", "diagonal"):
                with self.subTest(label=label, mode=mode):
                    self.make(label)
                    for i in range(155):
                        if mode != "startup" and i < 4:
                            self.tick(i, None)
                            continue
                        hand = synthetic_hand()
                        # Different synthetic finger configurations. Labels
                        # exercise arbitration, NOT verified real ASL shapes.
                        hand[12].y += (ord(label)%3)*.004
                        amount = 0. if mode == "startup" else min(max(i-4,0)/(8 if mode=="fast" else 42),1.)*.12
                        for p in hand:
                            p.x += amount
                            p.y += amount*.5 if mode == "diagonal" else 0.
                        self.tick(i, hand)
                    self.assertEqual(self.model.inputs, [])
                    self.assertIsNone(self.c.candidate)
                    self.assertFalse(self.c.blocked)
                    self.assertEqual(self.c.display_label,label)
                    self.assertNotIn("Z",self.c.composer.text)

    def test_small_scale_independent_tip_jitter_does_not_predict(self):
        for label in ("A","D","L","W"):
            with self.subTest(label=label):
                self.make(label)
                for i in range(155):
                    hand = synthetic_hand()
                    for k,p in enumerate(hand):
                        p.x = .5+(p.x-.5)*.20 + .00008*np.sin(i+k)
                        p.y = .5+(p.y-.5)*.20 + .00008*np.cos(i-k)
                    self.tick(i,hand)
                self.assertEqual(self.model.inputs,[])
                self.assertIsNone(self.c.candidate)

    def test_curved_entry_raw_result_does_not_cover_static_before_endpoint(self):
        # A curved rigid translation is inherently ambiguous without real
        # semantic evidence. Test separation, NOT that all arcs are rejected.
        for label in ("A","B","L","V","W"):
            with self.subTest(label=label):
                self.make(label)
                for i in range(100):
                    hand=synthetic_hand()
                    u=min(i/60,1.)
                    for p in hand:
                        p.x += .18*np.sin(np.pi*u)
                        p.y += .20*u
                    self.tick(i,hand)
                    if self.c.raw_dynamic_prediction:
                        self.assertIsNone(self.c.candidate)
                        self.assertEqual(self.c.display_label,label)
                        self.assertIsNone(self.c.jz_hold.hold_started_at)
                        self.assertEqual(self.c.validation["semantic_status"],"UNVERIFIED")
                        self.assertEqual(self.c.validation["geometry_policy"],"SHADOW_ONLY_NOT_RECALL_VALIDATED")
                        break
                else:
                    self.fail("Synthetic arc did not exercise the ambiguous raw-result path")

    def test_motion_criteria_cannot_be_accumulated_from_different_gap_runs(self):
        base = np.asarray([[0.,0.],[10.,20.],[20.,10.]])
        rows=[]
        for start in (0,10):
            for i in range(7):
                points=base.copy()
                points[1,0]+=i*9
                rows.append((start+i,(start+i)*.03,points))
        settings=MotionGateSettings()
        self.assertTrue(measure_motion(rows,100.,settings)["supported"])
        checked=coherent_motion_evidence(rows,100.,settings)
        self.assertFalse(checked["supported"])
        self.assertEqual(checked["run_count"],2)
        self.assertEqual(checked["qualifying_runs"],[])

    def test_failed_segment_validation_skips_model_and_restores_static(self):
        self.make("W",True)
        with patch("app.inference.unified_abc123_jz_auto_coordinator.validate_proposal",
                   return_value={"status":"REVIEW_REQUIRED","reason":"NO_SINGLE_COHERENT_MOTION_RUN"}):
            for i in range(70): self.tick(i,articulated_hand(i))
        self.assertEqual(self.model.inputs,[])
        self.assertIsNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.candidate)
        self.assertFalse(self.c.blocked)
        self.assertEqual(self.c.display_label,"W")
        self.assertEqual(self.c.composer.text,"")

    def test_shadow_geometry_failure_cannot_change_acceptance(self):
        self.make()
        with patch("app.inference.integrated_jz_validation.geometry_candidates",
                   side_effect=ValueError("shadow diagnostic unavailable")):
            for i in range(65):
                self.tick(i,articulated_hand(i))
                if self.c.pending_dynamic: break
        self.assertIsNotNone(self.c.pending_dynamic)
        self.assertEqual(self.c.validation["status"],"PASS")
        self.assertIn("shadow_geometry_error",self.c.validation)

    def test_short_gap_keeps_segment_but_long_gap_restores_static(self):
        self.make("P")
        for i in range(20): self.tick(i,articulated_hand(i))
        entries=self.c.session.segmenter.entries
        for i in range(20,23): self.tick(i,None)
        self.assertIs(self.c.session.segmenter.entries,entries)
        self.tick(23,articulated_hand(23))
        self.assertFalse({20,21,22}&{row[0] for row in entries})
        for i in range(24,31): self.tick(i,None)
        self.assertIsNone(self.c.candidate)
        self.assertFalse(self.c.blocked)
        self.tick(31,synthetic_hand())
        self.assertEqual(self.c.display_label,"P")

    def test_change_static_labels_does_not_reuse_dynamic_prediction(self):
        self.make()
        for i in range(180):
            self.c.static.predictors["ABC"].label="ABLPVW"[(i//30)%6]
            self.tick(i,synthetic_hand())
        self.assertEqual(self.model.inputs,[])
        self.assertIsNone(self.c.candidate)
        self.assertIsNone(self.c.raw_dynamic_prediction)

    def test_real_geometry_guard_before_endpoint_without_mock(self):
        from types import SimpleNamespace
        self.make()
        self.c.proposal_observation=SimpleNamespace(scale_px=100.,shape=np.zeros(63))
        self.assertTrue(self.c.proposal_pose_matches(SimpleNamespace(scale_px=100.,shape=np.zeros(63))))
        self.assertFalse(self.c.proposal_pose_matches(SimpleNamespace(scale_px=100.,shape=np.ones(63))))
        self.assertFalse(self.c.proposal_pose_matches(SimpleNamespace(scale_px=150.,shape=np.zeros(63))))

    def test_jz_then_release_then_static_does_not_resurrect_old_z(self):
        self.make("I",True)
        for i in range(200): self.tick(i,articulated_hand(min(i,65)))
        self.assertEqual(self.c.composer.text,"Z")
        for i in range(200,225): self.tick(i,None)
        self.c.static.predictors["ABC"].label="B"
        for i in range(225,350): self.tick(i,synthetic_hand())
        self.assertEqual(self.c.display_label,"B")
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.composer.text,"ZB")
        self.assertEqual(len(self.model.inputs),1)

    def test_trace_distinguishes_prediction_validation_display_and_confirmation(self):
        self.make("I",True)
        for i in range(200):
            self.tick(i,articulated_hand(min(i,65)))
            self.c.note_displayed(i/30)
        names=[event["event"] for event in self.events]
        for name in ("JZ_RAW_PREDICTION","JZ_SEGMENT_VALIDATION","JZ_CANDIDATE_CREATED",
                     "JZ_VALIDATED_CANDIDATE_DISPLAYED","STATIC_SUPPRESSION_CHANGED",
                     "JZ_CONFIRMED_EXPERIMENTAL"):
            self.assertIn(name,names)
        raw=next(e for e in self.events if e["event"]=="JZ_RAW_PREDICTION")
        candidate=next(e for e in self.events if e["event"]=="JZ_CANDIDATE_CREATED")
        self.assertIsNone(raw["candidate"])
        self.assertEqual(raw["raw_prediction"],"Z")
        self.assertEqual(raw["attempt_id"],candidate["attempt_id"])
        self.assertGreaterEqual(candidate["timestamp_monotonic"],raw["timestamp_monotonic"])
        self.assertEqual(names.count("JZ_VALIDATED_CANDIDATE_DISPLAYED"),1)


if __name__ == "__main__":
    unittest.main()
