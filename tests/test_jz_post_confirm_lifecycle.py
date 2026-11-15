"""Synthetic/mock lifecycle regressions, not actual-camera ASL accuracy tests."""
import copy
import unittest

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand


class PostConfirmTests(unittest.TestCase):
    def make(self,label="Z"):
        self.model=DynamicModel(label)
        self.letters=FakePredictor("I",0.)
        self.events=[]
        self.c=UnifiedAutoCoordinator(self.letters,FakePredictor("0",.5),self.model,
                                     enable_experimental_jz=True,event_sink=self.events.append)
        self.index=0

    def tick(self,raw,video=True):
        i=self.index
        self.index+=1
        return self.c.update(i,mirrored(raw) if raw and video else (synthetic_hand() if video else None),
                             raw,(480,640,3),i/30,raw_hand_count=1 if raw else 0)[1]

    def confirm_first(self,label="Z"):
        self.make(label)
        old_result=None
        for i in range(220):
            if self.c.session.last_result:
                old_result=copy.deepcopy(self.c.session.last_result)
            if self.tick(articulated_hand(min(i,65))):
                break
        self.assertEqual(self.c.composer.text,label)
        return old_result

    def test_commit_clears_actionable_bout_endpoint_and_result(self):
        self.confirm_first()
        for field in ("candidate","pending_dynamic","endpoint_observation","proposal_observation",
                      "candidate_at","candidate_attempt_id","raw_dynamic_prediction","validation"):
            with self.subTest(field=field): self.assertIsNone(getattr(self.c,field))
        self.assertIsNone(self.c.session.pending)
        self.assertIsNone(self.c.session.last_result)
        self.assertEqual(self.c.session.segmenter.entries,[])
        self.assertEqual(list(self.c.session.segmenter.history),[])
        self.assertEqual(self.c.capture_rows,[])
        self.assertTrue(self.c.post_jz_latch)
        self.tick(articulated_hand(65))
        self.assertEqual(self.c.jz_hold.state,"WAIT_RELEASE")

    def test_pose_rearm_discards_transition_buffer_without_requiring_release(self):
        self.confirm_first()
        # Real raw geometry changes, not just a mocked classifier label.
        for _ in range(3): self.tick(synthetic_hand())
        self.assertFalse(self.c.post_jz_latch)
        self.assertEqual(list(self.c.session.segmenter.history),[])
        self.assertFalse(self.c.session.segmenter.active)
        self.assertIsNone(self.c.session.segmenter.anchor)
        self.assertEqual(self.c.jz_hold.state,"IDLE")

    def test_replayed_result_cannot_unlock_consumed_attempt(self):
        old=self.confirm_first()
        self.assertIsNotNone(old)
        self.c.session.last_result=copy.deepcopy(old)
        self.tick(articulated_hand(65))
        self.assertTrue(self.c.post_jz_latch)
        self.assertIsNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.candidate)
        self.assertIsNone(self.c.raw_dynamic_prediction)
        self.assertEqual(self.c.composer.text,"Z")

    def test_delayed_pending_from_before_commit_never_invokes_model_again(self):
        self.confirm_first()
        self.c.session.pending=([],self.index,"MOTION_GATED_VALID_WINDOW",{})
        self.c.process_pending(self.index/30,articulated_hand(65),(480,640,3))
        self.assertIsNone(self.c.session.pending)
        self.assertIsNone(self.c.session.last_result)
        self.assertEqual(len(self.model.inputs),1)
        self.assertEqual(self.c.composer.text,"Z")
        self.assertTrue(self.c.post_jz_latch)

    def test_same_hand_after_commit_never_repeats_same_attempt(self):
        self.confirm_first()
        for _ in range(200): self.tick(articulated_hand(65))
        self.assertEqual(self.c.composer.text,"Z")
        self.assertEqual(len(self.model.inputs),1)

    def test_release_then_static_a_displays_and_confirms(self):
        self.confirm_first()
        for _ in range(23): self.tick(None,video=False)
        self.letters.label="A"
        for _ in range(125): self.tick(synthetic_hand())
        self.assertEqual(self.c.display_label,"A")
        self.assertEqual(self.c.composer.text,"ZA")
        self.assertEqual(len(self.model.inputs),1)

    def test_direct_static_transition_all_24_labels_after_j_and_z(self):
        for dynamic in ("J","Z"):
            with self.subTest(dynamic=dynamic):
                self.confirm_first(dynamic)
                expected=dynamic
                for n,label in enumerate("AURWKYBCDEFGHILMNOPQSTVX"):
                    self.letters.label=label
                    self.letters.shape=(n+1)*.1
                    for _ in range(135):
                        hand=synthetic_hand()
                        hand[8].x+=(n%4)*.015
                        self.tick(hand)
                    expected+=label
                    self.assertEqual(self.c.display_label,label)
                    self.assertIsNone(self.c.candidate)
                    self.assertIsNone(self.c.pending_dynamic)
                    self.assertEqual(self.c.composer.text,expected)
                self.assertEqual(len(self.model.inputs),1)

    def test_tracking_gap_does_not_reuse_finished_result(self):
        self.confirm_first()
        for _ in range(4): self.tick(None,video=True)
        self.assertTrue(self.c.post_jz_latch)  # IMAGE loss is not a physical release.
        for _ in range(40): self.tick(articulated_hand(65))
        for _ in range(2): self.tick(None,video=False)
        for _ in range(25): self.tick(articulated_hand(65))
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.composer.text,"Z")
        self.assertEqual(len(self.model.inputs),1)

    def test_fresh_same_or_other_dynamic_attempt_can_confirm_without_release(self):
        for first,second in (("Z","Z"),("Z","J"),("J","J"),("J","Z")):
            with self.subTest(first=first,second=second):
                self.confirm_first(first)
                self.model.label=second
                for k in range(220):
                    hand=articulated_hand(65)
                    for point in hand: point.x+=min(k,65)*.004
                    hand[20].x+=min(k,65)*.0035
                    self.tick(hand)
                self.assertEqual(self.c.composer.text,first+second)
                commits=[e for e in self.events if e["event"]=="JZ_CONFIRMED_EXPERIMENTAL"]
                self.assertEqual(len(commits),2)
                self.assertNotEqual(commits[0]["attempt_id"],commits[1]["attempt_id"])

    def test_trace_links_hold_and_display_to_fresh_candidate_and_last_commit(self):
        self.confirm_first()
        self.model.label="J"
        for k in range(220):
            hand=articulated_hand(65)
            for point in hand: point.x+=min(k,65)*.004
            hand[20].x+=min(k,65)*.0035
            self.tick(hand)
            self.c.note_displayed((self.index-1)/30)
        starts=[e for e in self.events if e["event"]=="JZ_DYNAMIC_HOLD_STARTED"]
        displays=[e for e in self.events if e["event"]=="JZ_VALIDATED_CANDIDATE_DISPLAYED"]
        self.assertEqual(len(starts),2)
        self.assertEqual(len(displays),1)
        for event in starts+displays:
            for field in ("timestamp_monotonic","current_attempt_id","candidate_source_attempt_id",
                          "candidate_created_time","last_confirmed_dynamic_attempt_id","static_instant",
                          "static_stable","dynamic_segmenter_state","proposal_candidate_state",
                          "post_jz_latch","hold_state","display_priority_reason",
                          "fresh_validated_dynamic_segment"):
                self.assertIn(field,event)
            self.assertTrue(event["fresh_validated_dynamic_segment"])
            self.assertEqual(event["display_priority_reason"],"FRESH_VALIDATED_DYNAMIC_CANDIDATE")
        self.assertEqual(starts[1]["last_confirmed_dynamic_attempt_id"],starts[0]["candidate_source_attempt_id"])
        self.assertNotEqual(starts[1]["candidate_source_attempt_id"],starts[1]["last_confirmed_dynamic_attempt_id"])
        self.assertEqual(starts[1]["static_stable"],"I")

    def test_same_number_in_new_mode_generation_is_not_a_replayed_attempt(self):
        self.confirm_first()
        self.c.switch_mode("123")
        self.c.switch_mode("ABC")
        for i in range(220): self.tick(articulated_hand(min(i,65)))
        self.assertEqual(self.c.composer.text,"ZZ")
        commits=[e for e in self.events if e["event"]=="JZ_CONFIRMED_EXPERIMENTAL"]
        self.assertEqual(commits[0]["attempt_id"],commits[1]["attempt_id"])
        self.assertNotEqual(commits[0]["session_generation"],commits[1]["session_generation"])


if __name__=="__main__":
    unittest.main()
