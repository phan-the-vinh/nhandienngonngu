"""Phase-A synthetic geometry/plumbing tests, not real Z recall evidence."""
from dataclasses import replace
from types import SimpleNamespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.inference.z_start_pose_experimental import (ZStartPoseValidator,ZStartPoseShadow,
    ZStartPoseSettings,TemporalReadyEvidence,replay_start_pose_shadow)
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.jz_attempt_capture import AttemptCapture,load_attempt,geometry_replay
from tests.test_unified_abc123_jz_auto import DynamicModel,mirrored
from tests.test_unified_camera_experiment import FakePredictor,synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand


def pose_hand(extended=("index",)):
    """Articulated synthetic bones, NOT labelled real ASL photographs."""
    a=np.zeros((21,3))
    a[0]=[.5,.78,0.]
    a[1:5]=[[.38,.69,0],[.32,.60,0],[.29,.56,0],[.25,.54,0]]
    for name,base,x in (("index",5,.4),("middle",9,.48),("ring",13,.56),("pinky",17,.64)):
        a[base]=[x,.5,0]
        a[base+1]=[x,.4,0]
        if name in extended:
            a[base+2]=[x,.32,0]
            a[base+3]=[x,.24,0]
        else:
            a[base+2]=[x+.02,.48,0]
            a[base+3]=[x+.03,.55,0]
    return a


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.validator=ZStartPoseValidator()

    def test_index_only_pose_valid_and_thumb_has_no_acceptance_constraint(self):
        result=self.validator.evaluate(pose_hand(),(640,480))
        self.assertEqual(result["pose_status"],"VALID")
        self.assertEqual(result["pose_score"],1.)
        self.assertEqual(result["thumb"]["acceptance_constraint"],"NONE_UNCALIBRATED_VARIANTS")
        self.assertEqual(result["landmark_confidence"],"NOT_PROVIDED")

    def test_many_static_like_configurations_not_z_pose(self):
        # Names exercise requested cases, not a claim these fixtures are ASL ground truth.
        cases={"A":(),"U":("index","middle"),"R":("index","middle"),
               "W":("index","middle","ring"),"K":("index","middle"),
               "Y":("pinky",),"V":("index","middle"),"J-like":("pinky",)}
        for name,extended in cases.items():
            with self.subTest(name=name):
                r=self.validator.evaluate(pose_hand(extended),(640,480))
                self.assertEqual(r["pose_status"],"INVALID")
                self.assertTrue(any(p["status"]=="FAIL" for p in r["predicates"].values()))

    def test_pose_invariant_to_mirror_scale_translation_and_pseudo_3d_rotation(self):
        points=pose_hand()*[640,480,640]
        center=points[0].copy()
        for theta in (0.,.4,1.1,-.7):
            for axis in ("z","x"):
                c,s=np.cos(theta),np.sin(theta)
                rotation=np.asarray([[c,-s,0],[s,c,0],[0,0,1]]) if axis=="z" else np.asarray([[1,0,0],[0,c,-s],[0,s,c]])
                raw=((points-center)@rotation.T*.7+center+[7.,-12.,0.])/[640,480,640]
                for reflected in (False,True):
                    a=raw.copy()
                    if reflected: a[:,0]=1-a[:,0]
                    self.assertEqual(self.validator.evaluate(a,(640,480),handedness="Left" if reflected else "Right")["pose_status"],"VALID")

    def test_aspect_ratio_correction_preserves_geometry(self):
        physical=pose_hand()*[640,480,640]
        original=self.validator.evaluate(pose_hand(),(640,480))
        wide=self.validator.evaluate(physical/[1280,480,1280],(1280,480))
        self.assertEqual(original["finger_geometry"],wide["finger_geometry"])

    def test_invalid_degenerate_missing_and_multiple_are_uncertain(self):
        for hand,count in ((None,0),(pose_hand(),2),(np.zeros((21,3)),1),(np.full((21,3),np.nan),1),(np.zeros((20,3)),1)):
            self.assertEqual(self.validator.evaluate(hand,(640,480),hand_count=count)["pose_status"],"UNCERTAIN")

    def test_low_presence_and_handedness_are_not_identity(self):
        self.assertEqual(self.validator.evaluate(pose_hand(),(640,480),presence_score=.1)["pose_status"],"UNCERTAIN")
        self.assertEqual(self.validator.evaluate(pose_hand(),(640,480),handedness="Left")["pose_status"],
                         self.validator.evaluate(pose_hand(),(640,480),handedness="Right")["pose_status"])

    def test_uncertain_band_and_config_validation(self):
        s=replace(ZStartPoseSettings(),index_pip_min_degrees=170.)
        a=pose_hand()
        # Slightly bent synthetic index, 165 degrees at PIP.
        length=.08
        a[7]=a[6]+[np.sin(np.radians(15))*length*480/640,-np.cos(np.radians(15))*length,0]
        a[8]=a[7]+(a[7]-a[6])
        self.assertEqual(ZStartPoseValidator(s).evaluate(a,(640,480))["pose_status"],"UNCERTAIN")
        with self.assertRaises(ValueError): ZStartPoseValidator(replace(s,stable_frames=0))


class TemporalShadowTests(unittest.TestCase):
    def ready(self,shadow):
        for i in range(6): shadow.observe(i,pose_hand(),(640,480),i*.04)
        self.assertEqual(shadow.state,"Z_POSE_READY")

    def test_short_stable_pose_ready_without_three_second_hold(self):
        shadow=ZStartPoseShadow()
        shadow.observe(0,pose_hand(),(640,480),0.)
        self.assertIsNone(shadow.ready)
        self.ready(shadow)
        self.assertLess(shadow.ready["timestamp"],.3)

    def test_new_onset_after_ready_links_but_semantic_motion_remains_unverified(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        shadow.begin_attempt(1,7,.28,.24)
        r=shadow.snapshot()
        self.assertTrue(r["attempt"]["start_pose_linked"])
        self.assertEqual(r["state"],"Z_CAPTURING")
        self.assertFalse(r["attempt"]["would_allow_z_candidate"])

    def test_before_ready_motion_not_reused_or_retroactively_linked(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        shadow.begin_attempt(1,7,.28,.04)
        self.assertFalse(shadow.attempt["start_pose_linked"])
        for i in range(8,15): shadow.observe(i,pose_hand(),(640,480),i*.04)
        self.assertFalse(shadow.attempt["start_pose_linked"])
        self.assertEqual(shadow.state,"IDLE")

    def test_static_entry_transitions_and_rotation_never_supply_semantic_pass(self):
        shadow=ZStartPoseShadow()
        for extended in ((),("index","middle"),("index","middle","ring"),("pinky",)):
            for i in range(8):
                shadow.observe(i,pose_hand(extended),(640,480),i*.04)
                self.assertIsNone(shadow.ready)
        shadow.begin_attempt(9,8,.32,.2)
        self.assertFalse(shadow.attempt["start_pose_linked"])
        self.assertFalse(shadow.attempt["would_allow_z_candidate"])

    def test_missing_tracking_invalidates_ready_and_does_not_prove_hand_identity(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        shadow.begin_attempt(1,7,.28,.24)
        shadow.observe(8,None,(640,480),.32,hand_count=0)
        self.assertIsNone(shadow.ready)
        self.assertEqual(shadow.attempt["missing_frames"],[8])
        self.assertEqual(shadow.attempt["continuity_status"],"UNVERIFIED_GAP")
        shadow.observe(9,pose_hand(),(640,480),.36)
        self.assertEqual(shadow.attempt["continuity_status"],"UNVERIFIED_GAP")

    def test_palm_jump_and_large_pose_change_clear_ready(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        moved=pose_hand();moved[:,0]+=.5
        shadow.observe(6,moved,(640,480),.24)
        self.assertIsNone(shadow.ready)
        self.assertGreater(shadow.track_generation,0)
        shadow.observe(7,pose_hand(("index","middle")),(640,480),.28)
        self.assertIsNone(shadow.ready)

    def test_rearm_without_hand_release_requires_fresh_readiness_and_keeps_no_old_attempt(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        shadow.begin_attempt(1,7,.28,.24)
        shadow.end_attempt()
        self.assertIsNone(shadow.ready)
        for i in range(8,14): shadow.observe(i,pose_hand(),(640,480),i*.04)
        shadow.begin_attempt(2,15,.60,.56)
        self.assertTrue(shadow.attempt["start_pose_linked"])
        self.assertEqual(shadow.attempt["attempt_id"],2)

    def test_slow_fast_rigid_hand_motion_preserves_pose_without_relative_finger_motion(self):
        for fps in (20,30,60):
            for speed in (.02,.20):
                shadow=ZStartPoseShadow()
                for i in range(fps//2): shadow.observe(i,pose_hand(),(640,480),i/fps)
                onset=fps//2
                shadow.begin_attempt(1,onset,onset/fps,(onset-1)/fps)
                self.assertTrue(shadow.attempt["start_pose_linked"])
                for i in range(onset,onset+fps):
                    hand=pose_hand();hand[:,0]+=speed*(i-onset)/fps
                    result=shadow.observe(i,hand,(640,480),i/fps)
                    self.assertEqual(result["pose"]["pose_status"],"VALID")
                self.assertEqual(shadow.attempt["continuity_status"],"OBSERVED_ONLY_IDENTITY_UNVERIFIED")

    def test_valid_index_pose_and_generic_motion_are_not_semantic_z_proof(self):
        shadow=ZStartPoseShadow();self.ready(shadow)
        shadow.begin_attempt(1,7,.28,.24)
        self.assertTrue(shadow.attempt["start_pose_linked"])
        self.assertIn("UNVERIFIED",shadow.attempt["motion_status"])
        self.assertFalse(shadow.attempt["would_allow_z_candidate"])


class TemporalEvidenceTests(unittest.TestCase):
    def test_legacy_ready_lifecycle_is_unchanged_by_temporal_instrumentation(self):
        old,new=ZStartPoseShadow(),ZStartPoseShadow()
        for i in range(45):
            hand=pose_hand() if i<12 or i>=24 else pose_hand(("index","middle"))
            if i in (18,19): hand=None
            old._observe_legacy(i,hand,(640,480),i*.04,hand_count=1 if hand is not None else 0)
            new.observe(i,hand,(640,480),i*.04,hand_count=1 if hand is not None else 0)
            if i in (7,32):
                for s in (old,new): s.begin_attempt(i,i,i*.04,(i-1)*.04)
            if i in (16,40):
                for s in (old,new): s.end_attempt()
            new.finish_frame(i,i*.04)
            left,right=old.snapshot(),new.snapshot()
            for key in ('state','pose','ready','attempt','settings'):
                self.assertEqual(left[key],right[key])

    def prepared(self):
        s=ZStartPoseShadow()
        for i in range(6):
            s.observe(i,pose_hand(),(640,480),i*.04)
            s.finish_frame(i,i*.04)
        self.assertEqual(s.temporal.state,"READY")
        return s

    def uncertain(self,s,index=6):
        pose=s.validator.evaluate(pose_hand(),(640,480))
        pose["pose_status"]="UNCERTAIN"
        with patch.object(s.validator,"evaluate",return_value=pose):
            return s.observe(index,pose_hand(),(640,480),index*.04)

    def test_uncertain_at_onset_preserves_prior_evidence_not_legacy_ready(self):
        s=self.prepared();self.uncertain(s)
        self.assertIsNone(s.ready)
        self.assertEqual(s.temporal.state,"ONSET_BOUNDARY_PENDING")
        s.begin_attempt(1,6,.24,.20)
        r=s.snapshot()
        self.assertFalse(r["attempt"]["start_pose_linked"])
        new=r["temporal_evidence"]["attempt"]
        self.assertTrue(new["temporal_start_pose_linked"])
        self.assertEqual(new["ready_evidence"]["last_valid_frame"],5)
        self.assertEqual(new["ready_evidence"]["linked_attempt_id"],1)
        self.assertAlmostEqual(new["ready_to_onset_seconds"],.12)
        self.assertFalse(new["would_allow_z_candidate"])

    def test_valid_new_onset_both_associations_match(self):
        s=self.prepared();s.observe(6,pose_hand(),(640,480),.24)
        s.begin_attempt(1,6,.24,.20)
        self.assertTrue(s.attempt["start_pose_linked"])
        self.assertTrue(s.temporal.attempt["temporal_start_pose_linked"])

    def test_uncertainty_without_same_frame_onset_expires_no_ttl_guess(self):
        s=self.prepared();self.uncertain(s)
        s.finish_frame(6,.24)
        self.assertIsNone(s.temporal.evidence)
        self.assertEqual(s.temporal.reason,"UNCERTAIN_FRAME_WITHOUT_ONSET")
        self.uncertain(s,7);s.begin_attempt(1,7,.28,.24)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])
        self.assertIsNone(s.temporal.snapshot()["ttl_seconds"])

    def test_clear_pose_change_invalidates_before_onset(self):
        s=self.prepared()
        s.observe(6,pose_hand(("index","middle")),(640,480),.24)
        self.assertIsNone(s.temporal.evidence)
        s.begin_attempt(1,6,.24,.20)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])

    def test_tracking_loss_never_supplies_rearm_or_identity(self):
        s=self.prepared();s.observe(6,None,(640,480),.24,hand_count=0)
        self.assertIsNone(s.temporal.evidence)
        s.begin_attempt(1,6,.24,.20)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])
        self.assertIn("UNVERIFIED",s.temporal.snapshot()["identity_status"])

    def test_joint_discontinuity_not_hidden_by_uncertain_boundary(self):
        s=self.prepared()
        # Existing angle guard remains authoritative for temporal diagnostics.
        pose=s.validator.evaluate(pose_hand(),(640,480));pose["pose_status"]="UNCERTAIN"
        pose["finger_geometry"]["pinky"]["dip_angle_degrees"]+=30
        with patch.object(s.validator,"evaluate",return_value=pose): s.observe(6,pose_hand(),(640,480),.24)
        self.assertEqual(s.temporal.reason,"JOINT_GEOMETRY_DISCONTINUITY")
        s.begin_attempt(1,6,.24,.20)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])

    def test_motion_before_readiness_cannot_be_retroactively_linked(self):
        s=self.prepared();s.observe(6,pose_hand(),(640,480),.24)
        s.begin_attempt(1,6,.24,.04)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])
        for i in range(7,20): s.observe(i,pose_hand(),(640,480),i*.04)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])
        self.assertIsNone(s.temporal.evidence)

    def test_completion_consumes_token_rearm_needs_new_streak(self):
        s=self.prepared();s.observe(6,pose_hand(),(640,480),.24)
        s.begin_attempt(1,6,.24,.20);old=s.temporal.attempt["ready_evidence"]["timestamp"]
        s.end_attempt()
        self.assertIsNone(s.temporal.evidence)
        s.observe(7,pose_hand(),(640,480),.28)
        self.assertIsNone(s.temporal.evidence)
        for i in range(8,12): s.observe(i,pose_hand(),(640,480),i*.04)
        s.begin_attempt(2,11,.44,.40)
        self.assertTrue(s.temporal.attempt["temporal_start_pose_linked"])
        self.assertGreater(s.temporal.attempt["ready_evidence"]["timestamp"],old)

    def test_session_generation_cannot_inherit_old_ready(self):
        s=self.prepared()
        s.observe(6,pose_hand(),(640,480),.24,session_generation=1)
        self.assertIsNone(s.ready);self.assertIsNone(s.temporal.evidence)
        s.begin_attempt(1,6,.24,.20)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])

    def test_stable_pose_without_motion_never_creates_attempt_or_candidate(self):
        s=self.prepared()
        for i in range(6,200): s.observe(i,pose_hand(),(640,480),i*.04);s.finish_frame(i,i*.04)
        self.assertIsNone(s.temporal.attempt)
        self.assertIsNone(s.temporal.last_association)
        self.assertEqual(s.temporal.evidence["last_valid_frame"],199)

    def test_static_to_z_and_z_to_z_need_fresh_observations_not_hand_withdrawal(self):
        for static in ((),("index","middle"),("index","middle","ring")):
            s=ZStartPoseShadow()
            for i in range(6): s.observe(i,pose_hand(static),(640,480),i*.04)
            for i in range(6,13): s.observe(i,pose_hand(),(640,480),i*.04)
            s.begin_attempt(1,12,.48,.44)
            self.assertTrue(s.temporal.attempt["temporal_start_pose_linked"])
            s.end_attempt()
            for i in range(13,20): s.observe(i,pose_hand(),(640,480),i*.04)
            s.begin_attempt(2,19,.76,.72)
            self.assertTrue(s.temporal.attempt["temporal_start_pose_linked"])

    def test_z_to_j_does_not_reuse_z_preparation(self):
        s=self.prepared();s.observe(6,pose_hand(),(640,480),.24)
        s.begin_attempt(1,6,.24,.20);s.end_attempt()
        for i in range(7,14): s.observe(i,pose_hand(("pinky",)),(640,480),i*.04)
        s.begin_attempt(2,13,.52,.48)
        self.assertFalse(s.temporal.attempt["temporal_start_pose_linked"])

    def test_pose_during_capture_does_not_erase_frozen_preparation(self):
        s=self.prepared();s.observe(6,pose_hand(),(640,480),.24)
        s.begin_attempt(1,6,.24,.20)
        s.observe(7,pose_hand(("index","middle")),(640,480),.28)
        self.assertTrue(s.temporal.attempt["temporal_start_pose_linked"])
        self.assertIsNone(s.temporal.evidence)

    def test_snapshot_mutation_cannot_modify_token_or_attempt(self):
        s=self.prepared();r=s.snapshot()
        r["temporal_evidence"]["evidence"]["last_valid_frame"]=999
        self.assertEqual(s.temporal.evidence["last_valid_frame"],5)


class IntegrationTests(unittest.TestCase):
    def tick(self,c,i,hand):
        return c.update(i,mirrored(hand) if hand else None,hand,(480,640,3),i/30,raw_hand_count=1 if hand else 0)

    def test_shadow_on_off_does_not_change_j_z_static_hold_features_or_text(self):
        for label in ("J","Z"):
            models=[DynamicModel(label),DynamicModel(label)]
            cs=[UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),m,
                                       enable_experimental_jz=True,z_start_pose_shadow=bool(n)) for n,m in enumerate(models)]
            for i in range(220):
                hand=articulated_hand(min(i,65))
                a=self.tick(cs[0],i,hand);b=self.tick(cs[1],i,hand)
                self.assertEqual(a[1],b[1])
                self.assertEqual(cs[0].candidate,cs[1].candidate)
                self.assertEqual(cs[0].hold.state,cs[1].hold.state)
                self.assertEqual(cs[0].composer.text,cs[1].composer.text)
                self.assertEqual(cs[0].blocked,cs[1].blocked)
            self.assertEqual(cs[1].composer.text,label)
            for a,b in zip(models[0].inputs,models[1].inputs): np.testing.assert_array_equal(a,b)

    def test_shadow_error_does_not_change_recognition(self):
        c=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),DynamicModel("J"),
                                 enable_experimental_jz=True,z_start_pose_shadow=True)
        with patch.object(c.z_start_pose_shadow,"observe",side_effect=RuntimeError("diagnostic error")):
            for i in range(220): self.tick(c,i,articulated_hand(min(i,65)))
        self.assertEqual(c.composer.text,"J")
        self.assertEqual(c.z_start_pose_error,"diagnostic error")

    def test_temporal_diagnostic_failure_leaves_recognition_and_model_inputs_identical(self):
        models=[DynamicModel("Z"),DynamicModel("Z")]
        cs=[UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),models[n],
                                  enable_experimental_jz=True,z_start_pose_shadow=bool(n)) for n in range(2)]
        with patch.object(cs[1].z_start_pose_shadow.temporal,"observe",side_effect=RuntimeError("temporal diagnostic failure")):
            for i in range(220):
                hand=articulated_hand(min(i,65))
                left=self.tick(cs[0],i,hand);right=self.tick(cs[1],i,hand)
                self.assertEqual(left[1],right[1])
                self.assertEqual(cs[0].display_label,cs[1].display_label)
                self.assertEqual(cs[0].blocked,cs[1].blocked)
                self.assertEqual(cs[0].hold.state,cs[1].hold.state)
                self.assertEqual(cs[0].composer.text,cs[1].composer.text)
        self.assertEqual(cs[1].composer.text,"Z")
        self.assertEqual(len(models[0].inputs),len(models[1].inputs))
        for a,b in zip(models[0].inputs,models[1].inputs): np.testing.assert_array_equal(a,b)

    def test_digits_and_mode_switch_do_not_run_z_pose_or_clear_text(self):
        c=UnifiedAutoCoordinator(FakePredictor("A",0.),FakePredictor("0",.5),DynamicModel(),z_start_pose_shadow=True)
        c.composer.append("A")
        c.switch_mode("123")
        with patch.object(c.z_start_pose_shadow,"observe",side_effect=AssertionError("Z pose in 123")) as observe:
            self.tick(c,0,None)
            observe.assert_not_called()
        c.switch_mode("ABC")
        self.assertEqual(c.composer.text,"A")
        self.assertIsNone(c.z_start_pose_shadow.ready)

    def test_check_only_shadow_flag_never_opens_camera_or_writes_capture(self):
        from app.inference import unified_abc123_jz_auto_experiment as camera
        out=io.StringIO()
        with (patch("cv2.VideoCapture",side_effect=AssertionError("camera forbidden")),
              patch.object(camera,"AttemptCapture") as capture,contextlib.redirect_stdout(out)):
            self.assertEqual(camera.main(["--check-only","--z-start-pose-shadow","--capture-jz-diagnostics"]),0)
        capture.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["z_start_pose_mode"],"SHADOW")
        self.assertFalse(json.loads(out.getvalue())["z_start_pose_acceptance_gate"])

    def test_capture_and_replay_keep_features_exact_and_shadow_association_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cap=AttemptCapture(root/"diagnostics",root=root,session_id="pose_shadow",model_sha256="fake",config={})
            c=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),DynamicModel("Z"),
                                     diagnostics=cap,z_start_pose_shadow=True)
            for i in range(90): self.tick(c,i,articulated_hand(min(i,65)))
            cap.close()
            self.assertEqual(cap.errors,[])
            folder=next(cap.path.glob("generation_*"))
            metadata,arrays=load_attempt(folder)
            self.assertTrue(geometry_replay(metadata,arrays)["exact_feature_match"])
            result=replay_start_pose_shadow(metadata,arrays)
            self.assertEqual(result["status"],"SHADOW_ONLY")
            self.assertFalse(result["acceptance_effect"])
            self.assertIn("z_start_pose_shadow",metadata["technical_validation"])

    def test_old_replay_missing_fields_cannot_be_inferred(self):
        result=replay_start_pose_shadow({}, {})
        self.assertEqual(result["status"],"UNAVAILABLE")
        self.assertIn("observed_landmarks",result["missing_fields"])

    def test_synthetic_jackz_without_release_has_identical_arbitration(self):
        # Mock labels validate lifecycle only, NOT actual JACKZ camera recall.
        models=[DynamicModel("J"),DynamicModel("J")]
        letters=[FakePredictor("I",0.),FakePredictor("I",0.)]
        cs=[UnifiedAutoCoordinator(letters[n],FakePredictor("0",.5),models[n],enable_experimental_jz=True,
                                   z_start_pose_shadow=bool(n)) for n in range(2)]
        index=0
        def feed(hand):
            nonlocal index
            a=self.tick(cs[0],index,hand);b=self.tick(cs[1],index,hand)
            self.assertEqual(a[1],b[1]);self.assertEqual(cs[0].composer.text,cs[1].composer.text)
            self.assertEqual(cs[0].hold.state,cs[1].hold.state)
            index+=1
        for i in range(220): feed(articulated_hand(min(i,65)))
        for n,label in enumerate("ACK"):
            for predictor in letters: predictor.label=label;predictor.shape=(n+1)*.3
            for _ in range(140): feed(synthetic_hand())
        for model in models: model.label="Z"
        for i in range(220): feed(articulated_hand(min(i,65)))
        self.assertEqual(cs[1].composer.text,"JACKZ")


if __name__=="__main__":
    unittest.main()
