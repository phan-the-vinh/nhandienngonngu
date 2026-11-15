"""Synthetic instrumentation/replay equivalence, NOT actual-camera ASL recall."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.inference.automatic_jz_experiment import geometry_candidates
from app.inference.jz_shadow_diagnostics import geometry_predicates,trajectory_diagnostics
from app.inference.jz_attempt_capture import AttemptCapture,load_attempt,geometry_replay,model_replay,checked_output,sha256
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_abc123_jz_auto import DynamicModel,mirrored
from tests.test_unified_camera_experiment import FakePredictor,synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand
from tools.replay_jz_attempt import main as replay_main


class ShadowGeometryTests(unittest.TestCase):
    def test_predicates_equal_existing_geometry_including_passing_j_z(self):
        rng=np.random.default_rng(42)
        values=[rng.normal(size=(16,69)) for _ in range(100)]
        for label in ("J","Z"):
            x=np.zeros((16,69))
            hand=x[:,:63].reshape(16,21,3)
            tip,pip,col=(20,18,67) if label=="J" else (8,6,65)
            hand[:,tip,1]=1.
            hand[:,pip,1]=.5
            for other in ((8,12,16) if label=="J" else (12,16,20)):
                hand[:,other,1]=.1
                hand[:,other-2,1]=.3
            if label=="J":
                x[:,col]=np.r_[np.zeros(8),np.linspace(0,.5,8)]
                x[:,col+1]=np.r_[np.linspace(0,.6,8),np.full(8,.6)]
            else:
                x[:,col]=np.interp(np.arange(16),[0,5,10,15],[0,.5,0,.5])
                x[:,col+1]=np.interp(np.arange(16),[0,5,10,15],[0,0,.3,.3])
            self.assertIn(label,geometry_candidates(x))
            values.append(x)
        for x in values:
            result=geometry_predicates(x)
            self.assertEqual(set(result["geometry_candidates"]),geometry_candidates(x))
            for label,item in result["labels"].items():
                self.assertEqual(item["pass"],not item["failed_predicates"])
                for name in item["failed_predicates"]:
                    self.assertTrue(item["predicates"][name]["failure_reason"])

    def test_invalid_features_fail_diagnostics_not_recognition(self):
        for x in (np.zeros((15,69)),np.full((16,69),np.nan)):
            with self.assertRaises(ValueError): geometry_predicates(x)

    def test_default_geometry_summary_does_not_contain_full_landmark_or_feature_series(self):
        result=geometry_predicates(np.zeros((16,69)))
        for item in result["labels"].values():
            self.assertFalse(item["details"]["full_series_included"])
            self.assertNotIn("path_xy",item["details"])
            self.assertNotIn("extended_tip_norm",item["details"])

    def test_raw_trajectory_does_not_bridge_missing_frames(self):
        a=np.asarray([[[p.x,p.y,p.z] for p in articulated_hand(i)] for i in (0,1,4,5)])
        result=trajectory_diagnostics(a,np.tile([640,480],(4,1)),[0,1,4,5],[0.,.1,.4,.5])
        self.assertEqual(result["contiguous_edges"],[True,False,True])
        self.assertIsNone(result["landmarks"]["0"]["speed_px_per_second"][1])
        self.assertIsNone(result["canonical_shape_step_rms"][1])
        self.assertIn("RAW_UNSMOOTHED",result["definition"])


class CaptureReplayTests(unittest.TestCase):
    def make(self,root,session="test_session",enabled=True):
        capture=AttemptCapture(root/"diagnostics",root=root,session_id=session,
                               model_sha256="synthetic-model",config={"seed":42})
        model=DynamicModel("Z")
        c=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),model,
                                 enable_experimental_jz=enabled,diagnostics=capture)
        return capture,c,model

    def tick(self,c,i,hand):
        return c.update(i,mirrored(hand) if hand else None,hand,(480,640,3),i/30,raw_hand_count=1 if hand else 0)

    def record(self,root,gaps=False):
        capture,c,model=self.make(root)
        for i in range(220):
            h=None if gaps and i in (20,21) else articulated_hand(min(i,65))
            self.tick(c,i,h)
        capture.close()
        self.assertEqual(capture.errors,[])
        folders=list(capture.path.glob("generation_*"))
        self.assertEqual(len(folders),1)
        return capture,c,model,folders[0]

    def test_capture_replays_exact_actual_model_input_and_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            capture,c,model,folder=self.record(Path(temp))
            metadata,arrays=load_attempt(folder)
            np.testing.assert_array_equal(model.inputs[0],arrays["features"].reshape(1,1104,order="C"))
            result=geometry_replay(metadata,arrays)
            self.assertTrue(result["exact_feature_match"])
            self.assertTrue(result["geometry_matches_saved"])
            self.assertEqual(result["feature_shape"],[16,69])
            replay=model_replay(metadata,arrays,DynamicModel("Z"),"synthetic-model")
            self.assertTrue(replay["prediction_match"])
            self.assertTrue(replay["probabilities_exact_match"])
            self.assertEqual(replay["max_probability_difference"],0.)
            self.assertEqual(metadata["candidate_status"],"COMMITTED")
            self.assertEqual(c.composer.text,"Z")
            self.assertIn("post_endpoint_stability",metadata)
            self.assertEqual(metadata["human_label"],"UNREVIEWED")
            with (capture.path/"human_review_manifest.csv").open(newline="",encoding="utf-8") as stream:
                review=list(csv.DictReader(stream))
            self.assertEqual(len(review),1)
            self.assertEqual(review[0]["human_label"],"UNREVIEWED")
            self.assertEqual(review[0]["committed_character"],"Z")

    def test_capture_preserves_gap_and_canonical_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,folder=self.record(Path(temp),gaps=True)
            metadata,arrays=load_attempt(folder)
            self.assertTrue({20,21}.issubset(set(metadata["tracking_gap_frame_indices"])))
            self.assertFalse({20,21}&set(arrays["frame_indices"].tolist()))
            self.assertFalse({20,21}&set(arrays["selected_frame_indices"].tolist()))
            self.assertTrue(geometry_replay(metadata,arrays)["exact_feature_match"])

    def test_replay_actual_frozen_classifier_on_synthetic_landmarks(self):
        from app.inference.dynamic_jz_camera_test import load_experiment_model
        model,model_hash=load_experiment_model()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cap=AttemptCapture(root/"diagnostics",root=root,session_id="actual_model_synthetic_input",
                               model_sha256=model_hash,config={"test_input":"SYNTHETIC_NOT_CAMERA_OR_DATASET"})
            c=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),model,diagnostics=cap)
            for i in range(100):
                self.tick(c,i,articulated_hand(i))
                if c.raw_dynamic_prediction:
                    break
            self.assertIsNotNone(c.raw_dynamic_prediction)
            cap.close()
            self.assertEqual(cap.errors,[])
            folder=next(cap.path.glob("generation_*"))
            metadata,arrays=load_attempt(folder)
            self.assertTrue(geometry_replay(metadata,arrays)["exact_feature_match"])
            result=model_replay(metadata,arrays,model,model_hash)
            self.assertTrue(result["prediction_match"])
            self.assertTrue(result["probabilities_exact_match"])
            self.assertEqual(result["max_probability_difference"],0.)

    def test_rejected_before_model_has_truthful_valid_count_and_no_invented_features(self):
        with tempfile.TemporaryDirectory() as temp:
            cap,c,model=self.make(Path(temp))
            for i in range(20): self.tick(c,i,articulated_hand(i))
            self.assertTrue(c.session.segmenter.active)
            for i in range(20,28): self.tick(c,i,None)
            cap.close()
            self.assertEqual(cap.errors,[])
            folder=next(cap.path.glob("generation_*"))
            metadata=json.loads((folder/"attempt.json").read_text())
            self.assertGreater(metadata["valid_frame_count"],0)
            self.assertFalse(metadata["replay_available"])
            self.assertEqual(model.inputs,[])
            self.assertEqual(metadata["human_label"],"UNREVIEWED")
            self.assertEqual(metadata["feature_shape"],[0,69])
            with self.assertRaisesRegex(ValueError,"missing capture/actual model features"):
                load_attempt(folder)

    def test_diagnostics_do_not_change_prediction_candidate_hold_text_or_features(self):
        with tempfile.TemporaryDirectory() as temp:
            cap,on,m1=self.make(Path(temp))
            m0=DynamicModel("Z")
            off=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),m0,enable_experimental_jz=True)
            for i in range(220):
                hand=articulated_hand(min(i,65))
                a=self.tick(off,i,hand)
                b=self.tick(on,i,hand)
                self.assertEqual(a[1],b[1])
                self.assertEqual(off.candidate,on.candidate)
                self.assertEqual(off.display_label,on.display_label)
                self.assertEqual(off.hold.state,on.hold.state)
                self.assertEqual(off.composer.text,on.composer.text)
                self.assertEqual(off.blocked,on.blocked)
            cap.close()
            self.assertEqual(cap.errors,[])
            self.assertEqual(len(m0.inputs),len(m1.inputs))
            for a,b in zip(m0.inputs,m1.inputs): np.testing.assert_array_equal(a,b)

    def test_static_and_digits_hold_and_text_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            cap,on,_=self.make(Path(temp))
            off=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),DynamicModel("Z"),enable_experimental_jz=True)
            for mode in ("ABC","123","ABC"):
                on.switch_mode(mode);off.switch_mode(mode)
                for i in range(130):
                    self.assertEqual(self.tick(on,i,synthetic_hand())[1],self.tick(off,i,synthetic_hand())[1])
                    self.assertEqual(on.composer.text,off.composer.text)
                    self.assertEqual(on.hold.state,off.hold.state)
            cap.close()
            self.assertEqual(list(cap.path.glob("*.npz")),[])

    def test_diagnostic_failure_cannot_change_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            cap,c,_=self.make(Path(temp))
            with patch.object(cap,"observe",side_effect=OSError("disk/diagnostic error")):
                for i in range(220): self.tick(c,i,articulated_hand(min(i,65)))
            self.assertEqual(c.composer.text,"Z")
            self.assertTrue(cap.errors)
            cap.close()

    def test_geometry_diagnostic_exception_does_not_change_technical_pass(self):
        c=UnifiedAutoCoordinator(FakePredictor("I",0.),FakePredictor("0",.5),DynamicModel("Z"))
        with patch("app.inference.integrated_jz_validation.geometry_predicates",side_effect=RuntimeError("shadow unavailable")):
            for i in range(80):
                self.tick(c,i,articulated_hand(min(i,65)))
                if c.pending_dynamic:
                    break
        self.assertIsNotNone(c.pending_dynamic)
        self.assertEqual(c.validation["status"],"PASS")
        self.assertIn("geometry_predicate_diagnostic_error",c.validation)

    def test_session_attempt_and_generation_ids_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cap,c,_=self.make(root)
            for i in range(220): self.tick(c,i,articulated_hand(min(i,65)))
            c.switch_mode("123");c.switch_mode("ABC")
            for i in range(220): self.tick(c,i+220,articulated_hand(min(i,65)))
            cap.close()
            folders=list(cap.path.glob("generation_*"))
            self.assertEqual(len(folders),2)
            keys={(json.loads((p/"attempt.json").read_text())["session_generation"],
                   json.loads((p/"attempt.json").read_text())["attempt_id"]) for p in folders}
            self.assertEqual(keys,{(0,1),(2,1)})
            other,_,_=self.make(root,session="other_session")
            other.close()
            self.assertNotEqual(cap.path,other.path)

    def test_two_dynamic_attempts_without_release_save_distinct_actual_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            cap,c,model=self.make(Path(temp))
            for i in range(220): self.tick(c,i,articulated_hand(min(i,65)))
            for k in range(220):
                hand=articulated_hand(65)
                for point in hand: point.x+=min(k,65)*.004
                hand[20].x+=min(k,65)*.0035
                self.tick(c,k+220,hand)
            cap.close()
            self.assertEqual(cap.errors,[])
            self.assertEqual(c.composer.text,"ZZ")
            folders=sorted(cap.path.glob("generation_*"))
            self.assertEqual(len(folders),2)
            self.assertEqual(len(model.inputs),2)
            for folder,flat in zip(folders,model.inputs):
                metadata,arrays=load_attempt(folder)
                np.testing.assert_array_equal(arrays["features"].reshape(1,1104),flat)
                self.assertTrue(geometry_replay(metadata,arrays)["exact_feature_match"])

    def test_exclusive_paths_and_hash_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cap,_,_,folder=self.record(root)
            before=(folder/"replay_data.npz").read_bytes()
            with self.assertRaises(FileExistsError): self.make(root)
            self.assertEqual(before,(folder/"replay_data.npz").read_bytes())
            with (folder/"replay_data.npz").open("ab") as stream: stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError,"SHA256"): load_attempt(folder)

    def test_replay_missing_data_and_changed_model_report_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,folder=self.record(Path(temp))
            metadata,arrays=load_attempt(folder)
            with self.assertRaisesRegex(ValueError,"hash differs"):
                model_replay(metadata,arrays,DynamicModel("Z"),"wrong-model")
            arrays["timestamps"][1]=arrays["timestamps"][0]
            with (folder/"replay_data.npz").open("wb") as stream: np.savez(stream,**arrays)
            metadata["array_sha256"]=sha256(folder/"replay_data.npz")
            (folder/"attempt.json").write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"chronological"): load_attempt(folder)
            metadata["replay_available"]=False
            (folder/"attempt.json").write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"missing capture"): load_attempt(folder)

    def test_replay_cli_check_only_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,folder=self.record(Path(temp))
            before={p.name:sha256(p) for p in folder.iterdir()}
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                result=replay_main(["--attempt-dir",str(folder),"--check-only"])
            self.assertEqual(result,0)
            self.assertEqual(json.loads(out.getvalue())["files_written"],0)
            self.assertEqual(before,{p.name:sha256(p) for p in folder.iterdir()})

    def test_default_has_no_capture_or_detail_files_and_check_only_never_constructs_capture(self):
        from app.inference import unified_abc123_jz_auto_experiment as camera
        c=UnifiedAutoCoordinator(FakePredictor("A",0.),FakePredictor("0",.5),DynamicModel())
        self.assertIsNone(c.diagnostics)
        with (patch("app.inference.unified_abc123_jz_auto_experiment.AttemptCapture") as create,
              patch("cv2.VideoCapture",side_effect=AssertionError("camera forbidden"))):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(camera.main(["--check-only","--capture-jz-diagnostics"]),0)
            create.assert_not_called()

    def test_output_confinement(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): checked_output(Path(temp)/"outside",Path(temp)/"allowed")


if __name__=="__main__":
    unittest.main()
