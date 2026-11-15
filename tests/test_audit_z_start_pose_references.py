"""Read-only preflight and timeline-summary tests; no real decoding/MediaPipe."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import numpy as np

from tools import audit_z_start_pose_references as audit


class ReferenceAuditTests(unittest.TestCase):
    def fake_video(self):
        cap=MagicMock()
        cap.isOpened.return_value=True
        cap.get.return_value=20.
        cap.read.side_effect=[(True,np.zeros((32,32,3),dtype=np.uint8)),
                              (True,np.zeros((32,32,3),dtype=np.uint8)),(False,None)]
        return cap

    def test_human_viewer_steps_and_saves_separate_json_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'380.avi';source.write_bytes(b'unchanged-video-fixture')
            reference={'filename':'380.avi','source_path':str(source),'sha256':audit.sha256(source)}
            paths=[]
            for _ in range(2):
                with (patch.object(audit.cv2,'VideoCapture',return_value=self.fake_video()),
                      patch.object(audit.cv2,'imshow'),patch.object(audit.cv2,'destroyAllWindows'),
                      patch.object(audit.cv2,'waitKey',side_effect=list(map(ord,'123d4s')))):
                    paths.append(audit.annotate_reference(reference,root/'annotations','human'))
            self.assertNotEqual(paths[0],paths[1])
            for path in paths:
                saved=json.loads(path.read_text())
                self.assertEqual(saved['marks'],dict(zip(audit.ANNOTATION_MARKS,(0,0,0,1))))
                self.assertEqual(saved['annotation_status'],'COMPLETE')
            self.assertEqual(source.read_bytes(),b'unchanged-video-fixture')

    def test_human_viewer_cancel_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'380.avi';source.write_bytes(b'video-fixture')
            reference={'filename':'380.avi','source_path':str(source),'sha256':audit.sha256(source)}
            with (patch.object(audit.cv2,'VideoCapture',return_value=self.fake_video()),
                  patch.object(audit.cv2,'imshow'),patch.object(audit.cv2,'destroyAllWindows'),
                  patch.object(audit.cv2,'waitKey',return_value=ord('q'))):
                self.assertIsNone(audit.annotate_reference(reference,root/'annotations','human'))
            self.assertFalse((root/'annotations').exists())

    def test_human_annotation_keeps_late_stable_pose_and_partial_marks_explicit(self):
        marks=dict(zip(audit.ANNOTATION_MARKS,(0,12,8,20)))
        record=audit.annotation_record({"filename":"380.avi"},90,30.,"human",marks,[[9,11]])
        self.assertEqual(record["marks"]["preparation_stable"],12)
        self.assertIn("NOT_DETECTOR",record["annotation_source"])
        self.assertFalse(record["acceptance_decision_changed"])
        marks["preparation_stable"]=None
        self.assertEqual(audit.annotation_record({},90,30.,"human",marks,[])["annotation_status"],"PARTIAL")

    def test_human_annotation_rejects_bad_ranges_missing_fields_and_reviewer(self):
        marks=dict(zip(audit.ANNOTATION_MARKS,(0,3,8,20)))
        for bad,reviewer,ranges in (({**marks,"true_z_motion_end":90},"human",[]),
                                   (marks,"",[]),(marks,"human",[[12,9]]),({},"human",[])):
            with self.assertRaises(ValueError): audit.annotation_record({},90,30.,reviewer,bad,ranges)

    def test_report_replay_uses_saved_rows_only_and_retains_old_new_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            for filename,digest in audit.REFERENCES.items():
                folder=run/Path(filename).stem;folder.mkdir()
                summary={"reference":{"filename":filename,"sha256":digest,"matches_user_reference":True},
                    "fps":25.,"decoded_frame_count":7,"pose_counts":{"VALID":6,"UNCERTAIN":1,"INVALID":0},
                    "true_motion_onset":"UNANNOTATED_REQUIRES_HUMAN_REVIEW"}
                rows=[{"frame_id":i,"seconds":i*.04,"pose_status":"VALID" if i<6 else "UNCERTAIN",
                    "image_hand_count":1,"image_error":"","ready":({"frame_id":3,"timestamp":.12,"track_generation":0} if 3<=i<6 else None),
                    "shadow_attempt":({"movement_since":.20,"start_pose_linked":False,"track_generation":0} if i==6 else None)} for i in range(7)]
                events=[{"event":"STATE_TRANSITION","frame_id":6,"previous_state":"READY","new_state":"CAPTURING","attempt_id":1}]
                (folder/"summary.json").write_text(json.dumps(summary))
                (folder/"frame_timeline.jsonl").write_text('\n'.join(map(json.dumps,rows)))
                (folder/"events.jsonl").write_text('\n'.join(map(json.dumps,events)))
            with (patch.object(audit,"checked_output",side_effect=lambda p:Path(p)),
                  patch.object(audit,"preflight",side_effect=AssertionError("raw hashing forbidden")),
                  patch.object(audit.cv2,"VideoCapture",side_effect=AssertionError("video forbidden")),
                  patch.object(audit,"create_hand_landmarker",side_effect=AssertionError("MediaPipe forbidden")),
                  patch.object(audit,"load_experiment_model",side_effect=AssertionError("model forbidden")),
                  patch.object(Path,"mkdir",side_effect=AssertionError("report write forbidden")),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(audit.main(["--replay-reports",str(run)]),0)
            result=json.loads(output.getvalue())
            self.assertFalse(result["acceptance_effect"])
            self.assertEqual(result["files_written"],0)
            for video in result["videos"]:
                association=video["associations"][0]
                self.assertFalse(association["legacy_start_pose_linked"])
                self.assertTrue(association["temporal_start_pose_linked"])
                self.assertIn("UNAVAILABLE",video["continuity_guard_recheck"])

    def test_declining_annotation_does_not_open_video_or_write(self):
        with (patch.object(audit,"preflight",return_value=([],[])),patch("builtins.input",return_value="n"),
              patch.object(audit.cv2,"VideoCapture",side_effect=AssertionError("video forbidden")),
              patch.object(Path,"mkdir",side_effect=AssertionError("write forbidden")),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(audit.main(["--annotate","--reference","380.avi","--reviewer","human"]),0)

    def test_hash_not_folder_or_filename_selects_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"J").mkdir();(root/"Z").mkdir()
            (root/"J/3.avi").write_bytes(b"different-source")
            (root/"Z/3.avi").write_bytes(b"reference")
            expected=hashlib.sha256(b"reference").hexdigest()
            with patch.object(audit,"REFERENCES",{"3.avi":expected}):
                matches,counterparts=audit.preflight(root)
            self.assertEqual(len(counterparts),2)
            self.assertEqual(matches[0]["source_folder_label"],"Z")
            self.assertEqual(matches[0]["provenance_status"],"USER_REFERENCE_SHA256_MATCHED")
            self.assertIn("NOT_INDEPENDENT",matches[0]["gesture_ground_truth"])

    def test_missing_or_multiple_hash_matches_stop_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"J").mkdir();(root/"Z").mkdir()
            expected=hashlib.sha256(b"reference").hexdigest()
            with patch.object(audit,"REFERENCES",{"3.avi":expected}):
                with self.assertRaisesRegex(ValueError,"found 0"): audit.preflight(root)
                (root/"J/3.avi").write_bytes(b"reference")
                (root/"Z/3.avi").write_bytes(b"reference")
                with self.assertRaisesRegex(ValueError,"found 2"): audit.preflight(root)

    def test_summary_keeps_valid_ranges_separate_and_does_not_invent_onset_or_recall(self):
        rows=[{"frame_id":i,"pose_status":s,"ready":None} for i,s in enumerate(["VALID","VALID","UNCERTAIN","VALID","INVALID"])]
        result=audit.summarize_timeline(rows,[],30.)
        self.assertEqual(result["first_valid_pose_frame"],0)
        self.assertEqual([(v["start_frame"],v["end_frame"]) for v in result["contiguous_valid_pose_ranges"]],[(0,1),(3,3)])
        self.assertEqual(result["ready_events"],[])
        self.assertEqual(result["detector_onsets"],[])
        self.assertIn("NOT_ESTIMABLE",result["recall"])
        self.assertIn("UNANNOTATED",result["true_motion_onset"])

    def test_summary_uses_frozen_ready_at_onset_not_any_earlier_valid_pose(self):
        rows=[{"frame_id":0,"pose_status":"VALID","ready":{"timestamp":0.,"frame_id":0}},
              {"frame_id":1,"pose_status":"INVALID","ready":None,
               "shadow_attempt":{"ready":None,"start_pose_linked":False,"movement_since":0.,"reason":"NO_READY_BEFORE_NEW_MOTION"}}]
        events=[{"event":"STATE_TRANSITION","previous_state":"READY","new_state":"CAPTURING","frame_id":1,"attempt_id":2}]
        result=audit.summarize_timeline(rows,events,30.)
        self.assertFalse(result["detector_onsets"][0]["start_pose_linked"])
        self.assertIsNone(result["detector_onsets"][0]["ready_at_onset"])

    def test_check_only_does_not_open_video_models_tasks_or_create_output(self):
        out=io.StringIO()
        with (patch.object(audit,"preflight",return_value=([{"filename":"3.avi"}],[])),
              patch.object(audit.cv2,"VideoCapture",side_effect=AssertionError("video open forbidden")),
              patch.object(audit,"create_hand_landmarker",side_effect=AssertionError("MediaPipe forbidden")),
              patch.object(audit,"load_experiment_model",side_effect=AssertionError("model loading forbidden")),
              patch.object(Path,"mkdir",side_effect=AssertionError("write forbidden")),
              contextlib.redirect_stdout(out)):
            self.assertEqual(audit.main(["--check-only"]),0)
        self.assertEqual(json.loads(out.getvalue())["files_written"],0)

    def test_declining_audit_never_loads_model_or_writes(self):
        with (patch.object(audit,"preflight",return_value=([],[])),patch("builtins.input",return_value="n"),
              patch.object(audit,"load_experiment_model",side_effect=AssertionError("model loading forbidden")),
              patch.object(Path,"mkdir",side_effect=AssertionError("write forbidden")),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(audit.main(["--audit"]),0)

    def test_synthetic_a_v_k_to_z_without_release_static_label_does_not_block_dynamic(self):
        from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
        from tests.test_unified_abc123_jz_auto import DynamicModel,mirrored
        from tests.test_unified_camera_experiment import FakePredictor,synthetic_hand
        from tests.test_dynamic_jz_motion_gate import articulated_hand
        # These are mock classifier labels / synthetic motion, not Z_TRUE trials.
        for label in ("A","V","K"):
            c=UnifiedAutoCoordinator(FakePredictor(label,0.),FakePredictor("0",.5),DynamicModel("Z"),
                                     enable_experimental_jz=True,z_start_pose_shadow=True)
            for i in range(140):
                h=synthetic_hand()
                c.update(i,mirrored(h),h,(480,640,3),i/30,raw_hand_count=1)
            self.assertEqual(c.composer.text,label)
            for k in range(220):
                h=articulated_hand(min(k,65));i=k+140
                c.update(i,mirrored(h),h,(480,640,3),i/30,raw_hand_count=1)
            self.assertEqual(c.composer.text,label+"Z")


if __name__=="__main__":
    unittest.main()
