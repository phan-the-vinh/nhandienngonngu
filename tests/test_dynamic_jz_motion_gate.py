"""Synthetic detector/UI tests, not ASL accuracy claims; no camera/MediaPipe."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from app.inference import dynamic_jz_auto_camera_experiment as auto
from app.inference import dynamic_jz_auto_camera_first16_v1 as previous
from app.inference.first16_jz_segmenter import MotionGateSettings, measure_motion
from tests.test_dynamic_jz_auto_camera_experiment import Model,hand_at


def articulated_hand(i):
    hand = hand_at(min(i,65)*.004)
    hand[20].x += min(max(i-4,0),60)*.0035
    hand[20].y += .035*np.sin(max(i-4,0)/12)
    return hand


def curved_hand(i, label="J"):
    u=min(i/60,1.)
    if label == "J":
        x=.16*max(u-.45,0)/.55
        y=.24*min(u/.45,1.)-.04*max(u-.45,0)/.55
    else:
        x=.22*(3*u if u<1/3 else 2-3*u if u<2/3 else 3*u-2)
        y=.16*min(max(3*u-1,0),1)
    hand=hand_at(x)
    for point in hand:
        point.y+=y
    return hand


class MotionGateTests(unittest.TestCase):
    def setUp(self):
        self.model=Model()
        self.events=[]
        self.session=auto.AutoJzSession(self.model,event_sink=self.events.append)

    def feed(self,i,hand,process=True):
        self.session.update(i,hand,(640,480),i/30,hand_count=0 if hand is None else 1)
        if process:
            return self.session.process_pending(i/30)

    def test_static_pose_five_seconds_no_result(self):
        for i in range(151):
            self.feed(i,hand_at())
        self.assertEqual(self.model.inputs,[])
        self.assertEqual(self.session.state,"READY")
        self.assertIsNone(self.session.last_result)

    def test_hand_entry_rigid_translation_then_hold_has_no_result(self):
        for i in range(170):
            self.feed(i,None if i<5 else hand_at(min(i-5,14)*.008))
        self.assertEqual(self.model.inputs,[])
        self.assertEqual(self.session.state,"READY")
        self.assertFalse(any(r["status"]=="RESULT" for r in self.session.results))

    def test_many_static_preroll_frames_do_not_count_as_motion_or_time(self):
        for i in range(90):
            self.feed(i,hand_at())
        for i in range(90,110):
            self.feed(i,articulated_hand(i-90))
        self.assertGreaterEqual(self.session.valid_frames,16)
        metrics=self.session.segmenter.trace_metrics(109/30)
        self.assertLess(metrics["capture_seconds_after_onset"],1.)
        self.assertFalse(metrics["motion_evidence"]["pre_roll_used_as_evidence"])
        self.assertLess(metrics["motion_evidence"]["evidence_frames"],self.session.valid_frames)
        self.assertEqual(self.model.inputs,[])

    def test_sixteen_without_minimum_time_keeps_collecting(self):
        for i in range(24):
            self.feed(i,articulated_hand(i))
        self.assertEqual(self.session.state,"CAPTURING")
        self.assertGreaterEqual(self.session.valid_frames,16)
        self.assertEqual(self.model.inputs,[])
        self.assertIsNone(self.session.pending)

    def test_sixteen_stationary_measurements_do_not_support_motion(self):
        rows=[(i,i/30,np.asarray([[0.,0.],[10.,20.],[20.,10.]])) for i in range(32)]
        evidence=measure_motion(rows,100.,MotionGateSettings())
        self.assertFalse(evidence["supported"])
        self.assertEqual(evidence["observed_moving_seconds"],0.)

    def test_high_frequency_small_jitter_not_motion_evidence(self):
        rows=[(i,i/30,np.asarray([[0.,0.],[10.,20.],[20.,10.]])+np.sin(i)*.5) for i in range(150)]
        self.assertFalse(measure_motion(rows,100.,MotionGateSettings())["supported"])
        for i in range(151):
            self.feed(i,hand_at(.0007*np.sin(i)))
        self.assertEqual(self.model.inputs,[])

    def test_articulated_motion_predicts_after_one_second_and_uniformly_samples(self):
        for i in range(100):
            self.feed(i,articulated_hand(i),process=False)
            if self.session.pending:
                break
        self.assertIsNotNone(self.session.pending)
        entries,index,endpoint,metrics=self.session.pending
        self.assertEqual(endpoint,"MOTION_GATED_VALID_WINDOW")
        self.assertGreaterEqual(metrics["capture_seconds_after_onset"],1.)
        self.assertTrue(metrics["motion_evidence"]["supported"])
        self.assertGreater(len(entries),16)
        indices=sample_valid_frame_indices([entry[0] for entry in entries])
        mapping={entry[0]:entry for entry in entries}
        expected,_=build_sequence_features([mapping[j][1] for j in indices],[(640,480)]*16)
        result=self.session.process_pending(index/30)
        self.assertEqual(result["status"],"RESULT")
        self.assertEqual(result["selected_frame_indices"],indices)
        np.testing.assert_array_equal(self.model.inputs[0],expected.reshape(1,1104,order="C"))

    def test_bent_j_and_z_like_paths_can_pass_without_finger_articulation(self):
        for label in ("J","Z"):
            with self.subTest(path=label):
                self.setUp()
                for i in range(100):
                    self.feed(i,curved_hand(i,label))
                    if self.session.last_result:
                        break
                self.assertIsNotNone(self.session.last_result)
                self.assertEqual(self.session.last_result["status"],"RESULT")
                evidence=self.session.last_result["segment"]["motion_evidence"]
                self.assertGreaterEqual(evidence["path_bend_palm_scales"],.1)

    def test_short_gap_preserves_valid_count_and_skips_missing_frames(self):
        for i in range(20):
            self.feed(i,articulated_hand(i))
        count=self.session.valid_frames
        entries=self.session.segmenter.entries
        for i in range(20,23):
            self.feed(i,None)
            self.assertEqual(self.session.valid_frames,count)
            self.assertIs(self.session.segmenter.entries,entries)
        self.feed(23,articulated_hand(23))
        self.assertEqual(self.session.valid_frames,count+1)
        self.assertEqual(self.session.segmenter.missing_frames,0)
        self.assertFalse({20,21,22}&{row[0] for row in entries})
        self.assertEqual(self.session.segmenter.motion_evidence["evidence_runs"],2)

    def test_missing_gap_is_never_credited_as_motion(self):
        points=np.asarray([[0.,0.],[10.,20.],[20.,10.]])
        rows=[(0,0.,points),(1,.03,points),(6,.2,points+100),(7,.23,points+100)]
        evidence=measure_motion(rows,100.,MotionGateSettings())
        self.assertEqual(evidence["observed_moving_seconds"],0.)
        self.assertFalse(evidence["supported"])

    def test_long_gap_still_rejects_no_prediction(self):
        for i in range(20):
            self.feed(i,articulated_hand(i))
        for i in range(20,27):
            self.feed(i,None)
        self.assertEqual(self.session.state,"REJECTED")
        self.assertEqual(self.session.reason,"MISSING_HAND_LIMIT")
        self.assertEqual(self.model.inputs,[])

    def test_result_then_held_pose_no_repeat_for_five_seconds(self):
        for i in range(80):
            self.feed(i,articulated_hand(i))
        for i in range(80,231):
            self.feed(i,articulated_hand(79))
        self.assertEqual(len(self.model.inputs),1)

    def test_new_motion_after_rearm_can_predict_again_without_cooldown(self):
        for i in range(240):
            hand=articulated_hand(min(i,70)) if i<110 else articulated_hand(min(i-110,70))
            # A real release avoids an artificial discontinuity when starting
            # the second synthetic gesture at a different position.
            self.feed(i,None if 100<=i<110 else hand)
        self.assertEqual(len(self.model.inputs),2)

    def test_configurable_duration_and_check_only_no_side_effects(self):
        with (patch.object(auto.cv2,"VideoCapture",side_effect=AssertionError("camera/video")),
              patch.object(auto,"create_hand_landmarker",side_effect=AssertionError("MediaPipe")),
              patch.object(auto,"DebugTrace",side_effect=AssertionError("write")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(auto.main(["--check-only","--debug","--min-capture-seconds","1.2"]),0)
        payload=json.loads(output.getvalue())
        self.assertEqual(payload["motion_gate"]["min_capture_seconds"],1.2)
        self.assertEqual(payload["settings"]["max_missing_frames"],6)
        self.assertEqual(payload["settings"]["max_frame_gap_seconds"],.4)
        self.assertFalse(payload["pre_roll_counts_as_motion_or_duration"])
        self.assertEqual(payload["files_written"],0)

    def test_invalid_motion_thresholds_fail_before_loading_model(self):
        with (patch.object(auto,"load_experiment_model",side_effect=AssertionError("load")),
              contextlib.redirect_stderr(io.StringIO())):
            for argv in (["--min-capture-seconds","0"],["--min-motion-speed","nan"],
                         ["--min-capture-seconds","6"]):
                with self.assertRaises(SystemExit):
                    auto.main(argv)

    def test_minimum_duration_uses_elapsed_time_at_different_fps(self):
        for fps in (15,60):
            with self.subTest(fps=fps):
                model=Model()
                session=auto.AutoJzSession(model)
                for i in range(int(3*fps)):
                    session.update(i,articulated_hand(i*30/fps),(640,480),i/fps,hand_count=1)
                    session.process_pending(i/fps)
                    if session.last_result:
                        break
                self.assertIsNotNone(session.last_result)
                self.assertEqual(session.last_result["status"],"RESULT")
                self.assertGreaterEqual(session.last_result["segment"]["capture_seconds_after_onset"],1.)

    def test_preserved_version_still_predicts_at_first_16(self):
        model=Model()
        session=previous.AutoJzSession(model)
        for i in range(16):
            session.update(i,hand_at(i*.008),(640,480),i/30,hand_count=1)
            session.process_pending(i/30)
        self.assertEqual(len(model.inputs),1)
        self.assertEqual(session.last_result["endpoint"],"FIRST_16_VALID_FRAMES")


if __name__=="__main__":
    unittest.main()
