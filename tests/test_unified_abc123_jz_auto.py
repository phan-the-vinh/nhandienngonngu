"""Offline integration tests. Synthetic classifier outputs test arbitration,
not real ASL accuracy. Actual model integrity is checked by --check-only.
"""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

import cv2
import numpy as np

from app.inference import unified_abc123_jz_auto_experiment as camera
from app.inference.dynamic_jz_auto_camera_experiment import AutoJzSession
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_camera_experiment import FakePredictor,synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand, curved_hand

DEFAULT_HAND = object()


class DynamicModel:
    classes_ = np.asarray(["J","Z"])
    def __init__(self,label="J"):
        self.label = label
        self.inputs = []
    def predict_proba(self,x):
        self.inputs.append(x.copy())
        return np.asarray([[.97,.03] if self.label=="J" else [.03,.97]])


def mirrored(hand):
    return [SimpleNamespace(x=1-p.x,y=p.y,z=p.z) for p in hand]


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.letters = FakePredictor("A",0.)
        self.digits = FakePredictor("0",.5)
        self.model = DynamicModel()
        self.events = []
        self.c = UnifiedAutoCoordinator(self.letters,self.digits,self.model,event_sink=self.events.append)
        self.index = 0

    def tick(self,now,raw=DEFAULT_HAND,video=DEFAULT_HAND):
        raw = synthetic_hand() if raw is DEFAULT_HAND else raw
        video = synthetic_hand() if video is DEFAULT_HAND else video
        result = self.c.update(self.index,video,raw,(480,640,3),now,raw_hand_count=0 if raw is None else 1)
        self.index += 1
        return result[1]

    def confirm_static(self,label,shape,start):
        predictor = self.letters if self.c.mode=="ABC" else self.digits
        predictor.label,predictor.shape = label,shape
        for moment in (start,start+.1,start+.2):
            self.assertIsNone(self.tick(moment))
        return self.tick(start+3.)

    def move(self,frames=80,video_visible=True,label="J"):
        step = 1.8 if label == "Z" else 1.0
        for k in range(frames):
            hand = curved_hand(min(int(k*step),60),label)
            self.tick(self.index/30,raw=hand,video=mirrored(hand) if video_visible else None)

    def test_static_a_hold_unchanged_and_no_dynamic_query(self):
        self.assertEqual(self.confirm_static("A",0.,0.),"A")
        self.assertEqual(self.c.composer.text,"A")
        self.assertEqual(self.c.hold.settings.hold_seconds,3.)
        self.assertEqual(self.model.inputs,[])

    def test_apple_two_ps_and_shared_text_after_mode_switch(self):
        self.assertEqual(self.confirm_static("A",0.,0.),"A")
        self.assertEqual(self.confirm_static("P",.3,3.1),"P")
        self.tick(6.2,raw=None,video=None)
        self.tick(6.9,raw=None,video=None)
        self.assertEqual(self.confirm_static("P",.3,7.),"P")
        self.assertEqual(self.confirm_static("L",.6,10.1),"L")
        self.assertEqual(self.confirm_static("E",.9,13.2),"E")
        self.assertEqual(self.c.composer.text,"APPLE")
        self.c.switch_mode("123")
        self.assertEqual(self.confirm_static("2",.8,16.3),"2")
        self.c.switch_mode("ABC")
        self.assertEqual(self.c.composer.text,"APPLE2")

    def test_digits_never_update_dynamic_session(self):
        self.c.switch_mode("123")
        with patch.object(self.c.session,"update",side_effect=AssertionError("JZ in 123")):
            self.assertEqual(self.confirm_static("8",.4,0.),"8")
        self.assertEqual(self.model.inputs,[])

    def test_static_pose_and_small_jitter_no_jz_candidate(self):
        for i in range(151):
            hand=synthetic_hand()
            for point in hand:
                point.x+=.0001*np.sin(i)
            self.tick(i/30,raw=hand,video=mirrored(hand))
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.model.inputs,[])

    def test_j_and_z_candidate_use_latest_session_not_old_coordinator(self):
        for label in ("J","Z"):
            with self.subTest(label=label):
                self.setUp()
                self.model.label=label
                split = 60 if label == "J" else 40
                self.move(split,label=label)
                self.assertIsNone(self.c.candidate)  # Raw prediction is not yet a candidate.
                self.assertEqual(self.c.raw_dynamic_prediction["prediction"],label)
                for i in range(split,100):
                    self.tick(i/30,raw=curved_hand(60,label))
                self.assertEqual(self.c.candidate["prediction"],label)
                self.assertEqual(self.c.display_label,label)
                self.assertIsInstance(self.c.session,AutoJzSession)
                self.assertEqual(self.c.session.segmenter.motion_gate.min_capture_seconds,1.)

    def test_integrated_features_equal_standalone_latest_on_same_raw_hands(self):
        standalone_model=DynamicModel()
        standalone=AutoJzSession(standalone_model)
        for i in range(70):
            hand=articulated_hand(i)
            self.tick(i/30,raw=hand,video=None)
            standalone.update(i,hand,(640,480),i/30,hand_count=1)
            standalone.process_pending(i/30)
        self.assertEqual(len(self.model.inputs),1)
        np.testing.assert_array_equal(self.model.inputs[0],standalone_model.inputs[0])
        self.assertEqual(self.model.inputs[0].shape,(1,1104))

    def test_video_missing_does_not_gate_raw_dynamic_candidate(self):
        self.move(65,video_visible=False)
        self.assertIsNotNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.display_label)  # No VIDEO hand, raw result cannot occupy main label.
        for i in range(65,100):
            self.tick(i/30,raw=curved_hand(60,"J"),video=None)
        self.assertEqual(self.c.display_label,"J")
        self.assertIsNone(self.c.static_prediction)

    def test_short_image_gap_retains_same_segment_even_when_video_still_detects(self):
        for i in range(20):
            hand=articulated_hand(i)
            self.tick(i/30,raw=hand,video=mirrored(hand))
        count=self.c.session.valid_frames
        entries=self.c.session.segmenter.entries
        for i in range(20,23):
            self.tick(i/30,raw=None)
            self.assertIs(self.c.session.segmenter.entries,entries)
            self.assertEqual(self.c.session.valid_frames,count)
        self.tick(23/30,raw=articulated_hand(23))
        self.assertEqual(self.c.session.valid_frames,count+1)
        self.assertEqual(self.c.session.segmenter.missing_frames,0)

    def test_sixteen_frames_not_enough_to_predict(self):
        self.move(24)
        self.assertGreaterEqual(self.c.session.valid_frames,16)
        self.assertEqual(self.model.inputs,[])
        self.assertIsNone(self.c.candidate)

    def test_suspect_does_not_reset_static_hold(self):
        original=self.c.static.hold
        saw_suspect=False
        for i in range(12):
            hand=articulated_hand(i)
            self.tick(i/30,raw=hand)
            if not self.c.session.segmenter.active:
                self.assertIs(self.c.static.hold,original)
                saw_suspect |= bool(self.c.session.segmenter.start_count)
        self.assertTrue(saw_suspect)

    def test_rejection_restores_static_not_locked_forever(self):
        # Unverified CAPTURING never blocks static ABC.
        for i in range(20):
            hand=articulated_hand(i)
            self.tick(i/30,raw=hand,video=mirrored(hand))
        self.assertFalse(self.c.blocked)
        # Once a trajectory-verified dynamic proposal blocks static (with commits enabled), losing it unlocks cleanly.
        self.setUp()
        self.c.enable_experimental_jz=True
        self.move(65)
        self.assertTrue(self.c.blocked)
        for i in range(65,72):
            self.tick(i/30,raw=None)
        self.assertFalse(self.c.blocked)
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.confirm_static("A",0.,3.),"A")

    def test_candidate_not_overwritten_by_static_model(self):
        self.move(65)
        self.letters.label="I"
        for i in range(65,100):
            self.tick(i/30,raw=curved_hand(60,"J"))
            if self.c.candidate:
                self.assertEqual(self.c.display_label,"J")
            else:
                self.assertEqual(self.c.display_label,"I")
        self.assertIsNotNone(self.c.candidate)
        self.assertEqual(self.c.composer.text,"")

    def test_hold_starts_only_after_observed_quiet_endpoint(self):
        self.c.enable_experimental_jz=True
        for i in range(66):
            self.tick(i/30,raw=curved_hand(min(i,60),"J"),video=None)
            if self.c.candidate:
                self.assertIsNone(self.c.jz_hold.hold_started_at)
                self.assertEqual(self.c.jz_hold.progress,0.)
        for i in range(66,100):
            self.tick(i/30,raw=curved_hand(60,"J"),video=None)
        self.assertTrue(self.c.candidate_settled)
        self.assertGreater(self.c.jz_hold.hold_started_at,60/30)
        self.assertEqual(self.c.composer.text,"")

    def test_experimental_hold_three_seconds_one_commit_no_static_i_or_repeat(self):
        self.c.enable_experimental_jz=True
        self.letters.label="I"
        for i in range(251):
            self.tick(i/30,raw=curved_hand(min(i,60),"J"),video=None)
        self.assertEqual(self.c.composer.text,"J")
        commits=[e for e in self.events if e["event"]=="JZ_CONFIRMED_EXPERIMENTAL"]
        endpoints=[e for e in self.events if e["event"]=="JZ_ENDPOINT_OBSERVED"]
        self.assertEqual(len(commits),1)
        self.assertGreaterEqual(commits[0]["timestamp_monotonic"]-endpoints[0]["timestamp_monotonic"],3.-1e-9)

    def test_default_diagnostic_never_commits_jz(self):
        for i in range(210):
            self.tick(i/30,raw=curved_hand(min(i,60),"J"))
        self.assertNotIn("J",self.c.composer.text)
        self.assertNotIn("Z",self.c.composer.text)
        self.assertFalse(self.c.enable_experimental_jz)
        self.assertIsNone(self.c.candidate)  # Bounded diagnostic display, not an endless static lock.

    def test_candidate_tracking_loss_cancels_no_commit(self):
        self.c.enable_experimental_jz=True
        self.move(60)
        self.assertIsNotNone(self.c.pending_dynamic)
        # Frame 1 IMAGE-only drop while VIDEO still detects hand uses 1-frame grace (Z-STEP-1).
        self.tick(60/30,raw=None)
        self.assertIsNotNone(self.c.pending_dynamic)
        self.assertTrue(self.c.blocked)
        # Frame 2 consecutive IMAGE drop cancels candidate cleanly with no commit.
        self.tick(61/30,raw=None)
        self.assertIsNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.composer.text,"")
        self.assertFalse(self.c.blocked)

    def test_candidate_pose_change_cancels_without_using_new_static_letter(self):
        self.c.enable_experimental_jz=True
        self.move(80)
        for i in range(80,100):
            self.tick(i/30,raw=curved_hand(60,"J"))
        self.assertTrue(self.c.candidate_settled)
        changed=curved_hand(60,"J")
        for point in changed:
            point.y+=.15
        self.tick(100/30,raw=changed)
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.composer.text,"")

    def test_mode_switch_clears_pending_and_candidate_preserves_text(self):
        self.c.composer.text="APP"
        self.move(60)
        self.assertIsNotNone(self.c.pending_dynamic)
        self.c.switch_mode("123")
        self.assertIsNone(self.c.candidate)
        self.assertIsNone(self.c.pending_dynamic)
        self.assertIsNone(self.c.raw_dynamic_prediction)
        self.assertEqual(self.c.session.segmenter.entries,[])
        self.assertEqual(self.c.composer.text,"APP")
        self.c.switch_mode("ABC")
        self.assertIsNone(self.c.candidate)
        self.assertEqual(self.c.hold.text,"APP")
        self.assertFalse(self.c.blocked)

    def test_two_jz_commits_then_short_loss_does_not_reuse_old_release_timer(self):
        self.c.enable_experimental_jz=True
        for i in range(200):
            self.tick(i/30,raw=curved_hand(min(i,60),"J"),video=None)
        self.assertEqual(self.c.composer.text,"J")
        for i in range(200,223):
            self.tick(i/30,raw=None,video=None)
        for i in range(223,480):
            self.tick(i/30,raw=curved_hand(min(i-223,60),"J"),video=None)
        self.assertEqual(self.c.composer.text,"JJ")
        self.tick(480/30,raw=None,video=None)
        self.assertTrue(self.c.post_jz_latch)
        self.assertIsNone(self.c.display_label)  # Committed J is in text/history, not a live prediction.

    def test_static_predictor_failure_does_not_cancel_raw_candidate(self):
        with patch.object(self.letters,"predict_landmarks",side_effect=ValueError("VIDEO features failed")):
            self.move(65)
            for i in range(65,100):
                self.tick(i/30,raw=curved_hand(60,"J"))
        self.assertIsNotNone(self.c.candidate)
        self.assertEqual(self.c.display_label,"J")

    def test_low_confidence_endpoint_does_not_lock_static_indefinitely(self):
        self.c.enable_experimental_jz=True
        with patch.object(self.model,"predict_proba",return_value=np.asarray([[.55,.45]])):
            for i in range(110):
                self.tick(i/30,raw=curved_hand(min(i,60),"J"),video=None)
        self.assertIsNone(self.c.candidate)
        self.assertFalse(self.c.blocked)
        self.assertEqual(self.c.composer.text,"")
        self.assertEqual(self.c.reason,"JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD")

    def test_image_tracking_loss_alone_is_not_physical_release_after_commit(self):
        self.c.enable_experimental_jz=True
        for i in range(200):
            self.tick(i/30,raw=curved_hand(min(i,60),"J"),video=None)
        self.assertEqual(self.c.composer.text,"J")
        for i in range(200,230):
            self.tick(i/30,raw=None,video=synthetic_hand())
        self.assertTrue(self.c.post_jz_latch)
        self.assertTrue(self.c.blocked)
        self.assertEqual(self.c.composer.text,"J")


class CameraTests(unittest.TestCase):
    def test_raw_landmark_overlay_mirrors_only_display_when_video_is_missing(self):
        raw=np.zeros((100,120,3),dtype=np.uint8)
        hand=[SimpleNamespace(x=.2,y=.4,z=0.) for _ in range(21)]
        c=UnifiedAutoCoordinator(FakePredictor("A",0.),FakePredictor("0",.5),DynamicModel())
        displayed=camera.render(raw[:,::-1].copy(),"ABC",None,raw,hand,c,0.)
        np.testing.assert_array_equal(displayed[40,95],[0,0,255])
        self.assertFalse(raw.any())
        self.assertTrue(all(point.x==.2 for point in hand))

    def test_check_only_validates_all_models_no_camera_task_or_files(self):
        with (patch.object(camera.cv2,"VideoCapture",side_effect=AssertionError("camera/video")),
              patch.object(camera,"create_hand_landmarker",side_effect=AssertionError("MediaPipe")),
              patch.object(camera,"DebugTrace",side_effect=AssertionError("write")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(camera.main(["--check-only","--debug"]),0)
        report=json.loads(output.getvalue())
        self.assertEqual(report["modes"],["ABC","123"])
        self.assertEqual(report["hold_seconds"],3.)
        self.assertEqual(report["gap_frames"],6)
        self.assertEqual(report["gap_ms"],400.)
        self.assertEqual(report["motion_gate"]["min_capture_seconds"],1.)
        self.assertEqual(report["landmarker_count_when_running"],2)
        self.assertFalse(report["jz_commits_enabled"])
        self.assertEqual(report["files_written"],0)

    def test_only_two_buttons_no_pause_jz_or_space_button(self):
        self.assertEqual(camera.mode_from_click(700,30,640),"ABC")
        self.assertEqual(camera.mode_from_click(850,30,640),"123")
        self.assertIsNone(camera.mode_from_click(850,65,640))

    def test_one_camera_two_independent_passes_mode_switch_mirror_and_safe_close(self):
        raw=np.arange(240*320*3,dtype=np.uint8).reshape(240,320,3)
        video,image,capture=MagicMock(),MagicMock(),MagicMock()
        video.__enter__.return_value=video
        image.__enter__.return_value=image
        capture.read.side_effect=[(True,raw.copy()) for _ in range(3)]
        video_inputs,image_inputs,displays,callback=[],[],[],[]
        def video_detect(mp_image,stamp):
            video_inputs.append((mp_image.numpy_view().copy(),stamp))
            if len(video_inputs)==1:
                raise RuntimeError("VIDEO detection temporarily failed")
            return SimpleNamespace(hand_landmarks=[])
        def image_detect(mp_image):
            image_inputs.append(mp_image.numpy_view().copy())
            return SimpleNamespace(hand_landmarks=[])
        video.detect_for_video.side_effect=video_detect
        image.detect.side_effect=image_detect
        def show(_name,display):
            displays.append(display.copy())
            if len(displays)<3:
                callback[0](cv2.EVENT_LBUTTONDOWN,320+(210 if len(displays)==1 else 60),30,0,None)
        c=UnifiedAutoCoordinator(FakePredictor("A",0.),FakePredictor("0",.5),DynamicModel())
        with (patch.object(camera,"create_hand_landmarker",side_effect=[video,image]) as create,
              patch.object(camera.cv2,"VideoCapture",return_value=capture) as opened,
              patch.object(camera.cv2,"namedWindow"),
              patch.object(camera.cv2,"setMouseCallback",side_effect=lambda _name,cb:callback.append(cb)),
              patch.object(camera.cv2,"imshow",side_effect=show),
              patch.object(camera.cv2,"waitKey",side_effect=[-1,-1,ord("q")]),
              patch.object(camera.cv2,"destroyAllWindows") as closed):
            self.assertEqual(camera.run_camera(c,0),0)
        opened.assert_called_once_with(0)
        self.assertEqual(create.call_count,2)
        self.assertEqual(create.call_args_list[0].kwargs["running_mode"],"VIDEO")
        self.assertEqual(create.call_args_list[1].kwargs["running_mode"],"IMAGE")
        self.assertEqual(create.call_args_list[1].kwargs["num_hands"],1)
        for i,mode in enumerate(("ABC","123","ABC")):
            expected=raw[:,::-1] if mode=="ABC" else raw
            np.testing.assert_array_equal(video_inputs[i][0],cv2.cvtColor(expected,cv2.COLOR_BGR2RGB))
            np.testing.assert_array_equal(displays[i][:,:320],raw[:,::-1])
        self.assertEqual(len(image_inputs),2)  # Not gated by VIDEO no-hand; NOT run in 123.
        for value in image_inputs:
            np.testing.assert_array_equal(value,cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
        stamps=[row[1] for row in video_inputs]
        self.assertTrue(all(b>a for a,b in zip(stamps,stamps[1:])))
        capture.release.assert_called_once()
        closed.assert_called_once()


if __name__=="__main__":
    unittest.main()
