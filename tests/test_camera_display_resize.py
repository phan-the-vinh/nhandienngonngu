"""Tests for resizable, fullscreen-capable, aspect-ratio-preserving camera window."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.inference import unified_abc123_jz_auto_experiment as camera
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from src.ui.window_display import (
    CameraWindowController,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    LETTERBOX_BGR,
    compute_initial_window_size,
    fit_frame_to_window,
)
from tests.test_unified_abc123_jz_auto import DynamicModel
from tests.test_unified_camera_experiment import FakePredictor


class WindowGeometryTests(unittest.TestCase):
    def test_initial_window_size_uses_1280x720_when_screen_fits(self):
        self.assertEqual(compute_initial_window_size((1536, 864)), (1280, 720))
        self.assertEqual(compute_initial_window_size((1920, 1080)), (1280, 720))
        self.assertEqual(compute_initial_window_size(None), (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT))

    def test_initial_window_size_shrinks_proportionally_for_small_screens(self):
        w, h = compute_initial_window_size((1024, 600))
        self.assertLessEqual(w, int(1024 * 0.90))
        self.assertLessEqual(h, int(600 * 0.85))
        self.assertAlmostEqual(w / h, 1280 / 720, delta=0.03)

    def test_fit_frame_preserves_aspect_ratio_with_letterbox_and_no_distortion(self):
        src = np.full((480, 950, 3), 200, dtype=np.uint8)
        # Mark synthetic landmark pixel inside camera feed (left 640px)
        src[240, 320] = [0, 0, 255]

        # Wide/tall target ratios: 1280x720, narrow 800x750, ultrawide 1400x480
        for dst_w, dst_h in ((1280, 720), (800, 750), (1400, 480)):
            canvas, vp = fit_frame_to_window(src, dst_w, dst_h)
            self.assertEqual(canvas.shape, (dst_h, dst_w, 3))
            self.assertAlmostEqual(vp.scaled_width / vp.scaled_height, 950 / 480, delta=0.01)

            # Uniform scale means X-scale == Y-scale (no shear/distortion)
            scale_x = vp.scaled_width / 950
            scale_y = vp.scaled_height / 480
            self.assertAlmostEqual(scale_x, scale_y, delta=0.005)

            # Corner of letterbox border remains untouched when padding exists
            if vp.offset_y > 0:
                np.testing.assert_array_equal(canvas[0, 0], LETTERBOX_BGR)
                self.assertIsNone(vp.map_point(10, 0))
            if vp.offset_x > 0:
                np.testing.assert_array_equal(canvas[0, 0], LETTERBOX_BGR)
                self.assertIsNone(vp.map_point(0, 10))

            # Landmark center maps back cleanly
            mapped_win_x = vp.offset_x + int(round(320 * scale_x))
            mapped_win_y = vp.offset_y + int(round(240 * scale_y))
            back = vp.map_point(mapped_win_x, mapped_win_y)
            self.assertIsNotNone(back)
            self.assertLessEqual(abs(back[0] - 320), 2)
            self.assertLessEqual(abs(back[1] - 240), 2)

    def test_mouse_click_maps_correctly_back_to_mode_and_suggestion_hitboxes(self):
        src = np.zeros((480, 950, 3), dtype=np.uint8)
        canvas, vp = fit_frame_to_window(src, 1280, 720)
        self.assertEqual(canvas.shape, (720, 1280, 3))

        # Logical center of "123" button is x = 640 + (165+296)//2 = 870, y = 27
        win_x = vp.offset_x + int(round(870 * vp.scaled_width / 950))
        win_y = vp.offset_y + int(round(27 * vp.scaled_height / 480))
        mapped = vp.map_point(win_x, win_y)
        self.assertIsNotNone(mapped)
        self.assertEqual(camera.mode_from_click(mapped[0], mapped[1], 640), "123")


class RunCameraControlsTests(unittest.TestCase):
    def test_window_normal_fullscreen_restore_esc_and_scaled_mouse_click(self):
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        video, image, capture = MagicMock(), MagicMock(), MagicMock()
        video.__enter__.return_value = video
        image.__enter__.return_value = image
        video.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        image.detect.return_value = SimpleNamespace(hand_landmarks=[])
        capture.read.side_effect = [(True, raw.copy()) for _ in range(5)]

        callbacks = []
        shown = []
        client_rect = [0, 0, 1280, 720]
        props = {}

        def fake_set_prop(_name, prop, val):
            props[prop] = val
            if prop == cv2.WND_PROP_FULLSCREEN:
                if val == cv2.WINDOW_FULLSCREEN:
                    client_rect[:] = [0, 0, 1536, 864]
                else:
                    client_rect[:] = [80, 80, 1280, 720]

        def fake_resize(_name, w, h):
            client_rect[:] = [80, 80, w, h]

        def fake_show(_name, frame):
            shown.append(frame.shape[:2])
            if len(shown) == 1:
                # Click "123" button using scaled 1280x720 window coordinates
                _, vp = fit_frame_to_window(np.zeros((480, 950, 3), dtype=np.uint8), 1280, 720)
                wx = vp.offset_x + int(round(870 * vp.scaled_width / 950))
                wy = vp.offset_y + int(round(27 * vp.scaled_height / 480))
                callbacks[0](cv2.EVENT_LBUTTONDOWN, wx, wy, 0, None)

        c = UnifiedAutoCoordinator(FakePredictor("A", 0.0), FakePredictor("0", 0.5), DynamicModel())
        with (
            patch.object(camera, "create_hand_landmarker", side_effect=[video, image]),
            patch.object(camera.cv2, "VideoCapture", return_value=capture),
            patch.object(camera.cv2, "namedWindow") as named_win,
            patch.object(camera.cv2, "resizeWindow", side_effect=fake_resize) as resize_win,
            patch.object(camera.cv2, "getWindowImageRect", side_effect=lambda _n: tuple(client_rect)),
            patch.object(camera.cv2, "getWindowProperty", return_value=1.0),
            patch.object(camera.cv2, "setWindowProperty", side_effect=fake_set_prop),
            patch.object(camera.cv2, "setMouseCallback", side_effect=lambda _n, cb: callbacks.append(cb)),
            patch.object(camera.cv2, "imshow", side_effect=fake_show),
            patch.object(camera.cv2, "waitKey", side_effect=[ord("f"), -1, ord("r"), 27]),
            patch.object(camera.cv2, "destroyAllWindows") as closed,
        ):
            rc = camera.run_camera(c, 0)

        self.assertEqual(rc, 0)
        named_win.assert_called_once_with(camera.WINDOW, cv2.WINDOW_NORMAL)
        # Called at creation and again by 'R' restore
        self.assertGreaterEqual(resize_win.call_count, 2)
        self.assertEqual(c.mode, "123")
        # Frame 1 & 2 before 'f' -> 720x1280; Frame 3 during fullscreen -> 864x1536; Frame 4 after 'r' -> 720x1280
        self.assertEqual(shown[0], (720, 1280))
        self.assertEqual(shown[2], (864, 1536))
        self.assertEqual(shown[3], (720, 1280))
        capture.release.assert_called_once()
        closed.assert_called_once()

    def test_real_opencv_highgui_window_resize_fullscreen_minimize_and_restore(self):
        ctrl = CameraWindowController("Test Camera Resize Real Window", initial_size=(1280, 720))
        sample = np.full((480, 950, 3), 180, dtype=np.uint8)
        try:
            ctrl.create()
            cv2.waitKey(20)
            init_disp = ctrl.prepare_display(sample)
            self.assertEqual(init_disp.shape, (720, 1280, 3))
            cv2.imshow(ctrl.window_name, init_disp)
            cv2.waitKey(20)

            # Drag resize simulation
            cv2.resizeWindow(ctrl.window_name, 900, 700)
            cv2.waitKey(20)
            drag_disp = ctrl.prepare_display(sample)
            self.assertEqual(drag_disp.shape, (700, 900, 3))
            self.assertAlmostEqual(ctrl.viewport.scaled_width / ctrl.viewport.scaled_height, 950 / 480, delta=0.02)

            # Toggle fullscreen ON -> larger or equal to 1280x720 screen bounds
            self.assertTrue(ctrl.toggle_fullscreen())
            cv2.waitKey(40)
            fs_disp = ctrl.prepare_display(sample)
            self.assertGreaterEqual(fs_disp.shape[1], 1280)
            self.assertGreaterEqual(fs_disp.shape[0], 720)

            # Toggle fullscreen OFF -> restores pre-fullscreen (900x700)
            self.assertFalse(ctrl.toggle_fullscreen())
            cv2.waitKey(40)
            unfs_disp = ctrl.prepare_display(sample)
            self.assertEqual(unfs_disp.shape, (700, 900, 3))

            # Restore initial size -> (1280, 720)
            ctrl.restore_initial_size()
            cv2.waitKey(30)
            reset_disp = ctrl.prepare_display(sample)
            self.assertEqual(reset_disp.shape, (720, 1280, 3))
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()
