"""Aspect-ratio-preserving OpenCV HighGUI window controller for camera display.

Display-only utilities: never loads models, opens cameras at import time, or
modifies camera capture resolution / landmark coordinates / inference inputs.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import sys

import cv2
import numpy as np

DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
LETTERBOX_BGR = (24, 24, 24)


def get_screen_resolution() -> tuple[int, int] | None:
    """Return primary display logical (width, height) when available."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        return None
    return None


def compute_initial_window_size(
    screen_size: tuple[int, int] | None = None,
    target_width: int = DEFAULT_WINDOW_WIDTH,
    target_height: int = DEFAULT_WINDOW_HEIGHT,
) -> tuple[int, int]:
    """Choose initial window dimensions around 1280x720 fitted to the screen."""
    if screen_size is None:
        return int(target_width), int(target_height)
    screen_w, screen_h = screen_size
    if screen_w <= 0 or screen_h <= 0:
        return int(target_width), int(target_height)
    max_w = max(320, int(screen_w * 0.90))
    max_h = max(240, int(screen_h * 0.85))
    if target_width <= max_w and target_height <= max_h:
        return int(target_width), int(target_height)
    scale = min(max_w / target_width, max_h / target_height)
    return (
        max(320, int(round(target_width * scale))),
        max(240, int(round(target_height * scale))),
    )


@dataclass(frozen=True)
class DisplayViewport:
    """Mapping between rendered UI canvas pixels and stretched/letterboxed window pixels."""

    src_width: int
    src_height: int
    dst_width: int
    dst_height: int
    offset_x: int
    offset_y: int
    scaled_width: int
    scaled_height: int

    def map_point(self, win_x: int, win_y: int) -> tuple[int, int] | None:
        """Map a window pixel coordinate back to source UI pixels, or None if on letterbox bar."""
        if (
            self.src_width <= 0
            or self.src_height <= 0
            or self.scaled_width <= 0
            or self.scaled_height <= 0
        ):
            return None
        rel_x = win_x - self.offset_x
        rel_y = win_y - self.offset_y
        if rel_x < 0 or rel_y < 0 or rel_x >= self.scaled_width or rel_y >= self.scaled_height:
            return None
        src_x = min(self.src_width - 1, int(rel_x * self.src_width / self.scaled_width))
        src_y = min(self.src_height - 1, int(rel_y * self.src_height / self.scaled_height))
        return src_x, src_y


def fit_frame_to_window(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    bg_color: tuple[int, int, int] = LETTERBOX_BGR,
) -> tuple[np.ndarray, DisplayViewport]:
    """Scale and letterbox `frame` to `(target_width, target_height)` without aspect distortion."""
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Source and target dimensions must be positive")
    if target_width == src_w and target_height == src_h:
        return frame, DisplayViewport(src_w, src_h, src_w, src_h, 0, 0, src_w, src_h)

    scale = min(target_width / src_w, target_height / src_h)
    scaled_w = max(1, min(target_width, int(round(src_w * scale))))
    scaled_h = max(1, min(target_height, int(round(src_h * scale))))
    offset_x = (target_width - scaled_w) // 2
    offset_y = (target_height - scaled_h) // 2

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=interpolation)

    canvas = np.empty((target_height, target_width, 3), dtype=frame.dtype)
    canvas[:, :] = bg_color
    canvas[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = resized
    return canvas, DisplayViewport(
        src_width=src_w,
        src_height=src_h,
        dst_width=target_width,
        dst_height=target_height,
        offset_x=offset_x,
        offset_y=offset_y,
        scaled_width=scaled_w,
        scaled_height=scaled_h,
    )


class CameraWindowController:
    """Manage a resizable, fullscreen-capable OpenCV window without distorting content."""

    def __init__(
        self,
        window_name: str,
        initial_size: tuple[int, int] | None = None,
    ) -> None:
        self.window_name = window_name
        if initial_size is None:
            initial_size = compute_initial_window_size(get_screen_resolution())
        self.init_width, self.init_height = int(initial_size[0]), int(initial_size[1])
        self.is_fullscreen = False
        self.viewport: DisplayViewport | None = None
        self._has_real_window = False

    def create(self) -> None:
        """Create the OpenCV window in resizable WINDOW_NORMAL mode at initial size."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except TypeError:
            cv2.namedWindow(self.window_name)
        try:
            cv2.resizeWindow(self.window_name, self.init_width, self.init_height)
        except Exception:
            pass

    def query_client_size(self) -> tuple[int, int] | None:
        """Return active OpenCV window client (width, height), or None if unavailable/minimized."""
        try:
            rect = cv2.getWindowImageRect(self.window_name)
        except Exception:
            return None
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            return None
        w, h = rect[2], rect[3]
        if (
            isinstance(w, bool)
            or isinstance(h, bool)
            or not isinstance(w, (int, float))
            or not isinstance(h, (int, float))
        ):
            return None
        iw, ih = int(w), int(h)
        if iw <= 0 or ih <= 0:
            return None
        self._has_real_window = True
        return iw, ih

    def prepare_display(self, display: np.ndarray) -> np.ndarray:
        """Fit rendered UI canvas to the current window size while preserving aspect ratio."""
        src_h, src_w = display.shape[:2]
        client_size = self.query_client_size()
        if client_size is None:
            self.viewport = DisplayViewport(src_w, src_h, src_w, src_h, 0, 0, src_w, src_h)
            return display
        fitted, viewport = fit_frame_to_window(display, client_size[0], client_size[1])
        self.viewport = viewport
        return fitted

    def map_mouse(self, win_x: int, win_y: int) -> tuple[int, int] | None:
        """Translate window mouse coordinates to logical UI coordinates."""
        if self.viewport is None:
            return win_x, win_y
        return self.viewport.map_point(win_x, win_y)

    def toggle_fullscreen(self) -> bool:
        """Toggle between normal resizable window and fullscreen."""
        self.is_fullscreen = not self.is_fullscreen
        prop_value = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop_value)
        except Exception:
            pass
        return self.is_fullscreen

    def restore_initial_size(self) -> None:
        """Exit fullscreen if active and restore the initial default window size."""
        if self.is_fullscreen:
            self.is_fullscreen = False
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            except Exception:
                pass
        try:
            cv2.resizeWindow(self.window_name, self.init_width, self.init_height)
        except Exception:
            pass

    def is_window_closed(self) -> bool:
        """Return True if a real window was explicitly closed via the titlebar X button."""
        if not self._has_real_window:
            return False
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except Exception:
            return False
        if isinstance(visible, (int, float)) and not isinstance(visible, bool):
            return visible < 1.0
        return False
