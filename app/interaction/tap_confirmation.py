"""Model-independent, experimental down-and-return TAP confirmation.

The caller supplies a stable label and 2D palm observations. No classifier,
MediaPipe task, camera, or absolute landmark-z distance is used here.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TapObservation(Protocol):
    label: str | None
    confidence: float
    scale_px: float
    wrist_px: tuple[float, float]
    palm_center_px: tuple[float, float] | None
    shape: tuple[float, ...]


@dataclass(frozen=True)
class TapSettings:
    hold_frames: int = 10
    down_frames: int = 2
    return_frames: int = 2
    reset_frames: int = 4
    min_confidence: float = 0.75
    onset_ratio: float = 0.12
    down_ratio: float = 0.35
    return_ratio: float = 0.10
    hysteresis_ratio: float = 0.08
    max_duration_ms: int = 1400
    max_ready_scale_cv: float = 0.06
    max_ready_center_span: float = 0.35
    max_horizontal_ratio: float = 0.45
    max_scale_change: float = 0.15
    max_shape_rms: float = 0.28

    def validate(self) -> None:
        if min(self.hold_frames, self.down_frames, self.return_frames,
               self.reset_frames, self.max_duration_ms) < 1:
            raise ValueError("TAP frame counts and duration must be positive")
        values = (self.min_confidence, self.onset_ratio, self.down_ratio,
                  self.return_ratio, self.hysteresis_ratio,
                  self.max_ready_scale_cv, self.max_ready_center_span,
                  self.max_horizontal_ratio, self.max_scale_change,
                  self.max_shape_rms)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("TAP thresholds must be finite")
        if not (0 < self.min_confidence <= 1
                and 0 <= self.return_ratio < self.onset_ratio < self.down_ratio
                and 0 < self.hysteresis_ratio < self.down_ratio - self.return_ratio
                and 0 < self.max_ready_scale_cv < 1
                and min(self.max_ready_center_span, self.max_horizontal_ratio,
                        self.max_scale_change, self.max_shape_rms) > 0):
            raise ValueError("Invalid TAP threshold ordering or tolerance")


def _center(obs: TapObservation) -> np.ndarray:
    point = obs.palm_center_px if obs.palm_center_px is not None else obs.wrist_px
    return np.asarray(point, dtype=np.float64)


class TapConfirmation:
    """One character per complete TAP; re-arm only after a neutral reset hold."""

    def __init__(self, settings: TapSettings):
        settings.validate()
        self.settings = settings
        self.state = "IDLE"
        self.pending: str | None = None
        self.text = ""
        self.vertical_ratio = 0.0
        self._hold: deque[TapObservation] = deque(maxlen=settings.hold_frames)
        self._baseline_center = np.zeros(2, dtype=np.float64)
        self._baseline_scale = 0.0
        self._baseline_shape = np.empty(0, dtype=np.float64)
        self._started_at_ms: int | None = None
        self._down_count = 0
        self._return_count = 0
        self._reset_count = 0

    def recalibrate(self) -> None:
        """Clear an uncommitted attempt; never bypass the post-confirm latch."""
        if self.state in ("CONFIRMED", "WAIT_RESET"):
            return
        self.state = "IDLE"
        self.pending = None
        self.vertical_ratio = 0.0
        self._hold.clear()
        self._started_at_ms = None
        self._down_count = self._return_count = 0

    def _valid(self, obs: TapObservation | None) -> bool:
        if obs is None or obs.label is None:
            return False
        shape = np.asarray(obs.shape, dtype=np.float64)
        center = _center(obs)
        return bool(obs.confidence >= self.settings.min_confidence
                    and math.isfinite(obs.scale_px) and obs.scale_px > 0
                    and center.shape == (2,) and np.isfinite(center).all()
                    and shape.ndim == 1 and shape.size > 0 and np.isfinite(shape).all())

    def update(self, obs: TapObservation | None, *, timestamp_ms: int | None = None,
               dynamic_active: bool = False) -> str | None:
        """Return a character exactly once, only after DOWN then RETURN."""
        now = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        if self.state == "CONFIRMED":
            self.state = "WAIT_RESET"
        if dynamic_active:
            self.recalibrate()
            return None
        if self.state == "WAIT_RESET":
            if self._valid(obs) and obs is not None:
                self.vertical_ratio = float((_center(obs)[1] - self._baseline_center[1])
                                            / self._baseline_scale)
                if abs(self.vertical_ratio) <= self.settings.return_ratio:
                    self._reset_count += 1
                    if self._reset_count >= self.settings.reset_frames:
                        self.state = "IDLE"
                        self.pending = None
                        self._hold.clear()
                        self._reset_count = 0
                        self._down_count = self._return_count = 0
                        self._started_at_ms = None
                        self.vertical_ratio = 0.0
                else:
                    self._reset_count = 0
            else:
                self._reset_count = 0  # missing hand cannot release the latch
            return None
        if not self._valid(obs):
            self.recalibrate()
            return None
        assert obs is not None
        if self.state == "IDLE":
            if self._hold and (self._hold[-1].label != obs.label
                               or len(self._hold[-1].shape) != len(obs.shape)):
                self._hold.clear()
            self._hold.append(obs)
            if len(self._hold) == self.settings.hold_frames:
                scales = np.asarray([item.scale_px for item in self._hold])
                centers = np.asarray([_center(item) for item in self._hold])
                shapes = np.asarray([item.shape for item in self._hold])
                center_median = np.median(centers, axis=0)
                center_span = float(np.max(np.linalg.norm(centers - center_median, axis=1))
                                    / np.median(scales))
                shape_median = np.median(shapes, axis=0)
                shape_span = float(np.max(np.sqrt(np.mean((shapes - shape_median) ** 2, axis=1))))
                if (float(np.std(scales) / np.mean(scales)) <= self.settings.max_ready_scale_cv
                        and center_span <= self.settings.max_ready_center_span
                        and shape_span <= self.settings.max_shape_rms):
                    self.pending = obs.label
                    self._baseline_center = center_median
                    self._baseline_scale = float(np.median(scales))
                    self._baseline_shape = shape_median
                    self.vertical_ratio = 0.0
                    self.state = "READY"
                else:
                    self._hold.popleft()
            return None

        if obs.label != self.pending or len(obs.shape) != len(self._baseline_shape):
            self.recalibrate()
            return None
        center = _center(obs)
        self.vertical_ratio = float((center[1] - self._baseline_center[1])
                                    / self._baseline_scale)  # image Y increases downward
        horizontal_ratio = float(abs(center[0] - self._baseline_center[0])
                                 / self._baseline_scale)
        scale_change = float(abs(obs.scale_px / self._baseline_scale - 1.0))
        shape_rms = float(np.sqrt(np.mean((np.asarray(obs.shape) - self._baseline_shape) ** 2)))
        if (horizontal_ratio > self.settings.max_horizontal_ratio
                or scale_change > self.settings.max_scale_change
                or shape_rms > self.settings.max_shape_rms):
            self.recalibrate()
            return None
        if self._started_at_ms is not None and (
                now < self._started_at_ms
                or now - self._started_at_ms > self.settings.max_duration_ms):
            self.recalibrate()
            return None

        if self.state == "READY":
            if self.vertical_ratio >= self.settings.onset_ratio:
                if self._started_at_ms is None:
                    self._started_at_ms = now
            elif self.vertical_ratio <= self.settings.return_ratio:
                self._started_at_ms = None
            if self.vertical_ratio >= self.settings.down_ratio:
                self._down_count += 1
                if self._down_count >= self.settings.down_frames:
                    self.state = "TAP_DOWN"
            else:
                self._down_count = 0
        elif self.state == "TAP_DOWN":
            if self.vertical_ratio <= self.settings.down_ratio - self.settings.hysteresis_ratio:
                self.state = "TAP_RETURN"
        elif self.state == "TAP_RETURN":
            if abs(self.vertical_ratio) <= self.settings.return_ratio:
                self._return_count += 1
                if self._return_count >= self.settings.return_frames:
                    self.state = "CONFIRMED"
                    self.text += self.pending or ""
                    self._reset_count = 0
                    return self.pending
            else:
                self._return_count = 0

        return None
