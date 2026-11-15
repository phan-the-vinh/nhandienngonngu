"""Experimental, model-independent downward-motion confirmation for a static pose."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DownObservation(Protocol):
    label: str | None
    confidence: float
    scale_px: float
    palm_center_px: tuple[float, float]
    shape: tuple[float, ...]


@dataclass(frozen=True)
class DownSettings:
    hold_frames: int = 7
    down_frames: int = 2
    return_frames: int = 4
    min_confidence: float = 0.75
    down_ratio: float = 0.25
    return_ratio: float = 0.10
    smoothing_alpha: float = 0.70
    max_ready_scale_cv: float = 0.06
    max_ready_center_span: float = 0.35
    max_horizontal_ratio: float = 0.45
    max_scale_change: float = 0.15
    max_shape_rms: float = 0.28

    def validate(self) -> None:
        if min(self.hold_frames, self.down_frames, self.return_frames) < 1:
            raise ValueError("Frame counts must be positive")
        values = (self.min_confidence, self.down_ratio, self.return_ratio,
                  self.smoothing_alpha, self.max_ready_scale_cv,
                  self.max_ready_center_span, self.max_horizontal_ratio,
                  self.max_scale_change, self.max_shape_rms)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Thresholds must be finite")
        if not (0 < self.min_confidence <= 1
                and 0 <= self.return_ratio < self.down_ratio
                and 0 < self.smoothing_alpha <= 1
                and 0 < self.max_ready_scale_cv < 1
                and min(self.max_ready_center_span, self.max_horizontal_ratio,
                        self.max_scale_change, self.max_shape_rms) > 0):
            raise ValueError("Invalid DOWN_CONFIRM threshold")


class DownConfirmation:
    """Stable pose -> down -> one commit -> require a physical return."""

    # Timing marker only; confirmation still requires down_ratio/down_frames.
    MOTION_ONSET_RATIO = 0.08

    def __init__(self, settings: DownSettings):
        settings.validate()
        self.settings = settings
        self.state = "IDLE"
        self.pending: str | None = None
        self.text = ""
        self.vertical_ratio = 0.0
        self._hold: deque[DownObservation] = deque(maxlen=settings.hold_frames)
        self._hold_times_ms: deque[float] = deque(maxlen=settings.hold_frames)
        self._baseline_center = np.zeros(2, dtype=np.float64)
        self._baseline_scale = 0.0
        self._baseline_shape = np.empty(0, dtype=np.float64)
        self._down_count = 0
        self._return_count = 0
        self._movement_started_ms: float | None = None
        self.last_ready_latency_ms: float | None = None
        self.last_confirm_latency_ms: float | None = None

    @staticmethod
    def _geometry_valid(obs: DownObservation | None) -> bool:
        if obs is None:
            return False
        center = np.asarray(obs.palm_center_px, dtype=np.float64)
        shape = np.asarray(obs.shape, dtype=np.float64)
        return bool(math.isfinite(obs.scale_px) and obs.scale_px > 0
                    and center.shape == (2,) and np.isfinite(center).all()
                    and shape.ndim == 1 and shape.size > 0 and np.isfinite(shape).all())

    def _ready_candidate(self, obs: DownObservation | None) -> bool:
        return bool(self._geometry_valid(obs) and obs is not None
                    and obs.label is not None
                    and math.isfinite(obs.confidence)
                    and obs.confidence >= self.settings.min_confidence)

    def recalibrate(self) -> None:
        """Clear a pending attempt, but never unlock a confirmed downward pull."""
        if self.state in ("CONFIRMED", "WAIT_RETURN"):
            return
        self.state = "IDLE"
        self.pending = None
        self.vertical_ratio = 0.0
        self._hold.clear()
        self._hold_times_ms.clear()
        self._down_count = 0
        self._movement_started_ms = None

    def _motion_compatible(self, obs: DownObservation) -> bool:
        center = np.asarray(obs.palm_center_px, dtype=np.float64)
        horizontal = abs(center[0] - self._baseline_center[0]) / self._baseline_scale
        scale_change = abs(obs.scale_px / self._baseline_scale - 1.0)
        shape_rms = float(np.sqrt(np.mean(
            (np.asarray(obs.shape, dtype=np.float64) - self._baseline_shape) ** 2)))
        return bool(horizontal <= self.settings.max_horizontal_ratio
                    and scale_change <= self.settings.max_scale_change
                    and shape_rms <= self.settings.max_shape_rms)

    def update(self, obs: DownObservation | None, *, dynamic_active: bool = False,
               timestamp_ms: float | None = None) -> str | None:
        """Commit only after two or more threshold frames; never on return alone."""
        now_ms = time.perf_counter() * 1000 if timestamp_ms is None else timestamp_ms
        if self.state == "CONFIRMED":
            self.state = "WAIT_RETURN"
        if dynamic_active:
            self.recalibrate()
            return None
        if self.state == "WAIT_RETURN":
            if self._geometry_valid(obs) and obs is not None:
                center = np.asarray(obs.palm_center_px, dtype=np.float64)
                raw_y = (center[1] - self._baseline_center[1]) / self._baseline_scale
                self.vertical_ratio = (self.settings.smoothing_alpha * raw_y
                                       + (1 - self.settings.smoothing_alpha) * self.vertical_ratio)
                horizontal = abs(center[0] - self._baseline_center[0]) / self._baseline_scale
                scale_change = abs(obs.scale_px / self._baseline_scale - 1.0)
                if (abs(self.vertical_ratio) <= self.settings.return_ratio
                        and horizontal <= self.settings.max_horizontal_ratio
                        and scale_change <= self.settings.max_scale_change):
                    self._return_count += 1
                    if self._return_count >= self.settings.return_frames:
                        self.state = "IDLE"
                        self.pending = None
                        self._hold.clear()
                        self._hold_times_ms.clear()
                        self._down_count = self._return_count = 0
                        self.vertical_ratio = 0.0
                        self._movement_started_ms = None
                else:
                    self._return_count = 0
            else:
                self._return_count = 0  # no hand is not evidence of return
            return None

        if self.state == "IDLE":
            if not self._ready_candidate(obs):
                self.recalibrate()
                return None
            assert obs is not None
            if self._hold and (self._hold[-1].label != obs.label
                               or len(self._hold[-1].shape) != len(obs.shape)):
                self._hold.clear()
                self._hold_times_ms.clear()
            self._hold.append(obs)
            self._hold_times_ms.append(now_ms)
            if len(self._hold) == self.settings.hold_frames:
                scales = np.asarray([item.scale_px for item in self._hold], dtype=np.float64)
                centers = np.asarray([item.palm_center_px for item in self._hold], dtype=np.float64)
                shapes = np.asarray([item.shape for item in self._hold], dtype=np.float64)
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
                    self.last_ready_latency_ms = now_ms - self._hold_times_ms[0]
                else:
                    self._hold.popleft()
                    self._hold_times_ms.popleft()
            return None

        if (not self._geometry_valid(obs) or obs is None
                or len(obs.shape) != len(self._baseline_shape)):
            self.recalibrate()  # lost tracking aborts an unconfirmed pull
            return None
        if not self._motion_compatible(obs):
            self.recalibrate()
            return None
        raw_y = ((obs.palm_center_px[1] - self._baseline_center[1])
                 / self._baseline_scale)  # image Y increases downward
        self.vertical_ratio = (self.settings.smoothing_alpha * raw_y
                               + (1 - self.settings.smoothing_alpha) * self.vertical_ratio)
        if self.vertical_ratio >= self.MOTION_ONSET_RATIO:
            if self._movement_started_ms is None:
                self._movement_started_ms = now_ms
        elif self.vertical_ratio <= self.MOTION_ONSET_RATIO / 2:
            self._movement_started_ms = None
        if self.vertical_ratio >= self.settings.down_ratio:
            self._down_count += 1
            if self._down_count >= self.settings.down_frames:
                self.state = "CONFIRMED"
                self.text += self.pending or ""  # frozen at READY
                self._return_count = 0
                onset_ms = self._movement_started_ms if self._movement_started_ms is not None else now_ms
                self.last_confirm_latency_ms = now_ms - onset_ms
                return self.pending
            self.state = "DOWN_DETECTED"
        else:
            self._down_count = 0
            self.state = "READY"
        return None
