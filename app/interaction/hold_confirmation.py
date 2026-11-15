"""Model-independent, time-based confirmation of a stable static hand pose."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HoldObservation(Protocol):
    label: str | None
    confidence: float
    scale_px: float
    palm_center_px: tuple[float, float]
    shape: tuple[float, ...]


@dataclass(frozen=True)
class HoldSettings:
    hold_seconds: float = 3.0
    stable_frames: int = 3
    min_confidence: float = 0.75
    max_center_drift_ratio: float = 0.25
    max_scale_change: float = 0.18
    max_shape_rms: float = 0.28
    release_seconds: float = 0.60
    min_new_pose_shape_rms: float = 0.06
    blip_grace_frames: int = 0
    blip_grace_seconds: float = 0.0
    max_total_blip_frames: int = 6

    def validate(self) -> None:
        numbers = (self.hold_seconds, self.min_confidence,
                   self.max_center_drift_ratio, self.max_scale_change,
                   self.max_shape_rms, self.release_seconds,
                   self.min_new_pose_shape_rms, self.blip_grace_seconds)
        if (self.stable_frames < 2 or self.blip_grace_frames < 0
                or self.max_total_blip_frames < 0
                or not all(math.isfinite(x) for x in numbers)
                or self.hold_seconds <= 0 or not 0 < self.min_confidence <= 1
                or self.blip_grace_seconds < 0
                or min(self.max_center_drift_ratio, self.max_scale_change,
                       self.max_shape_rms, self.release_seconds,
                       self.min_new_pose_shape_rms) <= 0):
            raise ValueError("Invalid HOLD_CONFIRM settings")


class HoldConfirmation:
    """IDLE -> STABILIZING -> HOLDING -> CONFIRMED -> WAIT_RELEASE.

    A changed pose or a sustained absence is required to rearm. The hold clock
    starts at the first eligible stable-prediction frame, so the pose-validation
    frames are included in the requested hold duration.
    """

    def __init__(self, settings: HoldSettings):
        settings.validate()
        self.settings = settings
        self.state = "IDLE"
        self.text = ""
        self.pending: str | None = None
        self.progress = 0.0
        self.last_confirmed: str | None = None
        self.last_confirmed_at: float | None = None
        self.hold_started_at: float | None = None
        self.rearmed = False
        self._last_timestamp: float | None = None
        self._candidate_started_at: float | None = None
        self._stable_count = 0
        self._anchor_center = np.zeros(2, dtype=np.float64)
        self._anchor_scale = 0.0
        self._anchor_shape = np.empty(0, dtype=np.float64)
        self._confirmed_shape = np.empty(0, dtype=np.float64)
        self._missing_since: float | None = None
        self._awaiting_rearm = False
        self._blip_count = 0
        self._total_blip_count = 0
        self._blip_started_at: float | None = None
        self._last_blip_at: float | None = None
        self._last_valid_timestamp: float | None = None
        self._grace_paused_seconds = 0.0

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def remaining_seconds(self) -> float:
        if self.state == "HOLDING":
            return self.settings.hold_seconds * (1.0 - self.progress)
        if self.state in ("CONFIRMED", "WAIT_RELEASE"):
            return 0.0
        return self.settings.hold_seconds

    @staticmethod
    def _geometry_valid(obs: HoldObservation | None) -> bool:
        if obs is None:
            return False
        center = np.asarray(obs.palm_center_px, dtype=np.float64)
        shape = np.asarray(obs.shape, dtype=np.float64)
        return bool(math.isfinite(obs.scale_px) and obs.scale_px > 0
                    and center.shape == (2,) and np.isfinite(center).all()
                    and shape.ndim == 1 and shape.size > 0 and np.isfinite(shape).all())

    def _eligible(self, obs: HoldObservation | None) -> bool:
        return bool(self._geometry_valid(obs) and obs is not None
                    and obs.label is not None and math.isfinite(obs.confidence)
                    and obs.confidence >= self.settings.min_confidence)

    @staticmethod
    def _shape_distance(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return math.inf
        return float(np.sqrt(np.mean((left - right) ** 2)))

    def _pose_compatible(self, obs: HoldObservation) -> bool:
        center = np.asarray(obs.palm_center_px, dtype=np.float64)
        shape = np.asarray(obs.shape, dtype=np.float64)
        drift = float(np.linalg.norm(center - self._anchor_center) / self._anchor_scale)
        scale_change = abs(obs.scale_px / self._anchor_scale - 1.0)
        shape_change = self._shape_distance(shape, self._anchor_shape)
        return bool(drift <= self.settings.max_center_drift_ratio
                    and scale_change <= self.settings.max_scale_change
                    and shape_change <= self.settings.max_shape_rms)

    def _distinct_from_confirmed(self, obs: HoldObservation) -> bool:
        return bool(obs.label != self.last_confirmed
                    and self._shape_distance(np.asarray(obs.shape, dtype=np.float64),
                                             self._confirmed_shape)
                    >= self.settings.min_new_pose_shape_rms)

    def _start_candidate(self, obs: HoldObservation, now: float) -> None:
        self.state = "STABILIZING"
        self.pending = obs.label
        self.progress = 0.0
        self.hold_started_at = None
        self._candidate_started_at = now
        self._stable_count = 1
        self._anchor_center = np.asarray(obs.palm_center_px, dtype=np.float64)
        self._anchor_scale = float(obs.scale_px)
        self._anchor_shape = np.asarray(obs.shape, dtype=np.float64)
        self._blip_count = 0
        self._total_blip_count = 0
        self._blip_started_at = None
        self._last_blip_at = None
        self._last_valid_timestamp = now
        self._grace_paused_seconds = 0.0

    def _clear_candidate(self) -> None:
        self.pending = None
        self.progress = 0.0
        self.hold_started_at = None
        self._candidate_started_at = None
        self._stable_count = 0
        self._blip_count = 0
        self._total_blip_count = 0
        self._blip_started_at = None
        self._last_blip_at = None
        self._last_valid_timestamp = None
        self._grace_paused_seconds = 0.0

    def _blip_grace_allowed(self, obs: HoldObservation | None, now: float) -> bool:
        """Allow a short bounded grace only for transient single-hand label blips."""
        if (self.settings.blip_grace_frames <= 0 or self.settings.blip_grace_seconds <= 0.0
                or self.state not in ("STABILIZING", "HOLDING")):
            return False
        if obs is None or not self._geometry_valid(obs) or not self._pose_compatible(obs):
            return False
        if (self._blip_count >= self.settings.blip_grace_frames
                or self._total_blip_count >= self.settings.max_total_blip_frames):
            return False
        if self._last_valid_timestamp is None:
            return False
        if (now - self._last_valid_timestamp) > self.settings.blip_grace_seconds:
            return False
        blip_start = now if self._blip_started_at is None else self._blip_started_at
        if (now - blip_start) > self.settings.blip_grace_seconds:
            return False
        raw_stable = getattr(obs, "raw_stable_label", None)
        if obs.label is None:
            return raw_stable in (None, self.pending)
        if obs.label == self.pending:
            return raw_stable in (None, self.pending)
        return False

    def _candidate_allowed(self, obs: HoldObservation | None) -> bool:
        if not self._eligible(obs) or obs is None:
            return False
        return not self._awaiting_rearm or self._distinct_from_confirmed(obs)

    def update(self, obs: HoldObservation | None, *, timestamp: float | None = None) -> str | None:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now) or (self._last_timestamp is not None and now < self._last_timestamp):
            raise ValueError("HOLD_CONFIRM requires a monotonic timestamp")
        self._last_timestamp = now
        self.rearmed = False

        if self.state == "CONFIRMED":
            self.state = "WAIT_RELEASE"

        if self.state == "WAIT_RELEASE":
            if not self._geometry_valid(obs):
                if self._missing_since is None:
                    self._missing_since = now
                if now - self._missing_since >= self.settings.release_seconds:
                    self._awaiting_rearm = False
                    self._missing_since = None
                    self.state = "IDLE"
                    self._clear_candidate()
                    self.rearmed = True
                return None
            self._missing_since = None
            if self._candidate_allowed(obs):
                assert obs is not None
                self.state = "IDLE"
                self._start_candidate(obs, now)
            return None

        if self.state == "IDLE":
            if self._candidate_allowed(obs):
                assert obs is not None
                self._start_candidate(obs, now)
            return None

        if (not self._eligible(obs) or obs is None or obs.label != self.pending
                or not self._pose_compatible(obs)):
            if self._blip_grace_allowed(obs, now):
                prev_t = self._last_blip_at if self._last_blip_at is not None else self._last_valid_timestamp
                assert prev_t is not None
                self._grace_paused_seconds += max(0.0, now - prev_t)
                if self._blip_started_at is None:
                    self._blip_started_at = now
                self._last_blip_at = now
                self._blip_count += 1
                self._total_blip_count += 1
                return None
            self.state = "WAIT_RELEASE" if self._awaiting_rearm else "IDLE"
            self._clear_candidate()
            if self._candidate_allowed(obs):
                assert obs is not None
                self.state = "IDLE"
                self._start_candidate(obs, now)
            return None

        if self._last_blip_at is not None:
            self._grace_paused_seconds += max(0.0, now - self._last_blip_at)
        self._blip_count = 0
        self._blip_started_at = None
        self._last_blip_at = None
        self._last_valid_timestamp = now

        if self.state == "STABILIZING":
            self._stable_count += 1
            if self._stable_count >= self.settings.stable_frames:
                self.state = "HOLDING"
                self.hold_started_at = self._candidate_started_at
                if self._awaiting_rearm:
                    self._awaiting_rearm = False
                    self.rearmed = True

        if self.state == "HOLDING":
            assert self.hold_started_at is not None
            elapsed = max(0.0, (now - self.hold_started_at) - self._grace_paused_seconds)
            self.progress = min(1.0, elapsed / self.settings.hold_seconds)
            if elapsed + 1e-9 >= self.settings.hold_seconds:
                self.progress = 1.0
                confirmed = self.pending
                assert confirmed is not None
                self.text += confirmed
                self.last_confirmed = confirmed
                self.last_confirmed_at = now
                self._confirmed_shape = np.asarray(obs.shape, dtype=np.float64).copy()
                self._awaiting_rearm = True
                self._missing_since = None
                self.state = "CONFIRMED"
                return confirmed
        return None
