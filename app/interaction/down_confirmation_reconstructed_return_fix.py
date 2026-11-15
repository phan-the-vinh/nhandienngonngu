"""Isolated return-unlock experiment for the reconstructed DOWN camera.

The shared DOWN state machine still handles READY and downward confirmation.
Only WAIT_RETURN differs: a verified upward release is measured from the lowest
post-confirmation hand position instead of requiring the previous letter's
position, horizontal placement, and scale to be reproduced.
"""

from __future__ import annotations

import math

import numpy as np

from app.interaction.down_confirmation import DownConfirmation, DownObservation, DownSettings


class DownConfirmationReturnFix(DownConfirmation):
    """Unlock after a sustained lift of both palm center and wrist."""

    def __init__(self, settings: DownSettings):
        super().__init__(settings)
        self._lowest_palm_y: float | None = None
        self._lowest_wrist_y: float | None = None
        self.return_up_ratio = 0.0

    @staticmethod
    def _wrist_y(obs: DownObservation | None) -> float | None:
        wrist = getattr(obs, "wrist_px", None)
        if wrist is None:
            return None
        coordinates = np.asarray(wrist, dtype=np.float64)
        if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
            return None
        return float(coordinates[1])

    def update(self, obs: DownObservation | None, *, dynamic_active: bool = False,
               timestamp_ms: float | None = None) -> str | None:
        if self.state not in ("CONFIRMED", "WAIT_RETURN"):
            committed = super().update(obs, dynamic_active=dynamic_active,
                                       timestamp_ms=timestamp_ms)
            if committed is not None and self._geometry_valid(obs) and obs is not None:
                self._lowest_palm_y = float(obs.palm_center_px[1])
                self._lowest_wrist_y = self._wrist_y(obs)
                self.return_up_ratio = 0.0
            return committed

        self.state = "WAIT_RETURN"
        if dynamic_active:
            return None
        wrist_y = self._wrist_y(obs)
        if (not self._geometry_valid(obs) or obs is None or wrist_y is None
                or self._baseline_scale <= 0 or not math.isfinite(self._baseline_scale)):
            self._return_count = 0  # Missing tracking never unlocks a pull.
            return None

        palm_y = float(obs.palm_center_px[1])
        assert self._lowest_palm_y is not None and self._lowest_wrist_y is not None
        self._lowest_palm_y = max(self._lowest_palm_y, palm_y)
        self._lowest_wrist_y = max(self._lowest_wrist_y, wrist_y)
        palm_lift = (self._lowest_palm_y - palm_y) / self._baseline_scale
        wrist_lift = (self._lowest_wrist_y - wrist_y) / self._baseline_scale
        self.return_up_ratio = min(palm_lift, wrist_lift)
        raw_y = (palm_y - self._baseline_center[1]) / self._baseline_scale
        self.vertical_ratio = (self.settings.smoothing_alpha * raw_y
                               + (1 - self.settings.smoothing_alpha) * self.vertical_ratio)

        # Same release distance implicit in the original DOWN/RETURN thresholds.
        required_lift = self.settings.down_ratio - self.settings.return_ratio
        if self.return_up_ratio >= required_lift:
            self._return_count += 1
            if self._return_count >= self.settings.return_frames:
                self.state = "IDLE"
                self.pending = None
                self._hold.clear()
                self._hold_times_ms.clear()
                self._down_count = self._return_count = 0
                self.vertical_ratio = self.return_up_ratio = 0.0
                self._movement_started_ms = None
                self._lowest_palm_y = self._lowest_wrist_y = None
        else:
            self._return_count = 0
        return None
