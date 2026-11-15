"""Experimental first-16 collection, isolated from the ABC/JZ detector.

Reuses motion onset and continuity guards. Only the completion policy changes:
first 16 real valid frames (including existing pre-roll), without quiet-end.
This can truncate a gesture; it is not the whole-gesture training sampler.
"""
from app.inference.automatic_jz_experiment import MotionSegmenter


class First16Segmenter(MotionSegmenter):
    def reset(self):
        super().reset()
        self.waiting_for_rearm = False
        self.rearm_detector = None

    def _quiet_window(self, now):
        if self.active:
            self.last_diagnostics.update(quiet_end_met=False,
                                         quiet_end_reason="NOT_REQUIRED_FIRST16")
            return False
        return super()._quiet_window(now)

    def trace_metrics(self, now):
        if self.waiting_for_rearm:
            return {**self.rearm_detector.trace_metrics(now), "valid_frames": 16,
                    "waiting_for_rearm": True, "quiet_end_required_for_prediction": False}
        return {**super().trace_metrics(now), "quiet_end_required_for_prediction": False}

    def update(self, index, hand, size, now):
        if self.waiting_for_rearm:
            state, reason, _ = self.rearm_detector.update(index, hand, size, now)
            metrics = dict(self.rearm_detector.last_diagnostics)
            quiet = (metrics.get("quiet_end_met")
                     or (state != "REJECTED" and self.rearm_detector._quiet_window(now)))
            # A real quiet observation or sufficiently long release opens the
            # next bout. Result display time never controls this re-arm rule.
            if (quiet
                    or (state == "REJECTED" and reason == "MISSING_HAND_LIMIT")):
                self.reset()
                return "IDLE", "REARMED_AFTER_STILL_HAND_OR_RELEASE", []
            self.missing_frames = self.rearm_detector.missing_frames
            self.motion_score = self.rearm_detector.motion_score
            self.last_diagnostics = self.trace_metrics(now)
            return "WAIT_REARM", "WAITING_FOR_STILL_HAND_OR_RELEASE", []

        state, reason, _ = super().update(index, hand, size, now)
        if state == "CAPTURING" and self.active and len(self.entries) >= 16:
            selected = self.entries[:16]
            diagnostics = {**self.trace_metrics(now), "completion": "FIRST_16_VALID_FRAMES",
                           "valid_frames": 16, "buffered_valid_frames": len(self.entries),
                           "selected_original_frame_indices": [row[0] for row in selected],
                           "endpoint_verified": False}
            last = self.entries[-1]
            self.reset()
            self.waiting_for_rearm = True
            self.rearm_detector = MotionSegmenter(self.settings)
            self.rearm_detector.update(last[0], last[1], last[2], now)
            self.last_diagnostics = diagnostics
            return "COMPLETE", "FIRST_16_VALID_FRAMES", selected
        return state, reason, []
