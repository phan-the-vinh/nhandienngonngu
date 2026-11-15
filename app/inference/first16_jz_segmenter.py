"""Motion-gated >=16 collection, isolated from the ABC/JZ detector.

The historical class/CLI name is retained. Pre-roll is inference context only:
it never contributes to post-onset observation duration or motion evidence.
Motion gates reject obvious static/entry cases, not certify ASL J/Z.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.inference.automatic_jz_experiment import MotionSegmenter, landmarks_to_array


@dataclass(frozen=True)
class MotionGateSettings:
    min_capture_seconds: float = 1.0
    max_capture_seconds: float = 1.88
    deceleration_window_seconds: float = .28
    max_pre_roll_frames: int = 4
    min_motion_seconds: float = .20
    min_motion_speed: float = .15  # anchored palm scales / second
    min_excursion: float = .25  # anchored palm scales
    min_relative_excursion: float = .12  # fingertip relative to wrist
    min_path_bend: float = .10


def _is_contiguous_frame(prev_index, next_index, skipped_duplicates):
    if next_index == prev_index + 1:
        return True
    if not skipped_duplicates or next_index <= prev_index + 1:
        return False
    return all(mid in skipped_duplicates for mid in range(prev_index + 1, next_index))


def measure_motion(rows, scale, settings, *, skipped_duplicates=None):
    """Measure only observed, post-onset contiguous pairs of pixel 0/8/20.

    Detector-only EMA does not modify landmarks/features. A single rigid entry
    translation fails the articulation/bent-path branch. Jitter must exceed
    excursion AND speed-supported duration; total noisy path length alone is
    never evidence. No displacement or duration is credited across a gap.
    """
    skipped = set(skipped_duplicates or ())
    runs = []
    for index, moment, points in rows:
        if not runs or not _is_contiguous_frame(runs[-1][-1][0], index, skipped):
            runs.append([])
        runs[-1].append((index, moment, points))
    moving_seconds = excursion = relative_excursion = bend = path_length = 0.
    for run in runs:
        if len(run) < 2:
            continue
        points = np.stack([row[2] for row in run])/scale
        filtered = [points[0]]
        for value in points[1:]:
            filtered.append(.35*value+.65*filtered[-1])
        points = np.stack(filtered)
        dt = np.diff([row[1] for row in run])
        steps = np.max(np.linalg.norm(np.diff(points,axis=0),axis=2),axis=1)
        speeds = steps/dt
        moving_seconds += float(np.sum(dt[speeds >= settings.min_motion_speed]))
        path_length += float(np.sum(steps))
        excursion = max(excursion,float(np.max(np.linalg.norm(points-points[0],axis=2))))
        relative = points[:,1:]-points[:,0:1]
        relative_excursion = max(relative_excursion,
                                 float(np.max(np.linalg.norm(relative-relative[0],axis=2))))
        for landmark in range(3):
            path = points[:,landmark]
            chord = path[-1]-path[0]
            length = np.linalg.norm(chord)
            if length <= 1e-9:
                continue
            delta = path-path[0]
            distance = np.abs(delta[:,0]*chord[1]-delta[:,1]*chord[0])/length
            turn = int(np.argmax(distance))
            first,last = path[turn]-path[0],path[-1]-path[turn]
            # Both legs must be meaningful; a noisy stationary endpoint cannot
            # turn an entry translation into gesture evidence.
            if (min(np.linalg.norm(first),np.linalg.norm(last)) >= .12
                    and np.dot(first,last) < .8*np.linalg.norm(first)*np.linalg.norm(last)):
                bend = max(bend,float(distance[turn]))
    supported = (moving_seconds >= settings.min_motion_seconds
                 and excursion >= settings.min_excursion
                 and (relative_excursion >= settings.min_relative_excursion
                      or bend >= settings.min_path_bend))
    return {"supported": bool(supported), "observed_moving_seconds": moving_seconds,
            "excursion_palm_scales": excursion, "relative_excursion_palm_scales": relative_excursion,
            "path_bend_palm_scales": bend, "filtered_path_length_palm_scales": path_length,
            "evidence_frames": len(rows), "evidence_runs": len(runs),
            "pre_roll_used_as_evidence": False}


class First16Segmenter(MotionSegmenter):
    def __init__(self, settings, motion_gate=MotionGateSettings()):
        if any(not np.isfinite(value) or value <= 0 for value in vars(motion_gate).values()):
            raise ValueError("Motion gate thresholds must be finite and positive")
        if motion_gate.min_capture_seconds >= settings.max_seconds:
            raise ValueError("Minimum capture duration must be below max-seconds")
        self.motion_gate = motion_gate
        super().__init__(settings)

    def reset(self):
        super().reset()
        self.waiting_for_rearm = False
        self.rearm_detector = None
        self.evidence_rows = []
        self.motion_evidence = {}
        self.completed_valid_frames = 0
        self.unique_entries = []
        self.observed_capture_entries = []
        self.capture_observations = deque(maxlen=64)
        self.total_observed_frames = 0
        self.duplicate_frames = 0
        self.mid_capture_duplicates = 0
        self.skipped_duplicate_indices = set()
        self.last_unique_raw = None

    def _quiet_window(self, now):
        if self.active:
            self.last_diagnostics.update(quiet_end_met=False,
                                         quiet_end_reason="MOTION_AND_DURATION_GATE_NOT_QUIET_END")
            return False
        return super()._quiet_window(now)

    def trace_metrics(self, now):
        if self.waiting_for_rearm:
            return {**self.rearm_detector.trace_metrics(now), "valid_frames": self.completed_valid_frames,
                    "waiting_for_rearm": True, "quiet_end_required_for_prediction": False}
        elapsed = max(0., now - self.started_at) if self.started_at is not None else 0.
        dup_ratio = self.duplicate_frames / max(1, self.total_observed_frames)
        return {**super().trace_metrics(now), "quiet_end_required_for_prediction": False,
                "minimum_capture_seconds": self.motion_gate.min_capture_seconds,
                "maximum_capture_seconds": self._effective_max_capture_seconds(),
                "capture_seconds_after_onset": elapsed, "motion_evidence": dict(self.motion_evidence),
                "unique_valid_frames": len(self.entries),
                "observed_valid_frames": self.total_observed_frames,
                "duplicate_frames": self.duplicate_frames,
                "duplicate_ratio": dup_ratio,
                "skipped_duplicate_indices": tuple(sorted(self.skipped_duplicate_indices)),
                "prediction_wait_reason": ("MINIMUM_CAPTURE_DURATION" if elapsed < self.motion_gate.min_capture_seconds
                                           else "MOTION_EVIDENCE_REQUIRED" if not self.motion_evidence.get("supported")
                                           else "VALID_FRAMES_REQUIRED" if len(self.entries) < 16
                                           else "WAITING_DECELERATION_OR_WINDOW")}

    def _effective_max_capture_seconds(self):
        return min(self.settings.max_seconds - 1e-3,
                   max(self.motion_gate.min_capture_seconds, self.motion_gate.max_capture_seconds))

    @staticmethod
    def _is_same_landmark(a, b):
        return a is not None and b is not None and float(np.max(np.abs(a - b))) <= 1e-7

    def _dedup_and_trim_onset(self):
        """Deduplicate pre-roll + onset frames and cap motionless lead-in while preserving >=1 pre-roll frame."""
        time_by_index = {entry[0]: moment for moment, entry in self.history}
        raw_entries = list(self.entries)
        self.observed_capture_entries = list(raw_entries)
        deduped = []
        last_raw = None
        for entry in raw_entries:
            arr = landmarks_to_array(entry[1])
            if self._is_same_landmark(arr, last_raw):
                # Prefer keeping the latest identical pre-roll timestamp closest to onset.
                if deduped and (time_by_index.get(entry[0], 0.) < (self.movement_since or 0.)):
                    self.skipped_duplicate_indices.add(deduped[-1][0])
                    deduped[-1] = entry
                else:
                    self.skipped_duplicate_indices.add(entry[0])
                continue
            deduped.append(entry)
            last_raw = arr
        if self.movement_since is not None:
            pre = [e for e in deduped if time_by_index.get(e[0], self.movement_since) < self.movement_since]
            post = [e for e in deduped if time_by_index.get(e[0], self.movement_since) >= self.movement_since]
            cap = int(self.motion_gate.max_pre_roll_frames)
            if len(pre) > cap:
                pre = pre[-cap:]
            deduped = pre + post
        self.entries[:] = deduped
        self.unique_entries = list(deduped)
        self.total_observed_frames = len(raw_entries)
        self.duplicate_frames = max(0, len(raw_entries) - len(deduped))
        self.last_unique_raw = landmarks_to_array(deduped[-1][1]) if deduped else None

    @staticmethod
    def _trajectory_completed_early(entries):
        """Return True when >=16 captured frames already form a valid complete J or Z trajectory."""
        if len(entries) < 16:
            return False
        from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
        from app.inference.integrated_jz_validation import validate_j_trajectory, validate_z_trajectory

        indices = sample_valid_frame_indices([entry[0] for entry in entries])
        mapping = {entry[0]: entry for entry in entries}
        features, _ = build_sequence_features(
            [mapping[i][1] for i in indices],
            [mapping[i][2] for i in indices],
        )
        return bool(validate_z_trajectory(features)["valid"] or validate_j_trajectory(features)["valid"])

    def _decelerated_at_endpoint(self, now, current_is_duplicate):
        """Detect end-of-gesture pause or completed J/Z shape between min and max capture seconds."""
        elapsed = now - self.started_at
        if current_is_duplicate and self.mid_capture_duplicates == 0 and self.total_observed_frames >= 25:
            return True, "FIRST_ENDPOINT_STATIONARY_FRAME"
        if self._trajectory_completed_early(self.entries):
            return True, "COMPLETE_JZ_TRAJECTORY_OBSERVED"
        window = self.motion_gate.deceleration_window_seconds
        rows = list(self.capture_observations)
        prior = [i for i, (moment, _) in enumerate(rows) if moment <= now - window + 1e-9]
        if prior and self.anchor is not None:
            values = np.stack([pts for _, pts in rows[prior[-1]:]])
            span = float(np.max(np.linalg.norm(values[:, None] - values[None, :], axis=-1))) / self.anchor[1]
            duration = max(1e-6, now - rows[prior[-1]][0])
            net_speed = float(np.max(np.linalg.norm(values[-1] - values[0], axis=-1))) / self.anchor[1] / duration
            if span <= self.settings.start_displacement * .35 and net_speed <= self.settings.stop_speed * .67:
                return True, "ENDPOINT_DECELERATION"
        if elapsed >= self._effective_max_capture_seconds():
            return True, "MAX_CAPTURE_SECONDS"
        return False, ""

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

        was_active = self.active
        prev_missing = self.missing_frames
        current_is_duplicate = False
        if was_active and hand is not None:
            raw = landmarks_to_array(hand)
            if self._is_same_landmark(raw, self.last_unique_raw):
                if not np.isfinite(now):
                    return self.reject("INVALID_TIMESTAMP")
                if self.frame_times and (now <= self.frame_times[-1][1] or index <= self.frame_times[-1][0]):
                    self.last_diagnostics.update(previous_frame_id=self.frame_times[-1][0],
                                                 previous_timestamp=self.frame_times[-1][1])
                    return self.reject("TEMPORAL_GAP")
                if self.last_time is not None and now - self.last_time > self.settings.max_frame_gap_seconds:
                    return self.reject("TEMPORAL_GAP")
                if now - self.started_at > self.settings.max_seconds or len(self.observed_capture_entries) >= 512:
                    return self.reject("SEGMENT_TIMEOUT")
                current_is_duplicate = True
                self.frame_times.append((index, now))
                self.missing_frames = 0
                if self.last_time is None or now - self.last_time > self.settings.max_frame_gap_seconds * .5:
                    self.last_time = now
                if self.filtered_points is not None:
                    self.signal_history.append((now, self.filtered_points.copy()))
                xy = raw[:, :2] * np.asarray(size)
                self.capture_observations.append((now, xy[[0, 8, 20]].copy()))
                self.total_observed_frames += 1
                self.duplicate_frames += 1
                self.skipped_duplicate_indices.add(index)
                if (now - self.started_at) < self.motion_gate.min_capture_seconds:
                    self.mid_capture_duplicates += 1
                snapshot = list(self.entries[-1][1]) if self.entries else hand
                self.observed_capture_entries.append((index, snapshot, size))
                state, reason = "CAPTURING", ""
        if not current_is_duplicate:
            state, reason, _ = super().update(index, hand, size, now)
            if not was_active and self.active and state == "CAPTURING":
                self._dedup_and_trim_onset()
            elif was_active and self.active and state == "CAPTURING" and hand is not None:
                self.observed_capture_entries.append(self.entries[-1])
                self.total_observed_frames += 1
                self.last_unique_raw = landmarks_to_array(self.entries[-1][1])

        if state == "CAPTURING" and self.active:
            if not current_is_duplicate and self.previous is not None and self.entries:
                pts = self.previous[1][[0, 8, 20]].copy()
                self.evidence_rows.append((self.entries[-1][0], now, pts))
                self.capture_observations.append((now, pts))
            self.motion_evidence = measure_motion(
                self.evidence_rows,
                self.anchor[1],
                self.motion_gate,
                skipped_duplicates=self.skipped_duplicate_indices,
            )
            elapsed = now - self.started_at
            supported = self.motion_evidence["supported"]
            if not supported and MotionSegmenter._quiet_window(self, now):
                diagnostics = {**self.trace_metrics(now), "rejection": "STATIC_OR_ENTRY_MOTION_ONLY"}
                self.reset()
                self.last_diagnostics = diagnostics
                return "IDLE", "STATIC_OR_ENTRY_MOTION_ONLY", []
            has_16 = len(self.entries) >= 16 or (
                elapsed >= self._effective_max_capture_seconds() and len(self.observed_capture_entries) >= 16
            )
            if not supported or elapsed < self.motion_gate.min_capture_seconds or not has_16:
                return state, "COLLECTING_MOTION_AND_DURATION_EVIDENCE", []
            ready, completion_trigger = self._decelerated_at_endpoint(now, current_is_duplicate)
            if not ready:
                return state, "COLLECTING_MOTION_AND_DURATION_EVIDENCE", []
            used_fallback = len(self.entries) < 16
            selected = list(self.observed_capture_entries) if used_fallback else list(self.entries)
            diagnostics = {**self.trace_metrics(now), "completion": "MOTION_GATED_VALID_WINDOW",
                           "completion_trigger": completion_trigger,
                           "valid_frames": len(selected), "buffered_valid_frames": len(selected),
                           "endpoint_verified": False}
            if used_fallback:
                diagnostics["skipped_duplicate_indices"] = ()
            last = selected[-1]
            self.reset()
            self.waiting_for_rearm = True
            self.completed_valid_frames = len(selected)
            self.rearm_detector = MotionSegmenter(self.settings)
            self.rearm_detector.update(last[0], last[1], last[2], now)
            self.last_diagnostics = diagnostics
            return "COMPLETE", "MOTION_GATED_VALID_WINDOW", selected
        return state, reason, []
