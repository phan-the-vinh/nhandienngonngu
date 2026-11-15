"""Unvalidated automatic J/Z segmentation and arbitration, NOT production.

Geometry gates are rejection heuristics, not an ASL verifier or a NONE model.
Dynamic commits require an explicit opt-in in the separate camera experiment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from ai.feature_extraction.hand_landmarks import landmarks_to_array
from ai.preprocessing.landmark_utils import normalize_landmarks
from app.inference.dynamic_jz_camera_test import predict_attempt
from app.inference.letter_inference import Prediction
from app.inference.push_confirmation_camera_test import observe_hand
from app.inference.unified_static_coordinator import RecognitionCoordinator
from app.interaction.hold_confirmation import HoldSettings


@dataclass(frozen=True)
class SegmentSettings:
    prime_seconds: float = .20  # historical compatibility; no mandatory quiet start
    start_displacement: float = .15
    stop_speed: float = .18  # palm scales / second, net speed over quiet window
    end_quiet_seconds: float = .35
    min_seconds: float = .25
    max_seconds: float = 6.0
    max_frame_gap_seconds: float = .20
    max_step: float = .65
    max_scale_change: float = .70
    max_shape_step: float = .50
    min_probability: float = .90  # not calibrated; never compared to static scores
    start_frames: int = 3
    start_seconds: float = .06
    pre_roll_seconds: float = .50
    max_missing_frames: int = 2
    min_valid_fraction: float = .90


class MotionSegmenter:
    """Buffered movement -> bounded quiet endpoint, never a rolling 16-frame guess.

    Uses raw pixel XY of 0/8/20, one anchored palm scale, bounded memory.
    Missing observations are skipped within explicit limits, never synthesized.
    Ambiguity must be rejected by the caller, not passed as a missing observation.
    """
    def __init__(self, settings: SegmentSettings = SegmentSettings()):
        self.settings = settings
        self.reset()

    def reset(self):
        self.active = False
        self.entries = []
        self.history = deque(maxlen=512)
        self.anchor = None
        self.previous = None
        self.quiet_since = None
        self.last_time = None
        self.started_at = None
        self.primed = False
        self.speed = 0.0
        self.filtered_points = None
        self.start_count = 0
        self.start_since = None
        self.movement_since = None
        self.motion_score = 0.0
        self.missing_frames = 0
        self.frame_times = deque(maxlen=16)
        self.signal_history = deque(maxlen=512)
        self.last_diagnostics = {}

    def reject(self, reason):
        now = self.frame_times[-1][1] if self.frame_times else 0.
        diagnostics = self.trace_metrics(now)
        self.reset()
        self.last_diagnostics = {**diagnostics, "rejection": reason}
        return "REJECTED", reason, []

    def trace_metrics(self, now):
        return {**self.last_diagnostics,
                "valid_frames": len(self.entries),
                "segment_duration_seconds": max(0., now - self.started_at) if self.started_at is not None else 0.,
                "first_frame": self.entries[0][0] if self.entries else None,
                "last_valid_frame": self.entries[-1][0] if self.entries else None,
                "consecutive_missing_frames": self.missing_frames,
                "motion_score": self.motion_score,
                "quiet_end_met": self.last_diagnostics.get("quiet_end_met", False),
                "quiet_end_reason": self.last_diagnostics.get("quiet_end_reason", "NOT_EVALUATED"),
                "max_frame_gap_seconds": self.settings.max_frame_gap_seconds,
                "max_missing_frames": self.settings.max_missing_frames,
                "max_segment_seconds": self.settings.max_seconds,
                "min_valid_fraction": self.settings.min_valid_fraction,
                "quiet_end_seconds_required": self.settings.end_quiet_seconds,
                "quiet_span_limit": self.settings.start_displacement,
                "quiet_net_speed_limit": self.settings.stop_speed}

    def update(self, index, hand, size, now):
        if not np.isfinite(now):
            return self.reject("INVALID_TIMESTAMP")
        if self.frame_times and (now <= self.frame_times[-1][1] or index <= self.frame_times[-1][0]):
            self.last_diagnostics.update(previous_frame_id=self.frame_times[-1][0],
                                         previous_timestamp=self.frame_times[-1][1])
            return self.reject("TEMPORAL_GAP")
        self.frame_times.append((index, now))
        if hand is None:
            self.missing_frames += 1
            self.quiet_since = None  # missing is never quiet-end evidence
            self.signal_history.clear()
            self.last_diagnostics = {**self.last_diagnostics, "missing_frames": self.missing_frames,
                                     "motion_score_valid": False, "quiet_end_met": False,
                                     "quiet_end_reason": "MISSING_OBSERVATION",
                                     "seconds_since_last_valid": now-self.last_time if self.last_time is not None else None}
            if (self.missing_frames > self.settings.max_missing_frames
                    or (self.last_time is not None and now - self.last_time > self.settings.max_frame_gap_seconds)):
                return self.reject("MISSING_HAND_LIMIT")
            return ("GAP", "", []) if self.active else ("SUSPECT", "", [])
        raw = landmarks_to_array(hand)
        xy = raw[:, :2] * np.asarray(size)
        scale = float(np.median(np.linalg.norm(xy[[5, 9, 17]] - xy[0], axis=1)))
        if scale <= 1e-6:
            return self.reject("DEGENERATE_SCALE")
        shape = np.asarray(normalize_landmarks(hand))
        if self.previous is not None:
            old_index, old_xy, old_scale, old_shape = self.previous
            dt = now - self.last_time
            self.last_diagnostics.update(seconds_since_last_valid=dt, frame_delta=index-old_index)
            if (index <= old_index or dt <= 0 or dt > self.settings.max_frame_gap_seconds):
                return self.reject("TEMPORAL_GAP")
            step = float(np.max(np.linalg.norm(xy[[0, 8, 20]] - old_xy[[0, 8, 20]], axis=1)))
            reference_scale = self.anchor[1] if self.anchor is not None else old_scale
            # Fingertips can rotate quickly while the palm remains continuous.
            # Foreshortening changes palm scale without changing hand identity.
            # Keep those signals diagnostic; do not reject on either alone.
            palm_ids = [0, 5, 9, 17]
            palm_step = np.linalg.norm(np.mean(xy[palm_ids] - old_xy[palm_ids], axis=0))
            intervals = [b[1] - a[1] for a, b in zip(self.frame_times, list(self.frame_times)[1:])]
            cadence = float(np.median(intervals)) if intervals else dt
            gap_factor = max(1., dt / cadence)
            palm_step_ratio = float(palm_step / reference_scale / gap_factor)
            scale_change = abs(scale / old_scale - 1)
            shape_step = float(np.sqrt(np.mean((shape - old_shape) ** 2)))
            self.last_diagnostics = dict(tip_step_ratio=step / reference_scale,
                                         palm_step_ratio=palm_step_ratio,
                                         previous_scale_change=scale_change,
                                         anchor_scale_change=abs(scale/reference_scale-1),
                                         shape_rms=shape_step, gap_factor=gap_factor,
                                         skipped_frames=self.missing_frames,
                                         seconds_since_last_valid=dt, frame_delta=index-old_index,
                                         motion_score_valid=True)
            if palm_step_ratio > self.settings.max_step:
                return self.reject("PALM_POSITION_DISCONTINUITY")
            if (scale_change / gap_factor > self.settings.max_scale_change
                    and shape_step / gap_factor > self.settings.max_shape_step):
                return self.reject("SCALE_AND_SHAPE_DISCONTINUITY")
        # Filter the detector signal, NEVER the landmarks stored for inference.
        # Palm center and fingertip-relative motion distinguish rigid translation
        # from articulation; displacement from an anchor is less jitter-sensitive
        # than the former max instantaneous tip-speed gate.
        center = np.mean(xy[[0, 5, 9, 17]], axis=0)
        points = np.stack((center, xy[8] - center, xy[20] - center))
        previous_filtered = self.filtered_points
        self.filtered_points = points.copy() if previous_filtered is None else .35 * points + .65 * previous_filtered
        if previous_filtered is not None:
            filtered_step = np.max(np.linalg.norm(self.filtered_points - previous_filtered, axis=1))
            self.speed = .5 * (filtered_step / reference_scale / dt) + .5 * self.speed
        # Snapshot coordinates: later renderer/task calls cannot mutate the bout.
        snapshot = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in raw]
        entry = (index, snapshot, size)
        self.previous = (index, xy, scale, shape)
        self.last_time = now
        self.missing_frames = 0
        self.signal_history.append((now, self.filtered_points.copy()))
        if not self.active:
            if self.anchor is None:
                self.anchor = (self.filtered_points.copy(), scale)
                self.quiet_since = now
            self.primed = True  # availability of an anchor, NOT proof of quiet
            displacement = float(np.max(np.linalg.norm(self.filtered_points - self.anchor[0], axis=1))) / self.anchor[1]
            self.motion_score = displacement
            if displacement >= self.settings.start_displacement * .5 and self.movement_since is None:
                self.movement_since = now
            if displacement >= self.settings.start_displacement:
                if self.start_since is None:
                    self.start_since = now
                self.start_count += 1
                self.history.append((now, entry))
                if (self.start_count < self.settings.start_frames
                        or now - self.start_since < self.settings.start_seconds):
                    return "SUSPECT", "", []
                self.active = True
                # Include pre-roll at the first threshold crossing, not just the
                # final debounce frame. Entries retain original, unfiltered data.
                self.entries = [value for moment, value in self.history
                                if moment >= self.movement_since - self.settings.pre_roll_seconds]
                self.started_at = now
                self.quiet_since = None
                return "CAPTURING", "", []
            if displacement < self.settings.start_displacement * .5:
                self.start_count = 0
                self.start_since = None
                self.movement_since = None
            # During real idle, re-anchor only within a bounded quiet window.
            # Do not continually chase a moving hand before it crosses start.
            if displacement < self.settings.start_displacement * .5 and self._quiet_window(now):
                self.anchor = (self.filtered_points.copy(), scale)
            self.history.append((now, entry))
            return "IDLE", "", []
        self.entries.append(entry)
        self.motion_score = self.speed
        # Evaluate the existing quiet rule before the timeout branch so a
        # rejection retains same-frame endpoint evidence. Timeout still wins.
        quiet = self._quiet_window(now)
        if now - self.started_at > self.settings.max_seconds or len(self.entries) > 512:
            return self.reject("SEGMENT_TIMEOUT")
        # Window displacement is bounded by the existing start threshold.
        # Unlike per-frame velocity, it does not amplify small IMAGE-mode jitter
        # by 1/dt. Require the full observed window, with no missing frames.
        if quiet:
            entries = self.entries
            duration = now - self.started_at
            span = entries[-1][0] - entries[0][0] + 1
            if len(entries) < 16 or duration < self.settings.min_seconds:
                return self.reject("TOO_SHORT")
            if len(entries) / span < self.settings.min_valid_fraction:
                return self.reject("INSUFFICIENT_SEGMENT_COVERAGE")
            diagnostics = {**self.trace_metrics(now), "completion": "OBSERVED_QUIET_END",
                           "valid_fraction": len(entries) / span}
            self.reset()
            self.last_diagnostics = diagnostics
            return "COMPLETE", "", entries
        return "CAPTURING", "", []

    def _quiet_window(self, now):
        # Include the observation immediately before the window boundary, so
        # floating-point/frame cadence cannot shorten the required duration.
        rows = list(self.signal_history)
        boundary = now - self.settings.end_quiet_seconds
        prior = [i for i, (moment, _) in enumerate(rows) if moment <= boundary + 1e-9]
        if not prior or self.anchor is None:
            self.last_diagnostics.update(quiet_end_met=False, quiet_end_reason="INSUFFICIENT_OBSERVED_WINDOW")
            return False
        values = np.stack([points for _, points in rows[prior[-1]:]])
        span = float(np.max(np.linalg.norm(values[:, None] - values[None, :], axis=-1))) / self.anchor[1]
        duration = now - rows[prior[-1]][0]
        net_speed = float(np.max(np.linalg.norm(values[-1]-values[0], axis=-1))) / self.anchor[1] / duration
        quiet = span <= self.settings.start_displacement and net_speed <= self.settings.stop_speed
        self.last_diagnostics = {**self.last_diagnostics, "quiet_window_span": span,
                                 "quiet_window_net_speed": net_speed, "quiet_end_met": quiet,
                                 "quiet_window_duration_seconds": duration,
                                 "quiet_end_reason": "OBSERVED_QUIET" if quiet else "WINDOW_STILL_MOVING"}
        return quiet

    def review_at_eof(self):
        """File diagnostic only. Never COMPLETE or evidence of a live endpoint."""
        entries = self.entries
        valid = (len(entries) >= 16 and len(entries) / (entries[-1][0]-entries[0][0]+1)
                 >= self.settings.min_valid_fraction) if entries else False
        self.reset()
        return ("REVIEW_REQUIRED", "EOF_WITHOUT_OBSERVED_QUIET_END", entries) if valid else (
            "REJECTED", "EOF_WITHOUT_VALID_SEGMENT", [])


def geometry_candidates(features):
    """Conservative path/handshape checks; no claim of ASL correctness.

    Reject pure rigid arm sweeps; require finger motion relative to wrist.
    J: extended little finger, bent path. Z: extended index, three alternating
    horizontal legs with downward progression. Either hand direction allowed.
    """
    if features.shape != (16, 69) or not np.isfinite(features).all():
        return set()
    shape = features[:, :63].reshape(16, 21, 3)
    wrist = features[:, 63:65]
    allowed = set()
    for label, tip, pip, col, folded in (("J", 20, 18, 67, (8, 12, 16)),
                                        ("Z", 8, 6, 65, (12, 16, 20))):
        extended = np.linalg.norm(shape[:, tip], axis=1) > 1.1 * np.linalg.norm(shape[:, pip], axis=1)
        folded_ok = np.ones(16, dtype=bool)
        for other_tip in folded:
            folded_ok &= (np.linalg.norm(shape[:, other_tip], axis=1)
                          < 1.3 * np.linalg.norm(shape[:, other_tip - 2], axis=1))
        if np.mean(extended & folded_ok) < .75:
            continue
        path = features[:, col:col + 2]
        relative = path - wrist
        if np.max(np.linalg.norm(relative - relative[0], axis=1)) < .08:
            continue
        if np.max(np.linalg.norm(path - path[0], axis=1)) < .30:
            continue
        if label == "J":
            # The corner is not necessarily at the ninth temporal sample.
            # Find maximum distance from the endpoint chord; keep the existing
            # directional, excursion, handshape and angle limits unchanged.
            chord = path[-1] - path[0]
            if np.linalg.norm(chord) <= 1e-9:
                continue
            relative_path = path - path[0]
            distance = np.abs(relative_path[:, 0]*chord[1] - relative_path[:, 1]*chord[0])
            turn = int(np.argmax(distance[1:-1])) + 1
            first, hook = path[turn] - path[0], path[-1] - path[turn]
            if (first[1] > .15 and abs(hook[0]) > .12
                    and np.dot(first, hook) < .6 * np.linalg.norm(first) * np.linalg.norm(hook)):
                allowed.add(label)
        else:
            legs = np.diff(path[[0, 5, 10, 15]], axis=0)
            dx, dy = legs[:, 0], legs[:, 1]
            if (np.all(np.abs(dx) > .12) and dx[0] * dx[1] < 0 and dx[1] * dx[2] < 0
                    and dy[1] > .10 and abs(dy[0]) < abs(dx[0]) * .6
                    and abs(dy[2]) < abs(dx[2]) * .6):
                allowed.add(label)
    return allowed


def evaluate_segment(entries, model, settings=SegmentSettings(), *, diagnostics=None):
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(model_called=False, geometry_candidates=[], candidate=None)
    indices = sample_valid_frame_indices([value[0] for value in entries])
    mapping = {index: (hand, size) for index, hand, size in entries}
    features, _ = build_sequence_features([mapping[i][0] for i in indices],
                                          [mapping[i][1] for i in indices])
    allowed = geometry_candidates(features)
    diagnostics.update(selected_frame_indices=indices, feature_shape=list(features.shape),
                       geometry_candidates=sorted(allowed))
    if not allowed:
        return None, "MOTION_OR_HANDSHAPE_NOT_SUPPORTED"
    diagnostics["model_called"] = True
    result = predict_attempt(entries, entries[-1][0] + 1, model)
    diagnostics["model_result"] = result
    if result["status"] != "PREDICTED":
        return None, result["failure_reason"]
    label = result["prediction"]
    confidence = result[f"probability_{label}"]
    if label not in allowed or confidence < settings.min_probability:
        return None, "MODEL_GEOMETRY_DISAGREEMENT_OR_LOW_CONFIDENCE"
    diagnostics["candidate"] = label
    return (label, confidence, indices), ""


class AbcDynamicCoordinator:
    """Priority arbitration, never a cross-model probability competition.

    Static predictions stay available even during dynamic diagnostics. Only a
    debounced CAPTURING bout or locked endpoint suppresses static confirmation.
    Rejection returns control immediately; only a successful commit is latched.
    """
    def __init__(self, letters, digits, model, enable_commits=False, dynamic_enabled=True):
        self.static = RecognitionCoordinator(letters, digits, HoldSettings())
        self.model = model
        self.enable_commits = enable_commits
        self.dynamic_enabled = dynamic_enabled
        self.segmenter = MotionSegmenter()
        self.locked = None
        self.endpoint = None
        self.blocked = False
        self.missing_since = None
        self.status = "STATIC"
        self.dynamic_state = "IDLE"
        self.suppression_reason = ""
        self.static_prediction = None
        self.last_observation = None
        self.jz_candidate = None
        self.suppressed = False
        self.event_sink = None
        self.frame_context = {}
        self.last_transition_reason = ""
        self.last_evaluation = {}
        self.identity_unverified = False

    def transition(self, state, now, reason="", metrics=None):
        previous = self.dynamic_state
        self.dynamic_state = state
        if self.event_sink and (previous != state or reason != self.last_transition_reason):
            self.event_sink({"event": "STATE_TRANSITION", **self.frame_context,
                             "timestamp_monotonic": now, "previous_state": previous,
                             "new_state": state, "reason": reason,
                             "segment": metrics if metrics is not None else self.segmenter.trace_metrics(now),
                             "model_called": False, "model_result": None, "geometry_candidates": None,
                             "candidate": self.jz_candidate,
                             "identity_unverified": self.identity_unverified})
        self.last_transition_reason = reason

    @property
    def mode(self):
        return self.static.mode

    @property
    def hold(self):
        return self.static.hold

    @property
    def composer(self):
        return self.static.composer

    def cancel_static_hold(self, now):
        # Valid geometry + no eligible label cancels progress WITHOUT treating
        # dynamic processing as hand release or discarding the one-shot latch.
        observation = (replace(self.last_observation, label=None, confidence=0.)
                       if self.last_observation is not None else None)
        self.hold.update(observation, timestamp=now)
        if not self.suppressed:
            self.static.predictors["ABC"].reset()
        self.suppressed = True
        self.endpoint = None

    def resume_static(self):
        if self.suppressed:
            self.static.predictors["ABC"].reset()
        self.suppressed = False
        self.suppression_reason = ""

    def cancel_dynamic(self, reason, now):
        metrics = (dict(self.segmenter.last_diagnostics)
                   if "rejection" in self.segmenter.last_diagnostics or "completion" in self.segmenter.last_diagnostics
                   else self.segmenter.trace_metrics(now))
        self.segmenter.reset()
        if self.locked or self.suppressed:
            self.cancel_static_hold(now)
        self.locked = None
        self.endpoint = None
        self.jz_candidate = None
        self.transition("REJECTED", now, reason, metrics)
        self.status = "REVIEW: " + reason
        self.resume_static()
        self.identity_unverified = False

    def update_static_hold(self, prediction, static_hand, frame_shape, now):
        observation = observe_hand(static_hand, prediction, frame_shape)
        self.last_observation = observation
        committed = self.hold.update(observation, timestamp=now)
        if committed is not None:
            self.composer.append(committed)
            self.static.predictors["ABC"].reset()
        elif self.hold.rearmed and self.hold.state == "IDLE":
            self.static.predictors["ABC"].reset()
        return prediction, committed

    def switch_mode(self, mode):
        if mode not in ("ABC", "123"):
            raise ValueError("Only ABC / 123 are supported")
        if mode != self.mode:
            self.static.switch_mode(mode)
            self.segmenter.reset()
            self.locked = None
            self.endpoint = None
            self.blocked = False
            self.missing_since = None
            self.status = "STATIC"
            self.dynamic_state = "IDLE"
            self.suppression_reason = ""
            self.static_prediction = None
            self.last_observation = None
            self.jz_candidate = None
            self.suppressed = False
            self.identity_unverified = False

    def missing_hand(self, now, index=None):
        if self.mode == "123":
            self.static.missing_hand(now)
            return
        tracking_reason = "TRACKING_LOST"
        if index is not None and self.dynamic_enabled and self.segmenter.active and not self.locked:
            state, reason, _ = self.segmenter.update(index, None, (0, 0), now)
            if state == "GAP":
                self.transition(state, now)
                self.status = "JZ_TRACKING_GAP: BOUNDED_WAIT"
                self.static_prediction = None
                self.cancel_static_hold(now)
                self.suppression_reason = "ACTIVE_DYNAMIC_TRACKING_GAP"
                return
            tracking_reason = reason or tracking_reason
        self.cancel_dynamic(tracking_reason, now)
        self.static_prediction = None
        self.static.missing_hand(now)
        if self.missing_since is None:
            self.missing_since = now
        if now - self.missing_since >= self.hold.settings.release_seconds:
            self.blocked = False
            self.status = "STATIC"

    def uncertain_hand(self, now):
        """Ambiguity is NOT evidence that the user physically released a pose."""
        if self.mode == "123":
            self.static.missing_hand(now)
            return
        self.cancel_dynamic("HAND_IDENTITY_UNCERTAIN", now)
        self.missing_since = None
        self.static_prediction = None
        self.static.predictors["ABC"].reset()
        # Invalidate countdown, but ambiguity must not count as physical release.
        observation = (replace(self.last_observation, label=None, confidence=0.)
                       if self.last_observation is not None else None)
        self.hold.update(observation, timestamp=now)

    def process_hand(self, static_hand, raw_hand, frame_shape, index, now, *, dynamic_missing=False,
                     dynamic_failure_reason=None):
        self.missing_since = None
        if self.mode == "123":
            result = self.static.process_hand(static_hand, frame_shape, now)
            self.static_prediction = result[0]
            self.dynamic_state = "DISABLED_IN_123"
            self.status = "DIGITS"
            return result
        if not self.dynamic_enabled:
            prediction, committed = self.static.process_hand(static_hand, frame_shape, now)
            self.static_prediction = prediction
            self.status = "STATIC: DYNAMIC_DISABLED"
            self.dynamic_state = "DISABLED"
            return prediction, committed
        # Always classify valid primary landmarks, regardless of secondary
        # MediaPipe failures. UI/debug must never lose a valid static prediction.
        prediction = self.static.predictors["ABC"].predict_landmarks(static_hand)
        self.static_prediction = prediction
        self.last_observation = observe_hand(static_hand, prediction, frame_shape)
        if self.blocked:
            self.suppression_reason = "WAIT_RELEASE_AFTER_JZ_COMMIT"
            # Only a successful dynamic commit has this one-shot latch. Keep
            # showing static predictions; physical release rearms it. Rejected
            # or diagnostic-only candidates NEVER enter this state.
            self.cancel_static_hold(now)
            return prediction, None
        if raw_hand is None and (not dynamic_missing or self.locked):
            self.cancel_dynamic(dynamic_failure_reason or "DYNAMIC_HAND_UNAVAILABLE_OR_MISMATCH", now)
            return self.update_static_hold(prediction, static_hand, frame_shape, now)
        if self.locked:
            label, confidence = self.locked
            features = tuple(normalize_landmarks(static_hand))
            prediction = Prediction(label, confidence, label, confidence, features, ())
            observation = observe_hand(static_hand, prediction, frame_shape)
            if self.endpoint is not None:
                settings = self.hold.settings
                drift = np.linalg.norm(np.asarray(observation.palm_center_px)
                                       - self.endpoint.palm_center_px) / self.endpoint.scale_px
                scale_change = abs(observation.scale_px / self.endpoint.scale_px - 1)
                shape_change = np.sqrt(np.mean((np.asarray(observation.shape) - self.endpoint.shape) ** 2))
                # Static HOLD intentionally may re-anchor a new stable pose.
                # A locked dynamic endpoint must instead cancel, not silently
                # restart on a different pose with the old J/Z label.
                if (drift > settings.max_center_drift_ratio
                        or scale_change > settings.max_scale_change
                        or shape_change > settings.max_shape_rms):
                    self.locked = None
                    self.cancel_dynamic("END_POSE_CHANGED", now)
                    return self.static_prediction, None
            else:
                self.endpoint = observation
            committed = self.hold.update(observation, timestamp=now)
            if committed:
                self.composer.append(committed)
                self.blocked = True
                self.locked = None
                self.static.predictors["ABC"].reset()
                self.status = "JZ_CONFIRMED: RELEASE_HAND"
            elif self.hold.state == "IDLE":
                self.locked = None
                self.cancel_dynamic("END_POSE_CHANGED", now)
            return prediction, committed
        return self.process_dynamic_segment(prediction, static_hand, raw_hand, frame_shape, index, now)

    def process_secondary_only(self, raw_hand, frame_shape, index, now):
        """Continue an already matched bout with REAL raw IMAGE landmarks.

        A missing VIDEO observation cannot verify cross-pass identity. Keep that
        fact latched for this bout: diagnostic candidates only, never auto-commit.
        Do not start a new unmatched bout or use mirrored VIDEO landmarks as raw.
        """
        self.static_prediction = None
        if not self.segmenter.active or self.locked or self.blocked:
            self.cancel_dynamic("VIDEO_MISSING_NO_MATCHED_ACTIVE_BOUT", now)
            self.static.missing_hand(now)
            return None, None
        self.identity_unverified = True
        self.cancel_static_hold(now)
        return self.process_dynamic_segment(None, None, raw_hand, frame_shape, index, now)

    def process_dynamic_segment(self, prediction, static_hand, raw_hand, frame_shape, index, now):
        def static_result():
            if static_hand is None:
                return None, None
            return self.update_static_hold(prediction, static_hand, frame_shape, now)

        height, width = frame_shape[:2]
        try:
            state, reason, entries = self.segmenter.update(index, raw_hand, (width, height), now)
        except (ValueError, TypeError, AttributeError) as exc:
            state, reason, entries = "REJECTED", str(exc), []
        metrics = (dict(self.segmenter.last_diagnostics) if state in ("COMPLETE", "REJECTED")
                   else self.segmenter.trace_metrics(now))
        if state in ("SUSPECT", "CAPTURING", "GAP", "REJECTED"):
            self.jz_candidate = None
        self.transition(state, now, reason, metrics)
        if state in ("IDLE", "SUSPECT"):
            self.resume_static()
            self.status = "STATIC" if state == "IDLE" else "STATIC: MOTION_DEBOUNCE"
            return static_result()
        if state in ("CAPTURING", "GAP"):
            self.jz_candidate = None
            self.cancel_static_hold(now)
            self.suppression_reason = "DEBOUNCED_DYNAMIC_MOTION"
            self.status = ("JZ_TRACKING_GAP: BOUNDED_WAIT" if state == "GAP"
                           else "JZ_MOTION: STATIC_HOLD_CANCELLED")
            return prediction, None
        if state == "REJECTED":
            self.cancel_dynamic(reason, now)
            return static_result()
        self.cancel_static_hold(now)
        self.last_evaluation = {"model_called": False, "model_result": None,
                                "geometry_candidates": [], "candidate": None}
        try:
            candidate, reason = evaluate_segment(entries, self.model, self.segmenter.settings,
                                                 diagnostics=self.last_evaluation)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            candidate, reason = None, "INVALID_SEQUENCE: " + str(exc)
        if self.event_sink:
            self.event_sink({"event": "SEGMENT_EVALUATION", **self.frame_context,
                             "timestamp_monotonic": now, "previous_state": "COMPLETE",
                             "new_state": "CANDIDATE" if candidate else "REJECTED",
                             "reason": reason, "segment": metrics, **self.last_evaluation,
                             "candidate": candidate[0] if candidate else None,
                             "identity_unverified": self.identity_unverified})
        if candidate is None:
            self.cancel_dynamic(reason, now)
            return prediction, None
        label, confidence, _indices = candidate
        self.jz_candidate = label
        if not self.enable_commits or self.identity_unverified or static_hand is None:
            self.resume_static()
            self.status = f"REVIEW_ONLY_{label}: " + ("IDENTITY_UNVERIFIED" if self.identity_unverified
                                                     else "AUTO_COMMIT_DISABLED")
            self.identity_unverified = False
            return prediction, None
        self.locked = (label, confidence)
        self.transition("ENDPOINT_HOLD", now)
        self.suppression_reason = "LOCKED_JZ_ENDPOINT"
        self.status = f"JZ_ENDPOINT_HOLD_{label}"
        # Start endpoint HOLD now; do not classify the same frame twice.
        dynamic_prediction = Prediction(label, confidence, label, confidence,
                                        tuple(normalize_landmarks(static_hand)), ())
        self.endpoint = observe_hand(static_hand, dynamic_prediction, frame_shape)
        self.hold.update(self.endpoint, timestamp=now)
        return dynamic_prediction, None
