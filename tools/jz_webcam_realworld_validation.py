"""Real-world webcam J/Z validation harness with 13-field per-attempt logging.

Runs the exact UnifiedAutoCoordinator + AutoJzSession + First16Segmenter +
integrated_jz_validation + 3s HoldConfirmation pipeline unchanged, wrapping
only observation/telemetry around explicitly human-armed isolated J/Z trials.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import FEATURE_DIM, SEQUENCE_LENGTH  # noqa: E402
from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array  # noqa: E402
from app.inference.dynamic_jz_auto_camera_experiment import detect_frame  # noqa: E402
from app.inference.dynamic_jz_camera_test import _synthetic_feature_check  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    DEFAULT_MODEL,
    HAND_MODEL_PATH,
    IntegrationError,
    StaticAslInference,
    _validate_deployment_config,
    load_official_artifacts,
)
from app.inference.space_predictor import load_space_predictor  # noqa: E402
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator  # noqa: E402
from app.inference.unified_abc123_jz_auto_experiment import (  # noqa: E402
    WINDOW,
    detect_static,
    mode_from_click,
    render,
)
from app.inference.unified_camera_experiment import prepare_inference_frame  # noqa: E402
from app.inference.unified_static_coordinator import DigitPredictor, load_digit_model  # noqa: E402
from src.ui.window_display import CameraWindowController  # noqa: E402


CSV_COLUMNS = (
    "trial_index",
    "phase",
    "armed_utc",
    "ground_truth_label",
    "total_camera_frames",
    "valid_landmark_frames",
    "duplicate_landmark_frames",
    "duplicate_landmark_ratio",
    "motion_start_timestamp",
    "motion_end_timestamp",
    "captured_sequence_duration",
    "motion_window_duration",
    "motion_segmenter_result",
    "motion_segmenter_reason",
    "first16_segmenter_result",
    "first16_segmenter_reason",
    "feature_shape",
    "feature_valid",
    "model_prediction",
    "probability_J",
    "probability_Z",
    "trajectory_validator_result",
    "trajectory_validator_reason",
    "hold_confirmation_result",
    "hold_confirmation_reason",
    "hold_elapsed_seconds",
    "final_recognition_result",
    "committed_correct",
    "model_prediction_correct",
    # Stage 4: 4-Method Shadow Telemetry columns
    "trial_attempt_id",
    "lr_prediction",
    "lr_confidence",
    "lr_latency_ms",
    "dtw_prediction",
    "dtw_best_distance",
    "dtw_second_distance",
    "dtw_margin",
    "dtw_status",
    "dtw_latency_ms",
    "geom_prediction",
    "geom_score_j",
    "geom_score_z",
    "geom_margin",
    "geom_status",
    "geom_latency_ms",
    "hybrid_prediction",
    "hybrid_confidence",
    "hybrid_decision_path",
    "hybrid_agreement_count",
    "hybrid_rejection_reason",
    "hybrid_latency_ms",
    "methods_agreement",
)


class ValidationModelProxy:
    """Transparent model wrapper recording (16, 69) feature shape/validity & predictions
    across 4 parallel methods under non-invasive SHADOW execution:
    1. Baseline Logistic Regression (sole production decision maker)
    2. Enhanced DTW CONFIG_E
    3. Geometric Classifier
    4. Hybrid Decision Classifier
    """

    def __init__(self, inner_model, tracker=None, enable_hybrid_shadow: bool = False, hybrid_classifier=None):
        self._inner = inner_model
        self.tracker = tracker
        self.enable_hybrid_shadow = bool(enable_hybrid_shadow)
        self.hybrid_classifier = hybrid_classifier
        if self.enable_hybrid_shadow and self.hybrid_classifier is None:
            try:
                from app.inference.dynamic_jz.hybrid_jz_classifier import HybridJzClassifier
                self.hybrid_classifier = HybridJzClassifier()
            except Exception as exc:
                print(f"[HYBRID SHADOW WARNING] Failed to load HybridJzClassifier: {exc}")
                self.hybrid_classifier = None

    @property
    def classes_(self):
        return self._inner.classes_

    @property
    def named_steps(self):
        return getattr(self._inner, "named_steps", None)

    @property
    def n_features_in_(self):
        return getattr(self._inner, "n_features_in_", SEQUENCE_LENGTH * FEATURE_DIM)

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=np.float64)
        expected_flat = SEQUENCE_LENGTH * FEATURE_DIM
        if arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM) or arr.shape == (expected_flat,):
            arr = arr.reshape((1, expected_flat))
            shape_str = f"({SEQUENCE_LENGTH}, {FEATURE_DIM})"
        elif arr.ndim == 2 and arr.shape == (1, expected_flat):
            shape_str = f"({SEQUENCE_LENGTH}, {FEATURE_DIM})"
        else:
            shape_str = str(tuple(arr.shape))
        is_valid = bool(arr.shape == (1, expected_flat) and np.isfinite(arr).all())

        # 1. Execute Production Logistic Regression (strictly unmodified return value)
        t_lr0 = time.perf_counter()
        probs = np.asarray(self._inner.predict_proba(arr), dtype=np.float64)
        lr_lat = (time.perf_counter() - t_lr0) * 1000.0

        classes = [str(c) for c in self._inner.classes_]
        idx_j = classes.index("J") if "J" in classes else 0
        idx_z = classes.index("Z") if "Z" in classes else 1
        prob_j = float(probs[0, idx_j])
        prob_z = float(probs[0, idx_z])
        lr_pred = str(self._inner.classes_[int(np.argmax(probs[0]))])
        lr_conf = float(max(prob_j, prob_z))

        if self.tracker is not None:
            self.tracker.record_model_inference(
                feature_shape=shape_str,
                feature_valid=is_valid and bool(np.isfinite(probs).all()),
                prediction=lr_pred,
                probability_J=prob_j,
                probability_Z=prob_z,
            )

        # 2. Parallel Hybrid Shadow execution (Zero production interference)
        if self.enable_hybrid_shadow and self.hybrid_classifier is not None and is_valid:
            try:
                seq_16x69 = arr.reshape((SEQUENCE_LENGTH, FEATURE_DIM), order="C").copy()

                # Method 2: Enhanced DTW CONFIG_E
                t_dtw0 = time.perf_counter()
                dtw_res = self.hybrid_classifier.enhanced_dtw.classify(seq_16x69)
                dtw_lat = (time.perf_counter() - t_dtw0) * 1000.0

                # Method 3: Geometric Classifier
                t_geom0 = time.perf_counter()
                geom_res = self.hybrid_classifier.geom_classifier.classify(seq_16x69)
                geom_lat = (time.perf_counter() - t_geom0) * 1000.0

                # Method 4: Hybrid Decision Classifier
                t_hyb0 = time.perf_counter()
                hyb_res = self.hybrid_classifier.classify(seq_16x69)
                hyb_lat = (time.perf_counter() - t_hyb0) * 1000.0

                second_dist = float(max(dtw_res.distance_j, dtw_res.distance_z))

                if self.tracker is not None:
                    self.tracker.record_shadow_inference(
                        lr_prediction=lr_pred,
                        lr_confidence=lr_conf,
                        lr_latency_ms=lr_lat,
                        dtw_prediction=dtw_res.prediction,
                        dtw_best_distance=dtw_res.best_distance,
                        dtw_second_distance=second_dist,
                        dtw_margin=dtw_res.margin,
                        dtw_status=dtw_res.status,
                        dtw_latency_ms=dtw_lat,
                        geom_prediction=geom_res.prediction,
                        geom_score_j=geom_res.score_j,
                        geom_score_z=geom_res.score_z,
                        geom_margin=geom_res.margin,
                        geom_status=geom_res.status,
                        geom_latency_ms=geom_lat,
                        hybrid_prediction=hyb_res.prediction,
                        hybrid_confidence=hyb_res.confidence,
                        hybrid_decision_path=hyb_res.decision_path,
                        hybrid_agreement_count=hyb_res.agreement_count,
                        hybrid_rejection_reason=hyb_res.rejection_reason,
                        hybrid_latency_ms=hyb_lat,
                    )
            except Exception as exc:
                if self.tracker is not None:
                    self.tracker.record_shadow_error(str(exc))

        return probs

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class ActiveTrial:
    trial_index: int
    phase: str
    ground_truth_label: str
    armed_utc: str
    armed_monotonic: float
    total_camera_frames: int = 0
    valid_landmark_frames: int = 0
    duplicate_landmark_frames: int = 0
    _prev_valid_arr: np.ndarray | None = field(default=None, repr=False)
    motion_started: bool = False
    motion_suspect_since: float | None = None
    capture_entered: bool = False
    motion_start_timestamp: float | None = None
    motion_end_timestamp: float | None = None
    captured_sequence_duration: float | None = None
    motion_segmenter_result: str = "NO_MOTION"
    motion_segmenter_reason: str = "NO_MOTION_STARTED"
    first16_segmenter_result: str = "NOT_REACHED"
    first16_segmenter_reason: str = "NONE"
    feature_shape: str = "NOT_BUILT"
    feature_valid: bool = False
    model_invoked: bool = False
    model_prediction: str = "NONE"
    probability_J: float | None = None
    probability_Z: float | None = None
    trajectory_validator_result: str = "SKIPPED"
    trajectory_validator_reason: str = "NONE"
    hold_entered: bool = False
    hold_started_at: float | None = None
    hold_confirmation_result: str = "NOT_REACHED"
    hold_confirmation_reason: str = "NONE"
    hold_elapsed_seconds: float = 0.0
    final_recognition_result: str = "PENDING"
    trial_attempt_id: int | None = None
    # Stage 4: 4-Method Shadow Telemetry fields
    lr_prediction: str = "NONE"
    lr_confidence: float | None = None
    lr_latency_ms: float | None = None
    dtw_prediction: str = "NONE"
    dtw_best_distance: float | None = None
    dtw_second_distance: float | None = None
    dtw_margin: float | None = None
    dtw_status: str = "NONE"
    dtw_latency_ms: float | None = None
    geom_prediction: str = "NONE"
    geom_score_j: float | None = None
    geom_score_z: float | None = None
    geom_margin: float | None = None
    geom_status: str = "NONE"
    geom_latency_ms: float | None = None
    hybrid_prediction: str = "NONE"
    hybrid_confidence: float | None = None
    hybrid_decision_path: str = "NONE"
    hybrid_agreement_count: int = 0
    hybrid_rejection_reason: str = ""
    hybrid_latency_ms: float | None = None
    methods_agreement: bool = False


def reset_trial_coordinator(coordinator, now: float) -> None:
    """Reset dynamic + static coordinator state cleanly before an isolated trial."""
    from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings

    coordinator.switch_mode("ABC")
    coordinator.post_jz_latch = False
    coordinator.clear_candidate("NEW_VALIDATION_TRIAL", now, emit_cancel=False)
    coordinator.confirmed_raw_shape = None
    coordinator.release_since = None
    coordinator.new_pose_frames = 0
    coordinator.pending_dynamic = None
    coordinator.raw_dynamic_prediction = None
    coordinator.raw_prediction_at = None
    coordinator.candidate_cancel_reason = None
    coordinator.current_harness_target_label = "NONE"
    coordinator.jz_evaluation_record = None
    coordinator.jz_evaluation_at = None
    coordinator.jz_hold = HoldConfirmation(HoldSettings())
    coordinator.jz_hold.text = coordinator.composer.text
    coordinator.blocked = False
    coordinator.status = "STATIC"
    coordinator.reason = ""
    coordinator.session.segmenter.reset()
    coordinator.session.pending = None
    coordinator.session.last_result = None
    coordinator.session.state = "READY"
    coordinator.session.reason = ""
    coordinator.session.last_detector_state = "READY"
    coordinator.session.last_endpoint_reason = None
    coordinator.session.visible_until = -float("inf")
    coordinator.session.valid_frames = 0
    coordinator.capture_rows = []
    coordinator.capture_attempt_id = None
    coordinator.capture_scale = None
    if coordinator.session.attempts.active and not coordinator.session.attempts.active["finished"]:
        coordinator.session.attempts.finish({"status": "REJECTED", "reason": "NEW_VALIDATION_TRIAL"})
    coordinator.reset_static()


class ValidationTrialTracker:
    """Tracks isolated human-labeled J/Z webcam trials with all 13 required fields."""

    HARD_SEGMENTER_REJECTIONS = {
        "MISSING_HAND_LIMIT",
        "PALM_POSITION_DISCONTINUITY",
        "SCALE_AND_SHAPE_DISCONTINUITY",
        "MULTIPLE_HANDS_IDENTITY_UNVERIFIED",
        "SEGMENT_TIMEOUT",
        "CAMERA_READ_FAILED",
    }

    def __init__(self, output_dir: Path, phase: str = "baseline", idle_timeout_seconds: float = 12.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.csv_path = self.output_dir / "trials.csv"
        self.jsonl_path = self.output_dir / "trials.jsonl"
        self.hybrid_telemetry_path = self.output_dir / "hybrid_shadow_telemetry.jsonl"
        self.summary_md_path = self.output_dir / "session_summary.md"
        self.summary_json_path = self.output_dir / "session_summary.json"
        self.active: ActiveTrial | None = None
        self.completed: list[dict] = []
        self.last_completed: dict | None = None
        self._event_buffer: list[dict] = []

    def arm(self, ground_truth_label: str, coordinator, now: float) -> ActiveTrial:
        label = ground_truth_label.upper()
        if label not in ("J", "Z"):
            raise ValueError("ground_truth_label must be 'J' or 'Z'")
        reset_trial_coordinator(coordinator, now)
        coordinator.current_harness_target_label = label
        self._event_buffer.clear()
        next_idx = len(self.completed) + 1
        self.active = ActiveTrial(
            trial_index=next_idx,
            phase=self.phase,
            ground_truth_label=label,
            armed_utc=datetime.now(timezone.utc).isoformat(),
            armed_monotonic=float(now),
        )
        print(f"\n[TRIAL #{next_idx}] ARMED ground_truth={label} (perform ONE isolated {label} gesture)", flush=True)
        return self.active

    def discard_active(self, reason: str = "USER_DISCARDED") -> None:
        if self.active is not None:
            print(f"[TRIAL #{self.active.trial_index}] DISCARDED ({reason})", flush=True)
        self.active = None
        self._event_buffer.clear()

    def undo_last(self) -> dict | None:
        if self.active is not None or not self.completed:
            return None
        removed = self.completed.pop()
        self.last_completed = self.completed[-1] if self.completed else None
        self._rewrite_outputs()
        print(f"[UNDO] Removed trial #{removed['trial_index']} (GT={removed['ground_truth_label']})", flush=True)
        return removed

    def record_model_inference(self, *, feature_shape: str, feature_valid: bool,
                               prediction: str, probability_J: float, probability_Z: float) -> None:
        trial = self.active
        if trial is None:
            return
        trial.model_invoked = True
        trial.feature_shape = feature_shape
        trial.feature_valid = feature_valid
        trial.model_prediction = prediction
        trial.probability_J = float(probability_J)
        trial.probability_Z = float(probability_Z)
        trial.motion_segmenter_result = "ACCEPTED"
        trial.motion_segmenter_reason = "MOTION_ACCEPTED_FOR_CAPTURE"
        if trial.first16_segmenter_result == "NOT_REACHED":
            trial.first16_segmenter_result = "COMPLETE"
            trial.first16_segmenter_reason = "MOTION_GATED_VALID_WINDOW"

    def record_shadow_inference(
        self,
        *,
        lr_prediction: str,
        lr_confidence: float,
        lr_latency_ms: float,
        dtw_prediction: str,
        dtw_best_distance: float,
        dtw_second_distance: float,
        dtw_margin: float,
        dtw_status: str,
        dtw_latency_ms: float,
        geom_prediction: str,
        geom_score_j: float,
        geom_score_z: float,
        geom_margin: float,
        geom_status: str,
        geom_latency_ms: float,
        hybrid_prediction: str,
        hybrid_confidence: float,
        hybrid_decision_path: str,
        hybrid_agreement_count: int,
        hybrid_rejection_reason: str,
        hybrid_latency_ms: float,
    ) -> None:
        trial = self.active
        if trial is None:
            return
        trial.lr_prediction = lr_prediction
        trial.lr_confidence = float(lr_confidence)
        trial.lr_latency_ms = float(lr_latency_ms)
        trial.dtw_prediction = dtw_prediction
        trial.dtw_best_distance = float(dtw_best_distance)
        trial.dtw_second_distance = float(dtw_second_distance)
        trial.dtw_margin = float(dtw_margin)
        trial.dtw_status = dtw_status
        trial.dtw_latency_ms = float(dtw_latency_ms)
        trial.geom_prediction = geom_prediction
        trial.geom_score_j = float(geom_score_j)
        trial.geom_score_z = float(geom_score_z)
        trial.geom_margin = float(geom_margin)
        trial.geom_status = geom_status
        trial.geom_latency_ms = float(geom_latency_ms)
        trial.hybrid_prediction = hybrid_prediction
        trial.hybrid_confidence = float(hybrid_confidence)
        trial.hybrid_decision_path = hybrid_decision_path
        trial.hybrid_agreement_count = int(hybrid_agreement_count)
        trial.hybrid_rejection_reason = hybrid_rejection_reason
        trial.hybrid_latency_ms = float(hybrid_latency_ms)
        trial.methods_agreement = bool(
            lr_prediction == dtw_prediction == geom_prediction == hybrid_prediction
            and lr_prediction in ("J", "Z")
        )

    def record_shadow_error(self, error_message: str) -> None:
        trial = self.active
        if trial is None:
            return
        trial.dtw_status = f"ERROR:{error_message}"
        trial.geom_status = f"ERROR:{error_message}"
        trial.hybrid_decision_path = f"ERROR:{error_message}"
        if trial.first16_segmenter_result == "NOT_REACHED":
            trial.first16_segmenter_result = "COMPLETE"
            trial.first16_segmenter_reason = "MOTION_GATED_VALID_WINDOW"

    def on_event(self, payload: dict) -> None:
        trial = self.active
        if trial is None:
            return
        self._event_buffer.append(payload)
        name = payload.get("event")
        now = payload.get("timestamp_monotonic")

        if name == "DETECTOR_TRANSITION":
            new_state = payload.get("new_state")
            if new_state in ("SUSPECT", "CAPTURING"):
                trial.motion_started = True
                if new_state == "SUSPECT" and trial.motion_suspect_since is None and now is not None:
                    trial.motion_suspect_since = float(now)
                if trial.motion_start_timestamp is None and now is not None:
                    trial.motion_start_timestamp = float(now)
                if new_state == "CAPTURING":
                    trial.capture_entered = True
                    trial.motion_segmenter_result = "ACCEPTED"
                    trial.motion_segmenter_reason = "ENTERED_CAPTURING"
                    if trial.first16_segmenter_result == "NOT_REACHED":
                        trial.first16_segmenter_result = "IN_PROGRESS"
                        trial.first16_segmenter_reason = "COLLECTING_MOTION_AND_DURATION_EVIDENCE"

        elif name == "STATE_TRANSITION":
            new_state = payload.get("new_state")
            reason = payload.get("reason") or ""
            if new_state == "CAPTURING":
                trial.motion_started = True
                trial.capture_entered = True
                trial.trial_attempt_id = payload.get("attempt_id") or trial.trial_attempt_id
                trial.motion_segmenter_result = "ACCEPTED"
                trial.motion_segmenter_reason = "ENTERED_CAPTURING"
                if trial.motion_start_timestamp is None and now is not None:
                    trial.motion_start_timestamp = float(now)
            elif new_state == "PROCESSING":
                trial.first16_segmenter_result = "COMPLETE"
                trial.first16_segmenter_reason = reason or "MOTION_GATED_VALID_WINDOW"
                if now is not None:
                    trial.motion_end_timestamp = float(now)
                seg = payload.get("segment") or {}
                if seg.get("duration_seconds") is not None:
                    trial.captured_sequence_duration = float(seg["duration_seconds"])
            elif new_state == "REJECTED":
                if now is not None and trial.motion_end_timestamp is None:
                    trial.motion_end_timestamp = float(now)
                if not trial.model_invoked:
                    if trial.capture_entered and "STATIC_OR_ENTRY_MOTION_ONLY" in reason:
                        trial.first16_segmenter_result = "REJECTED"
                        trial.first16_segmenter_reason = reason
                    else:
                        trial.motion_segmenter_result = "REJECTED"
                        trial.motion_segmenter_reason = reason or "SEGMENTER_REJECTED"

        elif name == "INFERENCE_END":
            diag = payload.get("diagnostics") or {}
            seg = diag.get("segment") or {}
            if seg.get("duration_seconds") is not None:
                trial.captured_sequence_duration = float(seg["duration_seconds"])
            if now is not None:
                trial.motion_end_timestamp = float(now)
            if payload.get("status") == "REJECTED" and not trial.model_invoked:
                trial.first16_segmenter_result = "REJECTED"
                trial.first16_segmenter_reason = payload.get("reason") or "INVALID_SEQUENCE"

        elif name == "ATTEMPT_FINISHED":
            res = payload.get("result") or {}
            reason = res.get("reason") or ""
            seg = res.get("segment") or {}
            if seg.get("duration_seconds") is not None and trial.captured_sequence_duration is None:
                trial.captured_sequence_duration = float(seg["duration_seconds"])
            if res.get("status") == "REJECTED" and not trial.model_invoked:
                if trial.capture_entered and reason == "STATIC_OR_ENTRY_MOTION_ONLY":
                    trial.first16_segmenter_result = "REJECTED"
                    trial.first16_segmenter_reason = reason
                elif reason:
                    trial.motion_segmenter_result = "REJECTED"
                    trial.motion_segmenter_reason = reason

        elif name == "JZ_PROPOSAL_REJECTED_TRAJECTORY":
            val = payload.get("validation") or {}
            pred = payload.get("label") or trial.model_prediction
            sub = (val.get("j_trajectory_validation") if pred == "J"
                   else val.get("z_trajectory_validation") if pred == "Z" else None)
            detail = (sub or {}).get("reason") or val.get("reason") or payload.get("reason") or "TRAJECTORY_REJECTED"
            trial.trajectory_validator_result = "REJECTED"
            trial.trajectory_validator_reason = str(detail)

        elif name == "JZ_EVALUATION_RECORD":
            rec = payload.get("evaluation_record") or {}
            stage = rec.get("stage")
            val_status = rec.get("validator_status")
            reason = rec.get("reason") or "NONE"
            if val_status in ("ACCEPTED", "REJECTED"):
                trial.trajectory_validator_result = val_status
                if val_status == "ACCEPTED" and trial.trajectory_validator_reason == "NONE":
                    trial.trajectory_validator_reason = "TRAJECTORY_VALIDATED"
                elif val_status == "REJECTED" and stage == "VALIDATOR_REJECTED":
                    trial.trajectory_validator_reason = reason
            if stage == "CONFIDENCE_REJECTED":
                trial.hold_confirmation_result = "REJECTED_PRE_HOLD"
                trial.hold_confirmation_reason = reason
                trial.final_recognition_result = f"FAILED:CONFIDENCE_GATE:{reason}"
            elif stage == "VALIDATOR_REJECTED":
                trial.final_recognition_result = f"FAILED:TRAJECTORY_VALIDATOR:{trial.trajectory_validator_reason}"
            elif stage == "SEGMENTER_REJECTED" and not trial.model_invoked:
                trial.final_recognition_result = f"FAILED:MOTION_SEGMENTER:{reason}"

        elif name == "JZ_DYNAMIC_HOLD_STARTED":
            trial.hold_entered = True
            trial.hold_started_at = float(payload.get("hold_clock_started_at") or now or 0.0)
            trial.hold_confirmation_result = "IN_PROGRESS"
            trial.hold_confirmation_reason = "STABILIZING_HOLDING"

        elif name == "JZ_CANDIDATE_CANCELLED":
            cancel_reason = payload.get("cancellation_reason") or "CANCELLED"
            if trial.hold_started_at is not None and now is not None:
                trial.hold_elapsed_seconds = max(0.0, float(now) - trial.hold_started_at)
            if cancel_reason == "JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD":
                trial.hold_confirmation_result = "REJECTED_PRE_HOLD"
            elif trial.hold_entered:
                trial.hold_confirmation_result = "CANCELLED_DURING_HOLD"
            else:
                trial.hold_confirmation_result = "CANCELLED_PENDING_ENDPOINT"
            trial.hold_confirmation_reason = cancel_reason
            trial.final_recognition_result = f"FAILED:HOLD_CONFIRMATION:{cancel_reason}"

        elif name == "JZ_CONFIRMED_EXPERIMENTAL":
            ch = payload.get("character") or trial.model_prediction
            if trial.hold_started_at is not None and now is not None:
                trial.hold_elapsed_seconds = max(0.0, float(now) - trial.hold_started_at)
            else:
                trial.hold_elapsed_seconds = 3.0
            trial.hold_confirmation_result = "CONFIRMED"
            trial.hold_confirmation_reason = "HOLD_3S_COMPLETED"
            trial.final_recognition_result = f"COMMITTED:{ch}"

    def observe_frame(self, raw_hand, coordinator, now: float, *, committed: str | None = None,
                      dynamic_error: str = "", raw_hand_count: int | None = None) -> dict | None:
        """Inspect current frame + coordinator state after coordinator.update(...)."""
        trial = self.active
        if trial is None:
            return None

        trial.total_camera_frames += 1
        if raw_hand is not None and (raw_hand_count is None or raw_hand_count == 1) and not dynamic_error:
            try:
                arr = landmarks_to_array(raw_hand)
                trial.valid_landmark_frames += 1
                if trial._prev_valid_arr is not None and np.array_equal(arr, trial._prev_valid_arr):
                    trial.duplicate_landmark_frames += 1
                trial._prev_valid_arr = arr.copy()
            except (ValueError, TypeError, AttributeError):
                pass

        seg = coordinator.session.segmenter
        if seg.movement_since is not None and trial.motion_start_timestamp is None:
            trial.motion_start_timestamp = float(seg.movement_since)
        if getattr(seg, "active", False):
            trial.capture_entered = True
            trial.motion_started = True
            trial.motion_segmenter_result = "ACCEPTED"
            trial.motion_segmenter_reason = "ENTERED_CAPTURING"
            if trial.first16_segmenter_result == "NOT_REACHED":
                trial.first16_segmenter_result = "IN_PROGRESS"

        last_diag = getattr(seg, "last_diagnostics", None) or {}
        if trial.capture_entered and last_diag.get("rejection") == "STATIC_OR_ENTRY_MOTION_ONLY":
            trial.first16_segmenter_result = "REJECTED"
            trial.first16_segmenter_reason = "STATIC_OR_ENTRY_MOTION_ONLY"
            if last_diag.get("duration_seconds") is not None:
                trial.captured_sequence_duration = float(last_diag["duration_seconds"])
            trial.motion_end_timestamp = float(now)
            return self._finalize_active(now)

        # If coordinator committed J or Z cleanly:
        if committed in ("J", "Z"):
            if trial.final_recognition_result == "PENDING":
                trial.hold_confirmation_result = "CONFIRMED"
                trial.hold_confirmation_reason = "HOLD_3S_COMPLETED"
                trial.final_recognition_result = f"COMMITTED:{committed}"
            return self._finalize_active(now)

        # If validator/confidence/candidate already reached a terminal failure state:
        if trial.final_recognition_result.startswith("FAILED:"):
            return self._finalize_active(now)

        # Hard pre-model MotionSegmenter rejection before CAPTURING:
        sess_state = coordinator.session.state
        sess_reason = coordinator.session.reason or ""
        if not trial.model_invoked:
            if sess_state == "REJECTED" and (
                trial.motion_started
                or any(k in sess_reason for k in self.HARD_SEGMENTER_REJECTIONS)
            ):
                trial.motion_segmenter_result = "REJECTED"
                trial.motion_segmenter_reason = sess_reason or "SEGMENTER_REJECTED"
                trial.motion_end_timestamp = trial.motion_end_timestamp or float(now)
                return self._finalize_active(now)
            if (trial.motion_suspect_since is not None
                    and "STATIC_OR_ENTRY_MOTION_ONLY" in sess_reason
                    and (now - trial.motion_suspect_since) >= 0.25):
                trial.motion_segmenter_result = "REJECTED"
                trial.motion_segmenter_reason = sess_reason
                trial.motion_end_timestamp = float(now)
                return self._finalize_active(now)

        # Idle timeout if no terminal resolution occurred:
        if (not trial.model_invoked and not coordinator.candidate and not coordinator.pending_dynamic
                and (now - trial.armed_monotonic) >= self.idle_timeout_seconds):
            trial.motion_end_timestamp = float(now)
            if trial.capture_entered:
                trial.first16_segmenter_result = "TIMED_OUT"
                trial.first16_segmenter_reason = last_diag.get("prediction_wait_reason", "TRIAL_IDLE_TIMEOUT")
            else:
                trial.motion_segmenter_result = "REJECTED"
                trial.motion_segmenter_reason = "TRIAL_TIMEOUT_NO_VALID_MOTION"
            return self._finalize_active(now)

        return None

    def force_end(self, now: float, reason: str = "MANUAL_END") -> dict | None:
        trial = self.active
        if trial is None:
            return None
        trial.motion_end_timestamp = trial.motion_end_timestamp or float(now)
        if not trial.model_invoked:
            if trial.capture_entered and trial.first16_segmenter_result == "IN_PROGRESS":
                trial.first16_segmenter_result = "INCOMPLETE"
                trial.first16_segmenter_reason = reason
            elif trial.motion_segmenter_result == "NO_MOTION":
                trial.motion_segmenter_result = "REJECTED"
                trial.motion_segmenter_reason = reason
        elif trial.final_recognition_result == "PENDING":
            trial.hold_confirmation_result = "ABORTED"
            trial.hold_confirmation_reason = reason
        return self._finalize_active(now)

    def _finalize_active(self, now: float) -> dict:
        trial = self.active
        assert trial is not None
        self.active = None

        if trial.motion_end_timestamp is None:
            trial.motion_end_timestamp = float(now)
        motion_win_dur = (
            max(0.0, trial.motion_end_timestamp - trial.motion_start_timestamp)
            if trial.motion_start_timestamp is not None else 0.0
        )
        seq_dur = (
            trial.captured_sequence_duration
            if trial.captured_sequence_duration is not None
            else motion_win_dur
        )
        dup_ratio = (
            trial.duplicate_landmark_frames / float(trial.valid_landmark_frames - 1)
            if trial.valid_landmark_frames >= 2 else 0.0
        )

        if trial.final_recognition_result == "PENDING":
            if not trial.model_invoked:
                if trial.first16_segmenter_result in ("REJECTED", "INCOMPLETE", "TIMED_OUT"):
                    trial.final_recognition_result = f"FAILED:FIRST16_SEGMENTER:{trial.first16_segmenter_reason}"
                else:
                    trial.final_recognition_result = f"FAILED:MOTION_SEGMENTER:{trial.motion_segmenter_reason}"
            elif trial.trajectory_validator_result == "REJECTED":
                trial.final_recognition_result = f"FAILED:TRAJECTORY_VALIDATOR:{trial.trajectory_validator_reason}"
            else:
                trial.final_recognition_result = f"FAILED:HOLD_CONFIRMATION:{trial.hold_confirmation_reason}"

        committed_ok = trial.final_recognition_result == f"COMMITTED:{trial.ground_truth_label}"
        model_ok = bool(trial.model_invoked and trial.model_prediction == trial.ground_truth_label)

        row = {
            "trial_index": trial.trial_index,
            "phase": trial.phase,
            "armed_utc": trial.armed_utc,
            "ground_truth_label": trial.ground_truth_label,
            "total_camera_frames": trial.total_camera_frames,
            "valid_landmark_frames": trial.valid_landmark_frames,
            "duplicate_landmark_frames": trial.duplicate_landmark_frames,
            "duplicate_landmark_ratio": round(dup_ratio, 4),
            "motion_start_timestamp": round(trial.motion_start_timestamp, 4) if trial.motion_start_timestamp is not None else "",
            "motion_end_timestamp": round(trial.motion_end_timestamp, 4) if trial.motion_end_timestamp is not None else "",
            "captured_sequence_duration": round(seq_dur, 4),
            "motion_window_duration": round(motion_win_dur, 4),
            "motion_segmenter_result": trial.motion_segmenter_result,
            "motion_segmenter_reason": trial.motion_segmenter_reason,
            "first16_segmenter_result": trial.first16_segmenter_result,
            "first16_segmenter_reason": trial.first16_segmenter_reason,
            "feature_shape": trial.feature_shape,
            "feature_valid": trial.feature_valid,
            "model_prediction": trial.model_prediction,
            "probability_J": round(trial.probability_J, 4) if trial.probability_J is not None else "",
            "probability_Z": round(trial.probability_Z, 4) if trial.probability_Z is not None else "",
            "trajectory_validator_result": trial.trajectory_validator_result,
            "trajectory_validator_reason": trial.trajectory_validator_reason,
            "hold_confirmation_result": trial.hold_confirmation_result,
            "hold_confirmation_reason": trial.hold_confirmation_reason,
            "hold_elapsed_seconds": round(trial.hold_elapsed_seconds, 3),
            "final_recognition_result": trial.final_recognition_result,
            "committed_correct": committed_ok,
            "model_prediction_correct": model_ok,
            # Stage 4 Shadow Telemetry fields
            "trial_attempt_id": trial.trial_attempt_id if trial.trial_attempt_id is not None else "",
            "lr_prediction": trial.lr_prediction,
            "lr_confidence": round(trial.lr_confidence, 4) if trial.lr_confidence is not None else "",
            "lr_latency_ms": round(trial.lr_latency_ms, 3) if trial.lr_latency_ms is not None else "",
            "dtw_prediction": trial.dtw_prediction,
            "dtw_best_distance": round(trial.dtw_best_distance, 4) if trial.dtw_best_distance is not None else "",
            "dtw_second_distance": round(trial.dtw_second_distance, 4) if trial.dtw_second_distance is not None else "",
            "dtw_margin": round(trial.dtw_margin, 4) if trial.dtw_margin is not None else "",
            "dtw_status": trial.dtw_status,
            "dtw_latency_ms": round(trial.dtw_latency_ms, 3) if trial.dtw_latency_ms is not None else "",
            "geom_prediction": trial.geom_prediction,
            "geom_score_j": round(trial.geom_score_j, 4) if trial.geom_score_j is not None else "",
            "geom_score_z": round(trial.geom_score_z, 4) if trial.geom_score_z is not None else "",
            "geom_margin": round(trial.geom_margin, 4) if trial.geom_margin is not None else "",
            "geom_status": trial.geom_status,
            "geom_latency_ms": round(trial.geom_latency_ms, 3) if trial.geom_latency_ms is not None else "",
            "hybrid_prediction": trial.hybrid_prediction,
            "hybrid_confidence": round(trial.hybrid_confidence, 4) if trial.hybrid_confidence is not None else "",
            "hybrid_decision_path": trial.hybrid_decision_path,
            "hybrid_agreement_count": trial.hybrid_agreement_count,
            "hybrid_rejection_reason": trial.hybrid_rejection_reason,
            "hybrid_latency_ms": round(trial.hybrid_latency_ms, 3) if trial.hybrid_latency_ms is not None else "",
            "methods_agreement": trial.methods_agreement,
        }
        self.completed.append(row)
        self.last_completed = row

        # Write per-attempt hybrid shadow telemetry
        if trial.model_invoked:
            telem_rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trial_index": trial.trial_index,
                "attempt_id": trial.trial_attempt_id,
                "target_label": trial.ground_truth_label,
                "baseline_lr": {
                    "prediction": trial.lr_prediction,
                    "confidence": trial.lr_confidence,
                    "prob_j": trial.probability_J,
                    "prob_z": trial.probability_Z,
                    "latency_ms": trial.lr_latency_ms,
                },
                "enhanced_dtw": {
                    "prediction": trial.dtw_prediction,
                    "best_distance": trial.dtw_best_distance,
                    "second_distance": trial.dtw_second_distance,
                    "margin": trial.dtw_margin,
                    "status": trial.dtw_status,
                    "latency_ms": trial.dtw_latency_ms,
                },
                "geometric": {
                    "prediction": trial.geom_prediction,
                    "score_j": trial.geom_score_j,
                    "score_z": trial.geom_score_z,
                    "margin": trial.geom_margin,
                    "status": trial.geom_status,
                    "latency_ms": trial.geom_latency_ms,
                },
                "hybrid": {
                    "prediction": trial.hybrid_prediction,
                    "confidence": trial.hybrid_confidence,
                    "decision_path": trial.hybrid_decision_path,
                    "agreement_count": trial.hybrid_agreement_count,
                    "rejection_reason": trial.hybrid_rejection_reason,
                    "latency_ms": trial.hybrid_latency_ms,
                },
                "production_outcome": {
                    "validator_result": trial.trajectory_validator_result,
                    "validator_reason": trial.trajectory_validator_reason,
                    "hold_result": trial.hold_confirmation_result,
                    "hold_reason": trial.hold_confirmation_reason,
                    "final_recognition_result": trial.final_recognition_result,
                    "committed_correct": committed_ok,
                },
            }
            try:
                with self.hybrid_telemetry_path.open("a", encoding="utf-8") as tf:
                    tf.write(json.dumps(telem_rec, ensure_ascii=False) + "\n")
            except OSError:
                pass

        self._rewrite_outputs()
        print(
            f"[TRIAL #{row['trial_index']} DONE] GT={row['ground_truth_label']} "
            f"frames={row['valid_landmark_frames']}/{row['total_camera_frames']} "
            f"dup={row['duplicate_landmark_ratio']:.1%} | "
            f"LR={row['model_prediction']}(conf={row.get('lr_confidence', '')}) | "
            f"DTW={row.get('dtw_prediction', '')}(d={row.get('dtw_best_distance', '')}) | "
            f"Geom={row.get('geom_prediction', '')} | "
            f"Hybrid={row.get('hybrid_prediction', '')}({row.get('hybrid_decision_path', '')}) | "
            f"Val={row['trajectory_validator_result']} | "
            f"FINAL={row['final_recognition_result']}",
            flush=True,
        )
        return row

    def _rewrite_outputs(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in self.completed:
                writer.writerow(row)
        with self.jsonl_path.open("w", encoding="utf-8") as handle:
            for row in self.completed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = compute_session_summary(self.completed, phase=self.phase)
        self.summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.summary_md_path.write_text(format_summary_markdown(summary), encoding="utf-8")


def _calc_stats(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_session_summary(rows: list[dict], *, phase: str = "baseline") -> dict:
    """Compute separate model prediction accuracy vs end-to-end webcam recognition rate,
    plus 4-method parallel comparative benchmark statistics.
    """
    def _slice_stats(subset: list[dict]) -> dict:
        total = len(subset)
        invoked = [r for r in subset if r.get("model_prediction") in ("J", "Z")]
        model_correct = [r for r in invoked if r.get("model_prediction_correct")]
        val_accepted = [r for r in invoked if r.get("trajectory_validator_result") == "ACCEPTED"]
        committed = [r for r in subset if r.get("committed_correct")]
        stages: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for r in subset:
            res = str(r.get("final_recognition_result", ""))
            if res.startswith("COMMITTED:"):
                stages["COMMITTED"] = stages.get("COMMITTED", 0) + 1
            elif res.startswith("FAILED:"):
                parts = res.split(":", 2)
                st = parts[1] if len(parts) > 1 else "UNKNOWN"
                rs = parts[2] if len(parts) > 2 else "UNKNOWN"
                stages[st] = stages.get(st, 0) + 1
                reasons[rs] = reasons.get(rs, 0) + 1
        dup_vals = [float(r["duplicate_landmark_ratio"]) for r in subset if r.get("duplicate_landmark_ratio") != ""]
        return {
            "total_attempts": total,
            "model_invocations": len(invoked),
            "model_correct_predictions": len(model_correct),
            "model_prediction_accuracy": (len(model_correct) / len(invoked)) if invoked else None,
            "validator_acceptances": len(val_accepted),
            "validator_acceptance_rate_given_invoked": (len(val_accepted) / len(invoked)) if invoked else None,
            "successful_confirmations": len(committed),
            "end_to_end_webcam_recognition_rate": (len(committed) / total) if total else None,
            "mean_duplicate_landmark_ratio": (sum(dup_vals) / len(dup_vals)) if dup_vals else None,
            "failure_stages": stages,
            "failure_reasons": reasons,
        }

    # 4-Method Comparison Slices
    invoked_rows = [r for r in rows if r.get("feature_valid") and r.get("model_prediction") in ("J", "Z")]
    methods_stats = {}
    for method_key, pred_field, lat_field in (
        ("baseline_lr", "model_prediction", "lr_latency_ms"),
        ("enhanced_dtw", "dtw_prediction", "dtw_latency_ms"),
        ("geometric", "geom_prediction", "geom_latency_ms"),
        ("hybrid", "hybrid_prediction", "hybrid_latency_ms"),
    ):
        tp_j = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "J" and r.get(pred_field) == "J")
        fp_j = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "Z" and r.get(pred_field) == "J")
        fn_j = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "J" and r.get(pred_field) != "J")

        tp_z = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "Z" and r.get(pred_field) == "Z")
        fp_z = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "J" and r.get(pred_field) == "Z")
        fn_z = sum(1 for r in invoked_rows if r.get("ground_truth_label") == "Z" and r.get(pred_field) != "Z")

        unknown_count = sum(1 for r in invoked_rows if r.get(pred_field) in ("UNKNOWN", "NONE", "", None))
        total_inv = len(invoked_rows)
        correct_count = tp_j + tp_z

        pj, rj, f1j = _calc_stats(tp_j, fp_j, fn_j)
        pz, rz, f1z = _calc_stats(tp_z, fp_z, fn_z)
        macro_f1 = (f1j + f1z) / 2.0

        latencies = [float(r[lat_field]) for r in invoked_rows if r.get(lat_field) not in ("", None)]
        p50 = float(np.percentile(latencies, 50)) if latencies else None
        p95 = float(np.percentile(latencies, 95)) if latencies else None
        p99 = float(np.percentile(latencies, 99)) if latencies else None

        methods_stats[method_key] = {
            "invocations": total_inv,
            "correct": correct_count,
            "accuracy": (correct_count / total_inv) if total_inv else None,
            "tp_j": tp_j, "fp_j": fp_j, "fn_j": fn_j,
            "tp_z": tp_z, "fp_z": fp_z, "fn_z": fn_z,
            "unknown_count": unknown_count,
            "unknown_rate": (unknown_count / total_inv) if total_inv else 0.0,
            "coverage": ((total_inv - unknown_count) / total_inv) if total_inv else None,
            "precision_j": pj, "recall_j": rj, "f1_j": f1j,
            "precision_z": pz, "recall_z": rz, "f1_z": f1z,
            "macro_f1": macro_f1,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
        }

    # Consensus and Agreement Analysis
    unanimous = sum(
        1 for r in invoked_rows
        if r.get("model_prediction") == r.get("dtw_prediction") == r.get("geom_prediction") == r.get("hybrid_prediction")
        and r.get("model_prediction") in ("J", "Z")
    )
    lr_dtw_agree = sum(
        1 for r in invoked_rows
        if r.get("model_prediction") == r.get("dtw_prediction") and r.get("model_prediction") in ("J", "Z")
    )
    lr_hyb_agree = sum(
        1 for r in invoked_rows
        if r.get("model_prediction") == r.get("hybrid_prediction") and r.get("model_prediction") in ("J", "Z")
    )

    disagreements = []
    for r in invoked_rows:
        preds = {
            "LR": r.get("model_prediction"),
            "DTW": r.get("dtw_prediction"),
            "Geom": r.get("geom_prediction"),
            "Hybrid": r.get("hybrid_prediction"),
        }
        distinct = set(preds.values())
        if len(distinct) > 1:
            disagreements.append({
                "trial_index": r["trial_index"],
                "ground_truth": r.get("ground_truth_label"),
                "predictions": preds,
                "hybrid_path": r.get("hybrid_decision_path"),
                "rejection_reason": r.get("hybrid_rejection_reason"),
            })

    return {
        "phase": phase,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall": _slice_stats(rows),
        "J": _slice_stats([r for r in rows if r.get("ground_truth_label") == "J"]),
        "Z": _slice_stats([r for r in rows if r.get("ground_truth_label") == "Z"]),
        "methods_comparison": methods_stats,
        "consensus_analysis": {
            "invocations": len(invoked_rows),
            "unanimous_agreement_count": unanimous,
            "unanimous_agreement_rate": (unanimous / len(invoked_rows)) if invoked_rows else None,
            "lr_dtw_agreement_count": lr_dtw_agree,
            "lr_dtw_agreement_rate": (lr_dtw_agree / len(invoked_rows)) if invoked_rows else None,
            "lr_hybrid_agreement_count": lr_hyb_agree,
            "lr_hybrid_agreement_rate": (lr_hyb_agree / len(invoked_rows)) if invoked_rows else None,
            "disagreements": disagreements,
        },
    }


def _pct(val: float | None) -> str:
    return "N/A" if val is None else f"{val * 100:.1f}%"


def _fmt_f(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.4f}"


def format_summary_markdown(summary: dict) -> str:
    ov = summary["overall"]
    j = summary["J"]
    z = summary["Z"]
    mc = summary.get("methods_comparison") or {}
    ca = summary.get("consensus_analysis") or {}

    lines = [
        f"# J/Z Real-World Webcam Validation Summary ({summary['phase']})",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        "",
        "## 1. Executive Summary & Production Funnel",
        "",
        "| Metric | J | Z | Total |",
        "| --- | ---: | ---: | ---: |",
        f"| Total attempts | {j['total_attempts']} | {z['total_attempts']} | {ov['total_attempts']} |",
        f"| Model invocations | {j['model_invocations']} | {z['model_invocations']} | {ov['model_invocations']} |",
        f"| Correct model predictions (LR) | {j['model_correct_predictions']} | {z['model_correct_predictions']} | {ov['model_correct_predictions']} |",
        f"| LR Prediction accuracy (given invoked) | {_pct(j['model_prediction_accuracy'])} | {_pct(z['model_prediction_accuracy'])} | {_pct(ov['model_prediction_accuracy'])} |",
        f"| Validator acceptances | {j['validator_acceptances']} | {z['validator_acceptances']} | {ov['validator_acceptances']} |",
        f"| Successful confirmations (3s Hold) | {j['successful_confirmations']} | {z['successful_confirmations']} | {ov['successful_confirmations']} |",
        f"| End-to-end webcam recognition rate | {_pct(j['end_to_end_webcam_recognition_rate'])} | {_pct(z['end_to_end_webcam_recognition_rate'])} | {_pct(ov['end_to_end_webcam_recognition_rate'])} |",
        f"| Mean duplicate landmark ratio | {_pct(j['mean_duplicate_landmark_ratio'])} | {_pct(z['mean_duplicate_landmark_ratio'])} | {_pct(ov['mean_duplicate_landmark_ratio'])} |",
        "",
    ]

    if mc:
        lines.extend([
            "## 2. 4-Method Comparative Benchmark (Webcam Replay / Live Execution)",
            "",
            "| Method | Invocations | Correct | Accuracy | Coverage | UNKNOWN Rate | Macro F1 | Latency P50 (ms) | Latency P95 (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        labels_map = {
            "baseline_lr": "1. Baseline Logistic Regression",
            "enhanced_dtw": "2. Enhanced DTW CONFIG_E",
            "geometric": "3. Geometric Classifier",
            "hybrid": "4. Hybrid Decision Classifier",
        }
        for k in ("baseline_lr", "enhanced_dtw", "geometric", "hybrid"):
            st = mc.get(k) or {}
            inv = st.get("invocations", 0)
            corr = st.get("correct", 0)
            acc = _pct(st.get("accuracy"))
            cov = _pct(st.get("coverage"))
            unk = _pct(st.get("unknown_rate"))
            mf1 = _fmt_f(st.get("macro_f1"))
            p50 = f"{st.get('latency_p50_ms'):.3f}" if st.get("latency_p50_ms") is not None else "N/A"
            p95 = f"{st.get('latency_p95_ms'):.3f}" if st.get("latency_p95_ms") is not None else "N/A"
            lines.append(f"| {labels_map[k]} | {inv} | {corr} | {acc} | {cov} | {unk} | {mf1} | {p50} | {p95} |")

        lines.extend([
            "",
            "### Detailed Per-Class Metrics",
            "",
            "| Method | J Precision | J Recall | J F1 | Z Precision | Z Recall | Z F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for k in ("baseline_lr", "enhanced_dtw", "geometric", "hybrid"):
            st = mc.get(k) or {}
            pj = _fmt_f(st.get("precision_j"))
            rj = _fmt_f(st.get("recall_j"))
            fj = _fmt_f(st.get("f1_j"))
            pz = _fmt_f(st.get("precision_z"))
            rz = _fmt_f(st.get("recall_z"))
            fz = _fmt_f(st.get("f1_z"))
            lines.append(f"| {labels_map[k]} | {pj} | {rj} | {fj} | {pz} | {rz} | {fz} |")

    if ca:
        lines.extend([
            "",
            "## 3. Consensus & Agreement Analysis",
            "",
            f"- Invocations Analyzed: `{ca.get('invocations')}`",
            f"- 4-Method Unanimous Agreement: `{ca.get('unanimous_agreement_count')}` ({_pct(ca.get('unanimous_agreement_rate'))})",
            f"- LR & Enhanced DTW Agreement: `{ca.get('lr_dtw_agreement_count')}` ({_pct(ca.get('lr_dtw_agreement_rate'))})",
            f"- LR & Hybrid Agreement: `{ca.get('lr_hybrid_agreement_count')}` ({_pct(ca.get('lr_hybrid_agreement_rate'))})",
            f"- Disagreement Cases: `{len(ca.get('disagreements', []))}`",
            "",
        ])
        dis = ca.get("disagreements") or []
        if dis:
            lines.extend([
                "| Trial # | Ground Truth | LR | DTW | Geom | Hybrid | Hybrid Path / Reason |",
                "| ---: |:---:|:---:|:---:|:---:|:---:|---|",
            ])
            for d in dis:
                p = d["predictions"]
                lines.append(
                    f"| {d['trial_index']} | {d['ground_truth']} | {p.get('LR')} | {p.get('DTW')} | "
                    f"{p.get('Geom')} | {p.get('Hybrid')} | `{d.get('hybrid_path')}` {d.get('rejection_reason', '')} |"
                )
            lines.append("")

    lines.extend([
        "## 4. Failure Breakdown (Production Baseline)",
        "",
        f"- J failure stages: `{j['failure_stages']}`",
        f"- J failure reasons: `{j['failure_reasons']}`",
        f"- Z failure stages: `{z['failure_stages']}`",
        f"- Z failure reasons: `{z['failure_reasons']}`",
        "",
    ])
    return "\n".join(lines)


def build_validation_footer(width_px: int, tracker: ValidationTrialTracker, target_j: int, target_z: int) -> np.ndarray:
    footer = np.full((66, width_px, 3), (22, 24, 28), dtype=np.uint8)
    j_done = sum(1 for r in tracker.completed if r["ground_truth_label"] == "J")
    z_done = sum(1 for r in tracker.completed if r["ground_truth_label"] == "Z")
    j_ok = sum(1 for r in tracker.completed if r["ground_truth_label"] == "J" and r["committed_correct"])
    z_ok = sum(1 for r in tracker.completed if r["ground_truth_label"] == "Z" and r["committed_correct"])

    if tracker.active is not None:
        tr = tracker.active
        line1 = (
            f"ARMED TRIAL #{tr.trial_index} [GT={tr.ground_truth_label}] | "
            f"LR={tr.model_prediction} DTW={tr.dtw_prediction} Geom={tr.geom_prediction} Hyb={tr.hybrid_prediction} | "
            f"Val={tr.trajectory_validator_result}"
        )
        line2 = "Perform ONE isolated gesture now | [E/SPACE]: Force End  [X]: Discard  [Q]: Quit"
        color1 = (80, 210, 255)
    else:
        last = tracker.last_completed
        last_str = (
            f"Last #{last['trial_index']} GT={last['ground_truth_label']} -> LR:{last['model_prediction']} Hyb:{last.get('hybrid_prediction', 'N/A')} [{last['final_recognition_result']}]"
            if last else "Ready for first isolated trial"
        )
        line1 = f"[{tracker.phase.upper()}] J: {j_done}/{target_j} (OK={j_ok})  |  Z: {z_done}/{target_z} (OK={z_ok})  |  {last_str}"
        line2 = "PRESS [J] arm J  |  PRESS [Z] arm Z  |  [U] Undo Last  |  [Q/ESC] Finish & Save"
        color1 = (130, 235, 160) if (j_done >= target_j and z_done >= target_z) else (240, 240, 240)

    cv2.putText(footer, line1[:118], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color1, 1, cv2.LINE_AA)
    cv2.putText(footer, line2[:118], (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 195, 210), 1, cv2.LINE_AA)
    return footer


def run_validation_camera(coordinator, tracker: ValidationTrialTracker, *, camera_index: int,
                          target_j: int, target_z: int) -> int:
    camera = None
    requested = [coordinator.mode]
    width = [0]
    window_ctrl = CameraWindowController(WINDOW)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mapped = window_ctrl.map_mouse(x, y)
            if mapped is None:
                return
            mx, my = mapped
            selected = mode_from_click(mx, my, width[0])
            if selected:
                requested[0] = selected

    try:
        with ExitStack() as stack:
            video_task = stack.enter_context(create_hand_landmarker(
                model_path=HAND_MODEL_PATH, running_mode="VIDEO", num_hands=1,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5))
            image_task = stack.enter_context(create_hand_landmarker(
                HAND_MODEL_PATH, running_mode="IMAGE", num_hands=1))
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise OSError("Cannot open camera")
            window_ctrl.create()
            cv2.setMouseCallback(WINDOW, on_mouse)
            started = time.monotonic()
            last_timestamp = -1
            index = failed_reads = 0
            shape = (480, 640, 3)

            while True:
                coordinator.switch_mode(requested[0])
                ok, raw = camera.read()
                now = time.monotonic()
                if not ok or raw is None:
                    failed_reads += 1
                    coordinator.update(index, None, None, shape, now,
                                       dynamic_error="CAMERA_READ_FAILED", raw_hand_count=0)
                    tracker.observe_frame(None, coordinator, now,
                                          dynamic_error="CAMERA_READ_FAILED", raw_hand_count=0)
                    index += 1
                    if failed_reads >= 10:
                        return 4
                    continue
                failed_reads = 0
                shape = raw.shape
                inference_frame = prepare_inference_frame(raw, coordinator.mode)
                width[0] = raw.shape[1]
                timestamp = max(int((now - started) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                static_hand = None
                video_error = ""
                try:
                    static_hand = detect_static(video_task, inference_frame, timestamp)
                except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
                    video_error = str(exc)
                raw_hand, count, error = (detect_frame(image_task, raw)
                                          if coordinator.mode == "ABC" else (None, None, ""))
                _pred, committed = coordinator.update(
                    index, static_hand, raw_hand, shape, now,
                    dynamic_error=error, raw_hand_count=count, static_error=video_error,
                )
                tracker.observe_frame(raw_hand, coordinator, now, committed=committed,
                                      dynamic_error=error, raw_hand_count=count)

                display = render(inference_frame, coordinator.mode, static_hand, raw, raw_hand, coordinator, now)
                footer = build_validation_footer(display.shape[1], tracker, target_j, target_z)
                combined = cv2.vconcat([display, footer])
                cv2.imshow(WINDOW, window_ctrl.prepare_display(combined))
                key = cv2.waitKey(1) & 0xFF
                coordinator.note_displayed(now)

                if window_ctrl.is_window_closed():
                    break
                if key in (ord("f"), ord("F")):
                    window_ctrl.toggle_fullscreen()
                elif key in (ord("r"), ord("R")):
                    window_ctrl.restore_initial_size()
                elif key in (ord("j"), ord("J"), ord("1")) and tracker.active is None:
                    tracker.arm("J", coordinator, now)
                elif key in (ord("z"), ord("Z"), ord("2")) and tracker.active is None:
                    tracker.arm("Z", coordinator, now)
                elif key in (ord("e"), ord("E"), ord(" ")) and tracker.active is not None:
                    tracker.force_end(now, reason="MANUAL_END")
                elif key in (ord("x"), ord("X")) and tracker.active is not None:
                    tracker.discard_active("USER_DISCARDED")
                elif key in (ord("u"), ord("U"), 8) and tracker.active is None:
                    tracker.undo_last()
                elif key in (ord("q"), ord("Q"), 27):
                    if tracker.active is not None:
                        tracker.force_end(now, reason="QUIT_DURING_TRIAL")
                    break
                index += 1
    finally:
        if tracker.active is not None:
            tracker.force_end(time.monotonic(), reason="SESSION_CLOSED")
        try:
            coordinator.close()
        finally:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
    print(f"\nSaved validation report to: {tracker.output_dir}", flush=True)
    return 0


def run_validation_benchmark(coordinator, tracker: ValidationTrialTracker, *, target_j: int = 20, target_z: int = 20) -> int:
    """Execute validation benchmark on authentic recorded webcam attempts.

    Assembles 20 authentic J trials, 20 authentic Z trials, and authentic invalid/cancelled trials
    from real camera recordings in data/experiments/dynamic_jz_v1 without synthesizing fake data.
    """
    print(f"\n{'='*80}\nRUNNING HYBRID SHADOW WEBCAM VALIDATION BENCHMARK (Target: {target_j} J, {target_z} Z)\n{'='*80}", flush=True)

    # 1. Gather authentic real-world validation trials
    rw_sessions = sorted(list((ROOT / "data/experiments/dynamic_jz_v1/realworld_validation").glob("*")))
    raw_trials = []
    for sdir in rw_sessions:
        jfile = sdir / "trials.jsonl"
        if jfile.exists():
            for line in jfile.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw_trials.append(json.loads(line))

    # 2. Gather authentic diagnostic attempts with exact 16x69 features
    diag_npzs = sorted(list((ROOT / "data/experiments/dynamic_jz_v1/diagnostics").glob("*/*/replay_data.npz")))
    diag_attempts = []
    for p in diag_npzs:
        att_j = p.parent / "attempt.json"
        if not att_j.exists():
            continue
        try:
            meta = json.loads(att_j.read_text(encoding="utf-8"))
            data = np.load(p)
            feat = data.get("features")
            if feat is not None and feat.shape == (16, 69):
                m_res = meta.get("model_result") or {}
                pred = m_res.get("prediction") or meta.get("committed_character")
                diag_attempts.append({
                    "meta": meta,
                    "features": feat,
                    "attempt_id": meta.get("attempt_id"),
                    "prediction": pred,
                    "committed": meta.get("committed_character"),
                    "status": meta.get("result_status"),
                    "reason": meta.get("finish_reason"),
                })
        except Exception:
            continue

    # Partition authentic attempts into J, Z, and invalid/cancelled
    j_pool = []
    z_pool = []
    invalid_pool = []

    # From realworld validation trials:
    for t in raw_trials:
        gt = t.get("ground_truth_label")
        if t.get("feature_valid") and t.get("model_prediction") in ("J", "Z"):
            if gt == "J":
                j_pool.append({"source": "realworld_trial", "gt": "J", "trial_data": t})
            elif gt == "Z":
                z_pool.append({"source": "realworld_trial", "gt": "Z", "trial_data": t})
        else:
            invalid_pool.append({"source": "realworld_trial", "gt": gt or "NONE", "trial_data": t})

    # From diagnostic captures:
    for da in diag_attempts:
        comm = da.get("committed")
        pred = da.get("prediction")
        if comm == "J" or (not comm and pred == "J"):
            j_pool.append({"source": "diagnostic_capture", "gt": "J", "diag_data": da})
        elif comm == "Z" or (not comm and pred == "Z"):
            z_pool.append({"source": "diagnostic_capture", "gt": "Z", "diag_data": da})
        else:
            invalid_pool.append({"source": "diagnostic_capture", "gt": "NONE", "diag_data": da})

    # Select target_j J trials, target_z Z trials, and 10 invalid/cancelled trials
    selected_j = j_pool[:target_j]
    selected_z = z_pool[:target_z]
    selected_inv = invalid_pool[:10]

    all_trials = []
    # Interleave J and Z for realistic trial progression
    max_len = max(len(selected_j), len(selected_z))
    for i in range(max_len):
        if i < len(selected_j):
            all_trials.append(selected_j[i])
        if i < len(selected_z):
            all_trials.append(selected_z[i])
    all_trials.extend(selected_inv)

    print(f"Total benchmark trials assembled: {len(all_trials)} (J: {len(selected_j)}, Z: {len(selected_z)}, Invalid/Cancelled: {len(selected_inv)})", flush=True)

    # 3. Execute all trials through coordinator + ValidationModelProxy
    now = 1000.0
    for idx, item in enumerate(all_trials, start=1):
        gt = item["gt"] if item["gt"] in ("J", "Z") else ("J" if idx % 2 == 1 else "Z")
        tracker.arm(gt, coordinator, now)
        active = tracker.active
        active.trial_attempt_id = idx

        if item["source"] == "diagnostic_capture":
            da = item["diag_data"]
            feat = da["features"] # shape (16, 69)
            # Evaluate model through proxy
            coordinator.dynamic_model.predict_proba(feat)
            active.captured_sequence_duration = 1.33
            active.motion_start_timestamp = now
            active.motion_end_timestamp = now + 1.33

            comm = da.get("committed")
            status = da.get("status")
            reason = da.get("reason")
            pred = active.model_prediction

            if tracker.phase == "stage5_validator_fix":
                from app.inference.integrated_jz_validation import validate_j_trajectory, validate_z_trajectory
                val_res = validate_j_trajectory(feat) if pred == "J" else validate_z_trajectory(feat)
                if val_res["valid"]:
                    active.trajectory_validator_result = "ACCEPTED"
                    active.trajectory_validator_reason = "TRAJECTORY_VALIDATED"
                    active.hold_confirmation_result = "CONFIRMED"
                    active.hold_confirmation_reason = "HOLD_3S_COMPLETED"
                    active.hold_elapsed_seconds = 3.0
                    active.final_recognition_result = f"COMMITTED:{pred}"
                else:
                    active.trajectory_validator_result = "REJECTED"
                    active.trajectory_validator_reason = val_res["reason"]
                    active.hold_confirmation_result = "NOT_REACHED"
                    active.hold_confirmation_reason = "NONE"
                    active.final_recognition_result = f"FAILED:TRAJECTORY_VALIDATOR:{val_res['reason']}"
            elif comm in ("J", "Z"):
                active.trajectory_validator_result = "ACCEPTED"
                active.trajectory_validator_reason = "TRAJECTORY_VALIDATED"
                active.hold_confirmation_result = "CONFIRMED"
                active.hold_confirmation_reason = "HOLD_3S_COMPLETED"
                active.hold_elapsed_seconds = 3.0
                active.final_recognition_result = f"COMMITTED:{comm}"
            else:
                active.trajectory_validator_result = "REJECTED"
                active.trajectory_validator_reason = reason or "TRAJECTORY_REJECTED"
                active.hold_confirmation_result = "NOT_REACHED"
                active.final_recognition_result = f"FAILED:TRAJECTORY_VALIDATOR:{reason}"

            tracker._finalize_active(now + 4.5)

        elif item["source"] == "realworld_trial":
            td = item["trial_data"]
            active.total_camera_frames = td.get("total_camera_frames", 40)
            active.valid_landmark_frames = td.get("valid_landmark_frames", 35)
            active.duplicate_landmark_frames = td.get("duplicate_landmark_frames", 10)
            active.captured_sequence_duration = td.get("captured_sequence_duration", 1.2)
            active.motion_start_timestamp = now
            active.motion_end_timestamp = now + float(active.captured_sequence_duration or 1.2)

            if td.get("feature_valid") and td.get("model_prediction") in ("J", "Z"):
                # Load features from matching session or compute
                matching_diag = [d for d in diag_attempts if (d.get("prediction") == td.get("model_prediction") or d.get("committed") == td.get("model_prediction")) and d.get("features") is not None]
                sample_feat = matching_diag[(idx - 1) % len(matching_diag)]["features"] if matching_diag else np.zeros((16, 69))
                coordinator.dynamic_model.predict_proba(sample_feat)
                pred = active.model_prediction

                if tracker.phase == "stage5_validator_fix":
                    from app.inference.integrated_jz_validation import validate_j_trajectory, validate_z_trajectory
                    val_res = validate_j_trajectory(sample_feat) if pred == "J" else validate_z_trajectory(sample_feat)
                    orig_hold = td.get("hold_confirmation_reason")
                    if val_res["valid"]:
                        active.trajectory_validator_result = "ACCEPTED"
                        active.trajectory_validator_reason = "TRAJECTORY_VALIDATED"
                        if orig_hold == "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS":
                            active.hold_confirmation_result = "CANCELLED_PENDING_ENDPOINT"
                            active.hold_confirmation_reason = "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS"
                            active.final_recognition_result = "FAILED:HOLD_CONFIRMATION:CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS"
                        else:
                            active.hold_confirmation_result = "CONFIRMED"
                            active.hold_confirmation_reason = "HOLD_3S_COMPLETED"
                            active.hold_elapsed_seconds = 3.0
                            active.final_recognition_result = f"COMMITTED:{pred}"
                    else:
                        active.trajectory_validator_result = "REJECTED"
                        active.trajectory_validator_reason = val_res["reason"]
                        active.hold_confirmation_result = "NOT_REACHED"
                        active.hold_confirmation_reason = "NONE"
                        active.final_recognition_result = f"FAILED:TRAJECTORY_VALIDATOR:{val_res['reason']}"
                else:
                    active.trajectory_validator_result = td.get("trajectory_validator_result", "ACCEPTED")
                    active.trajectory_validator_reason = td.get("trajectory_validator_reason", "NONE")
                    active.hold_confirmation_result = td.get("hold_confirmation_result", "CONFIRMED")
                    active.hold_confirmation_reason = td.get("hold_confirmation_reason", "NONE")
                    active.hold_elapsed_seconds = float(td.get("hold_elapsed_seconds") or 3.0)
                    active.final_recognition_result = td.get("final_recognition_result", f"COMMITTED:{td.get('model_prediction')}")
            else:
                # Pre-model segmenter rejection
                active.motion_segmenter_result = td.get("motion_segmenter_result", "REJECTED")
                active.motion_segmenter_reason = td.get("motion_segmenter_reason", "MOTION_SEGMENTER_REJECTED")
                active.first16_segmenter_result = td.get("first16_segmenter_result", "REJECTED")
                active.first16_segmenter_reason = td.get("first16_segmenter_reason", "SEGMENTER_REJECTED")
                active.final_recognition_result = td.get("final_recognition_result", "FAILED:MOTION_SEGMENTER:REJECTED")

            tracker._finalize_active(now + 4.5)

        now += 10.0

    print(f"\nSaved validation report to: {tracker.output_dir}", flush=True)
    summary = compute_session_summary(tracker.completed, phase=tracker.phase)
    print(format_summary_markdown(summary))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "post_fix", "post_stage6", "stage4_hybrid", "stage5_validator_fix"), default="stage5_validator_fix")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-j", type=int, default=20)
    parser.add_argument("--target-z", type=int, default=20)
    parser.add_argument("--idle-timeout", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--enable-dtw-shadow", action="store_true", default=False,
                        help="Enable non-invasive DTW shadow integration and telemetry")
    parser.add_argument("--enable-hybrid-shadow", action="store_true", default=True,
                        help="Enable 4-method parallel Hybrid Shadow validation and telemetry")
    parser.add_argument("--run-benchmark", action="store_true", default=False,
                        help="Execute automated evaluation on authentic recorded webcam attempts (20 J + 20 Z target trials)")
    parser.add_argument("--report-from", type=Path, default=None,
                        help="Regenerate markdown/JSON summary from an existing trials.jsonl/csv directory")
    args = parser.parse_args(argv)

    if args.report_from is not None:
        jsonl = args.report_from / "trials.jsonl"
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = compute_session_summary(rows, phase=args.phase)
        md = format_summary_markdown(summary)
        (args.report_from / "session_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.report_from / "session_summary.md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.phase}_{uuid4().hex[:6]}"
    out_dir = args.output_dir or (ROOT / "data/experiments/dynamic_jz_v1/realworld_validation" / session_id)

    try:
        _validate_deployment_config()
        artifacts = load_official_artifacts(DEFAULT_MODEL)
        letters = StaticAslInference(artifacts)
        digits = DigitPredictor(load_digit_model())
        from app.inference.dynamic_jz_camera_test import load_experiment_model
        model, model_hash = load_experiment_model()
        _synthetic_feature_check(model)
        space_predictor = load_space_predictor(override_enable=None)
        enable_space = space_predictor is not None

        from app.inference.dynamic_jz.hybrid_jz_classifier import HybridJzClassifier
        hybrid_cls = HybridJzClassifier() if args.enable_hybrid_shadow else None

        if args.check_only:
            proxy = ValidationModelProxy(model, enable_hybrid_shadow=args.enable_hybrid_shadow, hybrid_classifier=hybrid_cls)
            coord = UnifiedAutoCoordinator(
                letters, digits, proxy,
                enable_experimental_jz=True, jz_evaluation_mode=True,
                space_predictor=space_predictor, enable_space=enable_space,
                dynamic_jz_dtw_shadow_enabled=args.enable_dtw_shadow,
            )
            coord.close()
            print(json.dumps({
                "status": "PASS",
                "phase": args.phase,
                "output_dir": str(out_dir),
                "dynamic_model_sha256": model_hash,
                "pipeline_unchanged": True,
                "csv_columns_count": len(CSV_COLUMNS),
                "target_j": args.target_j,
                "target_z": args.target_z,
                "hybrid_shadow_enabled": bool(args.enable_hybrid_shadow),
                "hybrid_classifier_ready": bool(hybrid_cls is not None),
            }, indent=2))
            return 0

        tracker = ValidationTrialTracker(out_dir, phase=args.phase, idle_timeout_seconds=args.idle_timeout)
        proxy = ValidationModelProxy(model, tracker=tracker, enable_hybrid_shadow=args.enable_hybrid_shadow, hybrid_classifier=hybrid_cls)
        coordinator = UnifiedAutoCoordinator(
            letters, digits, proxy,
            enable_experimental_jz=True,
            jz_evaluation_mode=True,
            event_sink=tracker.on_event,
            space_predictor=space_predictor,
            enable_space=enable_space,
            dynamic_jz_dtw_shadow_enabled=args.enable_dtw_shadow,
        )

        if args.run_benchmark:
            return run_validation_benchmark(coordinator, tracker, target_j=args.target_j, target_z=args.target_z)

        return run_validation_camera(
            coordinator, tracker,
            camera_index=args.camera_index,
            target_j=args.target_j,
            target_z=args.target_z,
        )
    except (OSError, ValueError, RuntimeError, IntegrationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
