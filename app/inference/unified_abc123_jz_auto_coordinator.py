"""Experimental arbitration around the LATEST AutoJzSession and unchanged HOLD.

No camera, model loading or writes at import. Binary J/Z probabilities are never
compared to static probabilities, nor treated as evidence of gesture validity.
"""
from types import SimpleNamespace

import numpy as np

from ai.feature_extraction.hand_landmarks import landmarks_to_array
from ai.preprocessing.landmark_utils import normalize_landmarks
from app.inference.dynamic_jz_auto_camera_experiment import AutoJzSession
from app.inference.integrated_jz_validation import validate_proposal
from app.inference.push_confirmation_camera_test import observe_hand
from app.inference.unified_static_coordinator import RecognitionCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from app.inference.z_start_pose_experimental import ZStartPoseShadow


class UnifiedAutoCoordinator:
    def __init__(self, letters, digits, dynamic_model, *, enable_experimental_jz=False, event_sink=None,
                 diagnostics=None, z_start_pose_shadow=False, space_predictor=None, enable_space=False,
                 suggestion_manager=None, jz_evaluation_mode=False, dynamic_jz_dtw_shadow_enabled=None,
                 enable_jz_reject_option=True, jz_margin_threshold=0.10):
        self.static = RecognitionCoordinator(
            letters, digits, HoldSettings(blip_grace_frames=2, blip_grace_seconds=0.15),
            space_predictor=space_predictor, enable_space=enable_space,
        )
        self.diagnostics = diagnostics
        self.suggestion_manager = suggestion_manager
        self.suggestion_hitboxes = []
        self.sentence_hitboxes = []
        if self.suggestion_manager is not None:
            self.composer.on_change = self.suggestion_manager.update_text
            self.suggestion_manager.set_mode(self.mode)
            self.suggestion_manager.update_text(self.composer.text)
        self.z_start_pose_shadow = ZStartPoseShadow() if z_start_pose_shadow else None
        self.z_start_pose_report = None
        self.z_start_pose_error = None
        from app.inference.dynamic_jz.shadow_adapter import JzShadowAdapter, ShadowModelProxy
        self.shadow_adapter = JzShadowAdapter(enabled=dynamic_jz_dtw_shadow_enabled)
        if self.shadow_adapter.enabled:
            dynamic_model = ShadowModelProxy(dynamic_model, self.shadow_adapter)
        if diagnostics is not None:
            from app.inference.jz_attempt_capture import DiagnosticModelProxy
            dynamic_model = DiagnosticModelProxy(dynamic_model,diagnostics)
        self.enable_jz_reject_option = bool(enable_jz_reject_option)
        self.jz_margin_threshold = float(jz_margin_threshold)
        if enable_jz_reject_option:
            from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
            dynamic_model = ProbabilityMarginModelWrapper(
                dynamic_model,
                enable_reject_option=enable_jz_reject_option,
                margin_threshold=jz_margin_threshold,
            )
        self.dynamic_model = dynamic_model
        self.enable_experimental_jz = enable_experimental_jz
        self.event_sink = event_sink
        self.session = AutoJzSession(dynamic_model,event_sink=self.session_event if diagnostics else event_sink)
        self.candidate = None
        self.candidate_at = None
        self.candidate_settled = False
        self.endpoint_observation = None
        self.jz_hold = HoldConfirmation(HoldSettings())
        self.static_prediction = None
        self.status = "STATIC"
        self.reason = ""
        self.blocked = False
        self.seen_result = None
        self.post_jz_latch = False
        self.confirmed_raw_shape = None
        self.release_since = None
        self.new_pose_frames = 0
        self.frame_context = {}
        self.raw_dynamic_prediction = None
        self.raw_prediction_at = None
        self.pending_dynamic = None
        self.proposal_observation = None
        self.candidate_attempt_id = None
        self.validation = None
        self.capture_attempt_id = None
        self.capture_rows = []
        self.capture_scale = None
        self.session_generation = 0
        self.displayed_attempt_id = None
        self.last_onset_state = "READY"
        self.consumed_dynamic_attempts = set()
        self.last_confirmed_dynamic_attempt_id = None
        self.last_confirmed_dynamic_generation = None
        self.last_confirmed_dynamic_at = None
        self.candidate_created_at = None
        self.min_dynamic_j_confidence = 0.70
        self.min_dynamic_z_confidence = 0.75
        self.tracking_grace_frame_count = 0
        self.tracking_grace_used = False
        self.candidate_cancel_reason = None
        self.endpoint_mismatch_frames = 0
        self.max_endpoint_mismatch_tolerance = 3
        self.jz_evaluation_mode = bool(jz_evaluation_mode)
        self.jz_evaluation_record = None
        self.jz_evaluation_at = None
        self.jz_evaluation_ttl_seconds = 5.0

    def attempt_consumed(self, attempt_id):
        return (self.session_generation,attempt_id) in self.consumed_dynamic_attempts

    @staticmethod
    def _proposal_trajectory_valid(prediction_label, validation):
        if not validation or validation.get("status") != "PASS":
            return False
        if prediction_label == "Z":
            return bool(validation.get("z_trajectory_valid", False))
        if prediction_label == "J":
            return bool(validation.get("j_trajectory_valid", False))
        return False

    @property
    def fresh_validated_segment(self):
        is_fresh = bool(self.candidate and self.candidate_attempt_id is not None
                        and not self.attempt_consumed(self.candidate_attempt_id)
                        and self.validation and self.validation.get("status") == "PASS"
                        and self.candidate_settled and self.endpoint_observation is not None)
        if is_fresh:
            return self._proposal_trajectory_valid((self.candidate or {}).get("prediction"), self.validation)
        return is_fresh

    @property
    def display_priority_reason(self):
        if self.fresh_validated_segment:
            return "FRESH_VALIDATED_DYNAMIC_CANDIDATE"
        return "STATIC_PREDICTION" if self.static_prediction else "NO_CURRENT_PREDICTION"

    @property
    def mode(self):
        return self.static.mode

    @property
    def composer(self):
        return self.static.composer

    @property
    def hold(self):
        machine = self.jz_hold if self.fresh_validated_segment or self.post_jz_latch else self.static.hold
        machine.text = self.composer.text
        return machine

    @property
    def display_label(self):
        if self.fresh_validated_segment:
            return self.candidate["prediction"]
        # A previously committed J/Z is history, not the current prediction.
        prediction = self.static_prediction
        return (prediction.stable_label or prediction.instant_label) if prediction else None

    @property
    def display_dynamic_result(self):
        if self.fresh_validated_segment:
            return self.candidate
        return None

    @property
    def suggestions(self) -> list[str]:
        if self.suggestion_manager is not None:
            return self.suggestion_manager.get_suggestions()
        return []

    @property
    def sentence_suggestions(self) -> list[str]:
        if self.suggestion_manager is not None and hasattr(self.suggestion_manager, "get_sentence_suggestions"):
            return self.suggestion_manager.get_sentence_suggestions()
        return []

    def session_event(self,event):
        if self.diagnostics is not None:
            self.diagnostics.safe(self.diagnostics.event,event,self.session_generation)
        if self.event_sink:
            self.event_sink(event)

    def emit(self,event,now,**details):
        if self.event_sink or self.diagnostics is not None:
            payload = {"event":event,"timestamp_monotonic":now,**self.frame_context,
                             "attempt_id": (self.candidate_attempt_id if self.candidate or self.pending_dynamic
                                            else self.capture_attempt_id),
                             "current_attempt_id": self.capture_attempt_id,
                             "candidate_source_attempt_id": self.candidate_attempt_id,
                             "candidate_created_time": self.candidate_created_at,
                             "last_confirmed_dynamic_attempt_id": self.last_confirmed_dynamic_attempt_id,
                             "last_confirmed_dynamic_generation": self.last_confirmed_dynamic_generation,
                             "last_confirmed_dynamic_at": self.last_confirmed_dynamic_at,
                             "dynamic_segmenter_state": self.session.last_detector_state,
                             "proposal_candidate_state": ("VALIDATED_CANDIDATE" if self.candidate else
                                  "ENDPOINT_PENDING" if self.pending_dynamic else "EMPTY"),
                             "post_jz_latch": self.post_jz_latch,
                             "hold_state": self.hold.state,
                             "dynamic_hold_state": self.jz_hold.state,
                             "display_priority_reason": self.display_priority_reason,
                             "fresh_validated_dynamic_segment": self.fresh_validated_segment,
                             "session_generation": self.session_generation,
                             "mode":self.mode,"status":self.status,"reason":self.reason,
                             "candidate":self.candidate["prediction"] if self.candidate else None,
                             "raw_prediction": (self.raw_dynamic_prediction or {}).get("prediction"),
                             "raw_prediction_age_seconds": (now-self.raw_prediction_at
                                 if self.raw_prediction_at is not None else None),
                             "static_instant": getattr(self.static_prediction,"instant_label",None),
                             "static_stable": getattr(self.static_prediction,"stable_label",None),
                             **details}
            if self.z_start_pose_shadow is not None:
                payload["z_start_pose_shadow"]=self.z_start_pose_report
                payload["z_start_pose_shadow_error"]=self.z_start_pose_error
            if self.diagnostics is not None:
                self.diagnostics.safe(self.diagnostics.event,payload,self.session_generation)
            if self.event_sink:
                self.event_sink(payload)

    def reset_static(self):
        self.static.predictors[self.mode].reset()
        self.static.hold = HoldConfirmation(self.static.hold_settings)
        self.static.hold.text = self.composer.text

    def set_blocked(self,value):
        if value != self.blocked:
            # Only reset static state when entering suppression from a verified dynamic block or post-confirm latch.
            if value and (self.post_jz_latch or (self.enable_experimental_jz and (self.candidate or self.pending_dynamic))):
                self.reset_static()
            self.blocked = value
            self.emit("STATIC_SUPPRESSION_CHANGED",self.frame_context.get("timestamp",0.),
                      suppressed=value, suppression_reason=("MOTION_INTERLOCK_OR_VALIDATED_CANDIDATE"
                                                            if value else "STATIC_RESUMED"))

    def _sync_tracking_telemetry(self):
        self.frame_context.update({
            "pending_dynamic": bool(self.pending_dynamic is not None),
            "candidate_settled": bool(self.candidate_settled),
            "image_drop_streak": self.tracking_grace_frame_count,
            "tracking_grace_used": self.tracking_grace_used,
            "tracking_grace_frame_count": self.tracking_grace_frame_count,
            "candidate_cancel_reason": self.candidate_cancel_reason,
            "tracking_cancel_reason": self.candidate_cancel_reason,
        })

    def _log_shadow_telemetry(self, now, *, attempt_id, stage, model_pred=None,
                              final_output="NONE", committed=False):
        if not (self.shadow_adapter and self.shadow_adapter.enabled):
            return
        shadow_res = getattr(self.shadow_adapter, "latest_result", None)
        if shadow_res is None:
            proxy = getattr(self, "dynamic_model", None)
            shadow_res = getattr(proxy, "latest_shadow_result", None)
        if shadow_res is None:
            return
        pred_label = (model_pred or {}).get("prediction") or "NONE"
        prob = (model_pred or {}).get(f"probability_{pred_label}") if pred_label != "NONE" else None

        commit_status = (
            "COMMITTED" if committed
            else ("REJECTED" if stage in ("VALIDATOR_REJECTED", "CONFIDENCE_REJECTED", "SEGMENTER_REJECTED", "UNVERIFIED")
                  else ("CANCELLED" if stage == "CANCELLED" else "PENDING"))
        )
        agreement = bool(pred_label == shadow_res.prediction and pred_label in ("J", "Z"))
        target_label = getattr(self, "current_harness_target_label", "NONE")

        from datetime import datetime, timezone
        from app.inference.dynamic_jz.shadow_adapter import JzShadowTelemetryRecord
        record = JzShadowTelemetryRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            attempt_id=attempt_id,
            target_label=target_label,
            logistic_prediction=pred_label,
            logistic_confidence=float(prob) if prob is not None else None,
            dtw_prediction=shadow_res.prediction,
            dtw_best_distance=shadow_res.best_distance,
            dtw_second_distance=shadow_res.second_best_distance,
            dtw_margin=shadow_res.margin,
            dtw_status=shadow_res.status,
            tracking_quality=shadow_res.tracking_quality,
            feature_config=shadow_res.config_id,
            template_id=shadow_res.template_id,
            dtw_latency_ms=shadow_res.latency_ms,
            agreement=agreement,
            final_committed_label=final_output if committed else "NONE",
            final_commit_status=commit_status,
        )
        self.shadow_adapter.log_telemetry(record)
        self.emit("JZ_SHADOW_TELEMETRY", now, shadow_telemetry=record.to_dict())
        if stage in ("UNVERIFIED", "VALIDATOR_REJECTED", "CONFIDENCE_REJECTED", "SEGMENTER_REJECTED", "CONFIRMED", "CANCELLED"):
            if self.shadow_adapter:
                self.shadow_adapter.latest_result = None
            if hasattr(self.dynamic_model, "latest_shadow_result"):
                self.dynamic_model.latest_shadow_result = None

    def _record_jz_evaluation(self, now, *, attempt_id, stage, model_pred=None,
                              validator_status="SKIPPED", reason="NONE",
                              final_output="NONE", committed=False):
        if self.shadow_adapter and self.shadow_adapter.enabled:
            self._log_shadow_telemetry(now, attempt_id=attempt_id, stage=stage,
                                       model_pred=model_pred, final_output=final_output,
                                       committed=committed)
        if not self.jz_evaluation_mode:
            return None
        pred_label = (model_pred or {}).get("prediction")
        prob_j = (model_pred or {}).get("probability_J")
        prob_z = (model_pred or {}).get("probability_Z")
        confidence = (model_pred or {}).get(f"probability_{pred_label}") if pred_label else None
        record = {
            "attempt_id": attempt_id,
            "stage": stage,
            "raw_prediction": pred_label,
            "probability_J": float(prob_j) if prob_j is not None else None,
            "probability_Z": float(prob_z) if prob_z is not None else None,
            "confidence": float(confidence) if confidence is not None else None,
            "validator_status": validator_status,
            "reason": reason or "NONE",
            "final_output": final_output,
            "committed": bool(committed),
            "timestamp": float(now),
        }
        self.jz_evaluation_record = record
        self.jz_evaluation_at = float(now)
        if pred_label is not None and record["probability_J"] is not None and record["probability_Z"] is not None:
            print(
                f"[JZ_EVAL] attempt={attempt_id} raw={pred_label} "
                f"P(J)={record['probability_J']:.3f} P(Z)={record['probability_Z']:.3f} "
                f"validator={validator_status} reason={record['reason']} final={final_output}",
                flush=True,
            )
        else:
            print(
                f"[JZ_EVAL] attempt={attempt_id} stage={stage} raw=NONE "
                f"validator={validator_status} reason={record['reason']} final={final_output}",
                flush=True,
            )
        self.emit("JZ_EVALUATION_RECORD", now, evaluation_record=dict(record))
        return record

    def _update_jz_evaluation_outcome(self, reason, now):
        if not self.jz_evaluation_mode or self.jz_evaluation_record is None:
            return
        record = self.jz_evaluation_record
        if self.candidate_attempt_id is None or record.get("attempt_id") != self.candidate_attempt_id:
            return
        if reason == "CONFIRMED_ATTEMPT_CONSUMED":
            record["stage"] = "CONFIRMED"
            record["final_output"] = record.get("raw_prediction") or "CONFIRMED"
            record["reason"] = "NONE"
            record["committed"] = True
            record["timestamp"] = float(now)
            self.jz_evaluation_at = float(now)
            self._log_shadow_telemetry(now, attempt_id=record['attempt_id'], stage="CONFIRMED",
                                       model_pred={"prediction": record.get("raw_prediction")},
                                       final_output=record['final_output'], committed=True)
            if self.shadow_adapter:
                self.shadow_adapter.latest_result = None
            print(
                f"[JZ_EVAL] attempt={record['attempt_id']} raw={record['raw_prediction']} "
                f"validator={record['validator_status']} reason=NONE final={record['final_output']} (COMMITTED)",
                flush=True,
            )
            self.emit("JZ_EVALUATION_RECORD", now, evaluation_record=dict(record))
        elif reason != "REPLACED_BY_NEW_BOUT" and not record.get("committed"):
            record["stage"] = "CANCELLED"
            record["final_output"] = "NONE"
            if record.get("validator_status") == "ACCEPTED":
                record["reason"] = reason
            record["timestamp"] = float(now)
            self.jz_evaluation_at = float(now)
            self._log_shadow_telemetry(now, attempt_id=record['attempt_id'], stage="CANCELLED",
                                       model_pred={"prediction": record.get("raw_prediction")},
                                       final_output="NONE", committed=False)
            if self.shadow_adapter:
                self.shadow_adapter.latest_result = None
            print(
                f"[JZ_EVAL] attempt={record['attempt_id']} raw={record['raw_prediction']} "
                f"validator={record['validator_status']} reason={record['reason']} final=NONE",
                flush=True,
            )
            self.emit("JZ_EVALUATION_RECORD", now, evaluation_record=dict(record))

    def clear_candidate(self,reason,now,*,emit_cancel=True):
        had_candidate = bool(self.candidate or self.pending_dynamic)
        self.candidate_cancel_reason = reason
        self.tracking_grace_frame_count = 0
        self.endpoint_mismatch_frames = 0
        self._update_jz_evaluation_outcome(reason, now)
        if emit_cancel and had_candidate:
            self.frame_context["tracking_cancel_reason"] = reason
            self.frame_context["candidate_cancel_reason"] = reason
            self.emit("JZ_CANDIDATE_CANCELLED",now,cancellation_reason=reason,
                      tracking_grace_used=self.tracking_grace_used,
                      tracking_cancel_reason=reason)
        self.candidate = None
        self.candidate_at = None
        self.candidate_created_at = None
        self.candidate_settled = False
        self.endpoint_observation = None
        self.pending_dynamic = None
        self.proposal_observation = None
        self.candidate_attempt_id = None
        self.raw_dynamic_prediction = None
        self.raw_prediction_at = None
        self.validation = None
        self.capture_rows = []
        self.capture_attempt_id = None
        self._sync_tracking_telemetry()
        if not self.post_jz_latch:
            self.jz_hold = HoldConfirmation(HoldSettings())
            self.jz_hold.text = self.composer.text
        if had_candidate and reason in ("CONFIRMED_ATTEMPT_CONSUMED", "MODE_CHANGED", "POST_CONFIRM_REARMED"):
            self.reset_static()
        if self.shadow_adapter and reason in ("CONFIRMED_ATTEMPT_CONSUMED", "MODE_CHANGED"):
            self.shadow_adapter.latest_result = None
        self.reason = reason

    def switch_mode(self,mode):
        if mode not in ("ABC","123"):
            raise ValueError("Only ABC and 123 are supported")
        if mode == self.mode:
            return
        if self.suggestion_manager is not None:
            self.suggestion_manager.set_mode(mode)
        if self.z_start_pose_shadow is not None:
            self.z_start_pose_shadow.reset(session_generation=self.session_generation+1)
            self.z_start_pose_report=None
        if self.diagnostics is not None:
            self.diagnostics.safe(self.diagnostics.finish,"MODE_CHANGED")
            self.diagnostics.history.clear()
        self.static.switch_mode(mode)
        self.session_generation += 1
        self.session = AutoJzSession(self.dynamic_model,event_sink=self.session_event if self.diagnostics else self.event_sink)
        self.clear_candidate("MODE_CHANGED",self.frame_context.get("timestamp",0.))
        self.seen_result = None
        self.post_jz_latch = False
        self.confirmed_raw_shape = None
        self.release_since = None
        self.new_pose_frames = 0
        self.blocked = False
        self.static_prediction = None
        self.status = "STATIC"
        self.capture_attempt_id = None
        self.capture_rows = []
        self.capture_scale = None
        self.last_onset_state = "READY"
        if self.shadow_adapter:
            self.shadow_adapter.latest_result = None
        if self.suggestion_manager is not None and mode == "ABC":
            self.suggestion_manager.update_text(self.composer.text)

    def raw_observation(self,hand,shape):
        prediction_result = self.candidate or self.pending_dynamic
        label = prediction_result["prediction"]
        confidence = prediction_result[f"probability_{label}"]
        prediction = SimpleNamespace(instant_label=label,stable_label=label,
            stable_confidence=confidence,features=tuple(normalize_landmarks(hand)))
        return observe_hand(hand,prediction,shape)

    def proposal_pose_matches(self, observation):
        # Position and projected palm scale can shift during final J-hook supination
        # or Z-stroke deceleration before quiet-end; allow a bounded unsettled
        # settling margin (< 0.50 scale, < 0.40 shape RMS) while keeping settled
        # endpoint_matches and 3.0s HOLD tolerances strict.
        anchor = self.proposal_observation
        settings = self.jz_hold.settings
        return bool(anchor is not None
                    and abs(observation.scale_px/anchor.scale_px-1.) <= settings.max_scale_change * 2.5
                    and np.sqrt(np.mean((np.asarray(observation.shape)-anchor.shape)**2)) <= settings.max_shape_rms * 1.4)

    def process_pending(self, now, raw_hand, shape):
        """Integrated-only guard; standalone/counterpart sessions stay unchanged."""
        attempt = self.session.attempts.active
        attempt_id = attempt["attempt_id"] if attempt and not attempt["finished"] else None
        if attempt_id != self.capture_attempt_id:
            self.capture_attempt_id = attempt_id
            self.capture_rows = []
            self.capture_scale = None
            if self.z_start_pose_shadow is not None:
                try:
                    if attempt_id is None:
                        self.z_start_pose_shadow.end_attempt()
                    else:
                        self.z_start_pose_shadow.begin_attempt(attempt_id,attempt["start_frame"],attempt["started_at"],
                                                              self.session.segmenter.movement_since)
                    self.z_start_pose_report=self.z_start_pose_shadow.snapshot()
                    self.emit("Z_TEMPORAL_READY_ASSOCIATION",now,
                              temporal_evidence=self.z_start_pose_report["temporal_evidence"])
                except Exception as exc:
                    self.z_start_pose_error=str(exc)  # Shadow cannot change recognition.
        segmenter = self.session.segmenter
        if raw_hand is not None and (segmenter.active or self.session.pending):
            points = landmarks_to_array(raw_hand)[:,:2]*np.asarray((shape[1],shape[0]))
            self.capture_rows.append((self.frame_context["frame_id"],now,points[[0,8,20]].copy()))
        elif not (segmenter.active or self.session.pending):
            self.capture_rows = []
            self.capture_scale = None
            self.capture_attempt_id = None
        if segmenter.active:
            self.capture_scale = segmenter.anchor[1]
        if self.last_onset_state != self.session.last_detector_state:
            self.last_onset_state = self.session.last_detector_state
            self.emit("INTEGRATED_MOTION_STAGE",now,stage=self.last_onset_state,
                      attempt_id=(self.session.attempts.counts["TOTAL_ATTEMPTS"]+1
                                  if self.last_onset_state == "SUSPECT" else self.capture_attempt_id),
                      evidence=dict(segmenter.motion_evidence))
        context = None
        if self.session.pending:
            entries, index, _endpoint, metrics = self.session.pending
            if (self.attempt_consumed(attempt_id) or (attempt and self.last_confirmed_dynamic_at is not None
                    and attempt["started_at"] <= self.last_confirmed_dynamic_at)):
                self.emit("JZ_COMPLETED_ATTEMPT_REUSE_BLOCKED",now,attempt_id=attempt_id,
                          reuse_reason="CONSUMED_OR_STARTED_BEFORE_LAST_COMMIT")
                self.reset_motion_cycle(now,"COMPLETED_ATTEMPT_REUSE_BLOCKED")
                return None
            validation = validate_proposal(entries,self.capture_rows,self.capture_scale,
                                           metrics,segmenter.motion_gate)
            if self.z_start_pose_shadow is not None:
                validation["z_start_pose_shadow"]=self.z_start_pose_report
                validation["z_start_pose_shadow_error"]=self.z_start_pose_error
            if self.diagnostics is not None:
                self.diagnostics.safe(self.diagnostics.bout,entries,self.capture_rows,self.capture_scale,metrics,validation)
            self.emit("JZ_SEGMENT_VALIDATION",now,validation=validation)
            if validation["status"] != "PASS":
                self.session.pending = None
                self.session._finish("REJECTED",index,now,validation["reason"],
                                     {"candidate_validation":validation})
            else:
                context = (entries[-1][1], validation, attempt_id)
        self.session.process_pending(now)
        return context

    def note_displayed(self, now):
        if self.fresh_validated_segment and self.displayed_attempt_id != (self.session_generation,self.candidate_attempt_id):
            self.displayed_attempt_id = (self.session_generation,self.candidate_attempt_id)
            self.emit("JZ_VALIDATED_CANDIDATE_DISPLAYED",now,label=self.candidate["prediction"])
            self.session.note_displayed(now)

    def reset_motion_cycle(self, now, reason):
        """Forget actionable motion/context, preserve model/settings/stats and text.

Called after commit and static-pose/release rearm, NOT a cooldown. A fresh
meaningful-motion rearm retains its new segment so repeated J/Z remain possible.
        """
        if self.z_start_pose_shadow is not None:
            self.z_start_pose_shadow.end_attempt()
            self.z_start_pose_report=self.z_start_pose_shadow.snapshot()
        attempt = self.session.attempts.active
        if attempt and not attempt["finished"]:
            self.session._finish("UNCERTAIN",self.frame_context.get("frame_id",-1),now,reason)
        self.session.segmenter.reset()
        self.session.pending = None
        self.session.last_result = None
        self.session.valid_frames = 0
        self.session.visible_until = -float("inf")
        self.session.last_detector_state = "READY"
        self.session._state("READY",self.frame_context.get("frame_id",-1),now,reason)
        self.seen_result = None
        self.capture_attempt_id = None
        self.capture_rows = []
        self.capture_scale = None
        self.last_onset_state = "READY"
        if self.shadow_adapter:
            self.shadow_adapter.latest_result = None

    def consume_dynamic_attempt(self, observation, now):
        source_id = self.candidate_attempt_id
        self.consumed_dynamic_attempts.add((self.session_generation,source_id))
        self.last_confirmed_dynamic_attempt_id = source_id
        self.last_confirmed_dynamic_generation = self.session_generation
        self.last_confirmed_dynamic_at = now
        self.post_jz_latch = True
        self.confirmed_raw_shape = np.asarray(observation.shape).copy()
        self.release_since = None
        self.new_pose_frames = 0
        # Keep only the CONFIRMED UI/history; no candidate, endpoint, result or
        # buffered motion from this attempt can start another HOLD.
        self.clear_candidate("CONFIRMED_ATTEMPT_CONSUMED",now,emit_cancel=False)
        self.reset_motion_cycle(now,"CONFIRMED_ATTEMPT_CONSUMED")
        return source_id

    def endpoint_matches(self,observation):
        anchor = self.endpoint_observation
        if anchor is None:
            return False
        settings = self.jz_hold.settings
        center_drift = float(np.linalg.norm(np.asarray(observation.palm_center_px)-anchor.palm_center_px)/anchor.scale_px)
        scale_change = float(abs(observation.scale_px/anchor.scale_px-1.))
        shape_rms = float(np.sqrt(np.mean((np.asarray(observation.shape)-anchor.shape)**2)))
        return bool(center_drift <= settings.max_center_drift_ratio
                    and scale_change <= settings.max_scale_change
                    and shape_rms <= settings.max_shape_rms)

    def update_latch(self,raw_hand,now,*,video_present,observations_valid):
        if not self.post_jz_latch:
            return
        if self.jz_hold.state == "CONFIRMED":
            self.jz_hold.state = "WAIT_RELEASE"  # No fake missing observation/hold-time accumulation.
        rearm_reason = None
        if raw_hand is None:
            self.new_pose_frames = 0
            if video_present or not observations_valid:
                # IMAGE loss alone is not proof of a physical hand release.
                self.release_since = None
                return
            if self.release_since is None:
                self.release_since = now
            if now-self.release_since >= self.jz_hold.settings.release_seconds:
                self.post_jz_latch = False
                rearm_reason = "VALID_PHYSICAL_RELEASE"
        else:
            self.release_since = None
            current = np.asarray(normalize_landmarks(raw_hand))
            changed = np.sqrt(np.mean((current-self.confirmed_raw_shape)**2))
            meaningful = self.session.segmenter.motion_evidence.get("supported",False)
            self.new_pose_frames = (self.new_pose_frames+1 if changed >= self.jz_hold.settings.min_new_pose_shape_rms else 0)
            if self.new_pose_frames >= self.jz_hold.settings.stable_frames:
                self.post_jz_latch = False
                rearm_reason = "NEW_POSE"
            elif meaningful:
                self.post_jz_latch = False
                rearm_reason = "FRESH_MEANINGFUL_MOTION"
        if not self.post_jz_latch:
            if rearm_reason != "FRESH_MEANINGFUL_MOTION":
                self.reset_motion_cycle(now,"POST_CONFIRM_POSE_OR_RELEASE_TRANSITION_DISCARDED")
            self.clear_candidate("POST_CONFIRM_REARMED",now)
            self.reset_static()
            self.reason = "NEW_POSE_MOTION_OR_VALID_RELEASE"
            self.emit("POST_JZ_REARMED",now,rearm_reason=rearm_reason)

    def update(self,index,static_hand,raw_hand,shape,now,*,dynamic_error="",raw_hand_count=None,static_error=""):
        if self.diagnostics is not None:
            self.diagnostics.safe(self.diagnostics.observe,index,raw_hand,(shape[1],shape[0]),now,
                                  hand_count=raw_hand_count,error=dynamic_error,mode=self.mode)
        try:
            return self._update(index,static_hand,raw_hand,shape,now,dynamic_error=dynamic_error,
                                raw_hand_count=raw_hand_count,static_error=static_error)
        finally:
            if self.diagnostics is not None:
                self.diagnostics.safe(self.diagnostics.after_update)

    def _update(self,index,static_hand,raw_hand,shape,now,*,dynamic_error="",raw_hand_count=None,static_error=""):
        """Independent raw IMAGE input; VIDEO absence never gates this update."""
        self.frame_context = {"frame_id":index,"timestamp":now,"video_hand_detected":static_hand is not None,
                              "image_hand_detected":raw_hand is not None,"image_hand_count":raw_hand_count,
                              "video_error":static_error,"image_error":dynamic_error,
                              "pending_dynamic":bool(self.pending_dynamic is not None),
                              "candidate_settled":bool(self.candidate_settled),
                              "image_drop_streak":self.tracking_grace_frame_count,
                              "tracking_grace_used":self.tracking_grace_used,
                              "tracking_grace_frame_count":self.tracking_grace_frame_count,
                              "candidate_cancel_reason":self.candidate_cancel_reason,
                              "tracking_cancel_reason":self.candidate_cancel_reason}
        self.static_prediction = None  # Do not label last frame's prediction as current in event trace.
        committed = None
        hold_started = False
        confirmed_source_id = None
        if self.mode == "ABC":
            # Validation errors are missing observations for the latest session,
            # not mirrored substitutes. Actual API errors retain its rejection.
            if raw_hand is not None:
                try:
                    landmarks_to_array(raw_hand)
                except (ValueError,TypeError,AttributeError) as exc:
                    raw_hand = None
                    dynamic_error = "INVALID_LANDMARK_FRAME: "+str(exc)
            self.frame_context["image_hand_detected"] = raw_hand is not None
            if self.z_start_pose_shadow is not None:
                try:
                    previous=self.z_start_pose_report or {}
                    report=self.z_start_pose_shadow.observe(index,raw_hand,(shape[1],shape[0]),now,
                                                           hand_count=raw_hand_count,error=dynamic_error,
                                                           session_generation=self.session_generation)
                    self.z_start_pose_report=report
                    before=(previous.get("state"),(previous.get("pose") or {}).get("pose_status"),previous.get("ready"))
                    after=(report["state"],(report.get("pose") or {}).get("pose_status"),report.get("ready"))
                    if before!=after:
                        self.emit("Z_START_POSE_SHADOW_CHANGE",now)
                    old_temporal=previous.get("temporal_evidence") or {}
                    new_temporal=report["temporal_evidence"]
                    if (old_temporal.get("state"),old_temporal.get("reason")) != (new_temporal["state"],new_temporal["reason"]):
                        self.emit("Z_TEMPORAL_READY_CHANGE",now,temporal_evidence=new_temporal)
                except Exception as exc:
                    self.z_start_pose_error=str(exc)
            was_waiting = self.session.segmenter.waiting_for_rearm
            self.session.update(index,raw_hand,(shape[1],shape[0]),now,
                                detection_error=dynamic_error,hand_count=raw_hand_count)
            context = self.process_pending(now,raw_hand,shape)
            if self.z_start_pose_shadow is not None:
                try:
                    before_temporal=self.z_start_pose_report["temporal_evidence"] if self.z_start_pose_report else {}
                    self.z_start_pose_shadow.finish_frame(index,now)
                    self.z_start_pose_report=self.z_start_pose_shadow.snapshot()
                    after_temporal=self.z_start_pose_report["temporal_evidence"]
                    if (before_temporal.get("state"),before_temporal.get("reason")) != (after_temporal["state"],after_temporal["reason"]):
                        self.emit("Z_TEMPORAL_READY_CHANGE",now,temporal_evidence=after_temporal)
                except Exception as exc:
                    self.z_start_pose_error=str(exc)  # No recognition dependency.
            # Resolve post-confirm rearm BEFORE considering any new raw result.
            # A static-pose transition must not unlock via its own binary guess.
            self.update_latch(raw_hand,now,video_present=static_hand is not None,
                              observations_valid=not dynamic_error and not static_error)
            result = self.session.last_result
            if result is not None and result is not self.seen_result:
                self.seen_result = result
                if result["status"] == "RESULT" and self.attempt_consumed(result.get("attempt_id")):
                    self.emit("JZ_COMPLETED_ATTEMPT_REUSE_BLOCKED",now,
                              attempt_id=result.get("attempt_id"),reuse_reason="ALREADY_COMMITTED_RESULT")
                    self.session.last_result = None
                elif result["status"] == "RESULT":
                    model_pred = result["model_result"]
                    pred_label = (model_pred or {}).get("prediction")
                    pred_prob = float((model_pred or {}).get(f"probability_{pred_label}", 0.0) or 0.0)
                    required_conf = self.min_dynamic_j_confidence if pred_label == "J" else self.min_dynamic_z_confidence
                    trajectory_ok = (context is not None and result.get("attempt_id") == context[2]
                                     and self._proposal_trajectory_valid(pred_label, context[1]))
                    self.emit("JZ_RAW_PREDICTION",now,result=result,
                              attempt_id=result.get("attempt_id"),
                              validation_status=("TRAJECTORY_REJECTED" if (context is not None and not trajectory_ok)
                                                 else "ENDPOINT_PENDING" if context else "UNVERIFIED_NO_BOUT_CONTEXT"))
                    if context is None or result.get("attempt_id") != context[2]:
                        self.clear_candidate("UNVERIFIED_RAW_RESULT_NO_CANDIDATE",now)
                        self.raw_dynamic_prediction = model_pred
                        self.raw_prediction_at = now
                        self.reason = "UNVERIFIED_RAW_RESULT_NO_CANDIDATE"
                        self._record_jz_evaluation(
                            now, attempt_id=result.get("attempt_id"), stage="UNVERIFIED",
                            model_pred=model_pred, validator_status="REJECTED",
                            reason="UNVERIFIED_RAW_RESULT_NO_CANDIDATE", final_output="NONE",
                        )
                    elif not trajectory_ok:
                        val = context[1] or {}
                        traj_sub = (val.get("j_trajectory_validation") if pred_label == "J"
                                    else val.get("z_trajectory_validation") if pred_label == "Z" else None)
                        detail_reason = ((traj_sub or {}).get("reason") or val.get("reason")
                                         or (f"{pred_label}_TRAJECTORY_VALIDATION_FAILED" if pred_label in ("J", "Z")
                                             else "REJECT_OPTION_MARGIN_AMBIGUITY" if pred_label == "UNKNOWN"
                                             else "INVALID_DYNAMIC_LABEL"))
                        traj_reason = (f"{pred_label}_TRAJECTORY_VALIDATION_FAILED"
                                       if pred_label in ("J", "Z")
                                       else "REJECT_OPTION_MARGIN_AMBIGUITY" if pred_label == "UNKNOWN"
                                       else "INVALID_DYNAMIC_LABEL")
                        self.clear_candidate(traj_reason,now)
                        self.status = "REJECTED"
                        self.emit("JZ_PROPOSAL_REJECTED_TRAJECTORY",now,label=pred_label,
                                  reason=traj_reason,validation=context[1])
                        self._record_jz_evaluation(
                            now, attempt_id=result.get("attempt_id"), stage="VALIDATOR_REJECTED",
                            model_pred=model_pred, validator_status="REJECTED",
                            reason=detail_reason, final_output="NONE",
                        )
                    elif self.enable_experimental_jz and pred_prob < required_conf:
                        self.clear_candidate("JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD",now)
                        self.status = "REJECTED"
                        self._record_jz_evaluation(
                            now, attempt_id=result.get("attempt_id"), stage="CONFIDENCE_REJECTED",
                            model_pred=model_pred, validator_status="ACCEPTED",
                            reason="JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD", final_output="NONE",
                        )
                    else:
                        # Unverified/rejected raw data cannot release the post-commit latch.
                        self.post_jz_latch = False
                        self.clear_candidate("REPLACED_BY_NEW_BOUT",now)
                        self.raw_dynamic_prediction = model_pred
                        self.raw_prediction_at = now
                        self.candidate_at = now
                        self.tracking_grace_used = False
                        self.candidate_cancel_reason = None
                        terminal_hand,self.validation,self.candidate_attempt_id = context
                        self.pending_dynamic = self.raw_dynamic_prediction
                        self.jz_hold = HoldConfirmation(HoldSettings(min_confidence=required_conf))
                        self.jz_hold.text = self.composer.text
                        self.proposal_observation = self.raw_observation(terminal_hand,shape)
                        self._sync_tracking_telemetry()
                        self._record_jz_evaluation(
                            now, attempt_id=self.candidate_attempt_id, stage="ENDPOINT_PENDING",
                            model_pred=model_pred, validator_status="ACCEPTED",
                            reason="NONE", final_output="PENDING_ENDPOINT",
                        )
                else:
                    self.clear_candidate(result["reason"],now)
                    self.status = "REJECTED"
                    self._record_jz_evaluation(
                        now, attempt_id=result.get("attempt_id"), stage="SEGMENTER_REJECTED",
                        model_pred=None, validator_status="SKIPPED",
                        reason=result.get("reason", "UPSTREAM_REJECTED"), final_output="NONE",
                    )
            # The SAME latest detector's quiet re-arm is explicit endpoint
            # evidence only when a real valid hand was observed. A release/gap
            # never starts HOLD and no extra detector replaces the new one.
            quiet_endpoint = (was_waiting and not self.session.segmenter.waiting_for_rearm
                              and raw_hand is not None and self.session.reason == "REARMED_AFTER_STILL_HAND_OR_RELEASE")
            if self.raw_prediction_at is not None and now-self.raw_prediction_at >= self.session.result_seconds:
                self.raw_dynamic_prediction = None
                self.raw_prediction_at = None
            if self.candidate or self.pending_dynamic:
                is_single_image_drop = (
                    bool(self.candidate_settled or self.pending_dynamic)
                    and (raw_hand is None or (raw_hand_count is not None and raw_hand_count == 0))
                    and (static_hand is not None and not static_error)
                    and not dynamic_error
                )
                if (self.attempt_consumed(self.candidate_attempt_id)
                        or (self.candidate and not self.fresh_validated_segment)):
                    self.clear_candidate("CONSUMED_OR_INVALID_CANDIDATE_CONTEXT",now)
                    self.status = "REJECTED"
                elif is_single_image_drop and self.tracking_grace_frame_count < 1:
                    self.tracking_grace_frame_count += 1
                    self.tracking_grace_used = True
                    self._sync_tracking_telemetry()
                elif raw_hand is None or (raw_hand_count is not None and raw_hand_count != 1) or dynamic_error:
                    self.clear_candidate("CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS",now)
                    self.status = "REJECTED"
                elif self.candidate_settled and self.session.segmenter.active:
                    self.clear_candidate("NEW_DEBOUNCED_MOTION_DURING_ENDPOINT_HOLD",now)
                    self.status = "REJECTED"
                elif not self.enable_experimental_jz and now-self.candidate_at >= self.session.result_seconds:
                    self.clear_candidate("DIAGNOSTIC_DISPLAY_COMPLETED_NO_COMMIT",now)
                elif not self.candidate_settled and now-self.candidate_at > self.session.segmenter.settings.max_seconds:
                    self.clear_candidate("ENDPOINT_NOT_OBSERVED_TIMEOUT",now)
                    self.status = "REJECTED"
                else:
                    self.tracking_grace_frame_count = 0
                    self._sync_tracking_telemetry()
                    try:
                        observation = self.raw_observation(raw_hand,shape)
                    except (ValueError,TypeError,AttributeError):
                        observation = None
                        self.clear_candidate("INVALID_ENDPOINT_GEOMETRY",now)
                        self.status = "REJECTED"
                    if observation is not None and quiet_endpoint and not self.candidate_settled:
                        # Never adopt an unrelated newly formed static pose as
                        # the endpoint of an earlier binary prediction.
                        if self.proposal_pose_matches(observation):
                            self.candidate = self.pending_dynamic
                            self.pending_dynamic = None
                            self.candidate_created_at = now
                            self.candidate_settled = True
                            self.endpoint_observation = observation
                            self._sync_tracking_telemetry()
                            pred_label = (self.candidate or {}).get("prediction")
                            min_conf = self.min_dynamic_j_confidence if pred_label == "J" else self.min_dynamic_z_confidence
                            self.jz_hold = HoldConfirmation(HoldSettings(
                                min_confidence=min_conf,
                                hold_seconds=3.0,
                                max_center_drift_ratio=0.40,
                                max_scale_change=0.25,
                                max_shape_rms=0.28,
                                blip_grace_frames=2,
                                blip_grace_seconds=0.15,
                            ))
                            self.jz_hold.text = self.composer.text
                            self.endpoint_mismatch_frames = 0
                            self.emit("JZ_ENDPOINT_OBSERVED",now)
                            self.emit("JZ_CANDIDATE_CREATED",now,validation=self.validation)
                            if (self.jz_evaluation_mode and self.jz_evaluation_record is not None
                                    and self.jz_evaluation_record.get("attempt_id") == self.candidate_attempt_id):
                                self.jz_evaluation_record["stage"] = "CANDIDATE"
                                self.jz_evaluation_record["final_output"] = (
                                    f"CANDIDATE:{pred_label}" if self.enable_experimental_jz else "DIAGNOSTIC_ONLY"
                                )
                    if observation is not None and not self.candidate_settled and not self.proposal_pose_matches(observation):
                        self.clear_candidate("CAPTURE_TERMINAL_POSE_CHANGED_BEFORE_ENDPOINT",now)
                        self.status = "REJECTED"
                    elif observation is not None and self.candidate_settled and not self.endpoint_matches(observation):
                        self.endpoint_mismatch_frames += 1
                        if self.endpoint_mismatch_frames >= self.max_endpoint_mismatch_tolerance:
                            self.clear_candidate("ENDPOINT_POSE_CHANGED",now)
                            self.status = "REJECTED"
                    elif observation is not None and self.candidate_settled and self.enable_experimental_jz:
                        self.endpoint_mismatch_frames = 0
                        pred_label = (self.candidate or {}).get("prediction")
                        required_conf = self.min_dynamic_j_confidence if pred_label == "J" else self.min_dynamic_z_confidence
                        if not self._proposal_trajectory_valid(pred_label, self.validation):
                            traj_reason = (f"{pred_label}_TRAJECTORY_VALIDATION_FAILED"
                                           if pred_label in ("J", "Z") else "INVALID_DYNAMIC_LABEL")
                            self.clear_candidate(traj_reason, now)
                            self.status = "REJECTED"
                            self.emit("JZ_PROPOSAL_REJECTED_TRAJECTORY", now, label=pred_label,
                                      reason=traj_reason, validation=self.validation)
                        elif observation.confidence < required_conf:
                            self.clear_candidate("JZ_CONFIDENCE_BELOW_HOLD_THRESHOLD",now)
                            self.status = "REJECTED"
                        else:
                            previous_hold_state = self.jz_hold.state
                            committed = self.jz_hold.update(observation,timestamp=now)
                            hold_started = previous_hold_state == "IDLE" and self.jz_hold.state == "STABILIZING"
                            if (self.jz_evaluation_mode and self.jz_evaluation_record is not None
                                    and self.jz_evaluation_record.get("attempt_id") == self.candidate_attempt_id
                                    and not committed):
                                self.jz_evaluation_record["stage"] = self.jz_hold.state
                                self.jz_evaluation_record["final_output"] = (
                                    f"HOLDING:{pred_label}" if self.jz_hold.state in ("STABILIZING", "HOLDING")
                                    else self.jz_evaluation_record["final_output"]
                                )
                        if committed:
                            self.composer.append(committed)
                            self.jz_hold.text = self.composer.text
                            confirmed_source_id = self.consume_dynamic_attempt(observation,now)
                            self.status = "CONFIRMED"
            suppress = bool(self.post_jz_latch or (self.enable_experimental_jz and (self.candidate or self.pending_dynamic)))
        else:
            suppress = False  # No J/Z updates, inference or HOLD in 123.

        self.set_blocked(suppress)
        self.static_prediction = None
        try:
            if static_hand is None or (raw_hand_count is not None and raw_hand_count > 1):
                if not suppress:
                    self.static.missing_hand(now)
            elif suppress:
                self.static_prediction = self.static.predictors[self.mode].predict_landmarks(static_hand)
            else:
                self.static_prediction,static_commit = self.static.process_hand(static_hand,shape,now)
                committed = committed or static_commit
        except (ValueError,TypeError,AttributeError,RuntimeError):
            self.static.missing_hand(now)  # Never cancel a valid independent raw J/Z candidate.
            self.static_prediction = None
        if committed:
            self.status = "CONFIRMED"
            if self.suggestion_manager is not None:
                self.suggestion_manager.update_text(self.composer.text)
        elif self.candidate:
            self.status = "HOLDING" if self.jz_hold.state in ("STABILIZING","HOLDING") else "JZ CANDIDATE"
        elif suppress:
            self.status = ("WAIT RELEASE" if self.post_jz_latch else
                           "JZ ENDPOINT PENDING" if self.pending_dynamic else "JZ MOTION PROPOSAL")
        elif self.static.hold.state in ("STABILIZING","HOLDING"):
            self.status = "HOLDING"
        elif self.status != "REJECTED":
            self.status = "STATIC"
        self.static.hold.text = self.composer.text
        self.jz_hold.text = self.composer.text
        if hold_started:
            self.emit("JZ_DYNAMIC_HOLD_STARTED",now,hold_start_reason="FRESH_CAPTURE_BOUND_ENDPOINT",
                      hold_clock_started_at=self.jz_hold._candidate_started_at)
        if confirmed_source_id is not None:
            self.emit("JZ_CONFIRMED_EXPERIMENTAL",now,character=committed,attempt_id=confirmed_source_id)
        return self.static_prediction,committed

    def close(self):
        if self.suggestion_manager is not None:
            self.suggestion_manager.close()
