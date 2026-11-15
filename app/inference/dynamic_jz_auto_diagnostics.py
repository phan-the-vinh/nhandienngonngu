"""Observability only: no detector thresholds, classification or feature edits."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


class DebugTrace:
    """Buffered exclusive JSONL; no per-frame Terminal printing."""
    def __init__(self, path, root):
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(Path(root).resolve()):
            raise ValueError("Debug trace must stay inside the J/Z experiment")
        if self.path.suffix.lower() != ".jsonl":
            raise ValueError("Debug trace must be JSONL")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", buffering=65536)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stream.close()

    def __call__(self, event):
        self.stream.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event},
                                     allow_nan=False) + "\n")
        if event.get("event") in ("ATTEMPT_FINISHED", "RESULT_DISPLAYED", "SESSION_SUMMARY"):
            self.stream.flush()  # Preserve completed attempts without per-frame flush/printing.


class AttemptStatistics:
    """An attempt begins at debounced CAPTURING, not each rejected frame.

    SUSPECT/no-start episodes are separate evidence, not failed ground-truth
    gestures. Without user-labelled attempts no real success rate is known.
    """
    FIELDS = ("TOTAL_ATTEMPTS", "CAPTURE_STARTED", "CAPTURE_COMPLETED", "REJECTED",
              "UNCERTAIN", "TIMEOUT", "MODEL_INVOKED", "RESULT_DISPLAYED", "RESULT_CREATED")

    def __init__(self):
        self.counts = Counter({field: 0 for field in self.FIELDS})
        self.reasons = Counter()
        self.active = None

    def start(self, index, now):
        if self.active and not self.active["finished"]:
            return
        self.counts["TOTAL_ATTEMPTS"] += 1
        self.counts["CAPTURE_STARTED"] += 1
        self.active = dict(attempt_id=self.counts["TOTAL_ATTEMPTS"], started_at=now,
                           start_frame=index, finished=False, completed=False, invoked=False)

    def complete(self):
        if self.active and not self.active["completed"]:
            self.active["completed"] = True
            self.counts["CAPTURE_COMPLETED"] += 1

    def model_invoked(self):
        if self.active and not self.active["invoked"]:
            self.active["invoked"] = True
            self.counts["MODEL_INVOKED"] += 1

    def finish(self, result):
        if not self.active or self.active["finished"]:
            return None
        self.active["finished"] = True
        result["attempt_id"] = self.active["attempt_id"]
        status, reason = result["status"], result["reason"]
        if status in ("REJECTED", "UNCERTAIN"):
            self.counts[status] += 1
            self.reasons[reason] += 1
        if status == "RESULT":
            self.counts["RESULT_CREATED"] += 1
        if "TIMEOUT" in reason or result.get("endpoint") == "TIMEOUT_ENDPOINT_UNVERIFIED":
            self.counts["TIMEOUT"] += 1
        return result["attempt_id"]

    def displayed(self, result):
        if (not result or result.get("status") != "RESULT" or not result.get("attempt_id")
                or result.get("display_counted")):
            return False
        result["display_counted"] = True
        self.counts["RESULT_DISPLAYED"] += 1
        return True

    def snapshot(self):
        return {**dict(self.counts), "REJECTION_REASON": dict(self.reasons),
                "attempt_definition": "DEBOUNCED_CAPTURE; NOT_USER_LABELLED_GESTURE_COUNT"}


class PredictionProbe:
    """Transparent read-only predict_proba forwarding, never fit or transform."""
    def __init__(self, model, on_predict):
        self.model, self.on_predict = model, on_predict

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict_proba(self, features):
        self.on_predict()
        return self.model.predict_proba(features)


def handshape_evidence(features):
    """Explain the EXISTING gate's handshape branch, not an acceptance rule.

    Path acceptance is still computed solely by geometry_candidates(). These
    diagnostic fractions distinguish a failed pose prerequisite from a path
    failure; they never override the gate or claim ASL correctness.
    """
    shape = np.asarray(features)[:, :63].reshape(16, 21, 3)
    evidence = {}
    for label, tip, pip, folded in (("J", 20, 18, (8,12,16)), ("Z", 8, 6, (12,16,20))):
        extended = np.linalg.norm(shape[:,tip], axis=1) > 1.1*np.linalg.norm(shape[:,pip], axis=1)
        folded_ok = np.ones(16, dtype=bool)
        for other in folded:
            folded_ok &= np.linalg.norm(shape[:,other], axis=1) < 1.3*np.linalg.norm(shape[:,other-2], axis=1)
        fraction = float(np.mean(extended & folded_ok))
        evidence[label] = {"extended_fraction": float(np.mean(extended)),
                           "folded_fraction": float(np.mean(folded_ok)),
                           "joint_fraction": fraction, "existing_pose_gate_met": fraction >= .75}
    return evidence


def failure_stage(reason, evidence=None):
    if reason == "MOTION_OR_HANDSHAPE_NOT_SUPPORTED":
        if evidence:
            return ("TRAJECTORY_GATE" if any(v["existing_pose_gate_met"] for v in evidence.values())
                    else "HANDSHAPE_GATE")
        return "GEOMETRY_GATE_DETAILS_UNAVAILABLE"
    if "TIMEOUT" in reason or "QUIET_END" in reason or "EOF" in reason:
        return "END_DETECTION"
    if "TOO_SHORT" in reason or "INSUFFICIENT" in reason:
        return "FRAME_OR_COVERAGE_CHECK"
    if "DISAGREEMENT" in reason or "CONFIDENCE" in reason:
        return "CLASSIFIER_GATE"
    if any(word in reason for word in ("HAND", "TRACKING", "DISCONTINUITY", "TEMPORAL", "LANDMARK", "DETECTION")):
        return "TRACKING"
    return "NONE_OR_UNCLASSIFIED"


def readable_reason(reason, stage):
    if reason == "FIRST16_BINARY_RESULT_NOT_GESTURE_CONFIRMATION":
        return "16 valid frames: binary result, gesture not verified"
    if reason == "WAITING_FOR_STILL_HAND_OR_RELEASE":
        return "Hold still or release before next gesture"
    if reason == "SEGMENT_TIMEOUT":
        return "Capture timed out before completion"
    if stage == "HANDSHAPE_GATE":
        return "Hand pose did not meet existing J/Z gate"
    if stage == "TRAJECTORY_GATE":
        return "Pose passed, but path did not meet J/Z gate"
    if stage == "GEOMETRY_GATE_DETAILS_UNAVAILABLE":
        return "Hand pose/path gate rejected; inspect trace"
    if stage == "CLASSIFIER_GATE":
        return "Model disagreed with gate or confidence too low"
    if "TIMEOUT" in reason:
        return "Timed out without a verified motion endpoint"
    if stage == "END_DETECTION":
        return "No fully observed quiet endpoint"
    if stage == "FRAME_OR_COVERAGE_CHECK":
        return "Not enough valid frames or valid coverage"
    if stage == "TRACKING":
        return "Tracking gap/error or continuity guard failed"
    return reason.replace("_", " ")
