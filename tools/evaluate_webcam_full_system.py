"""Independent Full-System Webcam Evaluation Harness (Phase 1).

Orchestrates multi-modal evaluation sessions across ABC (24 letters), 123 (10 digits),
J/Z (2 dynamic gestures), and SPACE, logging structured trials with ground truth.
STRICTLY READ-ONLY with respect to production code. Never modifies existing models or logic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_models
from src.recognition.alphabet import AlphabetRecognizer
from app.inference.letter_inference import Prediction
from tools.calculate_webcam_metrics import calculate_metrics

# Standard target labels
ABC_CLASSES = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
DIGIT_CLASSES = tuple(str(i) for i in range(10))
DYNAMIC_CLASSES = ("J", "Z")
SPACE_CLASSES = ("SPACE",)

VALID_OUTCOMES = (
    "SUCCESS",
    "LANDMARK_DETECTION_FAILURE",
    "CLASSIFICATION_ERROR",
    "VALIDATOR_REJECTION",
    "HOLD_ABORTED",
    "TIMEOUT_MISSED",
    "DUPLICATE_COMMIT",
    "UNKNOWN",
)


def generate_planned_trials(
    abc_reps: int = 10,
    digit_reps: int = 10,
    jz_reps: int = 10,
    space_pos_reps: int = 15,
    space_neg_reps: int = 15,
) -> List[Dict[str, Any]]:
    """Generate the standardized 390-trial evaluation queue."""
    queue = []
    idx = 1

    # 1. ABC (240 trials)
    for rep in range(abc_reps):
        for ch in ABC_CLASSES:
            queue.append({
                "trial_id": f"TRIAL_{idx:04d}_ABC_{ch}_R{rep+1}",
                "branch": "ABC",
                "ground_truth": ch,
                "trial_type": "POSITIVE",
            })
            idx += 1

    # 2. 123 (100 trials)
    for rep in range(digit_reps):
        for d in DIGIT_CLASSES:
            queue.append({
                "trial_id": f"TRIAL_{idx:04d}_123_{d}_R{rep+1}",
                "branch": "123",
                "ground_truth": d,
                "trial_type": "POSITIVE",
            })
            idx += 1

    # 3. J/Z (20 trials)
    for rep in range(jz_reps):
        for jz in DYNAMIC_CLASSES:
            queue.append({
                "trial_id": f"TRIAL_{idx:04d}_JZ_{jz}_R{rep+1}",
                "branch": "JZ",
                "ground_truth": jz,
                "trial_type": "POSITIVE",
            })
            idx += 1

    # 4. SPACE (30 trials: 15 positive, 15 negative)
    for rep in range(space_pos_reps):
        queue.append({
            "trial_id": f"TRIAL_{idx:04d}_SPACE_POS_R{rep+1}",
            "branch": "SPACE",
            "ground_truth": "SPACE",
            "trial_type": "POSITIVE",
        })
        idx += 1

    for rep in range(space_neg_reps):
        queue.append({
            "trial_id": f"TRIAL_{idx:04d}_SPACE_NEG_R{rep+1}",
            "branch": "SPACE",
            "ground_truth": "NEGATIVE",
            "trial_type": "NEGATIVE",
        })
        idx += 1

    return queue


def determine_primary_outcome(
    ground_truth: str,
    commits: Sequence[Tuple[str, float]],
    landmarks_detected: bool,
    validator_rejected: bool,
    hold_aborted: bool,
    raw_prediction: Optional[str],
    candidate: Optional[str],
    timeout_occurred: bool,
) -> Tuple[str, List[str]]:
    """Determine the primary outcome with strict priority ordering and collect diagnostic flags."""
    flags: List[str] = []
    gt_clean = str(ground_truth).strip().upper()
    commit_texts = [c[0] for c in commits]

    # Diagnostic flag detection
    if not landmarks_detected:
        flags.append("NO_LANDMARKS_DETECTED")
    if validator_rejected:
        flags.append("VALIDATOR_REJECTED")
    if hold_aborted:
        flags.append("HOLD_STATE_ABORTED")
    if len(commits) > 1:
        flags.append("MULTIPLE_COMMITS_DETECTED")
    if raw_prediction and raw_prediction.upper() != gt_clean:
        flags.append("RAW_PREDICTION_MISMATCH")

    # Strict priority resolution for primary outcome:
    # 1. Total landmark detection failure across the trial
    if not landmarks_detected:
        return "LANDMARK_DETECTION_FAILURE", flags

    # 2. Duplicate commit
    if len(commits) > 1:
        return "DUPLICATE_COMMIT", flags

    # 3. Commit occurred
    if len(commits) == 1:
        committed_char = commit_texts[0].strip().upper()
        # For space, both ' ' and 'SPACE' are valid representations
        is_space_gt = gt_clean in ("SPACE", " ")
        is_space_commit = committed_char in ("SPACE", " ")

        if is_space_gt and is_space_commit:
            return "SUCCESS", flags
        elif gt_clean == "NEGATIVE":
            # A commit occurred during a negative trial -> false acceptance
            flags.append("NEGATIVE_TRIAL_COMMITTED")
            return "CLASSIFICATION_ERROR", flags
        elif committed_char == gt_clean:
            return "SUCCESS", flags
        else:
            return "CLASSIFICATION_ERROR", flags

    # 4. No commit occurred
    # Negative trial with no commit is successful rejection
    if gt_clean == "NEGATIVE":
        return "SUCCESS", flags

    if validator_rejected:
        return "VALIDATOR_REJECTION", flags

    if candidate == "UNKNOWN":
        return "UNKNOWN", flags

    if hold_aborted:
        return "HOLD_ABORTED", flags

    if raw_prediction and raw_prediction.upper() != gt_clean:
        return "CLASSIFICATION_ERROR", flags

    if timeout_occurred:
        return "TIMEOUT_MISSED", flags

    return "UNKNOWN", flags


class EvaluationHarness:
    """Manages evaluation sessions independently without altering production logic."""

    def __init__(
        self,
        coordinator: Optional[AlphabetRecognizer] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        data_source: str = "mock",
    ):
        self.data_source = data_source
        self.events: List[Dict[str, Any]] = []

        def internal_sink(event_payload: Dict[str, Any]):
            self.events.append(event_payload)
            if event_sink is not None:
                event_sink(event_payload)

        self.event_sink = internal_sink
        self.sim_clock: float = 1.0
        self.sim_frame_idx: int = 0

        if coordinator is None:
            bundle = load_models()
            self.coordinator = AlphabetRecognizer.from_models(
                bundle,
                enable_experimental_jz=True,
                enable_space=True,
                event_sink=self.event_sink,
            )
        else:
            self.coordinator = coordinator

    def run_trial_simulation(
        self,
        trial_spec: Dict[str, Any],
        scenario: str = "success",
        simulated_frames: int = 100,
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """Execute a single simulated trial in memory with complete observable telemetry."""
        from tests.test_unified_camera_experiment import synthetic_hand
        from tests.test_dynamic_jz_motion_gate import curved_hand, hand_at
        from tests.test_unified_abc123_jz_auto import mirrored

        trial_id = trial_spec["trial_id"]
        branch = trial_spec["branch"]
        ground_truth = trial_spec["ground_truth"]

        # Ensure mode alignment
        target_mode = "123" if branch == "123" else "ABC"
        if self.coordinator.mode != target_mode:
            self.coordinator.switch_mode(target_mode)

        # Clear composer and reset hold confirmation state between trials
        self.coordinator.composer.clear()
        hold = self.coordinator.hold
        hold.state = "IDLE"
        hold.text = ""
        hold.pending = None
        hold.progress = 0.0
        hold.last_confirmed = None
        hold.last_confirmed_at = None
        hold.hold_started_at = None
        hold.rearmed = True
        hold._last_timestamp = None
        hold._candidate_started_at = None
        hold._stable_count = 0
        hold._anchor_center = np.zeros(2, dtype=np.float64)
        hold._anchor_scale = 0.0
        hold._anchor_shape = np.empty(0, dtype=np.float64)
        hold._confirmed_shape = np.empty(0, dtype=np.float64)
        hold._missing_since = None
        hold._awaiting_rearm = False
        hold._blip_count = 0
        hold._total_blip_count = 0
        hold._blip_started_at = None
        hold._last_blip_at = None
        hold._last_valid_timestamp = None
        hold._grace_paused_seconds = 0.0

        # Inter-trial release / rearm settling (25 frames > 0.60s release_seconds)
        for _ in range(25):
            self.sim_frame_idx += 1
            self.sim_clock += 1.0 / fps
            self.coordinator.update(self.sim_frame_idx, None, None, (480, 640, 3), self.sim_clock, raw_hand_count=0)

        commits: List[Tuple[str, float]] = []
        raw_predictions: List[str] = []
        confidences: List[float] = []
        hold_states: List[str] = []
        landmarks_present = True
        validator_rejected = False
        hold_aborted = False
        t_start = time.perf_counter()

        # Execute scenario
        if scenario == "landmark_failure":
            landmarks_present = False
            for _ in range(simulated_frames):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                self.coordinator.update(self.sim_frame_idx, None, None, (480, 640, 3), self.sim_clock, raw_hand_count=0)

        elif scenario == "wrong_prediction":
            # Override prediction to wrong label
            wrong_char = "Z" if ground_truth != "Z" else "A"
            for _ in range(simulated_frames):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                now = self.sim_clock
                hand = synthetic_hand()
                self.coordinator.static.predictors[target_mode].predict_landmarks = lambda _h, w=wrong_char: Prediction(
                    instant_label=w, instant_confidence=0.95, stable_label=w,
                    stable_confidence=0.95, features=tuple([0.0]*63), probabilities=tuple([1.0 if i == 0 else 0.0 for i in range(24 if target_mode == 'ABC' else 10)])
                )
                pred, comm = self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), now, raw_hand_count=1)
                if pred:
                    raw_predictions.append(pred.instant_label)
                    confidences.append(pred.instant_confidence)
                hold_states.append(self.coordinator.hold.state)
                if comm:
                    commits.append((comm, now))

        elif scenario == "validator_rejection":
            validator_rejected = True
            # Simulate J/Z motion that fails trajectory check
            for f in range(simulated_frames):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                now = self.sim_clock
                hand = hand_at(f * 0.01)  # straight line horizontal, fails trajectory
                pred, comm = self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), now, raw_hand_count=1)
                hold_states.append(self.coordinator.hold.state)
                if comm:
                    commits.append((comm, now))

        elif scenario == "hold_aborted":
            hold_aborted = True
            # Hand moves violently midway through hold
            for f in range(simulated_frames):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                now = self.sim_clock
                pos = 0.0 if f < 40 else 0.8  # abrupt jump at frame 40 resets hold
                hand = hand_at(pos)
                pred, comm = self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), now, raw_hand_count=1)
                hold_states.append(self.coordinator.hold.state)
                if comm:
                    commits.append((comm, now))

        elif scenario == "timeout_missed":
            # Hand holds for only 1.0s then releases before 3.0s dwell
            for _ in range(30):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                hand = synthetic_hand()
                self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), self.sim_clock, raw_hand_count=1)
            for _ in range(simulated_frames - 30):
                self.sim_frame_idx += 1
                self.sim_clock += 1.0 / fps
                self.coordinator.update(self.sim_frame_idx, None, None, (480, 640, 3), self.sim_clock, raw_hand_count=0)

        elif scenario == "duplicate_commit":
            # Simulate double commit
            now1 = self.sim_clock + 1.0
            now2 = self.sim_clock + 2.0
            commits.append((ground_truth, now1))
            commits.append((ground_truth, now2))

        elif scenario == "unknown":
            candidate = "UNKNOWN"
            validator_rejected = True

        else:
            # Default: SUCCESS simulation
            if branch == "JZ":
                for k in range(60):
                    self.sim_frame_idx += 1
                    self.sim_clock += 1.0 / fps
                    hand = curved_hand(k, ground_truth)
                    self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), self.sim_clock, raw_hand_count=1)
                term = curved_hand(60, ground_truth)
                for _ in range(95):  # 3.16s hold
                    self.sim_frame_idx += 1
                    self.sim_clock += 1.0 / fps
                    now = self.sim_clock
                    pred, comm = self.coordinator.update(self.sim_frame_idx, mirrored(term), term, (480, 640, 3), now, raw_hand_count=1)
                    if comm:
                        commits.append((comm, now))
                    hold_states.append(self.coordinator.hold.state)
            else:
                for _ in range(simulated_frames):
                    self.sim_frame_idx += 1
                    self.sim_clock += 1.0 / fps
                    now = self.sim_clock
                    hand = synthetic_hand()
                    if ground_truth == "SPACE":
                        # Set space predictor mock
                        self.coordinator.space_predictor.predict_hand = lambda _h, mode="ABC": (0.95, True)
                    elif ground_truth == "NEGATIVE":
                        self.coordinator.space_predictor.predict_hand = lambda _h, mode="ABC": (0.05, False)
                    else:
                        self.coordinator.static.predictors[target_mode].predict_landmarks = lambda _h, g=ground_truth: Prediction(
                            instant_label=g, instant_confidence=0.98, stable_label=g,
                            stable_confidence=0.98, features=tuple([0.0]*63), probabilities=tuple([1.0 if i == 0 else 0.0 for i in range(24 if target_mode == 'ABC' else 10)])
                        )
                    pred, comm = self.coordinator.update(self.sim_frame_idx, mirrored(hand), hand, (480, 640, 3), now, raw_hand_count=1)
                    if pred:
                        raw_predictions.append(pred.instant_label)
                        confidences.append(pred.instant_confidence)
                    hold_states.append(self.coordinator.hold.state)
                    if comm:
                        commits.append((comm, now))

        t_elapsed = time.perf_counter() - t_start

        final_raw_pred = raw_predictions[-1] if raw_predictions else (ground_truth if scenario == "success" else None)
        final_conf = float(np.mean(confidences)) if confidences else (0.95 if scenario == "success" else None)
        candidate = getattr(self.coordinator, "display_label", None)

        primary_outcome, diag_flags = determine_primary_outcome(
            ground_truth=ground_truth,
            commits=commits,
            landmarks_detected=landmarks_present,
            validator_rejected=validator_rejected,
            hold_aborted=hold_aborted,
            raw_prediction=final_raw_pred,
            candidate=candidate,
            timeout_occurred=(scenario == "timeout_missed"),
        )

        commit_text = commits[0][0] if commits else ""

        return {
            "trial_id": trial_id,
            "branch": branch,
            "ground_truth": ground_truth,
            "raw_prediction": final_raw_pred,
            "prediction_confidence": final_conf,
            "candidate": candidate,
            "validation_result": getattr(self.coordinator, "validation", None),
            "hold_state": self.coordinator.hold.state,
            "commit_text": commit_text,
            "commit_count": len(commits),
            "elapsed_time": t_elapsed,
            "elapsed_time_ms": t_elapsed * 1000.0,
            "inference_latency_ms": 0.45,
            "coordinator_latency_ms": 1.25,
            "primary_outcome": primary_outcome,
            "diagnostic_flags": diag_flags,
            "data_source": self.data_source,
            "notes": f"Simulated scenario: {scenario}",
        }


def run_mock_evaluation(output_dir: Path, total_trials_limit: Optional[int] = None) -> Path:
    """Run Phase 1 mock evaluation session, outputting strictly mock-labeled results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    jsonl_path = output_dir / f"webcam_eval_trials_mock_{timestamp_str}.jsonl"
    summary_path = output_dir / f"webcam_eval_summary_mock_{timestamp_str}.json"

    harness = EvaluationHarness(data_source="mock")
    planned = generate_planned_trials()
    if total_trials_limit is not None:
        planned = planned[:total_trials_limit]

    trials_results = []
    with open(jsonl_path, "w", encoding="utf-8") as stream:
        for idx, spec in enumerate(planned):
            # Rotate scenarios for rich diagnostic coverage in mock mode
            if idx % 20 == 5:
                scenario = "wrong_prediction"
            elif idx % 20 == 11:
                scenario = "hold_aborted"
            elif idx % 20 == 17:
                scenario = "validator_rejection"
            elif idx % 50 == 49:
                scenario = "landmark_failure"
            else:
                scenario = "success"

            res = harness.run_trial_simulation(spec, scenario=scenario)
            trials_results.append(res)
            stream.write(json.dumps(res, ensure_ascii=False) + "\n")

    metrics = calculate_metrics(trials_results)
    metrics["metadata"] = {
        "session_timestamp_utc": timestamp_str,
        "data_source": "mock",
        "trials_evaluated": len(trials_results),
        "jsonl_file": str(jsonl_path),
        "warning": "MOCK BENCHMARK ONLY. NOT REAL WEBCAM ACCURACY.",
    }
    summary_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mock evaluation complete. Trials: {len(trials_results)}")
    print(f"Trials file: {jsonl_path}")
    print(f"Summary file: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Full-System Evaluation Harness (Phase 1)")
    parser.add_argument("--mock", action="store_true", default=True, help="Execute in mock synthetic mode (Default)")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 10-trial mock smoke test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/data", help="Output directory")
    args = parser.parse_args()

    limit = 10 if args.dry_run else args.limit
    run_mock_evaluation(args.output_dir, total_trials_limit=limit)


if __name__ == "__main__":
    main()
