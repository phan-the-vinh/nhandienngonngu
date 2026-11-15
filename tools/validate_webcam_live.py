"""Stage 13.1 Interactive Live Webcam Validation Protocol.

Runs a controlled live session comparing:
1. Baseline Production Logistic Regression
2. Production Logistic Regression + Probability Margin Reject Option (margin_threshold = 0.10)

Protocol:
- 20 J attempts
- 20 Z attempts
- 20 Negative control attempts (e.g. idle hand, pointing wave, fist, etc.)
- Total: 60 attempts

Computes separate metrics:
- J Recall, Z Recall
- J Precision, Z Precision
- Macro F1
- Coverage
- UNKNOWN rate
- Z -> J misclassification count
- Invoked Negative FAR & End-to-End Negative FAR
- End-to-End commit rate
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.feature_extraction.dynamic_sequence import (
    build_sequence_features,
    sample_valid_frame_indices,
)
from ai.feature_extraction.hand_landmarks import (
    create_hand_landmarker,
    landmarks_to_array,
)
from app.inference.dynamic_jz.stage13_margin_wrapper import ProbabilityMarginModelWrapper
from tools.evaluate_stage7_fusion import compute_metrics

MODEL_PATH = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
TASK_PATH = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
WINDOW_NAME = "STAGE 13.1 — Interactive Webcam Validation (J/Z)"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Webcam Validation Protocol for Stage 13.1")
    parser.add_argument("--trials-per-class", type=int, default=20,
                        help="Number of trials per class: J, Z, NEGATIVE (default: 20 -> 60 total)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--margin-threshold", type=float, default=0.10,
                        help="Reject option margin threshold (default: 0.10)")
    parser.add_argument("--preflight", action="store_true",
                        help="Perform hardware camera accessibility check only and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify pipeline and model with synthetic data without opening camera")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports/data/webcam_live_validation_results.json",
                        help="Path to save evaluation telemetry JSON")
    return parser.parse_args(argv)


def compute_session_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all required metrics from session telemetry records."""
    methods = ["baseline", "reject_option"]
    results = {}

    pos_records = [r for r in records if r["ground_truth"] in ("J", "Z")]
    neg_records = [r for r in records if r["ground_truth"] == "NEGATIVE"]

    # Pipeline diagnostics
    pos_pipeline_errors = sum(1 for r in pos_records if not r.get("model_invoked") and r.get("valid_frames", 0) >= 16)
    neg_pipeline_errors = sum(1 for r in neg_records if not r.get("model_invoked") and r.get("valid_frames", 0) >= 16)
    pos_insufficient_frames = sum(1 for r in pos_records if r.get("valid_frames", 0) < 16)
    neg_insufficient_frames = sum(1 for r in neg_records if r.get("valid_frames", 0) < 16)

    # Invoked positives (where features were successfully built and model was invoked)
    invoked_pos = [r for r in pos_records if r.get("model_invoked")]
    invoked_neg = [r for r in neg_records if r.get("model_invoked")]

    for method in methods:
        pred_key = f"{method}_prediction"
        y_true_pos = [r["ground_truth"] for r in invoked_pos]
        y_pred_pos = [r.get(pred_key, "UNKNOWN") for r in invoked_pos]

        m_pos = compute_metrics(y_true_pos, y_pred_pos, [0.0] * len(y_pred_pos)) if invoked_pos else {}

        # Z -> J misclassifications
        z_to_j = sum(1 for r in invoked_pos if r["ground_truth"] == "Z" and r.get(pred_key) == "J")
        j_to_z = sum(1 for r in invoked_pos if r["ground_truth"] == "J" and r.get(pred_key) == "Z")

        # Unknown count on positives (only among invoked attempts)
        unknown_pos = sum(1 for p in y_pred_pos if p == "UNKNOWN")
        unknown_rate_pos = unknown_pos / len(invoked_pos) if invoked_pos else 0.0

        # Invoked negatives
        neg_accepted = sum(1 for r in invoked_neg if r.get(pred_key) in ("J", "Z"))
        invoked_far = neg_accepted / len(invoked_neg) if invoked_neg else 0.0
        e2e_far = neg_accepted / len(neg_records) if neg_records else 0.0

        # End-to-end commits
        total_attempts = len(records)
        committed_count = sum(1 for r in records if r.get(f"{method}_committed"))
        e2e_commit_rate = committed_count / total_attempts if total_attempts else 0.0

        # End-to-end recalls
        total_j = sum(1 for r in pos_records if r["ground_truth"] == "J")
        total_z = sum(1 for r in pos_records if r["ground_truth"] == "Z")
        tp_j = sum(1 for r in pos_records if r.get("model_invoked") and r.get(pred_key) == "J")
        tp_z = sum(1 for r in pos_records if r.get("model_invoked") and r.get(pred_key) == "Z")

        e2e_j_recall = tp_j / total_j if total_j else 0.0
        e2e_z_recall = tp_z / total_z if total_z else 0.0

        results[method] = {
            "invoked_positives_count": len(invoked_pos),
            "total_positives_count": len(pos_records),
            "pipeline_error_positives_count": pos_pipeline_errors,
            "insufficient_frames_positives_count": pos_insufficient_frames,
            "invoked_accuracy": m_pos.get("accuracy", 0.0) if invoked_pos else 0.0,
            "coverage": m_pos.get("coverage", 0.0) if invoked_pos else 0.0,
            "j_recall": (m_pos.get("per_class", {}).get("J") or {}).get("recall", 0.0) if invoked_pos else 0.0,
            "z_recall": (m_pos.get("per_class", {}).get("Z") or {}).get("recall", 0.0) if invoked_pos else 0.0,
            "j_precision": (m_pos.get("per_class", {}).get("J") or {}).get("precision", 0.0) if invoked_pos else 0.0,
            "z_precision": (m_pos.get("per_class", {}).get("Z") or {}).get("precision", 0.0) if invoked_pos else 0.0,
            "macro_f1": m_pos.get("macro_f1", 0.0) if invoked_pos else 0.0,
            "unknown_count": unknown_pos,
            "unknown_rate": unknown_rate_pos,
            "z_to_j_errors": z_to_j,
            "j_to_z_errors": j_to_z,
            "invoked_negatives_count": len(invoked_neg),
            "total_negatives_count": len(neg_records),
            "pipeline_error_negatives_count": neg_pipeline_errors,
            "insufficient_frames_negatives_count": neg_insufficient_frames,
            "invoked_negatives_accepted": neg_accepted,
            "invoked_negative_far": invoked_far,
            "end_to_end_negative_far": e2e_far,
            "end_to_end_j_recall": e2e_j_recall,
            "end_to_end_z_recall": e2e_z_recall,
            "committed_count": committed_count,
            "end_to_end_commit_rate": e2e_commit_rate,
        }

    return results


def draw_hand_landmarks(frame: np.ndarray, hand: Any) -> None:
    """Draw hand landmarks and skeleton on the display frame."""
    h, w = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm
    ]
    points = []
    for lm in hand:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw lines
    for p1_idx, p2_idx in connections:
        if 0 <= p1_idx < len(points) and 0 <= p2_idx < len(points):
            cv2.line(frame, points[p1_idx], points[p2_idx], (0, 255, 180), 2, cv2.LINE_AA)

    # Draw circles at joints
    for idx, (px, py) in enumerate(points):
        if idx == 8:
            cv2.circle(frame, (px, py), 6, (255, 0, 255), -1, cv2.LINE_AA)  # Magenta for Z tip
        elif idx == 20:
            cv2.circle(frame, (px, py), 6, (255, 255, 0), -1, cv2.LINE_AA)  # Cyan for J tip
        else:
            cv2.circle(frame, (px, py), 3, (0, 200, 0), -1, cv2.LINE_AA)


def evaluate_attempt(
    trial_idx: int,
    target: str,
    valid_entries: List[Tuple[int, Any, Tuple[int, int]]],
    total_frames: int,
    wrapper_baseline: ProbabilityMarginModelWrapper,
    wrapper_reject: ProbabilityMarginModelWrapper,
    margin_threshold: float,
) -> Dict[str, Any]:
    """Process a recorded gesture bout and perform dual inference.

    Extracts 16 valid frames via uniform sampling, builds exact (16, 69) features
    without extra keyword arguments, and feeds the identical 1104-dim feature vector
    to both Baseline Production LR and Probability Margin Reject Option.
    """
    record: Dict[str, Any] = {
        "trial_index": trial_idx + 1,
        "ground_truth": target,
        "total_frames": total_frames,
        "valid_frames": len(valid_entries),
        "model_invoked": False,
        "baseline_prediction": "UNKNOWN",
        "reject_option_prediction": "UNKNOWN",
        "baseline_committed": False,
        "reject_option_committed": False,
        "probability_J": 0.0,
        "probability_Z": 0.0,
        "margin": 0.0,
        "rejection_reason": "",
    }

    if len(valid_entries) < 16:
        record["rejection_reason"] = f"INSUFFICIENT_VALID_FRAMES ({len(valid_entries)} < 16)"
        return record

    try:
        # Uniformly sample 16 valid frames across the bout
        indices = sample_valid_frame_indices([e[0] for e in valid_entries])
        mapping = {e[0]: e for e in valid_entries}
        sampled_hands = [mapping[i][1] for i in indices]
        sampled_sizes = [(int(mapping[i][2][0]), int(mapping[i][2][1])) for i in indices]

        # build_sequence_features takes (landmark_frames, frame_sizes, length=16)
        # returns (features, hand_scale_px) where features.shape == (16, 69)
        features, hand_scale_px = build_sequence_features(sampled_hands, sampled_sizes)

        if features.shape != (16, 69):
            raise ValueError(f"Unexpected features shape {features.shape}, expected (16, 69)")

        flat = features.reshape(1, 1104, order="C")

        # Baseline Production LR (Reject Option disabled)
        base_det = wrapper_baseline.predict_sample_detail(flat)
        base_pred = base_det["prediction"]

        # Production LR + Margin Reject Option
        reject_det = wrapper_reject.predict_sample_detail(flat)
        reject_pred = reject_det["prediction"]

        record.update({
            "model_invoked": True,
            "baseline_prediction": base_pred,
            "reject_option_prediction": reject_pred,
            "probability_J": float(base_det["probability_J"]),
            "probability_Z": float(base_det["probability_Z"]),
            "margin": float(base_det["margin"]),
            "hand_scale_px": float(hand_scale_px),
            "baseline_committed": (base_pred == target) if target in ("J", "Z") else False,
            "reject_option_committed": (reject_pred == target and reject_pred != "UNKNOWN") if target in ("J", "Z") else False,
            "rejection_reason": reject_det.get("reason", ""),
        })
    except Exception as exc:
        record["model_invoked"] = False
        record["rejection_reason"] = f"FEATURE_EXTRACTION_ERROR: {type(exc).__name__}: {exc}"

    return record


def render_hud(
    frame: np.ndarray,
    trial_idx: int,
    total_trials: int,
    target: str,
    state: str,
    frame_count: int,
    valid_count: int,
    last_eval: Optional[Dict[str, Any]],
    session_records: List[Dict[str, Any]],
    feedback_msg: str,
    margin_threshold: float,
) -> np.ndarray:
    """Render camera frame alongside a high-visibility interactive HUD."""
    height, width = frame.shape[:2]
    panel_width = 440
    canvas_height = max(height, 580)
    canvas = np.full((canvas_height, width + panel_width, 3), 30, dtype=np.uint8)

    # Place camera preview
    canvas[:height, :width] = frame

    # Live frame status banner
    if state == "RECORDING":
        cv2.rectangle(canvas, (2, 2), (width - 2, height - 2), (0, 0, 255), 4)
        cv2.circle(canvas, (30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    elif state == "READY":
        cv2.putText(canvas, "PRESS SPACE TO START", (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)
    elif state == "EVALUATED":
        cv2.putText(canvas, "EVALUATED - REVIEW RESULT", (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

    # Side panel helpers
    ox = width + 18

    def put_text(text: str, y: int, scale: float = 0.52, color: Tuple[int, int, int] = (240, 240, 240), thickness: int = 1):
        cv2.putText(canvas, text, (ox, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # Header
    put_text("STAGE 13.1 LIVE VALIDATION", 32, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (ox, 44), (ox + panel_width - 36, 44), (70, 70, 70), 1)

    # Progress
    put_text(f"Trial: {trial_idx + 1} / {total_trials}", 70, 0.60, (0, 220, 255), 2)

    # Target Card
    target_colors = {
        "J": (255, 255, 0),       # Cyan
        "Z": (255, 0, 255),       # Magenta
        "NEGATIVE": (0, 165, 255) # Orange
    }
    t_color = target_colors.get(target, (200, 200, 200))
    cv2.rectangle(canvas, (ox, 82), (ox + panel_width - 36, 126), (45, 45, 45), -1)
    cv2.rectangle(canvas, (ox, 82), (ox + panel_width - 36, 126), t_color, 2)
    cv2.putText(canvas, f"TARGET: {target}", (ox + 12, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.85, t_color, 2, cv2.LINE_AA)

    # Target Instruction
    if target == "J":
        instruction = "Extend pinky. Draw 'J' hook down & curve left."
    elif target == "Z":
        instruction = "Extend index. Draw 'Z' zigzag horizontally."
    else:
        instruction = "Hold idle hand / static pose / non-JZ motion."
    put_text(instruction, 146, 0.44, (180, 180, 180), 1)

    # State Badge
    state_badges = {
        "READY": ("READY [Press SPACE]", (0, 200, 0)),
        "RECORDING": ("● RECORDING GESTURE", (0, 0, 255)),
        "EVALUATED": ("EVALUATED [Review Below]", (0, 220, 255)),
    }
    badge_text, badge_color = state_badges.get(state, (state, (200, 200, 200)))
    put_text(badge_text, 176, 0.60, badge_color, 2)

    # Live frame counts
    put_text(f"Attempt Frames: {frame_count}  |  Valid Hand: {valid_count} / 16", 202, 0.48, (200, 200, 200), 1)

    cv2.line(canvas, (ox, 218), (ox + panel_width - 36, 218), (70, 70, 70), 1)

    # Evaluation Details
    if last_eval is not None:
        put_text("ATTEMPT INFERENCE:", 238, 0.52, (255, 255, 255), 2)
        if last_eval["model_invoked"]:
            p_j = last_eval["probability_J"]
            p_z = last_eval["probability_Z"]
            margin = last_eval["margin"]
            base_p = last_eval["baseline_prediction"]
            rej_p = last_eval["reject_option_prediction"]

            put_text(f"P(J)={p_j:.3f}   P(Z)={p_z:.3f}   Margin={margin:.3f}", 262, 0.46, (200, 220, 255), 1)
            base_col = (0, 255, 0) if (base_p == target and target in ("J", "Z")) else ((0, 200, 200) if target == "NEGATIVE" else (0, 0, 255))
            put_text(f"Baseline LR:       {base_p}", 286, 0.50, base_col, 2)

            rej_col = (0, 255, 0) if (rej_p == target and target in ("J", "Z")) else ((0, 165, 255) if rej_p == "UNKNOWN" else (0, 0, 255))
            put_text(f"LR + Margin 0.10:  {rej_p}", 310, 0.50, rej_col, 2)

            if rej_p == "UNKNOWN":
                put_text(f"Decision: REJECTED (Margin < {margin_threshold:.2f})", 332, 0.44, (0, 165, 255), 1)
            elif rej_p == target:
                put_text("Decision: CORRECT ACCEPTANCE", 332, 0.44, (0, 255, 0), 1)
            else:
                put_text("Decision: MISCLASSIFIED", 332, 0.44, (0, 0, 255), 1)
        else:
            put_text("No Inference: " + last_eval.get("rejection_reason", "Invalid"), 265, 0.46, (0, 100, 255), 1)
    else:
        put_text("Awaiting attempt...", 250, 0.48, (120, 120, 120), 1)

    cv2.line(canvas, (ox, 350), (ox + panel_width - 36, 350), (70, 70, 70), 1)

    # Key Controls
    put_text("KEYBOARD CONTROLS:", 372, 0.52, (255, 255, 255), 2)
    if state == "READY":
        put_text("[SPACE]   Start recording gesture", 395, 0.47, (0, 255, 180), 1)
        put_text("[Q / ESC] Quit early and save results", 418, 0.47, (180, 180, 180), 1)
    elif state == "RECORDING":
        put_text("[SPACE]   Stop recording & evaluate", 395, 0.47, (0, 255, 180), 1)
        put_text("[R]       Reset/restart recording", 418, 0.47, (255, 220, 0), 1)
        put_text("[Q / ESC] Quit early and save results", 441, 0.47, (180, 180, 180), 1)
    elif state == "EVALUATED":
        put_text("[SPACE]   ACCEPT & ADVANCE ->", 395, 0.47, (0, 255, 0), 2)
        put_text("[R]       DISCARD & REDO TRIAL", 418, 0.47, (255, 220, 0), 1)
        put_text("[Q / ESC] Quit early and save results", 441, 0.47, (180, 180, 180), 1)

    cv2.line(canvas, (ox, 460), (ox + panel_width - 36, 460), (70, 70, 70), 1)

    # Live cumulative summary
    done_count = len(session_records)
    put_text(f"SESSION PROGRESS: {done_count} / {total_trials} saved", 482, 0.50, (0, 220, 255), 1)
    if done_count > 0:
        invoked_pos = [r for r in session_records if r["ground_truth"] in ("J", "Z") and r.get("model_invoked")]
        if invoked_pos:
            base_acc = sum(1 for r in invoked_pos if r["baseline_prediction"] == r["ground_truth"]) / len(invoked_pos) * 100
            rej_acc = sum(1 for r in invoked_pos if r["reject_option_prediction"] == r["ground_truth"]) / len(invoked_pos) * 100
            z_to_j = sum(1 for r in invoked_pos if r["ground_truth"] == "Z" and r["baseline_prediction"] == "J")
            put_text(f"Invoked Acc: Base={base_acc:.1f}% | Reject={rej_acc:.1f}%", 504, 0.44, (200, 200, 200), 1)
            put_text(f"Z->J Errors: Base={z_to_j}", 524, 0.44, (200, 200, 200), 1)

    # Feedback message at bottom
    put_text(feedback_msg[:48], 552, 0.43, (0, 220, 220), 1)

    return canvas


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 80)
    print("STAGE 13.1 LIVE WEBCAM VALIDATION PROTOCOL")
    print("=" * 80)
    print(f"Trials per class: {args.trials_per_class} (Total: {args.trials_per_class * 3} attempts)")
    print(f"Margin Threshold: {args.margin_threshold}")
    print(f"Output path:      {args.output}")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model pipeline not found at {MODEL_PATH}", file=sys.stderr)
        return 1

    if not TASK_PATH.exists():
        print(f"ERROR: MediaPipe task not found at {TASK_PATH}", file=sys.stderr)
        return 1

    raw_pipeline = joblib.load(MODEL_PATH)
    wrapper_baseline = ProbabilityMarginModelWrapper(
        raw_pipeline, enable_reject_option=False, margin_threshold=args.margin_threshold
    )
    wrapper_reject = ProbabilityMarginModelWrapper(
        raw_pipeline, enable_reject_option=True, margin_threshold=args.margin_threshold
    )

    # -------------------------------------------------------------------------
    # MODE 1: DRY RUN (Pipeline and Synthetic Verification)
    # -------------------------------------------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] Verifying synthetic attempt inference through dual evaluator...")
        synthetic_feat = np.ones((1, 1104), dtype=np.float64)
        base_det = wrapper_baseline.predict_sample_detail(synthetic_feat)
        reject_det = wrapper_reject.predict_sample_detail(synthetic_feat)
        print(f"  Base prediction:   {base_det['prediction']} (P(J)={base_det['probability_J']:.4f}, P(Z)={base_det['probability_Z']:.4f})")
        print(f"  Reject prediction: {reject_det['prediction']} (margin: {reject_det['margin']:.4f})")

        # Verify metrics computation
        mock_records = [
            {"trial_index": 1, "ground_truth": "J", "valid_frames": 20, "model_invoked": True, "baseline_prediction": "J", "reject_option_prediction": "J", "baseline_committed": True, "reject_option_committed": True},
            {"trial_index": 2, "ground_truth": "Z", "valid_frames": 20, "model_invoked": True, "baseline_prediction": "Z", "reject_option_prediction": "Z", "baseline_committed": True, "reject_option_committed": True},
            {"trial_index": 3, "ground_truth": "NEGATIVE", "valid_frames": 5, "model_invoked": False, "baseline_prediction": "UNKNOWN", "reject_option_prediction": "UNKNOWN", "baseline_committed": False, "reject_option_committed": False},
        ]
        metrics = compute_session_metrics(mock_records)
        print(f"  Synthetic metrics test: baseline coverage={metrics['baseline']['coverage']:.2f}, reject coverage={metrics['reject_option']['coverage']:.2f}")
        print("\n[PASS] Dry run successful. All models, wrappers, and metrics functions verified.")
        return 0

    # -------------------------------------------------------------------------
    # MODE 2: CAMERA PREFLIGHT (Hardware check only)
    # -------------------------------------------------------------------------
    if args.preflight:
        print(f"\n[PREFLIGHT] Checking camera device accessibility on index {args.camera_index}...")
        test_camera = cv2.VideoCapture(args.camera_index)
        if not test_camera.isOpened():
            print(f"ERROR: Cannot open camera index {args.camera_index}.", file=sys.stderr)
            return 2
        ok, test_frame = test_camera.read()
        test_camera.release()
        if not ok or test_frame is None:
            print("ERROR: Camera opened but failed to capture a test frame.", file=sys.stderr)
            return 2
        print(f"[PASS] Webcam device {args.camera_index} is accessible (frame size: {test_frame.shape[1]}x{test_frame.shape[0]}).")
        return 0

    # -------------------------------------------------------------------------
    # MODE 3: DEFAULT (Interactive Live Webcam Session)
    # -------------------------------------------------------------------------
    print("\n[LIVE PROTOCOL] Launching interactive webcam session...")
    print(f"Target trials: {args.trials_per_class} J, {args.trials_per_class} Z, {args.trials_per_class} NEGATIVE (Total: {args.trials_per_class * 3})")
    print("Controls:")
    print("  [SPACE]  Start recording / Stop recording / Accept & next trial")
    print("  [R]      Redo current trial")
    print("  [Q/ESC]  Quit early and save completed trials")

    # Build trial sequence
    trial_targets: List[str] = (
        ["J"] * args.trials_per_class +
        ["Z"] * args.trials_per_class +
        ["NEGATIVE"] * args.trials_per_class
    )
    total_trials = len(trial_targets)

    trial_idx = 0
    state = "READY"
    valid_entries: List[Tuple[int, Any, Tuple[int, int]]] = []
    frame_count = 0
    last_eval: Optional[Dict[str, Any]] = None
    session_records: List[Dict[str, Any]] = []
    feedback_msg = "Press SPACE to begin recording."

    camera = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        camera = cv2.VideoCapture(args.camera_index)
        if not camera.isOpened():
            print(f"ERROR: Cannot open camera index {args.camera_index}.", file=sys.stderr)
            print("Please ensure webcam is connected and accessible.", file=sys.stderr)
            return 2

        with create_hand_landmarker(TASK_PATH, running_mode="IMAGE", num_hands=1) as landmarker:
            while trial_idx < total_trials:
                ok, frame = camera.read()
                if not ok or frame is None or frame.ndim != 3:
                    print("ERROR: Failed to read frame from webcam.", file=sys.stderr)
                    break

                height, width = frame.shape[:2]
                target = trial_targets[trial_idx]

                # Hand landmark extraction and visualization
                if state == "RECORDING":
                    frame_count += 1
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detection = landmarker.detect(mp_img)
                        if detection.hand_landmarks:
                            hand = detection.hand_landmarks[0]
                            try:
                                landmarks_to_array(hand)
                                valid_entries.append((frame_count, hand, (width, height)))
                                draw_hand_landmarks(frame, hand)
                            except (AttributeError, TypeError, ValueError):
                                pass
                    except Exception:
                        pass

                    # Automatic bout upper bound (150 frames ~ 5 seconds) to prevent infinite accumulation
                    if frame_count >= 150:
                        state = "EVALUATED"
                        last_eval = evaluate_attempt(
                            trial_idx, target, valid_entries, frame_count,
                            wrapper_baseline, wrapper_reject, args.margin_threshold
                        )
                        feedback_msg = "Max duration reached. Press SPACE to accept, R to redo."
                elif state == "READY":
                    # In READY state, still detect hand for preview skeleton
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detection = landmarker.detect(mp_img)
                        if detection.hand_landmarks:
                            draw_hand_landmarks(frame, detection.hand_landmarks[0])
                    except Exception:
                        pass

                # Render canvas with HUD
                canvas = render_hud(
                    frame=frame,
                    trial_idx=trial_idx,
                    total_trials=total_trials,
                    target=target,
                    state=state,
                    frame_count=frame_count,
                    valid_count=len(valid_entries),
                    last_eval=last_eval,
                    session_records=session_records,
                    feedback_msg=feedback_msg,
                    margin_threshold=args.margin_threshold,
                )

                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF

                # Key handling
                if key in (27, ord('q'), ord('Q')):
                    print("\n[INFO] Session terminated early by user (Q/ESC).")
                    break

                elif key == ord(' '):
                    if state == "READY":
                        state = "RECORDING"
                        valid_entries = []
                        frame_count = 0
                        last_eval = None
                        feedback_msg = f"Recording Trial {trial_idx + 1} ({target}). Perform gesture!"
                    elif state == "RECORDING":
                        state = "EVALUATED"
                        last_eval = evaluate_attempt(
                            trial_idx, target, valid_entries, frame_count,
                            wrapper_baseline, wrapper_reject, args.margin_threshold
                        )
                        feedback_msg = "Attempt evaluated. Press SPACE to accept, R to redo."
                    elif state == "EVALUATED":
                        if last_eval is not None:
                            session_records.append(last_eval)
                            status_tag = "INVOKED" if last_eval["model_invoked"] else "UNINVOKED"
                            print(f"[TRIAL {trial_idx + 1}/{total_trials}] Target: {target:<8} | Status: {status_tag:<9} | Base: {last_eval['baseline_prediction']:<7} | Reject: {last_eval['reject_option_prediction']:<7} | Margin: {last_eval['margin']:.3f}")
                        trial_idx += 1
                        state = "READY"
                        valid_entries = []
                        frame_count = 0
                        last_eval = None
                        if trial_idx < total_trials:
                            next_target = trial_targets[trial_idx]
                            feedback_msg = f"Trial accepted. Next: Trial {trial_idx + 1}/{total_trials} ({next_target})."
                        else:
                            feedback_msg = "All trials completed!"

                elif key in (ord('r'), ord('R')):
                    if state == "EVALUATED":
                        state = "READY"
                        valid_entries = []
                        frame_count = 0
                        last_eval = None
                        feedback_msg = f"Trial {trial_idx + 1} ({target}) discarded. Press SPACE to record again."
                    elif state == "RECORDING":
                        valid_entries = []
                        frame_count = 0
                        feedback_msg = f"Recording reset for Trial {trial_idx + 1}. Continue gesture."

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # POST-SESSION EVALUATION & REPORTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SESSION FINISHED")
    print("=" * 80)
    print(f"Total attempts recorded: {len(session_records)} / {total_trials}")

    if not session_records:
        print("[WARNING] No trials were completed. No results to evaluate.")
        return 0

    session_metrics = compute_session_metrics(session_records)

    # Print summary table
    print("\n" + "=" * 80)
    print("STAGE 13.1 LIVE WEBCAM VALIDATION RESULTS")
    print("=" * 80)
    b_res = session_metrics.get("baseline", {})
    r_res = session_metrics.get("reject_option", {})
    b_inv = b_res.get('invoked_positives_count', 0)
    r_inv = r_res.get('invoked_positives_count', 0)

    def fmt_pct(val: float, inv_count: int) -> str:
        return f"{val * 100:.1f}%" if inv_count > 0 else "N/A (0 invoked)"

    def fmt_num(val: float, inv_count: int) -> str:
        return f"{val:.4f}" if inv_count > 0 else "N/A (0 invoked)"

    print(f"{'Metric':<34} {'Baseline LR':<22} {'LR + Margin 0.10':<22}")
    print("-" * 78)
    print(f"{'Invoked Positives':<34} {b_inv:<22} {r_inv:<22}")
    print(f"{'Pipeline Errors (Positives)':<34} {b_res.get('pipeline_error_positives_count', 0):<22} {r_res.get('pipeline_error_positives_count', 0):<22}")
    print(f"{'Insufficient Frames (<16)':<34} {b_res.get('insufficient_frames_positives_count', 0):<22} {r_res.get('insufficient_frames_positives_count', 0):<22}")
    print(f"{'Invoked Accuracy':<34} {fmt_pct(b_res.get('invoked_accuracy', 0.0), b_inv):<22} {fmt_pct(r_res.get('invoked_accuracy', 0.0), r_inv):<22}")
    print(f"{'Coverage':<34} {fmt_pct(b_res.get('coverage', 0.0), b_inv):<22} {fmt_pct(r_res.get('coverage', 0.0), r_inv):<22}")
    print(f"{'Macro F1':<34} {fmt_num(b_res.get('macro_f1', 0.0), b_inv):<22} {fmt_num(r_res.get('macro_f1', 0.0), r_inv):<22}")
    print(f"{'J Recall':<34} {fmt_pct(b_res.get('j_recall', 0.0), b_inv):<22} {fmt_pct(r_res.get('j_recall', 0.0), r_inv):<22}")
    print(f"{'Z Recall':<34} {fmt_pct(b_res.get('z_recall', 0.0), b_inv):<22} {fmt_pct(r_res.get('z_recall', 0.0), r_inv):<22}")
    print(f"{'J Precision':<34} {fmt_pct(b_res.get('j_precision', 0.0), b_inv):<22} {fmt_pct(r_res.get('j_precision', 0.0), r_inv):<22}")
    print(f"{'Z Precision':<34} {fmt_pct(b_res.get('z_precision', 0.0), b_inv):<22} {fmt_pct(r_res.get('z_precision', 0.0), r_inv):<22}")
    print(f"{'UNKNOWN Count (Positives)':<34} {b_res.get('unknown_count', 0):<22} {r_res.get('unknown_count', 0):<22}")
    print(f"{'UNKNOWN Rate (Positives)':<34} {fmt_pct(b_res.get('unknown_rate', 0.0), b_inv):<22} {fmt_pct(r_res.get('unknown_rate', 0.0), r_inv):<22}")
    print(f"{'Z -> J Misclassifications':<34} {b_res.get('z_to_j_errors', 0):<22} {r_res.get('z_to_j_errors', 0):<22}")
    print(f"{'J -> Z Misclassifications':<34} {b_res.get('j_to_z_errors', 0):<22} {r_res.get('j_to_z_errors', 0):<22}")
    print(f"{'Invoked Negatives':<34} {b_res.get('invoked_negatives_count', 0):<22} {r_res.get('invoked_negatives_count', 0):<22}")
    print(f"{'Invoked Negative FAR':<34} {fmt_pct(b_res.get('invoked_negative_far', 0.0), b_res.get('invoked_negatives_count', 0)):<22} {fmt_pct(r_res.get('invoked_negative_far', 0.0), r_res.get('invoked_negatives_count', 0)):<22}")
    print(f"{'End-to-End Negative FAR':<34} {b_res.get('end_to_end_negative_far', 0.0)*100:.1f}%{'':<16} {r_res.get('end_to_end_negative_far', 0.0)*100:.1f}%")
    print(f"{'End-to-End J Recall':<34} {b_res.get('end_to_end_j_recall', 0.0)*100:.1f}%{'':<16} {r_res.get('end_to_end_j_recall', 0.0)*100:.1f}%")
    print(f"{'End-to-End Z Recall':<34} {b_res.get('end_to_end_z_recall', 0.0)*100:.1f}%{'':<16} {r_res.get('end_to_end_z_recall', 0.0)*100:.1f}%")
    print(f"{'End-to-End Commit Rate':<34} {b_res.get('end_to_end_commit_rate', 0.0)*100:.1f}%{'':<16} {r_res.get('end_to_end_commit_rate', 0.0)*100:.1f}%")
    print("=" * 78)

    # Save to JSON
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "margin_threshold": args.margin_threshold,
            "trials_per_class": args.trials_per_class,
            "total_trials_planned": total_trials,
            "total_trials_completed": len(session_records),
            "early_exit": len(session_records) < total_trials,
        },
        "metrics": session_metrics,
        "trials": session_records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[SAVED] Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
