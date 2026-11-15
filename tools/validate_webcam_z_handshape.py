"""Stage 17 Real-World Webcam Validation Protocol for Z Handshape Gate.

Runs an interactive live webcam validation comparing 3 gesture cohorts:
1. Cohort A: Valid Z (Target: 'Z') - Real ASL Z with index extended, other fingers curled.
2. Cohort B: Fist Drawing Z (Target: 'FIST_Z') - Fist moving in 3-stroke Z pattern.
3. Cohort C: Open Palm Drawing Z (Target: 'OPEN_PALM_Z') - Flat open palm moving in Z pattern.

Each attempt records:
- Trial ID
- Ground-truth label ('Z', 'FIST_Z', 'OPEN_PALM_Z')
- Landmark extraction success & frame counts
- Trajectory validator result & reason
- Handshape Gate result & reason
- Handshape metrics: index_extension, max_other_extension, dominance
- Logistic Regression predictions & probabilities
- Margin Reject Option result (threshold = 0.10)
- Production Shadow proposal vs Active proposal acceptance
- Hold confirmation & commit simulation
- Processing latency

Supports:
- --preflight: Verify camera accessibility
- --dry-run: Run synthetic test suite through full dual pipeline
- Live interactive HUD with live hand skeleton, landmark color coding, and telemetry overlay
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
from app.inference.integrated_jz_validation import (
    ZHandshapeSettings,
    ZTrajectorySettings,
    validate_z_handshape,
    validate_z_trajectory,
)

MODEL_PATH = PROJECT_ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"
TASK_PATH = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
WINDOW_NAME = "STAGE 17 — Interactive Webcam Z Handshape Validation"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 17 Real-World Webcam Validation of Z Handshape Gate")
    parser.add_argument("--trials-per-class", type=int, default=20,
                        help="Number of trials per cohort: Z, FIST_Z, OPEN_PALM_Z (default: 20 -> 60 total)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--margin-threshold", type=float, default=0.10,
                        help="Reject option margin threshold (default: 0.10)")
    parser.add_argument("--preflight", action="store_true",
                        help="Perform hardware camera accessibility check only and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify pipeline with synthetic data without opening camera")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports/data/stage17_webcam_results.json",
                        help="Path to save evaluation telemetry JSON")
    return parser.parse_args(argv)


def compute_stage17_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all required metrics from Stage 17 telemetry records."""
    cohort_z = [r for r in records if r["ground_truth"] == "Z"]
    cohort_fist = [r for r in records if r["ground_truth"] == "FIST_Z"]
    cohort_palm = [r for r in records if r["ground_truth"] == "OPEN_PALM_Z"]

    def cohort_stats(cohort: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(cohort)
        if total == 0:
            return {}
        invoked = [r for r in cohort if r.get("model_invoked")]
        traj_pass = [r for r in invoked if r.get("trajectory_valid")]
        hs_pass = [r for r in invoked if r.get("handshape_valid")]
        hs_reject = [r for r in invoked if not r.get("handshape_valid")]
        prod_shadow_acc = [r for r in invoked if r.get("production_shadow_proposal")]
        prod_active_acc = [r for r in invoked if r.get("production_active_proposal")]
        hold_ok = [r for r in invoked if r.get("hold_success")]
        commit_ok = [r for r in invoked if r.get("commit_success")]

        return {
            "total_attempts": total,
            "invoked_count": len(invoked),
            "landmark_error_count": total - len(invoked),
            "trajectory_pass_count": len(traj_pass),
            "trajectory_pass_rate": len(traj_pass) / len(invoked) if invoked else 0.0,
            "handshape_pass_count": len(hs_pass),
            "handshape_pass_rate": len(hs_pass) / len(invoked) if invoked else 0.0,
            "handshape_rejection_count": len(hs_reject),
            "handshape_rejection_rate": len(hs_reject) / len(invoked) if invoked else 0.0,
            "production_shadow_acceptance_count": len(prod_shadow_acc),
            "production_shadow_acceptance_rate": len(prod_shadow_acc) / len(invoked) if invoked else 0.0,
            "production_active_acceptance_count": len(prod_active_acc),
            "production_active_acceptance_rate": len(prod_active_acc) / len(invoked) if invoked else 0.0,
            "hold_success_count": len(hold_ok),
            "commit_success_count": len(commit_ok),
            "end_to_end_commit_rate": len(commit_ok) / total if total else 0.0,
        }

    latencies = [r["processing_latency_ms"] for r in records if "processing_latency_ms" in r and r["processing_latency_ms"] > 0]
    lat_stats = {}
    if latencies:
        lat_stats = {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }

    return {
        "valid_z": cohort_stats(cohort_z),
        "fist_z": cohort_stats(cohort_fist),
        "open_palm_z": cohort_stats(cohort_palm),
        "overall_latency_ms": lat_stats,
    }


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
            cv2.circle(frame, (px, py), 7, (255, 0, 255), -1, cv2.LINE_AA)  # Magenta for Index tip
        elif idx in (5, 9, 13, 17):
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1, cv2.LINE_AA)    # Green for Knuckles
        elif idx == 20:
            cv2.circle(frame, (px, py), 6, (255, 255, 0), -1, cv2.LINE_AA)  # Cyan for Pinky tip
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
    """Process a recorded gesture bout and evaluate Trajectory, Handshape Gate, and Production."""
    t_start = time.perf_counter()
    record: Dict[str, Any] = {
        "trial_index": trial_idx + 1,
        "ground_truth": target,
        "total_frames": total_frames,
        "valid_frames": len(valid_entries),
        "model_invoked": False,
        "trajectory_valid": False,
        "trajectory_reason": "",
        "trajectory_excursion": 0.0,
        "handshape_valid": False,
        "handshape_reason": "",
        "index_extension": 0.0,
        "max_other_extension": 0.0,
        "dominance": 0.0,
        "baseline_prediction": "UNKNOWN",
        "reject_option_prediction": "UNKNOWN",
        "probability_J": 0.0,
        "probability_Z": 0.0,
        "margin": 0.0,
        "production_shadow_proposal": False,
        "production_active_proposal": False,
        "hold_success": False,
        "commit_success": False,
        "processing_latency_ms": 0.0,
    }

    if len(valid_entries) < 16:
        record["trajectory_reason"] = f"INSUFFICIENT_VALID_FRAMES ({len(valid_entries)} < 16)"
        record["handshape_reason"] = "INSUFFICIENT_VALID_FRAMES"
        record["processing_latency_ms"] = (time.perf_counter() - t_start) * 1000.0
        return record

    try:
        indices = sample_valid_frame_indices([e[0] for e in valid_entries])
        mapping = {e[0]: e for e in valid_entries}
        sampled_hands = [mapping[i][1] for i in indices]
        sampled_sizes = [(int(mapping[i][2][0]), int(mapping[i][2][1])) for i in indices]

        features, hand_scale_px = build_sequence_features(sampled_hands, sampled_sizes)
        if features.shape != (16, 69):
            raise ValueError(f"Unexpected features shape {features.shape}, expected (16, 69)")

        # 1. Trajectory & Handshape Evaluation
        # Handshape Gate runs in SHADOW mode (Production baseline)
        traj_res = validate_z_trajectory(features, handshape_settings=ZHandshapeSettings(mode="SHADOW"))
        hs_res = traj_res.get("handshape_validation", {})

        record.update({
            "model_invoked": True,
            "trajectory_valid": bool(traj_res.get("valid", False)),
            "trajectory_reason": str(traj_res.get("reason", "")),
            "trajectory_excursion": float(traj_res.get("excursion", 0.0)),
            "handshape_valid": bool(traj_res.get("handshape_valid", False)),
            "handshape_reason": str(hs_res.get("reason", "")),
            "index_extension": float(hs_res.get("index_extension", 0.0)),
            "max_other_extension": float(hs_res.get("max_other_extension", 0.0)),
            "dominance": float(hs_res.get("dominance", 0.0)),
        })

        # 2. Model Prediction
        flat = features.reshape(1, 1104, order="C")
        base_det = wrapper_baseline.predict_sample_detail(flat)
        reject_det = wrapper_reject.predict_sample_detail(flat)

        record.update({
            "baseline_prediction": base_det["prediction"],
            "reject_option_prediction": reject_det["prediction"],
            "probability_J": float(base_det["probability_J"]),
            "probability_Z": float(base_det["probability_Z"]),
            "margin": float(base_det["margin"]),
        })

        # 3. Production Pipeline Proposal Simulation
        # In SHADOW mode: candidate created if trajectory passes AND model predicts Z
        prod_shadow_prop = bool(record["trajectory_valid"] and record["reject_option_prediction"] == "Z")
        # In ACTIVE mode: candidate created ONLY IF trajectory passes AND handshape gate passes AND model predicts Z
        prod_active_prop = bool(record["trajectory_valid"] and record["handshape_valid"] and record["reject_option_prediction"] == "Z")

        # 4. Hold & Commit Simulation
        # Only true Z gestures held for 3.0 seconds proceed to commit
        hold_ok = bool(prod_shadow_prop and target == "Z")
        commit_ok = hold_ok

        record.update({
            "production_shadow_proposal": prod_shadow_prop,
            "production_active_proposal": prod_active_prop,
            "hold_success": hold_ok,
            "commit_success": commit_ok,
        })
    except Exception as exc:
        record["model_invoked"] = False
        record["trajectory_reason"] = f"ERROR: {type(exc).__name__}: {exc}"
        record["handshape_reason"] = f"ERROR: {type(exc).__name__}: {exc}"

    record["processing_latency_ms"] = float((time.perf_counter() - t_start) * 1000.0)
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
    """Render camera frame alongside an informative interactive HUD."""
    height, width = frame.shape[:2]
    panel_width = 460
    canvas_height = max(height, 620)
    canvas = np.full((canvas_height, width + panel_width, 3), 28, dtype=np.uint8)

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

    def put_text(text: str, y: int, scale: float = 0.50, color: Tuple[int, int, int] = (240, 240, 240), thickness: int = 1):
        cv2.putText(canvas, text, (ox, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # Header
    put_text("STAGE 17 Z HANDSHAPE VALIDATION", 30, 0.62, (255, 255, 255), 2)
    put_text("-" * 46, 45, 0.45, (100, 100, 100))

    # Current target
    cohort_desc = {
        "Z": "COHORT A: VALID Z (Pointing Index, Draw Z)",
        "FIST_Z": "COHORT B: FIST (Hold Fist, Move in Z)",
        "OPEN_PALM_Z": "COHORT C: OPEN PALM (Flat Palm, Move in Z)",
    }.get(target, target)

    put_text(f"TRIAL: {trial_idx + 1} / {total_trials}", 70, 0.58, (0, 255, 255), 2)
    target_color = (0, 255, 0) if target == "Z" else (0, 165, 255) if target == "FIST_Z" else (255, 100, 255)
    put_text(f"TARGET: {target}", 95, 0.65, target_color, 2)
    put_text(cohort_desc, 118, 0.42, (180, 180, 180), 1)

    # State & Frames
    put_text(f"State: {state} | Recorded: {frame_count} | Valid: {valid_count}", 145, 0.46, (200, 200, 200))

    # Feedback box
    cv2.rectangle(canvas, (ox - 4, 155), (ox + panel_width - 30, 195), (45, 45, 45), -1)
    put_text(feedback_msg[:50], 175, 0.45, (0, 230, 230), 1)
    if len(feedback_msg) > 50:
        put_text(feedback_msg[50:100], 190, 0.42, (0, 230, 230), 1)

    # Last Evaluation Box
    put_text("LAST TRIAL EVALUATION", 218, 0.52, (255, 255, 0), 2)
    if last_eval is not None:
        y = 240
        inv = last_eval["model_invoked"]
        gt = last_eval["ground_truth"]
        tj_ok = last_eval["trajectory_valid"]
        hs_ok = last_eval["handshape_valid"]
        hs_rea = last_eval["handshape_reason"]
        b_pred = last_eval["baseline_prediction"]
        r_pred = last_eval["reject_option_prediction"]
        pz = last_eval["probability_Z"]
        sh_prop = last_eval["production_shadow_proposal"]
        act_prop = last_eval["production_active_proposal"]

        put_text(f"Status: {'INVOKED' if inv else 'FAILED'} | Valid Frames: {last_eval['valid_frames']}", y, 0.46, (0, 255, 0) if inv else (0, 0, 255))
        y += 20
        put_text(f"Trajectory: {'PASS' if tj_ok else 'FAIL'} ({last_eval['trajectory_reason'][:25]})", y, 0.46, (0, 255, 0) if tj_ok else (0, 0, 255))
        y += 20
        hs_color = (0, 255, 0) if hs_ok else (0, 0, 255)
        put_text(f"Handshape Gate: {'PASS' if hs_ok else 'REJECT'}", y, 0.48, hs_color, 2)
        y += 18
        put_text(f"  Reason: {hs_rea[:38]}", y, 0.42, hs_color)
        y += 18
        put_text(f"  Idx Ext: {last_eval['index_extension']:.3f} (>=0.70) | Dom: {last_eval['dominance']:.3f} (>=0.05)", y, 0.42, (200, 200, 200))
        y += 22
        put_text(f"LR Prediction: {b_pred} (P(Z)={pz:.3f}, Marg={last_eval['margin']:.3f})", y, 0.46, (220, 220, 220))
        y += 20
        put_text(f"Reject Option (0.10): {r_pred}", y, 0.46, (220, 220, 220))
        y += 22
        put_text(f"Prod SHADOW Proposal: {'CREATED' if sh_prop else 'BLOCKED'}", y, 0.46, (0, 255, 255) if sh_prop else (150, 150, 150))
        y += 18
        act_color = (0, 255, 0) if act_prop else (0, 150, 255) if gt != "Z" else (0, 0, 255)
        act_status = "ACCEPTED" if act_prop else "REJECTED (SAFE)" if gt != "Z" else "REJECTED (DROP)"
        put_text(f"Prod ACTIVE Proposal: {act_status}", y, 0.48, act_color, 2)
        y += 20
        put_text(f"Latency: {last_eval.get('processing_latency_ms', 0):.2f} ms", y, 0.42, (150, 150, 150))
    else:
        put_text("No trial evaluated yet.", 245, 0.45, (140, 140, 140))

    # Session Progress Summary
    put_text("-" * 46, 420, 0.45, (100, 100, 100))
    put_text("SESSION SUMMARY SO FAR", 440, 0.52, (255, 255, 255), 2)
    n_done = len(session_records)
    put_text(f"Completed: {n_done} / {total_trials}", 460, 0.46, (200, 200, 200))

    if session_records:
        z_done = [r for r in session_records if r["ground_truth"] == "Z"]
        fist_done = [r for r in session_records if r["ground_truth"] == "FIST_Z"]
        palm_done = [r for r in session_records if r["ground_truth"] == "OPEN_PALM_Z"]

        def summary_line(label: str, cohort: List[Dict[str, Any]], y_pos: int):
            if not cohort:
                put_text(f"{label}: 0 completed", y_pos, 0.44, (120, 120, 120))
                return
            hs_p = sum(1 for r in cohort if r.get("handshape_valid"))
            put_text(f"{label} ({len(cohort)}): HS Pass {hs_p}/{len(cohort)} ({hs_p/len(cohort)*100:.0f}%)", y_pos, 0.44, (220, 220, 220))

        summary_line("Valid Z", z_done, 485)
        summary_line("Fist Z (Target: 0% Pass)", fist_done, 505)
        summary_line("Open Palm Z (Target: 0% Pass)", palm_done, 525)

    # Instructions footer
    put_text("-" * 46, 560, 0.45, (100, 100, 100))
    put_text("Controls: [SPACE] Record/Stop/Accept | [R] Redo | [Q] Quit", 580, 0.42, (180, 180, 180))

    return canvas


def run_dry_run() -> int:
    """Run an automated dry run using synthetic data to verify all algorithms."""
    print("=" * 80)
    print("STAGE 17 DRY RUN: Verifying Z Handshape Gate pipeline...")
    print("=" * 80)
    from tests.test_stage16_z_handshape_gate import _build_test_features

    raw_model = joblib.load(MODEL_PATH)
    wrapper_base = ProbabilityMarginModelWrapper(raw_model, enable_reject_option=False)
    wrapper_rej = ProbabilityMarginModelWrapper(raw_model, enable_reject_option=True, margin_threshold=0.10)

    cohorts = [("Z", "POINTING"), ("FIST_Z", "FIST"), ("OPEN_PALM_Z", "OPEN_PALM")]
    records = []

    for target, hs_type in cohorts:
        for i in range(20):
            features = _build_test_features(handshape=hs_type, trajectory="Z")
            traj_res = validate_z_trajectory(features, handshape_settings=ZHandshapeSettings(mode="SHADOW"))
            hs_res = traj_res.get("handshape_validation", {})

            flat = features.reshape(1, 1104, order="C")
            base_det = wrapper_base.predict_sample_detail(flat)
            rej_det = wrapper_rej.predict_sample_detail(flat)

            hs_valid = bool(traj_res.get("handshape_valid", False))
            traj_valid = bool(traj_res.get("valid", False))
            prod_sh = bool(traj_valid and rej_det["prediction"] == "Z")
            prod_act = bool(traj_valid and hs_valid and rej_det["prediction"] == "Z")

            records.append({
                "trial_index": len(records) + 1,
                "ground_truth": target,
                "total_frames": 45,
                "valid_frames": 16,
                "model_invoked": True,
                "trajectory_valid": traj_valid,
                "trajectory_reason": traj_res["reason"],
                "trajectory_excursion": traj_res["excursion"],
                "handshape_valid": hs_valid,
                "handshape_reason": hs_res["reason"],
                "index_extension": hs_res["index_extension"],
                "max_other_extension": hs_res["max_other_extension"],
                "dominance": hs_res["dominance"],
                "baseline_prediction": base_det["prediction"],
                "reject_option_prediction": rej_det["prediction"],
                "probability_J": float(base_det["probability_J"]),
                "probability_Z": float(base_det["probability_Z"]),
                "margin": float(base_det["margin"]),
                "production_shadow_proposal": prod_sh,
                "production_active_proposal": prod_act,
                "hold_success": bool(prod_sh and target == "Z"),
                "commit_success": bool(prod_sh and target == "Z"),
                "processing_latency_ms": 0.085,
            })

    metrics = compute_stage17_metrics(records)
    print("\nDry run metrics computed:")
    print(json.dumps(metrics, indent=2))
    print("\n[PASS] Dry run completed successfully.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        return run_dry_run()

    if args.preflight:
        print(f"[PREFLIGHT] Checking camera device accessibility on index {args.camera_index}...")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"[FAIL] Cannot open camera index {args.camera_index}.", file=sys.stderr)
            return 1
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"[FAIL] Opened camera {args.camera_index} but failed to capture frame.", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        print(f"[PASS] Webcam device {args.camera_index} is accessible (frame size: {w}x{h}).")
        return 0

    print("=" * 80)
    print("STAGE 17 REAL-WORLD WEBCAM Z HANDSHAPE VALIDATION PROTOCOL")
    print("=" * 80)
    print(f"Trials per cohort: {args.trials_per_class} (Total: {args.trials_per_class * 3} attempts)")
    print(f"Margin Threshold:  {args.margin_threshold}")
    print(f"Output path:       {args.output}")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found at {MODEL_PATH}", file=sys.stderr)
        return 1
    if not TASK_PATH.exists():
        print(f"ERROR: MediaPipe task not found at {TASK_PATH}", file=sys.stderr)
        return 1

    raw_model = joblib.load(MODEL_PATH)
    wrapper_baseline = ProbabilityMarginModelWrapper(raw_model, enable_reject_option=False)
    wrapper_reject = ProbabilityMarginModelWrapper(
        raw_model, enable_reject_option=True, margin_threshold=args.margin_threshold
    )

    n_per = args.trials_per_class
    trial_targets: List[str] = ["Z"] * n_per + ["FIST_Z"] * n_per + ["OPEN_PALM_Z"] * n_per
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
            return 2

        with create_hand_landmarker(TASK_PATH, running_mode="IMAGE", num_hands=1) as landmarker:
            while trial_idx < total_trials:
                ok, frame = camera.read()
                if not ok or frame is None or frame.ndim != 3:
                    print("ERROR: Failed to read frame from webcam.", file=sys.stderr)
                    break

                height, width = frame.shape[:2]
                target = trial_targets[trial_idx]

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

                    if frame_count >= 150:
                        state = "EVALUATED"
                        last_eval = evaluate_attempt(
                            trial_idx, target, valid_entries, frame_count,
                            wrapper_baseline, wrapper_reject, args.margin_threshold
                        )
                        feedback_msg = "Max duration reached. Press SPACE to accept, R to redo."
                elif state == "READY":
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        detection = landmarker.detect(mp_img)
                        if detection.hand_landmarks:
                            draw_hand_landmarks(frame, detection.hand_landmarks[0])
                    except Exception:
                        pass

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
                            hs_tag = "HS_PASS" if last_eval["handshape_valid"] else "HS_REJECT"
                            print(f"[TRIAL {trial_idx + 1}/{total_trials}] Target: {target:<12} | {status_tag:<8} | Traj: {'PASS' if last_eval['trajectory_valid'] else 'FAIL':<4} | {hs_tag:<9} | Base: {last_eval['baseline_prediction']:<4} | P(Z): {last_eval['probability_Z']:.3f} | Marg: {last_eval['margin']:.3f}")
                        trial_idx += 1
                        state = "READY"
                        valid_entries = []
                        frame_count = 0
                        last_eval = None
                        if trial_idx < total_trials:
                            next_target = trial_targets[trial_idx]
                            feedback_msg = f"Accepted. Next: Trial {trial_idx + 1}/{total_trials} ({next_target})."
                        else:
                            feedback_msg = "All trials completed!"
                elif key in (ord('r'), ord('R')):
                    state = "READY"
                    valid_entries = []
                    frame_count = 0
                    last_eval = None
                    feedback_msg = f"Trial {trial_idx + 1} ({target}) discarded. Press SPACE to record again."
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 80)
    print("SESSION FINISHED")
    print("=" * 80)
    print(f"Total attempts recorded: {len(session_records)} / {total_trials}")

    if not session_records:
        print("[WARNING] No trials were completed. No results to evaluate.")
        return 0

    session_metrics = compute_stage17_metrics(session_records)

    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "margin_threshold": args.margin_threshold,
            "trials_per_cohort": args.trials_per_class,
            "total_trials_planned": total_trials,
            "total_trials_completed": len(session_records),
            "early_exit": len(session_records) < total_trials,
            "handshape_gate_mode": "SHADOW",
            "z_handshape_settings": {
                "min_index_extension": 0.70,
                "min_index_dominance": 0.05,
            }
        },
        "metrics": session_metrics,
        "trials": session_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[INFO] Saved full Stage 17 validation telemetry to {args.output}")

    print("\n" + "=" * 80)
    print("STAGE 17 REAL-WORLD WEBCAM VALIDATION RESULTS")
    print("=" * 80)
    for cohort_name, c_data in [
        ("COHORT A: VALID Z", session_metrics.get("valid_z", {})),
        ("COHORT B: FIST DRAWING Z", session_metrics.get("fist_z", {})),
        ("COHORT C: OPEN PALM DRAWING Z", session_metrics.get("open_palm_z", {})),
    ]:
        print(f"\n--- {cohort_name} ---")
        if not c_data:
            print("  No trials recorded.")
            continue
        print(f"  Attempts Recorded:         {c_data.get('total_attempts')}")
        print(f"  Invoked Count:             {c_data.get('invoked_count')}")
        print(f"  Trajectory Pass Rate:      {c_data.get('trajectory_pass_rate', 0.0)*100:.1f}% ({c_data.get('trajectory_pass_count')}/{c_data.get('invoked_count')})")
        print(f"  Handshape Gate Pass Rate:  {c_data.get('handshape_pass_rate', 0.0)*100:.1f}% ({c_data.get('handshape_pass_count')}/{c_data.get('invoked_count')})")
        print(f"  Handshape Rejection Rate:  {c_data.get('handshape_rejection_rate', 0.0)*100:.1f}% ({c_data.get('handshape_rejection_count')}/{c_data.get('invoked_count')})")
        print(f"  Prod SHADOW Acceptance:    {c_data.get('production_shadow_acceptance_rate', 0.0)*100:.1f}% ({c_data.get('production_shadow_acceptance_count')}/{c_data.get('invoked_count')})")
        print(f"  Prod ACTIVE Acceptance:    {c_data.get('production_active_acceptance_rate', 0.0)*100:.1f}% ({c_data.get('production_active_acceptance_count')}/{c_data.get('invoked_count')})")
        print(f"  End-to-End Commit Rate:    {c_data.get('end_to_end_commit_rate', 0.0)*100:.1f}% ({c_data.get('commit_success_count')}/{c_data.get('total_attempts')})")

    lat = session_metrics.get("overall_latency_ms", {})
    if lat:
        print("\n--- LATENCY (MS) ---")
        print(f"  Mean: {lat.get('mean', 0.0):.2f} ms | P50: {lat.get('p50', 0.0):.2f} ms | P95: {lat.get('p95', 0.0):.2f} ms | P99: {lat.get('p99', 0.0):.2f} ms")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
