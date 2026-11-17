"""Phase 2 Controlled Webcam Pilot Runner (37 Trials).

Executes the standardized 37-trial evaluation with live camera, MediaPipe tracking,
production coordinators, and interactive onscreen guidance.
STRICTLY READ-ONLY with respect to production code, models, and configurations.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker, landmarks_to_array
from app.inference.letter_inference import HAND_MODEL_PATH
from app.inference.dynamic_jz_auto_camera_experiment import detect_frame
from app.inference.unified_abc123_jz_auto_experiment import detect_static
from app.inference.unified_camera_experiment import prepare_inference_frame
from src.model import load_models
from src.recognition.alphabet import AlphabetRecognizer
from tools.calculate_webcam_metrics import calculate_metrics
from tools.evaluate_webcam_full_system import determine_primary_outcome

# 37 Planned Trials Specification
PLANNED_PILOT_TRIALS: List[Dict[str, Any]] = [
    # 1. ABC (12 trials: A, B, C, D, L, Y x 2)
    {"trial_id": "TRIAL_0001_ABC_A_R1", "branch": "ABC", "ground_truth": "A", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ A: Nắm tay, ngón cái áp sát ngón trỏ"},
    {"trial_id": "TRIAL_0002_ABC_A_R2", "branch": "ABC", "ground_truth": "A", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ A: Nắm tay, ngón cái áp sát ngón trỏ"},
    {"trial_id": "TRIAL_0003_ABC_B_R1", "branch": "ABC", "ground_truth": "B", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ B: Lòng bàn tay mở, 4 ngón khép thẳng, cái gấp"},
    {"trial_id": "TRIAL_0004_ABC_B_R2", "branch": "ABC", "ground_truth": "B", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ B: Lòng bàn tay mở, 4 ngón khép thẳng, cái gấp"},
    {"trial_id": "TRIAL_0005_ABC_C_R1", "branch": "ABC", "ground_truth": "C", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ C: Bàn tay khum hình chữ C"},
    {"trial_id": "TRIAL_0006_ABC_C_R2", "branch": "ABC", "ground_truth": "C", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ C: Bàn tay khum hình chữ C"},
    {"trial_id": "TRIAL_0007_ABC_D_R1", "branch": "ABC", "ground_truth": "D", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ D: Ngón trỏ chỉ thẳng, 3 ngón chạm ngón cái"},
    {"trial_id": "TRIAL_0008_ABC_D_R2", "branch": "ABC", "ground_truth": "D", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ D: Ngón trỏ chỉ thẳng, 3 ngón chạm ngón cái"},
    {"trial_id": "TRIAL_0009_ABC_L_R1", "branch": "ABC", "ground_truth": "L", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ L: Ngón trỏ và ngón cái tạo góc vuông chữ L"},
    {"trial_id": "TRIAL_0010_ABC_L_R2", "branch": "ABC", "ground_truth": "L", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ L: Ngón trỏ và ngón cái tạo góc vuông chữ L"},
    {"trial_id": "TRIAL_0011_ABC_Y_R1", "branch": "ABC", "ground_truth": "Y", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ Y: Ngón cái và ngón út xòe, 3 ngón giữa gập"},
    {"trial_id": "TRIAL_0012_ABC_Y_R2", "branch": "ABC", "ground_truth": "Y", "trial_type": "POSITIVE", "instruction": "Cử chỉ chữ Y: Ngón cái và ngón út xòe, 3 ngón giữa gập"},

    # 2. 123 (10 trials: 0..9 x 1)
    {"trial_id": "TRIAL_0013_123_0_R1", "branch": "123", "ground_truth": "0", "trial_type": "POSITIVE", "instruction": "Chữ số 0: Các ngón tay khum lại chạm đầu ngón cái thành số 0"},
    {"trial_id": "TRIAL_0014_123_1_R1", "branch": "123", "ground_truth": "1", "trial_type": "POSITIVE", "instruction": "Chữ số 1: Ngón trỏ giơ thẳng lên, các ngón khác gập"},
    {"trial_id": "TRIAL_0015_123_2_R1", "branch": "123", "ground_truth": "2", "trial_type": "POSITIVE", "instruction": "Chữ số 2: Ngón trỏ và ngón giữa giơ chữ V, các ngón khác gập"},
    {"trial_id": "TRIAL_0016_123_3_R1", "branch": "123", "ground_truth": "3", "trial_type": "POSITIVE", "instruction": "Chữ số 3: Ngón cái, trỏ, giữa giơ lên (ASL 3)"},
    {"trial_id": "TRIAL_0017_123_4_R1", "branch": "123", "ground_truth": "4", "trial_type": "POSITIVE", "instruction": "Chữ số 4: 4 ngón giơ thẳng, ngón cái gập vào lòng"},
    {"trial_id": "TRIAL_0018_123_5_R1", "branch": "123", "ground_truth": "5", "trial_type": "POSITIVE", "instruction": "Chữ số 5: Xòe cả 5 ngón tay"},
    {"trial_id": "TRIAL_0019_123_6_R1", "branch": "123", "ground_truth": "6", "trial_type": "POSITIVE", "instruction": "Chữ số 6: Ngón út chạm đầu ngón cái, 3 ngón kia giơ thẳng"},
    {"trial_id": "TRIAL_0020_123_7_R1", "branch": "123", "ground_truth": "7", "trial_type": "POSITIVE", "instruction": "Chữ số 7: Ngón đeo nhẫn chạm đầu ngón cái, 3 ngón kia giơ thẳng"},
    {"trial_id": "TRIAL_0021_123_8_R1", "branch": "123", "ground_truth": "8", "trial_type": "POSITIVE", "instruction": "Chữ số 8: Ngón giữa chạm đầu ngón cái, 3 ngón kia giơ thẳng"},
    {"trial_id": "TRIAL_0022_123_9_R1", "branch": "123", "ground_truth": "9", "trial_type": "POSITIVE", "instruction": "Chữ số 9: Ngón trỏ chạm đầu ngón cái (hình tròn), 3 ngón xòe"},

    # 3. J/Z (6 trials: 2 J, 2 Z, 2 Negative)
    {"trial_id": "TRIAL_0023_JZ_J_R1", "branch": "JZ", "ground_truth": "J", "trial_type": "POSITIVE", "instruction": "Cử chỉ J: Dùng ngón út vẽ đường cong chữ J trong không gian"},
    {"trial_id": "TRIAL_0024_JZ_J_R2", "branch": "JZ", "ground_truth": "J", "trial_type": "POSITIVE", "instruction": "Cử chỉ J: Dùng ngón út vẽ đường cong chữ J trong không gian"},
    {"trial_id": "TRIAL_0025_JZ_Z_R1", "branch": "JZ", "ground_truth": "Z", "trial_type": "POSITIVE", "instruction": "Cử chỉ Z: Dùng ngón trỏ vẽ đường zigzag chữ Z trong không gian"},
    {"trial_id": "TRIAL_0026_JZ_Z_R2", "branch": "JZ", "ground_truth": "Z", "trial_type": "POSITIVE", "instruction": "Cử chỉ Z: Dùng ngón trỏ vẽ đường zigzag chữ Z trong không gian"},
    {"trial_id": "TRIAL_0027_JZ_NEG_R1", "branch": "JZ", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính J/Z: Bàn tay tĩnh hoặc nắm tay, KHÔNG vẽ nét J/Z"},
    {"trial_id": "TRIAL_0028_JZ_NEG_R2", "branch": "JZ", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính J/Z: Bàn tay tĩnh hoặc nắm tay, KHÔNG vẽ nét J/Z"},

    # 4. SPACE (9 trials: 5 Positive, 4 Negative)
    {"trial_id": "TRIAL_0029_SPACE_POS_R1", "branch": "SPACE", "ground_truth": "SPACE", "trial_type": "POSITIVE", "instruction": "Cử chỉ SPACE: Mở rộng bàn tay theo cử chỉ quy định SPACE"},
    {"trial_id": "TRIAL_0030_SPACE_POS_R2", "branch": "SPACE", "ground_truth": "SPACE", "trial_type": "POSITIVE", "instruction": "Cử chỉ SPACE: Mở rộng bàn tay theo cử chỉ quy định SPACE"},
    {"trial_id": "TRIAL_0031_SPACE_POS_R3", "branch": "SPACE", "ground_truth": "SPACE", "trial_type": "POSITIVE", "instruction": "Cử chỉ SPACE: Mở rộng bàn tay theo cử chỉ quy định SPACE"},
    {"trial_id": "TRIAL_0032_SPACE_POS_R4", "branch": "SPACE", "ground_truth": "SPACE", "trial_type": "POSITIVE", "instruction": "Cử chỉ SPACE: Mở rộng bàn tay theo cử chỉ quy định SPACE"},
    {"trial_id": "TRIAL_0033_SPACE_POS_R5", "branch": "SPACE", "ground_truth": "SPACE", "trial_type": "POSITIVE", "instruction": "Cử chỉ SPACE: Mở rộng bàn tay theo cử chỉ quy định SPACE"},
    {"trial_id": "TRIAL_0034_SPACE_NEG_R1", "branch": "SPACE", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính SPACE: Tư thế tay nghỉ bình thường, không gõ space"},
    {"trial_id": "TRIAL_0035_SPACE_NEG_R2", "branch": "SPACE", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính SPACE: Tư thế tay nghỉ bình thường, không gõ space"},
    {"trial_id": "TRIAL_0036_SPACE_NEG_R3", "branch": "SPACE", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính SPACE: Tư thế tay nghỉ bình thường, không gõ space"},
    {"trial_id": "TRIAL_0037_SPACE_NEG_R4", "branch": "SPACE", "ground_truth": "NEGATIVE", "trial_type": "NEGATIVE", "instruction": "Âm tính SPACE: Tư thế tay nghỉ bình thường, không gõ space"},
]


def execute_pilot(
    camera_index: int = 0,
    output_dir: Optional[Path] = None,
    trials_subset: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute the Phase 2 Controlled Webcam Pilot."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports/data"
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_to_run = list(trials_subset) if trials_subset is not None else list(PLANNED_PILOT_TRIALS)
    total_trials = len(trials_to_run)

    jsonl_path = output_dir / "webcam_phase2_pilot_trials.jsonl"
    results_path = output_dir / "webcam_phase2_pilot_results.json"

    print(f"\n=======================================================")
    print(f" KHỞI ĐỘNG PHASE 2 WEBCAM PILOT: {total_trials} LƯỢT")
    print(f" Camera Index: {camera_index}")
    print(f" File JSONL: {jsonl_path}")
    print(f" File Results: {results_path}")
    print(f"=======================================================\n")

    # Load production models
    bundle = load_models()
    coordinator = AlphabetRecognizer.from_models(
        bundle,
        enable_experimental_jz=True,
        enable_space=True,
    )

    trials_results: List[Dict[str, Any]] = []
    window_name = "PHASE 2: WEBCAM PILOT (37 TRIALS)"

    try:
        with ExitStack() as stack:
            video_task = stack.enter_context(
                create_hand_landmarker(
                    model_path=HAND_MODEL_PATH,
                    running_mode="VIDEO",
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            image_task = stack.enter_context(
                create_hand_landmarker(
                    HAND_MODEL_PATH,
                    running_mode="IMAGE",
                    num_hands=1,
                )
            )
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise OSError(f"Cannot open webcam index {camera_index}")

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 600)

            frame_idx = 0
            session_start = time.monotonic()
            last_timestamp_ms = -1

            with open(jsonl_path, "w", encoding="utf-8") as stream:
                for trial_idx, spec in enumerate(trials_to_run):
                    trial_id = spec["trial_id"]
                    branch = spec["branch"]
                    gt = spec["ground_truth"]
                    ttype = spec["trial_type"]
                    instr = spec["instruction"]

                    print(f"[{trial_idx+1}/{total_trials}] {trial_id} | Nhánh: {branch} | Target: {gt}")

                    # 1. State Isolation: Switch mode & Reset
                    target_mode = "123" if branch == "123" else "ABC"
                    if coordinator.mode != target_mode:
                        coordinator.switch_mode(target_mode)

                    coordinator.composer.clear()
                    hold = coordinator.hold
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

                    # 2. Settling Phase: Guaranteed by time.monotonic() >= 0.80s (and >= 25 frames)
                    settle_start = time.monotonic()
                    settle_frames = 0
                    while (time.monotonic() - settle_start < 0.85) or settle_frames < 25:
                        ok, raw = camera.read()
                        if not ok or raw is None:
                            continue
                        settle_frames += 1
                        frame_idx += 1
                        now = time.monotonic()
                        coordinator.update(frame_idx, None, None, raw.shape, now, raw_hand_count=0)

                        # Render settling banner
                        disp = raw.copy()
                        disp = cv2.flip(disp, 1)
                        h_d, w_d = disp.shape[:2]
                        cv2.rectangle(disp, (0, 0), (w_d, 60), (30, 30, 30), -1)
                        cv2.putText(disp, f"TRIAL {trial_idx+1}/{total_trials}: THA LONG TAY (SETTLING)...",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                        cv2.imshow(window_name, disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            raise KeyboardInterrupt("Pilot aborted by user")

                    # 3. Preparation / Ready Countdown Stage (3.0 seconds)
                    prep_start = time.monotonic()
                    prep_duration = 3.0
                    while time.monotonic() - prep_start < prep_duration:
                        ok, raw = camera.read()
                        if not ok or raw is None:
                            continue
                        now = time.monotonic()
                        elapsed_prep = now - prep_start
                        remaining = max(0.0, prep_duration - elapsed_prep)

                        disp = raw.copy()
                        disp = cv2.flip(disp, 1)
                        h_d, w_d = disp.shape[:2]

                        # Overlay top bar
                        cv2.rectangle(disp, (0, 0), (w_d, 90), (20, 24, 28), -1)
                        cv2.putText(disp, f"LƯỢT {trial_idx+1}/{total_trials}: NHÁNH {branch} - MỤC TIÊU: [ {gt} ]",
                                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
                        cv2.putText(disp, f"{instr}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

                        # Center countdown
                        cd_text = f"CHUAN BI: {remaining:.1f}s (Nhan SPACE de bat dau)"
                        cv2.rectangle(disp, (w_d//2 - 250, h_d//2 - 35), (w_d//2 + 250, h_d//2 + 35), (0, 0, 0), -1)
                        cv2.rectangle(disp, (w_d//2 - 250, h_d//2 - 35), (w_d//2 + 250, h_d//2 + 35), (0, 215, 255), 2)
                        cv2.putText(disp, cd_text, (w_d//2 - 230, h_d//2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        cv2.imshow(window_name, disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 32:  # Space to skip countdown
                            break
                        elif key in (ord("q"), ord("Q"), 27):
                            raise KeyboardInterrupt("Pilot aborted by user")

                    # 4. Active Trial Window (Allows full 3.0s dwell without premature timeout)
                    # Window is 7.5 seconds total; guarantees plenty of time to position & hold
                    trial_start_mono = time.monotonic()
                    max_trial_duration = 7.5
                    commits: List[Tuple[str, float]] = []
                    raw_predictions: List[str] = []
                    confidences: List[float] = []
                    inference_times: List[float] = []
                    coord_times: List[float] = []
                    landmarks_detected_ever = False
                    validator_rejected = False
                    hold_aborted = False
                    candidate = None
                    hold_started_time: Optional[float] = None

                    while True:
                        now = time.monotonic()
                        elapsed_trial = now - trial_start_mono
                        if elapsed_trial >= max_trial_duration:
                            break

                        ok, raw = camera.read()
                        if not ok or raw is None:
                            continue
                        frame_idx += 1
                        shape = raw.shape

                        # Timing: Model inference
                        inference_frame = prepare_inference_frame(raw, coordinator.mode)
                        raw_timestamp_ms = int((now - session_start) * 1000)
                        raw_timestamp_ms = max(raw_timestamp_ms, last_timestamp_ms + 1)
                        last_timestamp_ms = raw_timestamp_ms

                        t0_inf = time.perf_counter()
                        static_hand = detect_static(video_task, inference_frame, raw_timestamp_ms)
                        t_inf = (time.perf_counter() - t0_inf) * 1000.0
                        inference_times.append(t_inf)

                        if static_hand is not None:
                            landmarks_detected_ever = True

                        raw_hand, count, dynamic_err = (
                            detect_frame(image_task, raw) if coordinator.mode == "ABC" else (None, None, "")
                        )
                        if raw_hand is not None:
                            landmarks_detected_ever = True

                        # Timing: Coordinator processing
                        t0_coord = time.perf_counter()
                        pred, comm = coordinator.update(
                            frame_idx,
                            static_hand,
                            raw_hand,
                            shape,
                            now,
                            dynamic_error=dynamic_err,
                            raw_hand_count=count,
                        )
                        t_coord = (time.perf_counter() - t0_coord) * 1000.0
                        coord_times.append(t_coord)

                        if pred:
                            raw_predictions.append(pred.instant_label)
                            confidences.append(pred.instant_confidence)

                        candidate = getattr(coordinator, "display_label", None)
                        if coordinator.hold.state in ("STABILIZING", "HOLDING") and hold_started_time is None:
                            hold_started_time = now

                        if comm:
                            commits.append((comm, now))
                            # Single commit fired, break to conclude trial
                            break

                        # Check validator status for dynamic J/Z
                        val_res = getattr(coordinator, "validation", None)
                        if val_res is not None and getattr(val_res, "status", None) == "REJECTED":
                            validator_rejected = True

                        # Display active HUD
                        disp = raw.copy()
                        disp = cv2.flip(disp, 1)
                        h_d, w_d = disp.shape[:2]

                        cv2.rectangle(disp, (0, 0), (w_d, 80), (20, 24, 28), -1)
                        cv2.putText(disp, f"LƯỢT {trial_idx+1}/{total_trials}: [ {gt} ] ({ttype})",
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120) if ttype=="POSITIVE" else (255, 150, 50), 2)
                        time_left = max(0.0, max_trial_duration - elapsed_trial)
                        cv2.putText(disp, f"Thoi gian con: {time_left:.1f}s | Nhan dien: {pred.instant_label if pred else 'None'}",
                                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                        # Hold progress bar
                        prog = coordinator.hold.progress
                        bar_w = int(w_d * 0.8)
                        bx1 = int(w_d * 0.1)
                        by = h_d - 45
                        cv2.rectangle(disp, (bx1, by), (bx1 + bar_w, by + 25), (40, 40, 40), -1)
                        fill_w = int(bar_w * prog)
                        bar_color = (0, 220, 255) if prog < 1.0 else (0, 255, 0)
                        if fill_w > 0:
                            cv2.rectangle(disp, (bx1, by), (bx1 + fill_w, by + 25), bar_color, -1)
                        cv2.rectangle(disp, (bx1, by), (bx1 + bar_w, by + 25), (200, 200, 200), 2)
                        hold_text = f"HOLD DWELL: {prog*100:.0f}% (State: {coordinator.hold.state})"
                        cv2.putText(disp, hold_text, (bx1 + 10, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if fill_w > 180 else (255, 255, 255), 1)

                        cv2.imshow(window_name, disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            raise KeyboardInterrupt("Pilot aborted by user")

                    # 5. Outcome & Telemetry Resolution
                    trial_elapsed = time.monotonic() - trial_start_mono
                    final_raw_pred = raw_predictions[-1] if raw_predictions else None
                    mean_conf = float(np.mean(confidences)) if confidences else 0.0
                    mean_inf = float(np.mean(inference_times)) if inference_times else 0.0
                    mean_coord = float(np.mean(coord_times)) if coord_times else 0.0

                    # True end-to-end commit latency includes user hold duration (usually >= 3000ms)
                    if commits and hold_started_time is not None:
                        e2e_commit_latency_ms = (commits[0][1] - hold_started_time) * 1000.0
                    elif commits:
                        e2e_commit_latency_ms = trial_elapsed * 1000.0
                    else:
                        e2e_commit_latency_ms = 0.0

                    outcome, diag_flags = determine_primary_outcome(
                        ground_truth=gt,
                        commits=commits,
                        landmarks_detected=landmarks_detected_ever,
                        validator_rejected=validator_rejected,
                        hold_aborted=(coordinator.hold.state == "IDLE" and hold_started_time is not None and not commits),
                        raw_prediction=final_raw_pred,
                        candidate=candidate,
                        timeout_occurred=(not commits and trial_elapsed >= max_trial_duration - 0.2),
                    )

                    commit_text = commits[0][0] if commits else ""

                    trial_record = {
                        "trial_id": trial_id,
                        "branch": branch,
                        "ground_truth": gt,
                        "trial_type": ttype,
                        "landmarks_detected": landmarks_detected_ever,
                        "raw_prediction": final_raw_pred,
                        "prediction_confidence": mean_conf,
                        "candidate": candidate,
                        "validation_result": getattr(coordinator, "validation", None),
                        "hold_state": coordinator.hold.state,
                        "commit_text": commit_text,
                        "commit_count": len(commits),
                        "elapsed_time": trial_elapsed,
                        "elapsed_time_ms": trial_elapsed * 1000.0,
                        "inference_latency_ms": mean_inf,
                        "coordinator_latency_ms": mean_coord,
                        "end_to_end_commit_latency_ms": e2e_commit_latency_ms,
                        "primary_outcome": outcome,
                        "diagnostic_flags": diag_flags,
                        "data_source": "webcam",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }

                    trials_results.append(trial_record)
                    stream.write(json.dumps(trial_record, ensure_ascii=False) + "\n")
                    stream.flush()

                    print(f"   -> Kết quả: {outcome} | Commit: '{commit_text}' | Flags: {diag_flags}")

                    # 6. Post-Trial Brief Feedback Screen (1.0s)
                    fb_start = time.monotonic()
                    while time.monotonic() - fb_start < 1.0:
                        ok, raw = camera.read()
                        if not ok or raw is None:
                            continue
                        disp = raw.copy()
                        disp = cv2.flip(disp, 1)
                        h_d, w_d = disp.shape[:2]

                        color = (0, 200, 0) if outcome == "SUCCESS" else (0, 0, 220)
                        cv2.rectangle(disp, (w_d//2 - 250, h_d//2 - 50), (w_d//2 + 250, h_d//2 + 50), (0, 0, 0), -1)
                        cv2.rectangle(disp, (w_d//2 - 250, h_d//2 - 50), (w_d//2 + 250, h_d//2 + 50), color, 3)
                        cv2.putText(disp, f"KET QUA: {outcome}", (w_d//2 - 230, h_d//2 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                        cv2.putText(disp, f"Commit: '{commit_text}' (Target: {gt})", (w_d//2 - 230, h_d//2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        cv2.imshow(window_name, disp)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            raise KeyboardInterrupt("Pilot aborted by user")

    except KeyboardInterrupt:
        print("\n[!] Thử nghiệm bị dừng sớm bởi người dùng.")
    finally:
        cv2.destroyAllWindows()
        print("\n[✓] Camera đã được đóng và giải phóng tài nguyên.")

    # Calculate overall pilot metrics
    metrics = calculate_metrics(trials_results)
    metrics["metadata"] = {
        "pilot_session_timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ"),
        "data_source": "webcam",
        "planned_trials": total_trials,
        "completed_trials": len(trials_results),
        "jsonl_path": str(jsonl_path),
        "note": "PHASE 2 CONTROLLED WEBCAM PILOT (37 TRIALS). REAL HARDWARE DATA.",
    }

    results_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[✓] Tổng kết Pilot đã ghi tại: {results_path}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 2 Webcam Pilot")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N trials")
    args = parser.parse_args()

    subset = PLANNED_PILOT_TRIALS[:args.limit] if args.limit else None
    execute_pilot(camera_index=args.camera_index, trials_subset=subset)
