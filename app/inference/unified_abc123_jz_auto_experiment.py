"""Separate EXPERIMENTAL two-button ABC (24 static + latest auto J/Z) / 123.

One camera, independent mirrored VIDEO and raw IMAGE passes. J/Z commits are
OFF by default: binary probabilities and motion gates are not a NONE model.
No production dispatcher, baseline camera, model or HOLD implementation changes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from ai.feature_extraction.hand_landmarks import create_hand_landmarker,landmarks_to_array  # noqa: E402
from app.inference.dynamic_jz_auto_camera_experiment import detect_frame  # noqa: E402
from app.inference.dynamic_jz_auto_diagnostics import DebugTrace  # noqa: E402
from app.inference.dynamic_jz_camera_test import load_experiment_model,_synthetic_feature_check  # noqa: E402
from app.inference.hold_confirmation_camera_test import _panel  # noqa: E402
from app.inference.letter_inference import (  # noqa: E402
    HAND_MODEL_PATH,DEFAULT_MODEL,IntegrationError,StaticAslInference,
    load_official_artifacts,_validate_deployment_config,
)
from app.inference.unified_camera_experiment import prepare_inference_frame,render_mirrored_frame  # noqa: E402
from app.inference.unified_static_coordinator import DigitPredictor,load_digit_model  # noqa: E402
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator  # noqa: E402
from app.inference.jz_attempt_capture import AttemptCapture,checked_output  # noqa: E402
from app.inference.space_predictor import load_space_predictor  # noqa: E402
from src.ui.window_display import CameraWindowController  # noqa: E402

WINDOW = "EXPERIMENTAL ABC A-Z / 123 - latest automatic J/Z"


def compute_suggestion_hitboxes(width: int, suggestions: Sequence[str]) -> list[tuple[int, int, int, int, str]]:
    """Compute (x1, y1, x2, y2, word) hitboxes for up to 5 word suggestions.

    Pills are laid out in the side panel (x: width + 14 to width + 296, y: 350 to 411).
    Guarantees:
    - Never overlaps with ABC/123 buttons (y: 8 to 47).
    - Never overlaps with camera frame (x < width).
    - Never overlaps with HOLD status or bottom TEXT line (y: 423).
    - Individual hitboxes are strictly non-overlapping.
    """
    if not suggestions:
        return []

    words = list(suggestions)[:5]
    n = len(words)
    hitboxes: list[tuple[int, int, int, int, str]] = []

    # Row 1: y1 = 350, y2 = 378 (height 28 px)
    # Row 2: y1 = 383, y2 = 411 (height 28 px)
    if n == 1:
        hitboxes.append((width + 60, 350, width + 250, 378, words[0]))
    elif n == 2:
        hitboxes.append((width + 14, 350, width + 149, 378, words[0]))
        hitboxes.append((width + 161, 350, width + 296, 378, words[1]))
    elif n == 3:
        hitboxes.append((width + 14, 350, width + 102, 378, words[0]))
        hitboxes.append((width + 111, 350, width + 199, 378, words[1]))
        hitboxes.append((width + 208, 350, width + 296, 378, words[2]))
    elif n == 4:
        hitboxes.append((width + 14, 350, width + 149, 378, words[0]))
        hitboxes.append((width + 161, 350, width + 296, 378, words[1]))
        hitboxes.append((width + 14, 383, width + 149, 411, words[2]))
        hitboxes.append((width + 161, 383, width + 296, 411, words[3]))
    elif n >= 5:
        hitboxes.append((width + 14, 350, width + 102, 378, words[0]))
        hitboxes.append((width + 111, 350, width + 199, 378, words[1]))
        hitboxes.append((width + 208, 350, width + 296, 378, words[2]))
        hitboxes.append((width + 14, 383, width + 149, 411, words[3]))
        hitboxes.append((width + 161, 383, width + 296, 411, words[4]))

    return hitboxes


def compute_sentence_hitboxes(width: int, sentences: Sequence[str]) -> list[tuple[int, int, int, int, str]]:
    """Compute (x1, y1, x2, y2, sentence) hitboxes for up to 3 sentence suggestions.

    Pills are laid out horizontally across the side panel (x: width + 14 to width + 296, y: 346 to 414).
    Guarantees:
    - Never overlaps with ABC/123 buttons (y: 8 to 47).
    - Never overlaps with camera frame (x < width).
    - Never overlaps with HOLD status (y: 266-325) or bottom TEXT line (y: 423).
    - Individual hitboxes are strictly non-overlapping.
    - Fits up to 3 sentence suggestions.
    """
    if not sentences:
        return []

    clean_sentences = list(sentences)[:3]
    n = len(clean_sentences)
    hitboxes: list[tuple[int, int, int, int, str]] = []

    x1 = width + 14
    x2 = width + 296

    if n == 1:
        hitboxes.append((x1, 352, x2, 405, clean_sentences[0]))
    elif n == 2:
        hitboxes.append((x1, 348, x2, 377, clean_sentences[0]))
        hitboxes.append((x1, 383, x2, 412, clean_sentences[1]))
    elif n >= 3:
        hitboxes.append((x1, 346, x2, 367, clean_sentences[0]))
        hitboxes.append((x1, 370, x2, 391, clean_sentences[1]))
        hitboxes.append((x1, 394, x2, 414, clean_sentences[2]))

    return hitboxes


def _render_sentence_pill(display, x1: int, y1: int, x2: int, y2: int, sentence: str, index_1_based: int):
    """Draw a single sentence suggestion pill with wrapping and clean formatting."""
    cv2.rectangle(display, (x1, y1), (x2, y2), (245, 238, 225), -1)
    cv2.rectangle(display, (x1, y1), (x2, y2), (160, 130, 90), 1)

    pill_w = x2 - x1
    pill_h = y2 - y1
    max_text_w = pill_w - 14
    display_text = f"{index_1_based}. {sentence}"

    font_scale = 0.33
    (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    if tw <= max_text_w and th <= pill_h - 4:
        tx = x1 + 8
        ty = y1 + (pill_h + th) // 2
        cv2.putText(display, display_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    else:
        font_scale = 0.27
        words = display_text.split()
        lines = []
        curr = []
        for w in words:
            cand = " ".join(curr + [w])
            (cw, _), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            if cw <= max_text_w:
                curr.append(w)
            else:
                if curr:
                    lines.append(" ".join(curr))
                    curr = [w]
                else:
                    lines.append(w)
                    curr = []
        if curr:
            lines.append(" ".join(curr))

        n_lines = len(lines)
        (_, line_h), _ = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        line_spacing = line_h + 2
        total_h = n_lines * line_h + (n_lines - 1) * 2
        y_start = y1 + max(line_h + 2, (pill_h - total_h) // 2 + line_h)

        for i, line in enumerate(lines[:2]):
            tx = x1 + 8
            ty = y_start + i * line_spacing
            cv2.putText(display, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)


def mode_from_click(x,y,width):
    if 8 <= y <= 47:
        if 14 <= x-width <= 145:
            return "ABC"
        if 165 <= x-width <= 296:
            return "123"
    return None


def text_action_from_click(x: int, y: int, width: int, height: int) -> str | None:
    """Determine if a canvas click at (x, y) hit Backspace or Clear button.

    Args:
        x: Logical canvas x coordinate.
        y: Logical canvas y coordinate.
        width: Camera frame width (offset to side panel).
        height: Frame/panel height.

    Returns:
        'BACKSPACE', 'CLEAR', or None.
    """
    panel_x = x - width
    if height - 60 <= y <= height - 36:
        if 156 <= panel_x <= 224:
            return "BACKSPACE"
        if 232 <= panel_x <= 296:
            return "CLEAR"
    return None


def render(inference_frame,mode,static_hand,raw_frame,raw_hand,coordinator,now):
    if mode == "ABC" and static_hand is None and raw_hand is not None:
        # Raw overlays are drawn BEFORE mirroring the display, not by changing
        # landmark coordinates used by J/Z or borrowing the VIDEO observation.
        frame = render_mirrored_frame(raw_frame,"123",raw_hand)
    else:
        frame = render_mirrored_frame(inference_frame,mode,static_hand)
    display = _panel(frame,coordinator.hold,coordinator.display_label,now)
    width = frame.shape[1]
    for label,left,right in (("ABC",14,145),("123",165,296)):
        color = (218,240,218) if mode == label else (235,235,235)
        cv2.rectangle(display,(width+left,8),(width+right,47),color,-1)
        cv2.rectangle(display,(width+left,8),(width+right,47),(65,65,65),1)
        cv2.putText(display,label,(width+left+34,36),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
    def text(value,y,scale=.36):
        cv2.putText(display,value,(width+8,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),1,cv2.LINE_AA)
    text(coordinator.status,61,.42)
    text("JZ PAUSED IN 123" if mode == "123" else
         "JZ commits: "+("EXPERIMENTAL ON" if coordinator.enable_experimental_jz else "OFF / DIAGNOSTIC"),76)
    if coordinator.z_start_pose_shadow is not None and mode == "ABC":
        report=coordinator.z_start_pose_report or {}
        pose=report.get("pose") or {}
        text(f"Z POSE SHADOW: {pose.get('pose_status','UNCERTAIN')} / {report.get('state','IDLE')}",91,.33)
    if coordinator.display_dynamic_result:
        result = coordinator.display_dynamic_result
        text(f"P(J)={result['probability_J']:.3f}  P(Z)={result['probability_Z']:.3f}",244,.42)
        text("Technical candidate / ASL unverified",260,.35)
    elif coordinator.raw_dynamic_prediction:
        result = coordinator.raw_dynamic_prediction
        text(f"RAW {result['prediction']} / NOT A CANDIDATE",244,.39)
        text(f"P(J)={result['probability_J']:.3f} P(Z)={result['probability_Z']:.3f}",260,.36)
    elif mode == "ABC" and coordinator.session.segmenter.active:
        text(f"JZ valid={coordinator.session.valid_frames}  gap={coordinator.session.segmenter.missing_frames}",244,.40)

    # Word & Sentence suggestions section (strictly mutually exclusive)
    current_pfx = getattr(coordinator.composer, "current_prefix", "") if hasattr(coordinator, "composer") else ""
    suggestions = getattr(coordinator, "suggestions", [])
    sentence_suggestions = getattr(coordinator, "sentence_suggestions", [])

    if current_pfx and suggestions:
        # Priority 1: User is typing a word -> Word Suggestions
        hitboxes = compute_suggestion_hitboxes(width, suggestions)
        coordinator.suggestion_hitboxes = hitboxes
        coordinator.sentence_hitboxes = []

        cv2.rectangle(display, (width + 8, 332), (width + 302, 416), (242, 244, 246), -1)
        cv2.putText(display, "SUGGESTIONS (CLICK TO SELECT):", (width + 12, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1, cv2.LINE_AA)
        for x1, y1, x2, y2, word in hitboxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (238, 228, 208), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (160, 130, 90), 1)
            font_scale = 0.48
            text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pill_w = x2 - x1
            if text_size[0] > pill_w - 6:
                font_scale = max(0.30, font_scale * (pill_w - 6) / max(1, text_size[0]))
                text_size, _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = x1 + (pill_w - text_size[0]) // 2
            ty = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(display, word, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
    elif not current_pfx and mode == "ABC" and sentence_suggestions:
        # Priority 2: Word boundary with committed context -> Sentence Suggestions
        sentence_hitboxes = compute_sentence_hitboxes(width, sentence_suggestions)
        coordinator.sentence_hitboxes = sentence_hitboxes
        coordinator.suggestion_hitboxes = []

        cv2.rectangle(display, (width + 8, 330), (width + 302, 416), (242, 244, 246), -1)
        cv2.putText(display, "SENTENCE SUGGESTIONS (CLICK):", (width + 12, 342),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1, cv2.LINE_AA)
        for i, (x1, y1, x2, y2, sentence) in enumerate(sentence_hitboxes):
            _render_sentence_pill(display, x1, y1, x2, y2, sentence, i + 1)
    else:
        coordinator.suggestion_hitboxes = []
        coordinator.sentence_hitboxes = []
        text("Binary score is NOT gesture validity",display.shape[0]-80,.36)
    if getattr(coordinator, "jz_evaluation_mode", False):
        h = display.shape[0]
        box_w = min(width - 10, 460)
        if box_w > 120 and h > 120:
            y0 = h - 108
            cv2.rectangle(display, (10, y0), (box_w, h - 10), (24, 28, 34), -1)
            cv2.rectangle(display, (10, y0), (box_w, h - 10), (95, 190, 255), 1)
            rec = getattr(coordinator, "jz_evaluation_record", None)
            if rec is None:
                lines = [
                    ("JZ EVALUATION MODE: WAITING FOR MOTION", (120, 220, 255)),
                    ("Model P(J)/P(Z) + Validator + Reason shown here", (210, 210, 210)),
                ]
            else:
                raw_pred = rec.get("raw_prediction") or "NONE"
                conf = rec.get("confidence")
                pj = rec.get("probability_J")
                pz = rec.get("probability_Z")
                val_status = rec.get("validator_status", "SKIPPED")
                reason = rec.get("reason", "NONE")
                final_out = rec.get("final_output", "NONE")
                pred_str = (f"MODEL PREDICTION: {raw_pred} ({conf*100:.1f}%)  "
                            f"P(J)={pj:.3f} P(Z)={pz:.3f}"
                            if raw_pred != "NONE" and conf is not None and pj is not None and pz is not None
                            else f"MODEL PREDICTION: {raw_pred}")
                val_prefix = f"{raw_pred} VALIDATOR" if raw_pred in ("J", "Z") else "VALIDATOR"
                val_color = ((110, 235, 130) if val_status == "ACCEPTED"
                             else (110, 145, 255) if val_status == "REJECTED"
                             else (210, 210, 210))
                lines = [
                    (pred_str, (255, 255, 255)),
                    (f"{val_prefix}: {val_status}", val_color),
                    (f"REASON: {reason[:48]}", (175, 215, 255)),
                    (f"FINAL OUTPUT: {final_out}", (120, 240, 170) if rec.get("committed") else (240, 240, 240)),
                ]
            for idx, (line, color) in enumerate(lines):
                cv2.putText(display, line, (18, y0 + 22 + idx * 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    if display.shape[0] >= 460:
        cv2.putText(display, "F: Fullscreen | R: Reset | Q/ESC: Quit",
                    (width + 8, display.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (85, 85, 85), 1, cv2.LINE_AA)
    return display


def detect_static(task,frame,timestamp_ms):
    """Baseline VIDEO input contract; failures do not gate the raw IMAGE pass."""
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = task.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp_ms)
    if len(result.hand_landmarks) != 1:
        return None
    hand = result.hand_landmarks[0]
    landmarks_to_array(hand)
    return hand


def run_camera(coordinator,camera_index):
    camera = None
    requested = [coordinator.mode]
    width = [0]
    height = [480]
    window_ctrl = CameraWindowController(WINDOW)
    def on_mouse(event,x,y,_flags,_userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            mapped = window_ctrl.map_mouse(x, y)
            if mapped is None:
                return
            x, y = mapped
            selected = mode_from_click(x,y,width[0])
            if selected:
                requested[0] = selected
                return
            action = text_action_from_click(x, y, width[0], height[0])
            if action == "BACKSPACE":
                coordinator.backspace()
                print(f"BUTTON_CLICK=BACKSPACE TEXT={coordinator.composer.text}", flush=True)
                return
            elif action == "CLEAR":
                coordinator.clear()
                print(f"BUTTON_CLICK=CLEAR TEXT={coordinator.composer.text}", flush=True)
                return
            hitboxes = getattr(coordinator, "suggestion_hitboxes", [])
            if hitboxes:
                for x1, y1, x2, y2, word in list(hitboxes):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        current_valid = coordinator.suggestions
                        if word in current_valid:
                            committed = coordinator.composer.commit_word(word, valid_suggestions=current_valid)
                            if committed:
                                if coordinator.suggestion_manager is not None:
                                    coordinator.suggestion_manager.update_text(coordinator.composer.text)
                                coordinator.hold.text = coordinator.composer.text
                                coordinator.suggestion_hitboxes = []
                                print(f"WORD_SELECTED={word} TEXT={coordinator.composer.text}", flush=True)
                        break
                return
            sentence_hitboxes = getattr(coordinator, "sentence_hitboxes", [])
            if sentence_hitboxes:
                for x1, y1, x2, y2, sentence in list(sentence_hitboxes):
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        current_valid = getattr(coordinator, "sentence_suggestions", [])
                        if coordinator.suggestion_manager is not None and hasattr(coordinator.suggestion_manager, "current_sentence_context"):
                            expected_ctx = coordinator.suggestion_manager.current_sentence_context
                        else:
                            expected_ctx = coordinator.composer.committed_context
                        if sentence in current_valid:
                            committed = coordinator.composer.commit_sentence(
                                sentence,
                                valid_suggestions=current_valid,
                                expected_context=expected_ctx
                            )
                            if committed:
                                if coordinator.suggestion_manager is not None:
                                    coordinator.suggestion_manager.update_text(coordinator.composer.text)
                                coordinator.hold.text = coordinator.composer.text
                                coordinator.sentence_hitboxes = []
                                print(f"SENTENCE_SELECTED={sentence} TEXT={coordinator.composer.text}", flush=True)
                        break
                return
    try:
        with ExitStack() as stack:
            video_task = stack.enter_context(create_hand_landmarker(
                model_path=HAND_MODEL_PATH,running_mode="VIDEO",num_hands=1,
                min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5))
            image_task = stack.enter_context(create_hand_landmarker(HAND_MODEL_PATH,running_mode="IMAGE",num_hands=1))
            camera = cv2.VideoCapture(camera_index)  # The ONLY physical capture.
            if not camera.isOpened():
                raise OSError("Cannot open camera")
            window_ctrl.create()
            cv2.setMouseCallback(WINDOW,on_mouse)
            started = time.monotonic()
            last_timestamp = -1
            previous_time = None
            last_debug = -float("inf")
            index = failed_reads = 0
            shape = (480,640,3)
            while True:
                coordinator.switch_mode(requested[0])
                ok,raw = camera.read()
                now = time.monotonic()
                if not ok or raw is None:
                    failed_reads += 1
                    coordinator.update(index,None,None,shape,now,dynamic_error="CAMERA_READ_FAILED",raw_hand_count=0)
                    index += 1
                    if failed_reads >= 10:
                        return 4
                    continue
                failed_reads = 0
                shape = raw.shape
                inference_frame = prepare_inference_frame(raw,coordinator.mode)
                width[0] = raw.shape[1]
                height[0] = raw.shape[0]
                timestamp = max(int((now-started)*1000),last_timestamp+1)
                last_timestamp = timestamp
                static_hand = None
                video_error = ""
                try:
                    static_hand = detect_static(video_task,inference_frame,timestamp)
                except (ValueError,RuntimeError,TypeError,AttributeError) as exc:
                    video_error = str(exc)
                # This call is independent of static_hand/video_error. No
                # cross-pass correspondence gate and no mirrored substitutes.
                raw_hand,count,error = (detect_frame(image_task,raw) if coordinator.mode == "ABC"
                                        else (None,None,""))
                _prediction,committed = coordinator.update(index,static_hand,raw_hand,shape,now,
                                                          dynamic_error=error,raw_hand_count=count,static_error=video_error)
                if committed:
                    print(f"CONFIRMED={committed} TEXT={coordinator.composer.text}",flush=True)
                display = render(inference_frame,coordinator.mode,static_hand,raw,raw_hand,coordinator,now)
                frame_to_show = window_ctrl.prepare_display(display)
                cv2.imshow(WINDOW,frame_to_show)
                key = cv2.waitKey(1)&0xff
                coordinator.note_displayed(now)
                if coordinator.event_sink and now-last_debug >= .5:
                    fps = 1/(now-previous_time) if previous_time is not None and now>previous_time else 0.
                    coordinator.emit("RUNTIME_SAMPLE",now,fps=fps,video_error=video_error,
                                     jz_state=coordinator.session.state,hold_state=coordinator.hold.state)
                    last_debug = now
                previous_time = now
                if window_ctrl.is_window_closed():
                    return 0
                if key in (ord("f"),ord("F")):
                    window_ctrl.toggle_fullscreen()
                elif key in (ord("r"),ord("R")):
                    window_ctrl.restore_initial_size()
                elif key in (ord("q"),ord("Q"),27):
                    return 0
                elif key in (8, 127):
                    coordinator.backspace()
                index += 1
    finally:
        try:
            coordinator.close()
            if coordinator.event_sink:
                coordinator.session.close_debug_session(time.monotonic())
                coordinator.emit("UNIFIED_SESSION_END",time.monotonic(),text=coordinator.composer.text)
        finally:
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()


def main(argv=None, *, coordinator_factory=UnifiedAutoCoordinator, artifact_loader=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--camera-index",type=int,default=0)
    parser.add_argument("--enable-experimental-jz",action="store_true",help="Opt in to J/Z endpoint HOLD commits; unvalidated")
    space_group = parser.add_mutually_exclusive_group()
    space_group.add_argument("--enable-space",dest="enable_space",action="store_true",default=None,help="Enable SPACE gesture recognition in ABC and 123 modes.")
    space_group.add_argument("--disable-space",dest="enable_space",action="store_false",default=None,help="Disable SPACE gesture recognition; keep baseline ABC/123 behavior.")
    suggestion_group = parser.add_mutually_exclusive_group()
    suggestion_group.add_argument("--enable-word-suggestion",dest="enable_word_suggestion",action="store_true",default=True,help="Enable offline word suggestion and click-to-select (default: True).")
    suggestion_group.add_argument("--disable-word-suggestion",dest="enable_word_suggestion",action="store_false",help="Disable offline word suggestion.")
    parser.add_argument("--debug",action="store_true",help="Temporary buffered event JSONL, never per-frame Terminal logs")
    parser.add_argument("--capture-jz-diagnostics",action="store_true",
                        help="Opt in to landmark/feature-only diagnostic capture; no images/video or training data")
    parser.add_argument("--z-start-pose-shadow",action="store_true",
                        help="Phase-A Z pose/READY/onset diagnostics ONLY; no acceptance effect, not a gate")
    parser.add_argument("--jz-evaluation-mode",action="store_true",
                        help="Surface raw J/Z probabilities, validator decision, rejection reason and final output without auto-committing unverified J/Z")
    parser.add_argument("--no-jz-reject-option",dest="enable_jz_reject_option",action="store_false",default=True,
                        help="Disable Probability Margin Reject Option; revert to raw baseline Logistic Regression")
    parser.add_argument("--jz-margin-threshold",type=float,default=0.10,
                        help="Probability Margin threshold |P(J) - P(Z)| for Reject Option (default: 0.10)")
    parser.add_argument("--jz-diagnostics-dir",type=Path,
                        default=ROOT/"data/experiments/dynamic_jz_v1/diagnostics",
                        help="Temporary output inside dynamic_jz_v1; exclusive session/attempt paths")
    args = parser.parse_args(argv)
    try:
        diagnostic_output=checked_output(args.jz_diagnostics_dir)
        _validate_deployment_config()
        if artifact_loader is None:
            artifacts = load_official_artifacts(DEFAULT_MODEL)
            letters = StaticAslInference(artifacts)
            digits = DigitPredictor(load_digit_model())
            model,model_hash = load_experiment_model()
        else:
            artifacts,letters,digits,model,model_hash = artifact_loader()
        _synthetic_feature_check(model)
        space_predictor = load_space_predictor(override_enable=args.enable_space)
        enable_space = (space_predictor is not None)
        suggestion_manager = None
        if args.enable_word_suggestion:
            try:
                from app.suggestion.dictionary import OfflineDictionary
                from app.suggestion.qwen_ranker import QwenRanker
                from app.suggestion.sentence_generator import SentenceGenerator
                from app.suggestion.manager import SuggestionManager
                dict_candidates = [
                    ROOT / "data/dictionary/english_words_25k.tsv",
                    Path("data/dictionary/english_words_25k.tsv").resolve(),
                ]
                dict_path = next((p for p in dict_candidates if p.is_file()), None)
                if dict_path is not None:
                    dict_inst = OfflineDictionary.load(dict_path)
                else:
                    dict_inst = OfflineDictionary()
                ranker_inst = QwenRanker()
                sentence_gen_inst = SentenceGenerator()
                suggestion_manager = SuggestionManager(
                    dictionary=dict_inst,
                    ranker=ranker_inst,
                    sentence_generator=sentence_gen_inst
                )
            except Exception as exc:
                print(f"WARNING: Offline suggestion initialization failed: {exc}", file=sys.stderr)
                suggestion_manager = None
        extra_kwargs = {}
        if args.jz_evaluation_mode:
            extra_kwargs["jz_evaluation_mode"] = True
        extra_kwargs["enable_jz_reject_option"] = args.enable_jz_reject_option
        extra_kwargs["jz_margin_threshold"] = args.jz_margin_threshold
        coordinator = coordinator_factory(letters,digits,model,enable_experimental_jz=args.enable_experimental_jz,
                                             z_start_pose_shadow=args.z_start_pose_shadow,
                                             space_predictor=space_predictor, enable_space=enable_space,
                                             suggestion_manager=suggestion_manager, **extra_kwargs)
        if args.check_only:
            coordinator.close()
            print(json.dumps({"status":"PASS","experimental":True,"camera_opened":False,
                "mediapipe_executed":False,"files_written":0,"modes":["ABC","123"],
                "word_suggestion_enabled":args.enable_word_suggestion,
                "word_suggestion_initialized":suggestion_manager is not None,
                "space_enabled":enable_space,"space_model":str(space_predictor.model_path) if space_predictor else None,
                "camera_count_when_running":1,"landmarker_count_when_running":2,
                "letters_model":artifacts.model_name,"letters_epoch":artifacts.best_epoch,
                "digits_model":"models/digits/production/digits_final.joblib",
                "dynamic_model_sha256":model_hash,"dynamic_feature_shape":[16,69],
                "dynamic_implementation":"AutoJzSession + First16Segmenter / latest motion gate",
                "dynamic_pipeline":"IMAGE/raw/no-flip/independent of static VIDEO",
                "static_pipeline":"baseline VIDEO/mirrored ABC, raw 123; mirrored display always",
                "motion_gate":vars(coordinator.session.segmenter.motion_gate),
                "gap_frames":coordinator.session.segmenter.settings.max_missing_frames,
                "gap_ms":1000*coordinator.session.segmenter.settings.max_frame_gap_seconds,
                "hold_seconds":coordinator.hold.settings.hold_seconds,
                "jz_hold_requires_observed_endpoint":True,"jz_commits_enabled":args.enable_experimental_jz,
                "candidate_validation":"single coherent motion run + capture-bound endpoint; semantic UNVERIFIED",
                "geometry_gate":"SHADOW_ONLY_NOT_RECALL_VALIDATED",
                "z_start_pose_mode":"SHADOW" if args.z_start_pose_shadow else "OFF",
                "z_start_pose_acceptance_gate":False,
                "z_temporal_evidence":"SHADOW comparison / same-onset-frame boundary; no calibrated TTL",
                "diagnostic_capture_enabled":args.capture_jz_diagnostics,
                "diagnostic_output":str(diagnostic_output),"diagnostic_capture_writes_images_or_video":False,
                "jz_evaluation_mode":bool(args.jz_evaluation_mode),
                "jz_reject_option_enabled":bool(coordinator.enable_jz_reject_option),
                "jz_margin_threshold":float(coordinator.jz_margin_threshold),
                "raw_prediction_overrides_static":False,
                "no_none_class":True,"mirror_display":True}))
            return 0
        print("EXPERIMENTAL: J/Z motion gating is unvalidated; binary scores do not prove J/Z.")
        with ExitStack() as stack:
            if args.debug:
                root = ROOT/"data/experiments/dynamic_jz_v1"
                path = root/"metadata"/("unified_abc123_jz_events_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                                        +uuid4().hex[:8]+".jsonl")
                trace=stack.enter_context(DebugTrace(path,root))
                print(f"UNIFIED_JZ_TRACE={path}",flush=True)
                coordinator.event_sink=trace
                coordinator.session.event_sink=trace
            if args.capture_jz_diagnostics:
                capture=AttemptCapture(diagnostic_output,model_sha256=model_hash,config={
                    "motion_gate":vars(coordinator.session.segmenter.motion_gate),
                    "segment_settings":vars(coordinator.session.segmenter.settings),
                    "hold_settings":vars(coordinator.hold.settings),
                    "z_start_pose_shadow":args.z_start_pose_shadow,
                    "z_start_pose_settings":vars(coordinator.z_start_pose_shadow.settings) if coordinator.z_start_pose_shadow else None,
                    "enable_experimental_jz":args.enable_experimental_jz,"collection_mode":"first16"})
                def close_capture():
                    capture.close()
                    if capture.errors:
                        print("JZ_DIAGNOSTIC_ERRORS="+json.dumps(capture.errors),file=sys.stderr)
                stack.callback(close_capture)
                print(f"JZ_DIAGNOSTIC_SESSION={capture.path}",flush=True)
                coordinator=coordinator_factory(letters,digits,model,
                    enable_experimental_jz=args.enable_experimental_jz,event_sink=coordinator.event_sink,diagnostics=capture,
                    z_start_pose_shadow=args.z_start_pose_shadow,
                    space_predictor=space_predictor,enable_space=enable_space,
                    suggestion_manager=suggestion_manager,**extra_kwargs)
            return run_camera(coordinator,args.camera_index)
    except (OSError,ValueError,RuntimeError,IntegrationError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
