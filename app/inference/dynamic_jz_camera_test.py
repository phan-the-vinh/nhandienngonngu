"""Manual, experimental J/Z camera test. Not part of the production dispatcher.

SPACE starts/stops one complete gesture bout. This binary model has no NONE
class, so it must never be used for continuous gesture detection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction.dynamic_sequence import (  # noqa: E402
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    DynamicSequenceError,
    build_sequence_features,
    sample_valid_frame_indices,
)
from ai.feature_extraction.hand_landmarks import (  # noqa: E402
    create_hand_landmarker,
    landmarks_to_array,
)

EXPERIMENT_DIR = ROOT / "data/experiments/dynamic_jz_binary_baseline_v1"
TASK_PATH = ROOT / "models/mediapipe/hand_landmarker.task"
LOG_PATH = EXPERIMENT_DIR / "camera_tests/trials.csv"
LOG_COLUMNS = (
    "session_id", "trial_index", "timestamp_utc", "true_label", "status",
    "prediction", "probability_J", "probability_Z", "frames_total",
    "frames_valid", "selected_original_frame_indices", "failure_reason",
    "model_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_experiment_model() -> tuple[Pipeline, str]:
    """Load only the frozen experimental artifacts, including their scaler."""
    manifest = json.loads((EXPERIMENT_DIR / "training_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status") != "EXPERIMENTAL_NOT_PRODUCTION"
            or manifest.get("feature_shape") != [16, 69]
            or manifest.get("flattened_feature_count") != 1104
            or manifest.get("pipeline") != ["StandardScaler", "LogisticRegression"]):
        raise ValueError("Unexpected J/Z experiment contract")
    artifacts = manifest.get("artifacts", {})
    for name in ("model_pipeline.joblib", "scaler.joblib", "split_manifest.json", "config_snapshot.json"):
        expected = artifacts.get(name)
        if not isinstance(expected, str) or _sha256(EXPERIMENT_DIR / name) != expected:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    model = load(EXPERIMENT_DIR / "model_pipeline.joblib")
    separate_scaler = load(EXPERIMENT_DIR / "scaler.joblib")
    if (not isinstance(model, Pipeline)
            or list(model.named_steps) != ["scaler", "classifier"]
            or not isinstance(model.named_steps["scaler"], StandardScaler)
            or not isinstance(model.named_steps["classifier"], LogisticRegression)
            or not isinstance(separate_scaler, StandardScaler)
            or getattr(model, "n_features_in_", None) != SEQUENCE_LENGTH * FEATURE_DIM
            or list(model.classes_) != ["J", "Z"]
            or not np.array_equal(model.named_steps["scaler"].mean_, separate_scaler.mean_)
            or not np.array_equal(model.named_steps["scaler"].scale_, separate_scaler.scale_)):
        raise ValueError("Unexpected J/Z model or scaler")
    if not TASK_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe task missing: {TASK_PATH}")
    return model, artifacts["model_pipeline.joblib"]


def predict_attempt(
    valid_entries: list[tuple[int, list[object], tuple[int, int]]],
    total_frames: int,
    model: Pipeline,
    detection_error: str = "",
) -> dict:
    """Apply the official valid-frame sampler and 16x69 feature builder."""
    result = {
        "status": "INVALID_ATTEMPT", "frames_total": total_frames,
        "frames_valid": len(valid_entries), "selected_original_frame_indices": [],
        "prediction": "", "probability_J": "", "probability_Z": "",
        "failure_reason": "",
    }
    if detection_error:
        result["failure_reason"] = detection_error
        return result
    if len(valid_entries) < SEQUENCE_LENGTH:
        result["failure_reason"] = "INSUFFICIENT_VALID_LANDMARK_FRAMES"
        return result
    try:
        original_indices = sample_valid_frame_indices([entry[0] for entry in valid_entries])
        by_index = {index: (hand, size) for index, hand, size in valid_entries}
        if (len(by_index) != len(valid_entries) or original_indices[-1] >= total_frames):
            raise ValueError("Invalid capture frame indices")
        frames = [by_index[index][0] for index in original_indices]
        sizes = [by_index[index][1] for index in original_indices]
        features, _ = build_sequence_features(frames, sizes)
        if features.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(features).all():
            raise ValueError("Invalid sequence features")
        flat = features.reshape(1, SEQUENCE_LENGTH * FEATURE_DIM, order="C")
        probabilities = np.asarray(model.predict_proba(flat), dtype=np.float64)
        if (probabilities.shape != (1, 2) or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("Invalid model probabilities")
        result.update(
            status="PREDICTED",
            selected_original_frame_indices=original_indices,
            prediction=str(model.classes_[int(np.argmax(probabilities[0]))]),
            probability_J=float(probabilities[0, 0]),
            probability_Z=float(probabilities[0, 1]),
        )
    except (DynamicSequenceError, ValueError, IndexError, KeyError) as exc:
        result["failure_reason"] = str(exc)
    return result


def _synthetic_feature_check(model: Pipeline) -> None:
    """Exercise the exact feature builder and inference without MediaPipe/camera."""
    frames = []
    for step in range(SEQUENCE_LENGTH):
        hand = [SimpleNamespace(x=0.35 + (index % 5) * 0.02 + step * 0.001,
                                y=0.35 + (index // 5) * 0.02,
                                z=index * 0.001) for index in range(21)]
        frames.append((step, hand, (640, 480)))
    result = predict_attempt(frames, SEQUENCE_LENGTH, model)
    if result["status"] != "PREDICTED" or len(result["selected_original_frame_indices"]) != 16:
        raise ValueError(f"Synthetic feature/inference check failed: {result['failure_reason']}")


def _append_trial(session_id: str, trial_index: int, true_label: str, result: dict,
                  model_sha256: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    if exists:
        with LOG_PATH.open("r", encoding="utf-8", newline="") as stream:
            if next(csv.reader(stream), None) != list(LOG_COLUMNS):
                raise ValueError(f"Existing log has incompatible columns: {LOG_PATH}")
    with LOG_PATH.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "session_id": session_id, "trial_index": trial_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "true_label": true_label, "status": result["status"],
            "prediction": result["prediction"],
            "probability_J": result["probability_J"],
            "probability_Z": result["probability_Z"],
            "frames_total": result["frames_total"],
            "frames_valid": result["frames_valid"],
            "selected_original_frame_indices": json.dumps(result["selected_original_frame_indices"]),
            "failure_reason": result["failure_reason"],
            "model_sha256": model_sha256,
        })


def run_camera(model: Pipeline, model_sha256: str, camera_index: int,
               save_results: bool) -> None:
    """Camera access occurs only here, never during import or --check-only."""
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trial_index = 0
    true_label = ""
    active_label = ""
    recording = False
    total_frames = 0
    valid_entries: list[tuple[int, list[object], tuple[int, int]]] = []
    detection_error = ""
    last_result: dict | None = None
    camera = None
    try:
        with create_hand_landmarker(TASK_PATH, running_mode="IMAGE", num_hands=1) as landmarker:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera index {camera_index}")
            while True:
                ok, frame = camera.read()
                if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError("Cannot read a valid camera frame")
                height, width = frame.shape[:2]
                if recording:
                    frame_index = total_frames
                    total_frames += 1
                    try:
                        # Same as official extraction: BGR -> RGB, no flip,
                        # IMAGE mode, first detected hand, 21 finite landmarks.
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        detected = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                        if detected.hand_landmarks:
                            hand = detected.hand_landmarks[0]
                            try:
                                landmarks_to_array(hand)
                            except (AttributeError, TypeError, ValueError):
                                pass  # Invalid landmark frame: same skip policy as extraction.
                            else:
                                valid_entries.append((frame_index, hand, (width, height)))
                    except Exception as exc:
                        if not detection_error:
                            detection_error = f"DETECTION_ERROR_AT_FRAME_{frame_index}:{type(exc).__name__}"
                display = frame.copy()  # Never flip the inference frame or display.
                color = (0, 0, 255) if recording else (0, 220, 0)
                state = (f"REC frames={total_frames} valid={len(valid_entries)}"
                         if recording else "READY - SPACE to start")
                cv2.putText(display, state, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"True label: {active_label if recording else (true_label or 'UNSET')}",
                            (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, "SPACE start/stop | J/Z label | C clear | Q quit",
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(display, "Binary J/Z only: probability is not gesture confirmation",
                            (15, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2)
                if last_result is not None:
                    if last_result["status"] == "PREDICTED":
                        line = (f"Pred: {last_result['prediction']}  "
                                f"P(J)={last_result['probability_J']:.3f}  "
                                f"P(Z)={last_result['probability_Z']:.3f}")
                    else:
                        line = f"No prediction: {last_result['failure_reason']}"
                    cv2.putText(display, line, (15, 135), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (0, 220, 255), 2)
                cv2.imshow("Experimental J/Z camera test", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    if not recording:
                        trial_index += 1
                        active_label = true_label
                        valid_entries = []
                        total_frames = 0
                        detection_error = ""
                        last_result = None
                        recording = True
                    else:
                        last_result = predict_attempt(valid_entries, total_frames, model, detection_error)
                        recording = False
                        if save_results:
                            _append_trial(session_id, trial_index, active_label, last_result, model_sha256)
                elif not recording and key in (ord("j"), ord("J"), ord("z"), ord("Z")):
                    true_label = chr(key).upper()
                elif not recording and key in (ord("c"), ord("C")):
                    true_label = ""
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify artifacts and inference; never open camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--save-results", action="store_true", help=f"Append trial metadata to {LOG_PATH}")
    args = parser.parse_args(argv)
    model, model_sha256 = load_experiment_model()
    if args.check_only:
        _synthetic_feature_check(model)
        print(json.dumps({"status": "CHECK_ONLY_PASS", "camera_opened": False,
                          "model_path": str(EXPERIMENT_DIR / "model_pipeline.joblib"),
                          "model_sha256": model_sha256, "classes": list(model.classes_),
                          "feature_shape": [SEQUENCE_LENGTH, FEATURE_DIM],
                          "media_pipe_task_exists": True}))
        return 0
    run_camera(model, model_sha256, args.camera_index, args.save_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
