"""Realtime camera inference for the frozen 24-class static ASL model.

This module reuses ``landmark_utils.normalize_landmarks`` -- the exact
normalization function used to create the frozen dataset. It never fits or
updates the scaler/model and does not compose predictions into words.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import joblib
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
# landmark_utils is imported directly by path so the camera does not depend on
# how the `ai` package directory is cased or on `import ai` succeeding.
PREPROCESSING_DIR = ROOT / "ai" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from landmark_utils import normalize_landmarks  # noqa: E402


MODEL_DIR = ROOT / "models" / "letters" / "fallback"
UVW_MODEL_DIR = ROOT / "models" / "letters" / "production"
TEST_EVALUATION_MANIFEST = MODEL_DIR / "evaluation_manifest.json"
TRAINING_MANIFEST = MODEL_DIR / "training_manifest.json"
TRAINING_CONFIG = MODEL_DIR / "training_config.json"
SELECTED_MODEL_METADATA = MODEL_DIR / "selected_model.json"
HAND_MODEL_PATH = ROOT / "models" / "mediapipe" / "hand_landmarker.task"
DEPLOYMENT_CONFIG_PATH = ROOT / "configs" / "letters.json"
FEATURES = tuple(f"f{i}" for i in range(1, 64))
STATIC_LABELS = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
DEBUG_LABELS = ("P", "Q", "U", "V")
DEFAULT_MODEL = "uvw_medium"
FALLBACK_MODEL = "v1"
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class IntegrationError(RuntimeError):
    """Raised when an artifact or inference contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"Cannot read JSON artifact: {path}") from exc


def _resolve_project_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise IntegrationError(f"Artifact path escapes project root: {relative_path}") from exc
    return candidate


def _require_hash(path: Path, expected: str, artifact_name: str) -> None:
    if not path.is_file():
        raise IntegrationError(f"Missing {artifact_name}: {path}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise IntegrationError(
            f"SHA-256 mismatch for {artifact_name}: expected {expected}, got {actual}"
        )


def _validate_deployment_config() -> dict:
    config = _load_json(DEPLOYMENT_CONFIG_PATH)
    if not (
        config.get("status") == "ACTIVE"
        and config.get("task") == "24_static_ASL_alphabet"
        and config.get("default_model", {}).get("cli_value") == DEFAULT_MODEL
        and config.get("default_model", {}).get("name") == "uvw_medium"
        and config.get("default_model", {}).get("best_epoch") == 81
        and config.get("fallback_model", {}).get("cli_value") == FALLBACK_MODEL
        and config.get("fallback_model", {}).get("name") == "v1_baseline"
        and config.get("fallback_model", {}).get("best_epoch") == 143
        and config.get("hand_landmarker_path")
        == "models/mediapipe/hand_landmarker.task"
        and config.get("active_deployment_manifest_path")
        == "models/letters/production/deployment_manifest_v3.json"
        and config.get("features") == list(FEATURES)
        and config.get("labels") == list(STATIC_LABELS)
    ):
        raise IntegrationError("Static-ASL deployment configuration is inconsistent")
    return config


@dataclass(frozen=True)
class ArtifactBundle:
    model: object
    scaler: object
    labels: tuple[str, ...]
    model_path: Path
    scaler_path: Path
    label_mapping_path: Path
    hand_model_path: Path
    hashes: dict[str, str]
    model_name: str
    best_epoch: int
    scaler_train_rows: int


def _load_v1_artifacts() -> ArtifactBundle:
    """Load and validate the already-selected Baseline epoch-143 artifacts."""
    test_manifest = _load_json(TEST_EVALUATION_MANIFEST)
    training_manifest = _load_json(TRAINING_MANIFEST)
    training_config = _load_json(TRAINING_CONFIG)
    selected_metadata = _load_json(SELECTED_MODEL_METADATA)

    model_meta = test_manifest.get("model", {})
    if test_manifest.get("status") != "TEST_EVALUATION_COMPLETE":
        raise IntegrationError("TEST evaluation manifest is not complete")
    if not (
        model_meta.get("name") == "baseline"
        and model_meta.get("checkpoint_epoch") == 143
        and model_meta.get("selected_before_test") is True
        and training_manifest.get("status") == "VALIDATION_COMPARISON_COMPLETE"
        and training_manifest.get("selection", {}).get("selected_model") == "baseline"
        and selected_metadata.get("selected") == "baseline"
    ):
        raise IntegrationError("Official Baseline epoch-143 selection metadata is inconsistent")
    if training_config.get("features") != list(FEATURES):
        raise IntegrationError("Training feature order is not f1..f63")
    if training_config.get("labels") != list(STATIC_LABELS):
        raise IntegrationError("Training label mapping is not the 24 static ASL classes")

    historical_model_dir = "AI/dataset/mlp_validation_comparison_v1/"
    expected_evidence_paths = {
        "model_path": historical_model_dir + "selected_model.joblib",
        "scaler_path": historical_model_dir + "scaler.joblib",
        "label_mapping_path": historical_model_dir + "label_mapping.json",
    }
    if any(model_meta.get(key) != path for key, path in expected_evidence_paths.items()):
        raise IntegrationError("Baseline TEST evidence artifact paths are inconsistent")
    model_path = MODEL_DIR / "selected_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    mapping_path = MODEL_DIR / "label_mapping.json"
    hand_hash = training_config.get("input_snapshots", {}).get(
        "AI/models/hand_landmarker.task"
    )
    if not hand_hash:
        raise IntegrationError("Hand Landmarker hash is absent from training configuration")

    expected_hashes = {
        "model": model_meta["model_sha256"],
        "scaler": model_meta["scaler_sha256"],
        "label_mapping": model_meta["label_mapping_sha256"],
        "hand_landmarker": hand_hash,
    }
    for path, name in (
        (model_path, "model"),
        (scaler_path, "scaler"),
        (mapping_path, "label_mapping"),
        (HAND_MODEL_PATH, "hand_landmarker"),
    ):
        _require_hash(path, expected_hashes[name], name)

    mapping = _load_json(mapping_path)
    labels = tuple(mapping.get("classes_in_model_order", ()))
    expected_label_to_index = {label: index for index, label in enumerate(STATIC_LABELS)}
    if (
        labels != STATIC_LABELS
        or mapping.get("label_to_index") != expected_label_to_index
        or "J" in labels
        or "Z" in labels
    ):
        raise IntegrationError("Label mapping contract is invalid")

    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
    except Exception as exc:
        raise IntegrationError(f"Cannot load model/scaler: {exc}") from exc

    if int(getattr(scaler, "n_features_in_", -1)) != 63:
        raise IntegrationError("Scaler does not accept 63 features")
    if int(getattr(scaler, "n_samples_seen_", -1)) != 62591:
        raise IntegrationError("Scaler is not the TRAIN-fitted artifact")
    if int(getattr(model, "n_features_in_", -1)) != 63:
        raise IntegrationError("Model does not accept 63 features")
    if not np.array_equal(getattr(model, "classes_", None), np.arange(24)):
        raise IntegrationError("Model class order does not match label mapping")

    return ArtifactBundle(
        model=model,
        scaler=scaler,
        labels=labels,
        model_path=model_path,
        scaler_path=scaler_path,
        label_mapping_path=mapping_path,
        hand_model_path=HAND_MODEL_PATH,
        hashes=expected_hashes,
        model_name="v1_baseline",
        best_epoch=143,
        scaler_train_rows=62591,
    )


def _manifest_output(manifest: dict, relative_path: str) -> dict:
    matches = [item for item in manifest.get("outputs", ()) if item.get("path") == relative_path]
    if len(matches) != 1:
        raise IntegrationError(f"Manifest output is missing or ambiguous: {relative_path}")
    return matches[0]


def _load_uvw_medium_artifacts() -> ArtifactBundle:
    """Load the validation-selected uvw_medium candidate for explicit camera testing."""
    manifest_path = UVW_MODEL_DIR / "training_manifest.json"
    config_path = UVW_MODEL_DIR / "training_config.json"
    selection_path = UVW_MODEL_DIR / "selected_candidate.json"
    manifest = _load_json(manifest_path)
    config = _load_json(config_path)
    selection = _load_json(selection_path)
    selected = manifest.get("selection", {})
    if not (
        manifest.get("status") == "UVW_VALIDATION_COMPARISON_COMPLETE"
        and manifest.get("integrity", {}).get("status") == "PASS"
        and selected.get("validation_winner") == "uvw_medium"
        and selected.get("recommended_for_camera_test") == "uvw_medium"
        and selected.get("deployment_authorized") is False
        and selection.get("validation_winner") == "uvw_medium"
        and selection.get("recommended_for_camera_test") == "uvw_medium"
    ):
        raise IntegrationError("uvw_medium selection/integrity metadata is inconsistent")
    if config.get("features") != list(FEATURES):
        raise IntegrationError("uvw_medium feature order is not f1..f63")
    if config.get("labels") != list(STATIC_LABELS):
        raise IntegrationError("uvw_medium label mapping is not the 24 static ASL classes")
    if config.get("variants", {}).get("uvw_medium") != "weight_supplement_medium":
        raise IntegrationError("uvw_medium weight configuration is inconsistent")

    model_relative = selection.get("source_model_path")
    scaler_relative = selection.get("source_scaler_path")
    mapping_relative = selection.get("label_mapping_path")
    if not all(isinstance(value, str) and value for value in
               (model_relative, scaler_relative, mapping_relative)):
        raise IntegrationError("uvw_medium artifact paths are absent from selection metadata")
    model_path = UVW_MODEL_DIR / model_relative
    scaler_path = UVW_MODEL_DIR / scaler_relative
    mapping_path = UVW_MODEL_DIR / mapping_relative

    expected_hashes = {}
    for name, relative_path in (
        ("model", model_relative),
        ("scaler", scaler_relative),
        ("label_mapping", mapping_relative),
        ("selection_metadata", "selected_candidate.json"),
        ("training_config", "training_config.json"),
    ):
        item = _manifest_output(manifest, relative_path)
        path = UVW_MODEL_DIR / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact '{relative_path}' not found at '{path}'. "
                "This public repository is distributed source-only without unverified pre-trained model weights. "
                "Please see docs/MODEL_DISTRIBUTION.md for details."
            )
        if path.stat().st_size != int(item["bytes"]):
            raise IntegrationError(f"uvw_medium artifact size mismatch: {relative_path}")
        _require_hash(path, item["sha256"], name)
        expected_hashes[name] = item["sha256"]

    v1_training_config = _load_json(TRAINING_CONFIG)
    hand_hash = v1_training_config.get("input_snapshots", {}).get(
        "AI/models/hand_landmarker.task"
    )
    if not hand_hash:
        raise IntegrationError("Hand Landmarker hash is absent from v1 training configuration")
    _require_hash(HAND_MODEL_PATH, hand_hash, "hand_landmarker")
    expected_hashes["hand_landmarker"] = hand_hash

    mapping = _load_json(mapping_path)
    labels = tuple(mapping.get("classes_in_model_order", ()))
    expected_label_to_index = {label: index for index, label in enumerate(STATIC_LABELS)}
    if (
        labels != STATIC_LABELS
        or mapping.get("label_to_index") != expected_label_to_index
        or "J" in labels
        or "Z" in labels
    ):
        raise IntegrationError("uvw_medium label mapping contract is invalid")
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
    except Exception as exc:
        raise IntegrationError(f"Cannot load uvw_medium model/scaler: {exc}") from exc
    if int(getattr(scaler, "n_features_in_", -1)) != 63:
        raise IntegrationError("uvw_medium scaler does not accept 63 features")
    if int(getattr(scaler, "n_samples_seen_", -1)) != 62648:
        raise IntegrationError("uvw_medium scaler is not fitted on all TRAIN UVW rows")
    if int(getattr(model, "n_features_in_", -1)) != 63:
        raise IntegrationError("uvw_medium model does not accept 63 features")
    if not np.array_equal(getattr(model, "classes_", None), np.arange(24)):
        raise IntegrationError("uvw_medium model class order does not match label mapping")

    evaluation = _load_json(UVW_MODEL_DIR / "evaluation_summary.json")
    best_epoch = evaluation.get("uvw_medium", {}).get("best_epoch")
    if best_epoch != 81:
        raise IntegrationError("uvw_medium best epoch does not match the validated checkpoint")
    return ArtifactBundle(
        model=model,
        scaler=scaler,
        labels=labels,
        model_path=model_path,
        scaler_path=scaler_path,
        label_mapping_path=mapping_path,
        hand_model_path=HAND_MODEL_PATH,
        hashes=expected_hashes,
        model_name="uvw_medium",
        best_epoch=best_epoch,
        scaler_train_rows=62648,
    )


def load_official_artifacts(model_name: str = DEFAULT_MODEL) -> ArtifactBundle:
    """Load a hash-verified inference bundle; uvw_medium is the active default."""
    if model_name == "v1":
        return _load_v1_artifacts()
    if model_name == "uvw_medium":
        return _load_uvw_medium_artifacts()
    raise IntegrationError(f"Unsupported model selection: {model_name}")


def features_from_landmarks(hand_landmarks: Sequence[object]) -> np.ndarray:
    """Validate 21 landmarks, then call the project's canonical normalizer."""
    if len(hand_landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(hand_landmarks)}")
    raw = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in hand_landmarks],
        dtype=np.float64,
    )
    if raw.shape != (21, 3) or not np.isfinite(raw).all():
        raise ValueError("Landmarks contain invalid/non-finite coordinates")
    if float(np.linalg.norm(raw - raw[0], axis=1).max()) <= 1e-12:
        raise ValueError("Degenerate landmarks have zero hand extent")

    # Keep one canonical normalization implementation: landmark_utils.py.
    features = np.asarray(normalize_landmarks(hand_landmarks), dtype=np.float64)
    if features.shape != (63,) or not np.isfinite(features).all():
        raise ValueError("Canonical normalization did not produce 63 finite features")
    return features


@dataclass(frozen=True)
class Prediction:
    instant_label: str
    instant_confidence: float
    stable_label: str | None
    stable_confidence: float | None
    features: tuple[float, ...]
    probabilities: tuple[float, ...]


class TemporalSmoother:
    """Vote/probability smoothing for display only; it never changes the model."""

    def __init__(self, class_count: int, window_size: int = 7, minimum_votes: int = 4):
        if window_size < 1 or not 1 <= minimum_votes <= window_size:
            raise ValueError("Require 1 <= minimum_votes <= window_size")
        self.class_count = class_count
        self.window_size = window_size
        self.minimum_votes = minimum_votes
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, instant_index: int, probabilities: np.ndarray) -> tuple[int | None, float | None]:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (self.class_count,) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid probability vector")
        self._history.append((int(instant_index), probabilities.copy()))
        if len(self._history) < self.minimum_votes:
            return None, None

        average = np.mean([item[1] for item in self._history], axis=0)
        candidate = int(np.argmax(average))
        votes = sum(index == candidate for index, _ in self._history)
        if votes < self.minimum_votes:
            return None, None
        return candidate, float(average[candidate])


class StaticAslInference:
    def __init__(self, artifacts: ArtifactBundle, window_size: int = 7, minimum_votes: int = 4):
        self.artifacts = artifacts
        self.smoother = TemporalSmoother(len(artifacts.labels), window_size, minimum_votes)

    def reset(self) -> None:
        """Clear temporal state immediately when no valid hand is present."""
        self.smoother.reset()

    def predict_features(self, features: Sequence[float]) -> Prediction:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (63,) or not np.isfinite(values).all():
            self.reset()
            raise ValueError("Inference requires exactly 63 finite features")
        try:
            scaled = self.artifacts.scaler.transform(values.reshape(1, -1))
            probabilities = np.asarray(
                self.artifacts.model.predict_proba(scaled)[0], dtype=np.float64
            )
        except Exception as exc:
            self.reset()
            raise IntegrationError(f"Scaler/model inference failed: {exc}") from exc
        if probabilities.shape != (24,) or not np.isfinite(probabilities).all():
            self.reset()
            raise IntegrationError("Model returned an invalid probability vector")
        instant_index = int(np.argmax(probabilities))
        stable_index, stable_confidence = self.smoother.update(instant_index, probabilities)
        return Prediction(
            instant_label=self.artifacts.labels[instant_index],
            instant_confidence=float(probabilities[instant_index]),
            stable_label=None if stable_index is None else self.artifacts.labels[stable_index],
            stable_confidence=stable_confidence,
            features=tuple(map(float, values)),
            probabilities=tuple(map(float, probabilities)),
        )

    def predict_landmarks(self, hand_landmarks: Sequence[object]) -> Prediction:
        try:
            return self.predict_features(features_from_landmarks(hand_landmarks))
        except Exception:
            self.reset()
            raise


class DebugCollector:
    """Persist an explicitly requested, manually labelled camera sample."""

    def __init__(
        self,
        output_dir: Path,
        manual_label: str,
        view: str,
        mirror: bool,
        model_name: str = "v1_baseline",
        required_instant_prediction: str | None = None,
    ):
        if manual_label not in DEBUG_LABELS:
            raise ValueError(f"Debug manual label must be one of: {', '.join(DEBUG_LABELS)}")
        if view not in {"front", "horizontal"}:
            raise ValueError("Debug view must be front or horizontal")
        if (
            required_instant_prediction is not None
            and required_instant_prediction not in STATIC_LABELS
        ):
            raise ValueError("Debug prediction filter must be a static ASL label")
        self.manual_label = manual_label
        self.view = view
        self.mirror = mirror
        self.model_name = model_name
        self.required_instant_prediction = required_instant_prediction
        self.output_dir = Path(output_dir) / f"{manual_label}_{view}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "captures.csv"

    def capture(
        self,
        raw_frame: np.ndarray,
        overlay_frame: np.ndarray,
        hand_landmarks: Sequence[object],
        prediction: Prediction,
        handedness: str,
    ) -> Path:
        if (
            self.required_instant_prediction is not None
            and prediction.instant_label != self.required_instant_prediction
        ):
            raise ValueError(
                "Capture rejected: Instant prediction is "
                f"{prediction.instant_label}, expected "
                f"{self.required_instant_prediction}"
            )
        now = datetime.now(timezone.utc)
        capture_id = now.strftime("%Y%m%dT%H%M%S_%fZ")
        raw_name = f"{capture_id}_raw.png"
        overlay_name = f"{capture_id}_overlay.png"
        json_name = f"{capture_id}.json"
        raw_path = self.output_dir / raw_name
        overlay_path = self.output_dir / overlay_name
        json_path = self.output_dir / json_name
        if any(path.exists() for path in (raw_path, overlay_path, json_path)):
            raise OSError(f"Debug capture ID collision: {capture_id}")
        if not cv2.imwrite(str(raw_path), raw_frame):
            raise OSError(f"Cannot save debug raw frame: {raw_path}")
        if not cv2.imwrite(str(overlay_path), overlay_frame):
            raise OSError(f"Cannot save debug overlay: {overlay_path}")

        probability_map = dict(zip(STATIC_LABELS, prediction.probabilities, strict=True))
        payload = {
            "schema_version": 1,
            "capture_id": capture_id,
            "captured_utc": now.isoformat(),
            "manual_label": self.manual_label,
            "manual_view": self.view,
            "manual_label_source": "command_line_user_confirmation",
            "required_instant_prediction": self.required_instant_prediction,
            "mirrored_before_mediapipe": self.mirror,
            "handedness_reported_by_mediapipe": handedness,
            "instant_prediction": prediction.instant_label,
            "instant_confidence": prediction.instant_confidence,
            "stable_prediction": prediction.stable_label,
            "stable_confidence": prediction.stable_confidence,
            "probability_P": probability_map["P"],
            "probability_Q": probability_map["Q"],
            "probability_manual_label": probability_map[self.manual_label],
            "probabilities": probability_map,
            "features": dict(zip(FEATURES, prediction.features, strict=True)),
            "raw_landmarks": [
                {"index": index, "x": float(point.x), "y": float(point.y), "z": float(point.z)}
                for index, point in enumerate(hand_landmarks)
            ],
            "raw_image": raw_name,
            "overlay_image": overlay_name,
            "normalization_function": "AI/landmark_utils.py::normalize_landmarks",
            "model": self.model_name,
        }
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")

        row = {
            "capture_id": capture_id,
            "captured_utc": payload["captured_utc"],
            "manual_label": self.manual_label,
            "manual_view": self.view,
            "handedness": handedness,
            "instant_prediction": prediction.instant_label,
            "instant_confidence": prediction.instant_confidence,
            "stable_prediction": prediction.stable_label or "",
            "stable_confidence": "" if prediction.stable_confidence is None else prediction.stable_confidence,
            "probability_P": probability_map["P"],
            "probability_Q": probability_map["Q"],
            "probability_manual_label": probability_map[self.manual_label],
            "raw_image": raw_name,
            "overlay_image": overlay_name,
            "details_json": json_name,
        }
        write_header = not self.index_path.exists()
        with self.index_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return json_path


def draw_hand(
    frame: np.ndarray,
    hand_landmarks: Sequence[object],
    show_indices: bool = False,
) -> None:
    if len(hand_landmarks) != 21:
        return
    height, width = frame.shape[:2]
    points = [
        (int(np.clip(point.x * width, 0, width - 1)),
         int(np.clip(point.y * height, 0, height - 1)))
        for point in hand_landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for index, point in enumerate(points):
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
        if show_indices:
            cv2.putText(
                frame, str(index), (point[0] + 4, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA,
            )


def _put_status(frame: np.ndarray, text: str, line: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame, text, (20, 40 + line * 37), cv2.FONT_HERSHEY_SIMPLEX,
        0.78 if line else 0.9, color, 2, cv2.LINE_AA,
    )


def run_camera(
    pipeline: StaticAslInference,
    camera_index: int = 0,
    mirror: bool = True,
    debug_collector: DebugCollector | None = None,
) -> int:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        print(f"ERROR: Cannot open camera index {camera_index}.", file=sys.stderr)
        return 3

    started = time.perf_counter()
    last_timestamp = -1
    read_failures = 0
    detection_failures = 0
    window_name = (
        f"Static ASL - {pipeline.artifacts.model_name} "
        f"epoch {pipeline.artifacts.best_epoch}"
    )
    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pipeline.artifacts.hand_model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    pipeline.reset()
                    read_failures += 1
                    if read_failures >= 10:
                        print("ERROR: Camera stopped returning frames.", file=sys.stderr)
                        return 4
                    time.sleep(0.03)
                    continue
                read_failures = 0
                if mirror:
                    frame = cv2.flip(frame, 1)
                detector_input_frame = frame.copy()
                capture_data = None

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = int((time.perf_counter() - started) * 1000)
                    timestamp = max(timestamp, last_timestamp + 1)
                    last_timestamp = timestamp
                    result = landmarker.detect_for_video(mp_image, timestamp)
                    detection_failures = 0
                except Exception as exc:
                    pipeline.reset()
                    detection_failures += 1
                    _put_status(frame, "Landmarker error - state cleared", 0, (0, 0, 255))
                    _put_status(frame, str(exc)[:70], 1, (0, 0, 255))
                    result = None

                if result is not None and result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    try:
                        prediction = pipeline.predict_landmarks(hand)
                        draw_hand(frame, hand)
                        debug_overlay = detector_input_frame.copy()
                        draw_hand(debug_overlay, hand, show_indices=True)
                        handedness = "Unknown"
                        if result.handedness and result.handedness[0]:
                            handedness = str(result.handedness[0][0].category_name)
                        capture_data = (
                            detector_input_frame,
                            debug_overlay,
                            hand,
                            prediction,
                            handedness,
                        )
                        _put_status(
                            frame,
                            f"Instant: {prediction.instant_label}  {prediction.instant_confidence * 100:.1f}%",
                            0,
                            (0, 255, 255),
                        )
                        if prediction.stable_label is None:
                            _put_status(frame, "Stable: collecting frames...", 1, (255, 255, 255))
                        else:
                            _put_status(
                                frame,
                                f"Stable: {prediction.stable_label}  {prediction.stable_confidence * 100:.1f}%",
                                1,
                                (0, 255, 0),
                            )
                    except (ValueError, IntegrationError) as exc:
                        pipeline.reset()
                        _put_status(frame, "Invalid landmarks - prediction cleared", 0, (0, 0, 255))
                        _put_status(frame, str(exc)[:70], 1, (0, 0, 255))
                elif result is not None:
                    pipeline.reset()
                    _put_status(frame, "No hand - prediction cleared", 0, (0, 0, 255))

                _put_status(frame, "Static classes only: A-I, K-Y | Q/Esc: quit", 2, (220, 220, 220))
                if debug_collector is not None:
                    _put_status(
                        frame,
                        f"Debug {debug_collector.manual_label}/{debug_collector.view} | S: save",
                        3,
                        (255, 200, 0),
                    )
                    if debug_collector.required_instant_prediction is not None:
                        _put_status(
                            frame,
                            "Save filter: Instant must be "
                            f"{debug_collector.required_instant_prediction}",
                            4,
                            (255, 200, 0),
                        )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("s"):
                    if debug_collector is None:
                        print("Debug capture is not enabled.", file=sys.stderr)
                    elif capture_data is None:
                        print("No valid hand/landmarks to save.", file=sys.stderr)
                    else:
                        try:
                            saved = debug_collector.capture(*capture_data)
                            print(f"DEBUG_CAPTURE_SAVED={saved}", flush=True)
                        except (OSError, ValueError) as exc:
                            print(f"ERROR: Debug capture failed: {exc}", file=sys.stderr)
                if detection_failures >= 30:
                    print("ERROR: Repeated Hand Landmarker failures.", file=sys.stderr)
                    return 5
    except Exception as exc:
        pipeline.reset()
        print(f"ERROR: Hand Landmarker could not run: {exc}", file=sys.stderr)
        return 5
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime 24-class static ASL inference")
    parser.add_argument(
        "--model", choices=("v1", "uvw_medium"), default=DEFAULT_MODEL,
        help="Inference model; default is uvw_medium and v1 remains available as fallback",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the camera frame")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--smoothing-min-votes", type=int, default=4)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--debug-label", choices=DEBUG_LABELS)
    parser.add_argument("--debug-view", choices=("front", "horizontal"))
    parser.add_argument(
        "--debug-prediction-filter", choices=STATIC_LABELS,
        help="Save only when the instantaneous model prediction matches this label",
    )
    parser.add_argument("--check-only", action="store_true", help="Verify/load artifacts without opening camera")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        deployment_config = _validate_deployment_config()
        artifacts = load_official_artifacts(args.model)
        pipeline = StaticAslInference(
            artifacts,
            window_size=args.smoothing_window,
            minimum_votes=args.smoothing_min_votes,
        )
        debug_values = (args.debug_dir, args.debug_label, args.debug_view)
        if any(value is not None for value in debug_values) and not all(
            value is not None for value in debug_values
        ):
            raise ValueError(
                "Debug capture requires --debug-dir, --debug-label and --debug-view together"
            )
        if args.debug_prediction_filter is not None and args.debug_dir is None:
            raise ValueError("--debug-prediction-filter requires debug capture options")
        debug_collector = None
        if args.debug_dir is not None:
            debug_collector = DebugCollector(
                args.debug_dir,
                manual_label=args.debug_label,
                view=args.debug_view,
                mirror=not args.no_mirror,
                model_name=artifacts.model_name,
                required_instant_prediction=args.debug_prediction_filter,
            )
    except (IntegrationError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.check_only:
        print(json.dumps({
            "status": "CAMERA_INFERENCE_CHECK_PASS",
            "model": artifacts.model_name,
            "best_epoch": artifacts.best_epoch,
            "features": len(FEATURES),
            "classes": list(artifacts.labels),
            "scaler_train_rows": int(artifacts.scaler.n_samples_seen_),
            "hashes_verified": artifacts.hashes,
            "camera_opened": False,
            "training_performed": False,
            "configured_default_model": deployment_config["default_model"]["name"],
            "configured_fallback_model": deployment_config["fallback_model"]["name"],
        }))
        return 0
    return run_camera(
        pipeline,
        camera_index=args.camera_index,
        mirror=not args.no_mirror,
        debug_collector=debug_collector,
    )


if __name__ == "__main__":
    raise SystemExit(main())
