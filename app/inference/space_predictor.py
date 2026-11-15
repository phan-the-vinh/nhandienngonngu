"""SPACE gesture prediction module for additive ABC + 123 runtime integration.

Encapsulates the verified candidate model from Phase 3.9.8:
- LogisticRegression (C=10.0, class_weight='balanced')
- 63 features, wrist-relative, max Euclidean distance (Branch A)
- Zero wrist weights verification
- Shared landmark reuse from the active camera loop (detect_static)
- Optional horizontal reflection adapter for ABC mode (mirrored video capture)
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

# Canonical feature normalization
from ai.preprocessing.landmark_utils import normalize_landmarks

logger = logging.getLogger("SpacePredictor")

ROOT = Path(__file__).resolve().parents[2]
SPACE_CONFIG_PATH = ROOT / "configs" / "space.json"
DEFAULT_MODEL_PATH = ROOT / "models" / "space" / "production" / "final_space_model.joblib"
DEFAULT_MODEL_SHA256 = "453162d8aa0212016216a6c02798d3dcaa445c1fbb7a6340c896bb828e076121"
DEFAULT_THRESHOLD = 0.889063295339523


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


class SpacePredictor:
    """Evaluates 21 MediaPipe hand landmarks against the binary SPACE classifier."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        enable_adapter: bool = True,
        expected_sha256: str | None = DEFAULT_MODEL_SHA256,
    ):
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.enable_adapter = bool(enable_adapter)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"SPACE model file not found at '{self.model_path}'. "
                "This public repository is distributed source-only without unverified pre-trained weights. "
                "Please see docs/MODEL_DISTRIBUTION.md for details."
            )

        # Integrity verification
        if expected_sha256 is not None:
            actual_sha = compute_sha256(self.model_path)
            if actual_sha.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SPACE model SHA-256 mismatch! Expected {expected_sha256}, got {actual_sha}"
                )

        self.model = joblib.load(self.model_path)

        # Enforce runtime model contract
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Loaded SPACE model does not implement predict_proba()")
        if getattr(self.model, "n_features_in_", None) != 63:
            raise ValueError(
                f"Expected 63 features, model expects {getattr(self.model, 'n_features_in_', None)}"
            )
        if not np.array_equal(self.model.classes_, [0, 1]):
            raise ValueError(f"Expected classes [0, 1], got {self.model.classes_}")

        # Verify wrist landmark invariance (features 0, 1, 2)
        wrist_weights = self.model.coef_[0, :3]
        if not np.allclose(wrist_weights, 0.0, atol=1e-12):
            logger.warning(f"Wrist weights are non-zero: {wrist_weights}")

        logger.info(
            f"SpacePredictor initialized successfully: threshold={self.threshold:.6f}, adapter={self.enable_adapter}"
        )

    def reset(self) -> None:
        """Reset internal state. Stateless model predictor; hook provided for interface parity."""
        pass

    def predict_features(
        self, features_63: np.ndarray | Sequence[float], mode: str = "123"
    ) -> tuple[float, bool]:
        """
        Evaluates a 63-feature normalized vector.
        If mode == 'ABC' and adapter is enabled, reflects relative X coordinates.
        Returns: (probability_space, is_space_predicted)
        """
        feats = np.asarray(features_63, dtype=np.float64).copy()
        if feats.shape != (63,) or not np.isfinite(feats).all():
            return 0.0, False

        # Apply horizontal reflection adapter for ABC mode
        if mode == "ABC" and self.enable_adapter:
            # Features are interleaved [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
            # Reflect relative horizontal displacement: x -> -x
            feats[0::3] = -feats[0::3]

        prob = float(self.model.predict_proba(feats.reshape(1, -1))[0, 1])
        is_space = bool(prob >= self.threshold)
        return prob, is_space

    def predict_hand(
        self, hand: Sequence[object], mode: str = "123"
    ) -> tuple[float, bool]:
        """
        Extracts canonical 63 features from a sequence of 21 MediaPipe landmarks
        and evaluates SPACE probability.
        """
        if len(hand) != 21:
            return 0.0, False

        try:
            norm_features = normalize_landmarks(hand)
        except Exception as exc:
            logger.debug(f"Normalization failed: {exc}")
            return 0.0, False

        if len(norm_features) != 63:
            return 0.0, False

        return self.predict_features(norm_features, mode=mode)


def load_space_predictor(
    config_path: Path = SPACE_CONFIG_PATH,
    override_enable: bool | None = None,
) -> SpacePredictor | None:
    """
    Fail-safe factory function to initialize SpacePredictor from configuration.
    Returns None if disabled or if model loading fails, ensuring zero crashes.
    """
    try:
        config = {}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        enabled = config.get("enable_space", False)
        if override_enable is not None:
            enabled = override_enable

        if not enabled:
            logger.info("SPACE gesture recognition is disabled by configuration.")
            return None

        raw_path = config.get("model_path", DEFAULT_MODEL_PATH)
        model_path = Path(raw_path)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        threshold = float(config.get("threshold", DEFAULT_THRESHOLD))
        apply_adapter = bool(config.get("apply_abc_x_flip", True))
        expected_sha = config.get("model_sha256", DEFAULT_MODEL_SHA256)

        return SpacePredictor(
            model_path=model_path,
            threshold=threshold,
            enable_adapter=apply_adapter,
            expected_sha256=expected_sha,
        )
    except Exception as exc:
        logger.error(f"Failed to load SpacePredictor (fail-safe to baseline): {exc}")
        return None
