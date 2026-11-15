"""Canonical project-wide constants for the NhanDienCuChi recognizers.

This module is the single place for values shared by the letter and digit
pipelines. It does not replace ``ai.preprocessing.landmark_utils``: the
normalization implementation itself stays in that module as the only source of
truth.
"""
from __future__ import annotations

from pathlib import Path

# Project root: <root>/ai/common/constants.py -> parents[2].
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------
# Feature contract
# ------------------------------------------------------------------
FEATURE_COUNT: int = 63
FEATURE_NAMES: tuple[str, ...] = tuple(f"f{i}" for i in range(1, FEATURE_COUNT + 1))

# ------------------------------------------------------------------
# Label sets
# ------------------------------------------------------------------
# 24 static ASL classes; J and Z are dynamic gestures and are not modeled here.
LETTER_LABELS: tuple[str, ...] = tuple("ABCDEFGHIKLMNOPQRSTUVWXY")
DIGIT_LABELS: tuple[str, ...] = tuple("0123456789")

# ------------------------------------------------------------------
# Runtime artifact locations (see SYSTEM_CONTEXT.md)
# ------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
MEDIAPIPE_DIR: Path = MODELS_DIR / "mediapipe"
HAND_LANDMARKER_PATH: Path = MEDIAPIPE_DIR / "hand_landmarker.task"

LETTERS_MODEL_DIR: Path = MODELS_DIR / "letters"
LETTERS_PRODUCTION_DIR: Path = LETTERS_MODEL_DIR / "production"
LETTERS_FALLBACK_DIR: Path = LETTERS_MODEL_DIR / "fallback"

DIGITS_MODEL_DIR: Path = MODELS_DIR / "digits"
DIGITS_PRODUCTION_DIR: Path = DIGITS_MODEL_DIR / "production"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
LETTERS_CONFIG_PATH: Path = CONFIGS_DIR / "letters.json"
DIGITS_CONFIG_PATH: Path = CONFIGS_DIR / "digits.json"
CAMERA_CONFIG_PATH: Path = CONFIGS_DIR / "camera.json"

# ------------------------------------------------------------------
# MediaPipe hand skeleton (21-landmark topology)
# ------------------------------------------------------------------
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# Canonical normalization contract string recorded in metadata.
NORMALIZATION_MODULE: str = "ai/preprocessing/landmark_utils.py::normalize_landmarks"
