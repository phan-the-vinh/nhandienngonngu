"""MediaPipe Hand Landmarker helpers.

This module is the canonical home for MediaPipe hand-landmark extraction. The
two frozen runtime cameras (``app/inference/letter_inference.py`` and
``app/inference/digit_inference.py``) currently keep their own inline
HandLandmarker setup so their verified behavior is preserved exactly. The
helpers here are provided for new code and for a future, behavior-checked
consolidation; wiring them into the frozen cameras is intentionally out of scope
for the structural refactor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path-based import so this module does not depend on `import ai` succeeding.
_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from constants import HAND_LANDMARKER_PATH  # noqa: E402

_RUNNING_MODES = {
    "IMAGE": vision.RunningMode.IMAGE,
    "VIDEO": vision.RunningMode.VIDEO,
}


def create_hand_landmarker(
    model_path: Path | str = HAND_LANDMARKER_PATH,
    running_mode: str = "IMAGE",
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float | None = None,
) -> vision.HandLandmarker:
    """Create a ``HandLandmarker`` for the requested mediapipe running mode."""
    if running_mode not in _RUNNING_MODES:
        raise ValueError(f"running_mode must be one of {sorted(_RUNNING_MODES)}")
    options_kwargs = dict(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=_RUNNING_MODES[running_mode],
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
    )
    if min_tracking_confidence is not None:
        options_kwargs["min_tracking_confidence"] = min_tracking_confidence
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**options_kwargs))


def landmarks_to_array(hand_landmarks: Sequence[object]) -> np.ndarray:
    """Return a validated ``(21, 3)`` float64 array of landmark coordinates."""
    if len(hand_landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(hand_landmarks)}")
    raw = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in hand_landmarks],
        dtype=np.float64,
    )
    if raw.shape != (21, 3) or not np.isfinite(raw).all():
        raise ValueError("Landmarks contain invalid/non-finite coordinates")
    return raw


__all__ = ["create_hand_landmarker", "landmarks_to_array", "mp"]
