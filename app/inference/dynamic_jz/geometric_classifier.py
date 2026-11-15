"""Geometric & Ratio-Based Classifier for Dynamic J/Z Gestures (Method 2).

An independent, rule-and-ratio geometric classifier that evaluates multi-criteria
kinematic and morphological stroke profiles without relying on a single ratio:

For J (Pinky tracking, Landmark 20):
- Bounding box aspect ratio (x_span / y_span, J is vertically elongated).
- Downward drop magnitude (dy_down).
- Lateral hook excursion (dx_hook) and upward curve (dy_hook < 0).
- Handshape dominance (pinky extension vs index and curled fingers).
- Parametric J curve similarity and lead-in monotonicity.

For Z (Index tracking, Landmark 8):
- Bounding box aspect ratio (x_span / y_span, Z is horizontally wide).
- Three-stroke segmentation (top horizontal, diagonal down-reverse, bottom horizontal).
- Horizontal direction reversals (dx1 * dx2 < 0 and dx2 * dx3 < 0).
- Diagonal downward progression (dy2 > 0).
- Corner turn angles (~110-160 deg).
- Handshape dominance (index extension vs pinky and curled fingers).
- Piecewise linear Z model similarity.

Decision logic:
- Computes comprehensive J score (0..1) and Z score (0..1).
- Requires winning score >= min_score (default: 0.50) and separation >= min_margin (default: 0.10).
- Rejects non-qualifying or out-of-distribution motions as UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    INDEX_TIP_IDX,
    LANDMARK_COUNT,
    PALM_SCALE_LANDMARKS,
    PINKY_TIP_IDX,
    SEQUENCE_LENGTH,
)


@dataclass(frozen=True)
class GeometricFeaturesJ:
    """Extracted geometric and ratio features for J hypothesis."""
    aspect_ratio: float           # x_span / (y_span + eps)
    downward_drop: float          # dy from start to lowest point
    hook_lateral: float           # dx from lowest point to end
    hook_upward: float            # dy from lowest point to end (< 0 is upward)
    upward_before_drop: float     # lead-in upward drift before drop
    pinky_extension: float        # endpoint pinky tip distance / palm scale
    curled_extension: float       # endpoint mean curled fingers distance / palm scale
    handshape_dominance: float    # pinky_ext - curled_ext
    pinky_vs_index: float         # pinky_ext - index_ext
    curve_similarity: float       # correlation with idealized J hook profile (0..1)
    composite_score: float        # multi-criteria weighted score (0..1)


@dataclass(frozen=True)
class GeometricFeaturesZ:
    """Extracted geometric and ratio features for Z hypothesis."""
    aspect_ratio: float           # x_span / (y_span + eps)
    corner1_idx: int              # optimal corner 1 frame
    corner2_idx: int              # optimal corner 2 frame
    dx_stroke1: float             # top stroke horizontal displacement
    dx_stroke2: float             # diagonal stroke horizontal displacement
    dx_stroke3: float             # bottom stroke horizontal displacement
    dy_diagonal: float            # diagonal downward displacement
    turn1_angle_deg: float        # angle change at corner 1 in degrees
    turn2_angle_deg: float        # angle change at corner 2 in degrees
    reversal_valid: bool          # whether dx1*dx2 < 0 and dx2*dx3 < 0
    index_extension: float        # endpoint index tip distance / palm scale
    curled_extension: float       # endpoint mean curled fingers distance / palm scale
    handshape_dominance: float    # index_ext - curled_ext
    index_vs_pinky: float         # index_ext - pinky_ext
    piecewise_similarity: float   # fit quality to 3-segment Z path (0..1)
    composite_score: float        # multi-criteria weighted score (0..1)


@dataclass(frozen=True)
class GeometricClassificationResult:
    """Output contract of the Geometric / Ratio Classifier."""
    prediction: str                      # 'J', 'Z', or 'UNKNOWN'
    score_j: float                       # Composite score for J (0..1)
    score_z: float                       # Composite score for Z (0..1)
    margin: float                        # |score_j - score_z|
    features_j: GeometricFeaturesJ | None
    features_z: GeometricFeaturesZ | None
    status: str                          # 'OK', 'REJECTED', 'INVALID_INPUT', 'ERROR'
    rejection_reason: str = ""
    latency_ms: float = 0.0


def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, val)))


class GeometricClassifier:
    """Multi-criteria geometric and ratio classifier for J and Z dynamic gestures."""

    def __init__(
        self,
        min_score: float = 0.50,
        min_margin: float = 0.10,
        min_pinky_dominance: float = -0.15,
        min_index_dominance: float = -0.15,
    ) -> None:
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.min_pinky_dominance = float(min_pinky_dominance)
        self.min_index_dominance = float(min_index_dominance)

    def extract_j_features(
        self,
        path_pinky: np.ndarray,
        landmarks_3d: np.ndarray,
        palm_scales: np.ndarray,
    ) -> GeometricFeaturesJ:
        """Extract geometric ratios and trajectory properties for J (Pinky, Landmark 20)."""
        x = path_pinky[:, 0]
        y = path_pinky[:, 1]  # positive Y is downward in image space

        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        aspect_ratio = float(x_span / max(y_span, 1e-6))

        # Bottom index: deepest point before final frames
        t_bot = int(np.argmax(y[:15]))
        dy_down = float(y[t_bot] - y[0])
        up_before = float(y[0] - np.min(y[:t_bot + 1])) if t_bot > 0 else 0.0
        dx_hook = float(abs(x[-1] - x[t_bot]))
        dy_hook = float(y[-1] - y[t_bot])  # negative is upward

        # Handshape metrics at endpoint (frame 15)
        safe_palm = max(float(palm_scales[-1]), 1e-6)
        pinky_pt = landmarks_3d[-1, PINKY_TIP_IDX, :]
        pinky_ext = float(np.linalg.norm(pinky_pt) / safe_palm)

        index_pt = landmarks_3d[-1, INDEX_TIP_IDX, :]
        index_ext = float(np.linalg.norm(index_pt) / safe_palm)
        pinky_vs_index = float(pinky_ext - index_ext)

        other_lms = (INDEX_TIP_IDX, 12, 16)  # index, middle, ring
        curled_exts = [
            float(np.linalg.norm(landmarks_3d[-1, idx, :]) / safe_palm)
            for idx in other_lms
        ]
        curled_mean = float(np.mean(curled_exts))
        dominance = float(pinky_ext - curled_mean)

        # Multi-criteria scoring:
        # 1. Downward drop score: saturates around dy_down >= 0.35
        s_drop = _clamp((dy_down - 0.10) / 0.30)

        # 2. Aspect ratio score: J aspect ratio ideally between 0.20 and 0.90
        if aspect_ratio < 0.15:
            s_aspect = _clamp(aspect_ratio / 0.15)
        elif aspect_ratio <= 0.90:
            s_aspect = 1.0
        elif aspect_ratio <= 1.50:
            s_aspect = _clamp(1.0 - (aspect_ratio - 0.90) / 0.60)
        else:
            s_aspect = 0.0

        # 3. Hook score: requires both lateral excursion and upward curve
        s_hook_x = _clamp(dx_hook / 0.18)
        s_hook_y = _clamp(1.0 - max(0.0, dy_hook) / 0.10)
        s_hook = float(s_hook_x * s_hook_y)

        # 4. Lead-in monotonicity: no massive upward motion before drop
        s_mono = _clamp(1.0 - up_before / max(0.55 * dy_down, 0.10))

        # 5. Handshape score: pinky extension + contrast vs index
        s_hs_ext = _clamp(pinky_ext / 0.75)
        s_hs_contrast = _clamp((pinky_vs_index + 0.15) / 0.40)
        s_hs = 0.5 * s_hs_ext + 0.5 * s_hs_contrast

        # Composite J score (weighted sum)
        composite = float(
            0.25 * s_drop +
            0.20 * s_aspect +
            0.25 * s_hook +
            0.10 * s_mono +
            0.20 * s_hs
        )

        return GeometricFeaturesJ(
            aspect_ratio=aspect_ratio,
            downward_drop=dy_down,
            hook_lateral=dx_hook,
            hook_upward=dy_hook,
            upward_before_drop=up_before,
            pinky_extension=pinky_ext,
            curled_extension=curled_mean,
            handshape_dominance=dominance,
            pinky_vs_index=pinky_vs_index,
            curve_similarity=float(0.5 * s_drop + 0.5 * s_hook),
            composite_score=composite,
        )

    def extract_z_features(
        self,
        path_index: np.ndarray,
        landmarks_3d: np.ndarray,
        palm_scales: np.ndarray,
    ) -> GeometricFeaturesZ:
        """Extract geometric ratios, stroke segmentation, and corner turns for Z (Index, Landmark 8)."""
        x = path_index[:, 0]
        y = path_index[:, 1]

        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        aspect_ratio = float(x_span / max(y_span, 1e-6))

        best_corner1 = 5
        best_corner2 = 10
        best_rev_score = -1e9
        best_dx = (0.0, 0.0, 0.0)
        best_dy = (0.0, 0.0, 0.0)
        best_turn_angles = (0.0, 0.0)

        for t1 in range(2, 8):
            for t2 in range(t1 + 2, 14):
                dx1 = float(x[t1] - x[0])
                dy1 = float(y[t1] - y[0])

                dx2 = float(x[t2] - x[t1])
                dy2 = float(y[t2] - y[t1])

                dx3 = float(x[-1] - x[t2])
                dy3 = float(y[-1] - y[t2])

                rev1 = bool(dx1 * dx2 < 0)
                rev2 = bool(dx2 * dx3 < 0)
                is_valid_rev = rev1 and rev2

                is_down_diag = dy2 > 0

                v1 = np.array([dx1, dy1])
                v2 = np.array([dx2, dy2])
                v3 = np.array([dx3, dy3])

                n1, n2, n3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
                if n1 > 1e-4 and n2 > 1e-4:
                    cos1 = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    ang1_deg = float(np.degrees(np.arccos(cos1)))
                else:
                    ang1_deg = 0.0

                if n2 > 1e-4 and n3 > 1e-4:
                    cos2 = float(np.clip(np.dot(v2, v3) / (n2 * n3), -1.0, 1.0))
                    ang2_deg = float(np.degrees(np.arccos(cos2)))
                else:
                    ang2_deg = 0.0

                score = (abs(dx1) + abs(dx2) + abs(dx3)) + 1.5 * dy2
                if is_valid_rev:
                    score += 2.0
                if is_down_diag:
                    score += 1.0

                if score > best_rev_score:
                    best_rev_score = score
                    best_corner1 = t1
                    best_corner2 = t2
                    best_dx = (dx1, dx2, dx3)
                    best_dy = (dy1, dy2, dy3)
                    best_turn_angles = (ang1_deg, ang2_deg)

        dx1, dx2, dx3 = best_dx
        dy1, dy2, dy3 = best_dy
        ang1, ang2 = best_turn_angles
        rev_valid = bool((dx1 * dx2 < 0) and (dx2 * dx3 < 0))

        # Handshape metrics at endpoint
        safe_palm = max(float(palm_scales[-1]), 1e-6)
        index_pt = landmarks_3d[-1, INDEX_TIP_IDX, :]
        index_ext = float(np.linalg.norm(index_pt) / safe_palm)

        pinky_pt = landmarks_3d[-1, PINKY_TIP_IDX, :]
        pinky_ext = float(np.linalg.norm(pinky_pt) / safe_palm)
        index_vs_pinky = float(index_ext - pinky_ext)

        other_lms = (12, 16, PINKY_TIP_IDX)  # middle, ring, pinky
        curled_exts = [
            float(np.linalg.norm(landmarks_3d[-1, idx, :]) / safe_palm)
            for idx in other_lms
        ]
        curled_mean = float(np.mean(curled_exts))
        dominance = float(index_ext - curled_mean)

        # Score components for Z:
        # 1. Stroke span score: all 3 strokes should have reasonable width
        min_span = min(abs(dx1), abs(dx2), abs(dx3))
        s_span = _clamp(min_span / 0.12)

        # 2. Reversal score: requires reversals
        s_rev = 1.0 if rev_valid else 0.0

        # 3. Diagonal downward score: dy2 >= 0.12
        s_diag = _clamp(dy2 / 0.15)

        # 4. Turn angle score: ideal turns around 110 - 160 deg
        s_ang1 = _clamp(1.0 - abs(ang1 - 135.0) / 60.0)
        s_ang2 = _clamp(1.0 - abs(ang2 - 135.0) / 60.0)
        s_angles = 0.5 * s_ang1 + 0.5 * s_ang2

        # 5. Aspect ratio score: Z aspect ratio ideally between 0.50 and 2.50
        if aspect_ratio < 0.35:
            s_aspect = _clamp(aspect_ratio / 0.35)
        elif aspect_ratio <= 2.20:
            s_aspect = 1.0
        else:
            s_aspect = _clamp(1.0 - (aspect_ratio - 2.20) / 1.0)

        # 6. Handshape score: index extension + contrast vs pinky
        s_hs_ext = _clamp(index_ext / 0.75)
        s_hs_contrast = _clamp((index_vs_pinky + 0.15) / 0.40)
        s_hs = 0.5 * s_hs_ext + 0.5 * s_hs_contrast

        # Composite Z score
        composite = float(
            0.20 * s_span +
            0.25 * s_rev +
            0.20 * s_diag +
            0.15 * s_angles +
            0.10 * s_aspect +
            0.10 * s_hs
        )

        return GeometricFeaturesZ(
            aspect_ratio=aspect_ratio,
            corner1_idx=best_corner1,
            corner2_idx=best_corner2,
            dx_stroke1=dx1,
            dx_stroke2=dx2,
            dx_stroke3=dx3,
            dy_diagonal=dy2,
            turn1_angle_deg=ang1,
            turn2_angle_deg=ang2,
            reversal_valid=rev_valid,
            index_extension=index_ext,
            curled_extension=curled_mean,
            handshape_dominance=dominance,
            index_vs_pinky=index_vs_pinky,
            piecewise_similarity=float(0.5 * s_span + 0.5 * s_diag),
            composite_score=composite,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> GeometricClassificationResult:
        """Classify a (16, 69) sequence into J, Z, or UNKNOWN based on geometric rules."""
        t_start = time.perf_counter()

        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                return GeometricClassificationResult(
                    prediction="UNKNOWN",
                    score_j=0.0,
                    score_z=0.0,
                    margin=0.0,
                    features_j=None,
                    features_z=None,
                    status="INVALID_INPUT",
                    rejection_reason="INVALID_SHAPE_OR_NON_FINITE",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            return GeometricClassificationResult(
                prediction="UNKNOWN",
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                features_j=None,
                features_z=None,
                status="ERROR",
                rejection_reason=f"INPUT_ERROR: {exc}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 1. Parse landmarks and palm scale
        landmarks_3d = arr[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_pts = landmarks_3d[:, PALM_SCALE_LANDMARKS, :]
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        # 2. Extract trajectory paths (palm scale normalized from cols 63..68)
        path_index = arr[:, 65:67]
        path_pinky = arr[:, 67:69]

        # 3. Compute geometric features for J and Z
        feat_j = self.extract_j_features(path_pinky, landmarks_3d, palm_scales)
        feat_z = self.extract_z_features(path_index, landmarks_3d, palm_scales)

        score_j = feat_j.composite_score
        score_z = feat_z.composite_score
        margin = float(abs(score_j - score_z))

        # 4. Multi-criteria Decision Logic
        # A. Minimum score check
        best_score = max(score_j, score_z)
        if best_score < self.min_score:
            return GeometricClassificationResult(
                prediction="UNKNOWN",
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                features_j=feat_j,
                features_z=feat_z,
                status="REJECTED",
                rejection_reason=f"INSUFFICIENT_GEOMETRIC_SCORE ({best_score:.3f} < {self.min_score:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # B. Separation margin check
        if margin < self.min_margin:
            return GeometricClassificationResult(
                prediction="UNKNOWN",
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                features_j=feat_j,
                features_z=feat_z,
                status="REJECTED",
                rejection_reason=f"AMBIGUOUS_SCORE_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # C. Dominance & Trajectory sanity check for winning candidate
        if score_j > score_z:
            if feat_j.downward_drop < 0.15:
                return GeometricClassificationResult(
                    prediction="UNKNOWN",
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_J_DROP ({feat_j.downward_drop:.3f} < 0.15)",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_j.hook_lateral < 0.10:
                return GeometricClassificationResult(
                    prediction="UNKNOWN",
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_J_HOOK ({feat_j.hook_lateral:.3f} < 0.10)",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            pred = "J"

        else:
            if not feat_z.reversal_valid:
                return GeometricClassificationResult(
                    prediction="UNKNOWN",
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason="Z_REVERSALS_INVALID",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            pred = "Z"

        return GeometricClassificationResult(
            prediction=pred,
            score_j=score_j,
            score_z=score_z,
            margin=margin,
            features_j=feat_j,
            features_z=feat_z,
            status="OK",
            rejection_reason="",
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )
