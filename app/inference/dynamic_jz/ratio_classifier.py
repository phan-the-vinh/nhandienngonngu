"""Ratio-Based Trajectory Recognition Engine for Dynamic J and Z Gestures (Stage 6).

An independent, rule-and-ratio geometric classifier that recognizes dynamic J/Z gestures
based on morphological ratios, trajectory progression, stroke proportions, and kinematic continuity:

For J (Pinky tracking, Landmark 20):
- Full trajectory vertical drop from Apex (highest point) to Nadir (deepest point).
- Horizontal hook excursion (dx_hook) across all post-nadir frames (supporting both left/right variants).
- Ratio of downward drop to hook lateral span (down_to_hook_ratio).
- Curvature profile and turning angle at nadir transition.
- Downward stroke linearity (distinguishing straight downward plunge from circular arc loops).
- Temporal apex, nadir, hook-peak, and endpoint coordinates and monotonicity.
- Handshape dominance (pinky extension vs index and curled fingers).

For Z (Index tracking, Landmark 8):
- Optimal 3-stroke segmentation (top horizontal, diagonal downward-reverse, bottom horizontal).
- Horizontal direction reversals (dx1 * dx2 < 0 and dx2 * dx3 < 0).
- Diagonal downward progression (dy2 > 0).
- Turning corner angles (~100-165 deg).
- Stroke length balance ratios (L1/L_tot, L2/L_tot, L3/L_tot).
- Piecewise linear trajectory similarity.
- Strict discrimination against:
  * Horizontal lines (insufficient dy2 or y_span).
  * Diagonal lines (insufficient horizontal stroke spans, no reversals).
  * Inverted/wrong zigzags (dy2 <= 0).
  * Circular motions (constant curvature, loop closure, no sharp corners).
  * Random hand tremor and noise (insufficient stroke path length).
- Handshape dominance (index extension vs pinky and curled fingers).

Data handling:
- Palm-scale normalized (dimensionless, camera resolution and distance invariant).
- Sequence-relative displacement (wrist/hand position invariant).
- Detection and robust handling of MediaPipe duplicate frames and micro-tremors (~35%).
- Outlier spike discrimination (out-and-back glitch vs persistent stroke progression).
- Conservative UNKNOWN rejection for weak, incomplete, or out-of-distribution motions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class RatioFeaturesJ:
    """Extracted geometric and ratio features for the J hypothesis (Pinky, Landmark 20)."""
    x_span: float                           # Full horizontal span in palm scales
    y_span: float                           # Full vertical span in palm scales
    dx_total: float                         # Net horizontal displacement |x_end - x_start|
    dy_total: float                         # Net vertical displacement y_end - y_start
    aspect_ratio: float                     # x_span / max(y_span, eps) (< 1.0 for vertical J)
    t_apex: int                             # Frame index of apex (highest point)
    t_nadir: int                            # Frame index of nadir (lowest point)
    t_hook_peak: int                        # Frame index of maximum lateral hook excursion
    t_end: int                              # Final frame index (15)
    apex_coord: tuple[float, float]         # (x, y) at apex
    nadir_coord: tuple[float, float]        # (x, y) at nadir
    hook_peak_coord: tuple[float, float]    # (x, y) at hook peak
    endpoint_coord: tuple[float, float]     # (x, y) at endpoint
    downward_drop: float                    # dy from apex to nadir (palm scales)
    downward_ratio: float                   # downward_drop / y_span
    hook_lateral: float                     # max lateral excursion from nadir for t >= nadir
    hook_ratio: float                       # hook_lateral / x_span
    down_to_hook_ratio: float               # downward_drop / hook_lateral
    hook_direction: str                     # 'RIGHT' or 'LEFT'
    turn_angle_deg: float                   # Angle between incoming drop vector and outgoing hook vector
    drop_linearity: float                   # x_span_during_drop / downward_drop
    is_circular_loop: bool                  # Whether motion curves in a closed circle
    motion_continuity: float                # Smoothness measure (1.0 - max_step / span)
    is_teleport_jump: bool                  # Whether a single frame jump exceeds teleport threshold
    duplicate_frame_ratio: float            # Fraction of frozen/duplicate frames (< 0.008 step)
    pinky_extension: float                  # Endpoint pinky tip distance / palm scale
    curled_extension: float                 # Endpoint mean curled finger distance / palm scale
    handshape_dominance: float              # pinky_ext - curled_ext
    pinky_vs_index: float                   # pinky_ext - index_ext
    composite_score: float                  # Multi-criteria weighted score (0..1)


@dataclass(frozen=True)
class RatioFeaturesZ:
    """Extracted geometric and ratio features for the Z hypothesis (Index, Landmark 8)."""
    x_span: float                           # Full horizontal span in palm scales
    y_span: float                           # Full vertical span in palm scales
    aspect_ratio: float                     # x_span / max(y_span, eps) (~0.6-2.5 for wide Z)
    corner1_idx: int                        # Frame index of corner 1 (end of top stroke)
    corner2_idx: int                        # Frame index of corner 2 (end of diagonal stroke)
    corner1_coord: tuple[float, float]      # (x, y) at corner 1
    corner2_coord: tuple[float, float]      # (x, y) at corner 2
    dx_stroke1: float                       # Top stroke horizontal displacement
    dy_stroke1: float                       # Top stroke vertical displacement
    dx_stroke2: float                       # Diagonal stroke horizontal displacement
    dy_stroke2: float                       # Diagonal stroke downward displacement
    dx_stroke3: float                       # Bottom stroke horizontal displacement
    dy_stroke3: float                       # Bottom stroke vertical displacement
    stroke1_len: float                      # Euclidean length of stroke 1
    stroke2_len: float                      # Euclidean length of stroke 2 (diagonal)
    stroke3_len: float                      # Euclidean length of stroke 3
    total_path_len: float                   # L1 + L2 + L3
    stroke_length_ratios: tuple[float, float, float]  # (L1/L, L2/L, L3/L)
    horizontal_ratio: float                 # (|dx1| + |dx3|) / (2 * x_span)
    diagonal_ratio: float                   # L2 / total_path_len
    diagonal_downward: float                # dy2 (must be positive)
    reversals_count: int                    # Number of valid reversals (target 2)
    reversals_valid: bool                   # True if dx1*dx2 < 0 and dx2*dx3 < 0
    turn1_angle_deg: float                  # Turning angle at corner 1 in degrees
    turn2_angle_deg: float                  # Turning angle at corner 2 in degrees
    piecewise_similarity: float             # Trajectory fit to 3-segment Z path (0..1)
    is_horizontal_line: bool                # Rejection flag: purely horizontal motion
    is_diagonal_line: bool                  # Rejection flag: purely diagonal swipe
    is_circular_motion: bool                # Rejection flag: circular wave
    is_jitter_noise: bool                   # Rejection flag: static noise / micro-tremor
    motion_continuity: float                # Step continuity measure (0..1)
    duplicate_frame_ratio: float            # Fraction of duplicate frames
    index_extension: float                  # Endpoint index tip distance / palm scale
    curled_extension: float                 # Endpoint mean curled finger distance / palm scale
    handshape_dominance: float              # index_ext - curled_ext
    index_vs_pinky: float                   # index_ext - pinky_ext
    composite_score: float                  # Multi-criteria weighted score (0..1)


@dataclass(frozen=True)
class RatioClassificationResult:
    """Output contract of the Ratio-Based Trajectory Classifier."""
    prediction: str                         # 'J', 'Z', or 'UNKNOWN'
    confidence: float                       # Winning confidence score (0..1)
    score_j: float                          # Composite ratio score for J (0..1)
    score_z: float                          # Composite ratio score for Z (0..1)
    margin: float                           # |score_j - score_z|
    ratios: dict[str, Any]                  # Detailed ratio metrics dictionary
    features_j: RatioFeaturesJ | None
    features_z: RatioFeaturesZ | None
    status: str                             # 'OK', 'REJECTED', 'INVALID_INPUT', 'ERROR'
    rejection_reason: str = ""
    latency_ms: float = 0.0


def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, val)))


class RatioBasedJZClassifier:
    """Independent Ratio-Based Classifier for Dynamic ASL J and Z gestures.

    Operates purely on morphological ratios, geometric stroke segmentation,
    and kinematic continuity criteria.
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        min_margin: float = 0.08,
        min_j_drop: float = 0.25,
        min_j_hook: float = 0.10,
        min_z_span: float = 0.06,
        min_z_diag_drop: float = 0.06,
        min_pinky_ext: float = 0.50,
        min_index_ext: float = 0.50,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.min_j_drop = float(min_j_drop)
        self.min_j_hook = float(min_j_hook)
        self.min_z_span = float(min_z_span)
        self.min_z_diag_drop = float(min_z_diag_drop)
        self.min_pinky_ext = float(min_pinky_ext)
        self.min_index_ext = float(min_index_ext)

    def extract_j_features(
        self,
        path_pinky: np.ndarray,
        landmarks_3d: np.ndarray,
        palm_scales: np.ndarray,
    ) -> RatioFeaturesJ:
        """Extract multi-ratio features and trajectory progression for J (Pinky tip, Landmark 20)."""
        x = path_pinky[:, 0]
        y = path_pinky[:, 1]  # positive Y is downward in camera/image coordinates

        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        dx_total = float(abs(x[-1] - x[0]))
        dy_total = float(y[-1] - y[0])
        aspect_ratio = float(x_span / max(y_span, 1e-6))

        # 1. Apex, Nadir, and Hook progression
        # Nadir is lowest point before frame 15 to allow a hook phase
        t_nadir = int(np.argmax(y[:15]))
        # Apex is highest point prior to reaching nadir
        t_apex = int(np.argmin(y[: t_nadir + 1]))
        t_end = len(y) - 1

        downward_drop = float(y[t_nadir] - y[t_apex])
        downward_ratio = float(downward_drop / max(y_span, 1e-6))

        # Lateral hook excursion across all frames t >= t_nadir
        lateral_offsets = np.abs(x[t_nadir:] - x[t_nadir])
        if len(lateral_offsets) > 0:
            hook_lateral = float(np.max(lateral_offsets))
            t_hook_peak = t_nadir + int(np.argmax(lateral_offsets))
            hook_direction = "RIGHT" if (x[t_hook_peak] - x[t_nadir]) > 0 else "LEFT"
        else:
            hook_lateral = 0.0
            t_hook_peak = t_nadir
            hook_direction = "NONE"

        hook_ratio = float(hook_lateral / max(x_span, 1e-6))
        down_to_hook_ratio = float(downward_drop / max(hook_lateral, 1e-6))

        # 2. Downward stroke linearity vs circular loop
        if t_nadir > t_apex:
            drop_x_span = float(np.ptp(x[t_apex : t_nadir + 1]))
        else:
            drop_x_span = 0.0
        drop_linearity = float(drop_x_span / max(downward_drop, 1e-6))

        # Loop closure distance: circle returns back to start; J terminates at hook
        net_dist = float(np.linalg.norm(path_pinky[-1] - path_pinky[0]))
        is_circular_loop = bool(
            net_dist < 0.40 * max(y_span, 1e-6) and drop_linearity > 0.65 and aspect_ratio > 0.60
        )

        # 3. Turning angle at nadir transition
        v_in = path_pinky[t_nadir] - path_pinky[t_apex]
        v_out = path_pinky[t_hook_peak] - path_pinky[t_nadir]
        n_in = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        if n_in > 1e-4 and n_out > 1e-4:
            cos_turn = float(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))
            turn_angle_deg = float(np.degrees(np.arccos(cos_turn)))
        else:
            turn_angle_deg = 90.0

        # 4. Motion continuity and duplicate frame check
        step_diffs = np.linalg.norm(np.diff(path_pinky, axis=0), axis=1)
        duplicate_count = int(np.sum(step_diffs < 0.008))
        duplicate_ratio = float(duplicate_count / max(len(step_diffs), 1))

        max_step = float(np.max(step_diffs)) if len(step_diffs) > 0 else 0.0
        motion_continuity = _clamp(1.0 - max_step / max(y_span, 0.20))
        is_teleport = bool(max_step >= 0.96 * max(downward_drop, 0.20) and len(step_diffs) <= 2)

        # 5. Handshape features at endpoint
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

        # 6. Scoring components for J
        # A. Downward drop score: saturates around downward_drop >= 0.80
        s_drop = _clamp((downward_drop - 0.20) / 0.60)

        # B. Hook lateral score: saturates around hook_lateral >= 0.40
        s_hook = _clamp((hook_lateral - 0.10) / 0.30)

        # C. Down-to-hook ratio score: ideal between 1.2 and 4.5
        if down_to_hook_ratio < 0.6:
            s_ratio = _clamp(down_to_hook_ratio / 0.6)
        elif down_to_hook_ratio <= 4.5:
            s_ratio = 1.0
        else:
            s_ratio = _clamp(1.0 - (down_to_hook_ratio - 4.5) / 3.0)

        # D. Turn angle score: ideal around 75 - 135 deg
        s_turn = _clamp(1.0 - abs(turn_angle_deg - 105.0) / 60.0)

        # E. Aspect ratio score: J is vertically elongated (aspect_ratio <= 0.90)
        if aspect_ratio < 0.15:
            s_aspect = _clamp(aspect_ratio / 0.15)
        elif aspect_ratio <= 1.0:
            s_aspect = 1.0
        else:
            s_aspect = _clamp(1.0 - (aspect_ratio - 1.0) / 0.80)

        # F. Handshape score: pinky extension + contrast vs index
        s_hs_ext = _clamp(pinky_ext / 0.70)
        s_hs_contrast = _clamp((pinky_vs_index + 0.15) / 0.35)
        s_hs = 0.5 * s_hs_ext + 0.5 * s_hs_contrast

        # G. Rejection multiplier for circular motion or teleport
        s_circ = 0.0 if is_circular_loop or is_teleport else 1.0

        composite_score = float(
            (0.25 * s_drop +
             0.20 * s_hook +
             0.15 * s_ratio +
             0.15 * s_turn +
             0.10 * s_aspect +
             0.15 * s_hs) * s_circ
        )

        return RatioFeaturesJ(
            x_span=x_span,
            y_span=y_span,
            dx_total=dx_total,
            dy_total=dy_total,
            aspect_ratio=aspect_ratio,
            t_apex=t_apex,
            t_nadir=t_nadir,
            t_hook_peak=t_hook_peak,
            t_end=t_end,
            apex_coord=(float(x[t_apex]), float(y[t_apex])),
            nadir_coord=(float(x[t_nadir]), float(y[t_nadir])),
            hook_peak_coord=(float(x[t_hook_peak]), float(y[t_hook_peak])),
            endpoint_coord=(float(x[-1]), float(y[-1])),
            downward_drop=downward_drop,
            downward_ratio=downward_ratio,
            hook_lateral=hook_lateral,
            hook_ratio=hook_ratio,
            down_to_hook_ratio=down_to_hook_ratio,
            hook_direction=hook_direction,
            turn_angle_deg=turn_angle_deg,
            drop_linearity=drop_linearity,
            is_circular_loop=is_circular_loop,
            motion_continuity=motion_continuity,
            is_teleport_jump=is_teleport,
            duplicate_frame_ratio=duplicate_ratio,
            pinky_extension=pinky_ext,
            curled_extension=curled_mean,
            handshape_dominance=dominance,
            pinky_vs_index=pinky_vs_index,
            composite_score=composite_score,
        )

    def extract_z_features(
        self,
        path_index: np.ndarray,
        landmarks_3d: np.ndarray,
        palm_scales: np.ndarray,
    ) -> RatioFeaturesZ:
        """Extract multi-ratio features and 3-stroke geometry for Z (Index tip, Landmark 8)."""
        x = path_index[:, 0]
        y = path_index[:, 1]

        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        aspect_ratio = float(x_span / max(y_span, 1e-6))

        # 1. Search for optimal 3-stroke corner partition (corner1, corner2)
        best_score = -1e9
        best_t1, best_t2 = 4, 10
        best_dx = (0.0, 0.0, 0.0)
        best_dy = (0.0, 0.0, 0.0)
        best_angles = (0.0, 0.0)
        best_rev = False

        for t1 in range(1, 8):
            for t2 in range(t1 + 2, 15):
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
                n1, n2, n3 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2)), float(np.linalg.norm(v3))

                ang1 = (
                    float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / max(n1 * n2, 1e-6), -1.0, 1.0))))
                    if n1 > 1e-4 and n2 > 1e-4 else 0.0
                )
                ang2 = (
                    float(np.degrees(np.arccos(np.clip(np.dot(v2, v3) / max(n2 * n3, 1e-6), -1.0, 1.0))))
                    if n2 > 1e-4 and n3 > 1e-4 else 0.0
                )

                score = (abs(dx1) + abs(dx2) + abs(dx3)) + 1.5 * dy2
                if is_valid_rev:
                    score += 3.0
                if is_down_diag:
                    score += 1.5

                if score > best_score:
                    best_score = score
                    best_t1, best_t2 = t1, t2
                    best_dx = (dx1, dx2, dx3)
                    best_dy = (dy1, dy2, dy3)
                    best_angles = (ang1, ang2)
                    best_rev = is_valid_rev

        dx1, dx2, dx3 = best_dx
        dy1, dy2, dy3 = best_dy
        ang1, ang2 = best_angles

        l1 = float(np.hypot(dx1, dy1))
        l2 = float(np.hypot(dx2, dy2))
        l3 = float(np.hypot(dx3, dy3))
        total_len = max(l1 + l2 + l3, 1e-6)
        r1, r2, r3 = l1 / total_len, l2 / total_len, l3 / total_len

        min_span = min(abs(dx1), abs(dx2), abs(dx3))
        reversals_count = (1 if dx1 * dx2 < 0 else 0) + (1 if dx2 * dx3 < 0 else 0)

        # 2. Discrimination against invalid motion archetypes
        # A. Horizontal line: no diagonal drop or y_span too small
        is_horiz = bool(dy2 < 0.06 or y_span < 0.12 or abs(dy1) + abs(dy3) > 1.5 * abs(dy2))
        # B. Diagonal line: no horizontal strokes
        is_diag = bool(min_span < 0.05 or abs(dx1) < 0.05 or abs(dx3) < 0.05 or reversals_count < 1)
        # C. Jitter / noise: overall movement too tiny
        is_jitter = bool(total_len < 0.25 or x_span < 0.15)
        # D. Circular motion: continuous circular wave without sharp reversals
        net_dist = float(np.linalg.norm(path_index[-1] - path_index[0]))
        is_circ = bool(net_dist < 0.35 * max(x_span, 1e-6) and abs(ang1 - 90.0) < 25.0 and abs(ang2 - 90.0) < 25.0 and not best_rev)

        # 3. Piecewise model similarity (deviation from idealized 3-line segments)
        devs = []
        for t in range(len(path_index)):
            pt = path_index[t]
            if t <= best_t1:
                p_start, p_end = path_index[0], path_index[best_t1]
            elif t <= best_t2:
                p_start, p_end = path_index[best_t1], path_index[best_t2]
            else:
                p_start, p_end = path_index[best_t2], path_index[-1]
            seg_vec = p_end - p_start
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len > 1e-4:
                # 2D perpendicular distance: |ax*by - ay*bx| / len
                d = float(abs(seg_vec[0] * (pt[1] - p_start[1]) - seg_vec[1] * (pt[0] - p_start[0])) / seg_len)
            else:
                d = float(np.linalg.norm(pt - p_start))
            devs.append(d)
        mean_dev = float(np.mean(devs))
        piecewise_sim = _clamp(float(np.exp(-mean_dev / 0.15)))

        # 4. Motion continuity and duplicate frames
        step_diffs = np.linalg.norm(np.diff(path_index, axis=0), axis=1)
        duplicate_count = int(np.sum(step_diffs < 0.008))
        duplicate_ratio = float(duplicate_count / max(len(step_diffs), 1))
        max_step = float(np.max(step_diffs)) if len(step_diffs) > 0 else 0.0
        motion_continuity = _clamp(1.0 - max_step / max(total_len, 0.20))

        # 5. Handshape metrics at endpoint
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

        # 6. Scoring components for Z
        # A. Direction reversal score: strictly requires 2 horizontal reversals
        s_rev = 1.0 if best_rev else 0.0

        # B. Stroke span score: all 3 strokes must have sufficient horizontal extent
        s_span = _clamp((min_span - 0.06) / 0.20)

        # C. Diagonal downward score: dy2 must progress downward
        s_diag = _clamp((dy2 - 0.06) / 0.25) if dy2 > 0 else 0.0

        # D. Turn angle score: ideal turns around 110 - 160 deg
        s_ang1 = _clamp(1.0 - abs(ang1 - 135.0) / 60.0)
        s_ang2 = _clamp(1.0 - abs(ang2 - 135.0) / 60.0)
        s_ang = 0.5 * s_ang1 + 0.5 * s_ang2

        # E. Stroke balance score: all 3 strokes well-proportioned
        s_bal = _clamp(1.0 - 2.0 * max(0.0, 0.18 - min(r1, r2, r3)))

        # F. Handshape score: index extension + contrast vs pinky
        s_hs_ext = _clamp(index_ext / 0.70)
        s_hs_contrast = _clamp((index_vs_pinky + 0.15) / 0.35)
        s_hs = 0.5 * s_hs_ext + 0.5 * s_hs_contrast

        # G. Rejection multiplier for invalid patterns
        s_invalid = 0.0 if (is_horiz or is_diag or is_jitter or is_circ) else 1.0

        composite_score = float(
            (0.25 * s_rev +
             0.20 * s_span +
             0.20 * s_diag +
             0.15 * s_ang +
             0.10 * s_bal +
             0.10 * s_hs) * s_invalid
        )

        return RatioFeaturesZ(
            x_span=x_span,
            y_span=y_span,
            aspect_ratio=aspect_ratio,
            corner1_idx=best_t1,
            corner2_idx=best_t2,
            corner1_coord=(float(x[best_t1]), float(y[best_t1])),
            corner2_coord=(float(x[best_t2]), float(y[best_t2])),
            dx_stroke1=dx1,
            dy_stroke1=dy1,
            dx_stroke2=dx2,
            dy_stroke2=dy2,
            dx_stroke3=dx3,
            dy_stroke3=dy3,
            stroke1_len=l1,
            stroke2_len=l2,
            stroke3_len=l3,
            total_path_len=total_len,
            stroke_length_ratios=(r1, r2, r3),
            horizontal_ratio=float((abs(dx1) + abs(dx3)) / (2.0 * max(x_span, 1e-6))),
            diagonal_ratio=float(l2 / total_len),
            diagonal_downward=dy2,
            reversals_count=reversals_count,
            reversals_valid=best_rev,
            turn1_angle_deg=ang1,
            turn2_angle_deg=ang2,
            piecewise_similarity=piecewise_sim,
            is_horizontal_line=is_horiz,
            is_diagonal_line=is_diag,
            is_circular_motion=is_circ,
            is_jitter_noise=is_jitter,
            motion_continuity=motion_continuity,
            duplicate_frame_ratio=duplicate_ratio,
            index_extension=index_ext,
            curled_extension=curled_mean,
            handshape_dominance=dominance,
            index_vs_pinky=index_vs_pinky,
            composite_score=composite_score,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> RatioClassificationResult:
        """Classify a 16x69 sequence into J, Z, or UNKNOWN based on geometric ratios."""
        t_start = time.perf_counter()

        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=0.0,
                    score_j=0.0,
                    score_z=0.0,
                    margin=0.0,
                    ratios={},
                    features_j=None,
                    features_z=None,
                    status="INVALID_INPUT",
                    rejection_reason="INVALID_SHAPE_OR_NON_FINITE",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            return RatioClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                ratios={},
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

        # 2. Extract trajectory paths (palm scale normalized cols 63..68)
        path_index = arr[:, 65:67]
        path_pinky = arr[:, 67:69]

        # 3. Extract ratio features for J and Z
        feat_j = self.extract_j_features(path_pinky, landmarks_3d, palm_scales)
        feat_z = self.extract_z_features(path_index, landmarks_3d, palm_scales)

        score_j = feat_j.composite_score
        score_z = feat_z.composite_score
        margin = float(abs(score_j - score_z))
        best_score = max(score_j, score_z)

        # Build detailed ratios dictionary for inspection & telemetry
        ratios_dict: dict[str, Any] = {
            "j": {
                "aspect_ratio": round(feat_j.aspect_ratio, 3),
                "downward_drop": round(feat_j.downward_drop, 3),
                "hook_lateral": round(feat_j.hook_lateral, 3),
                "down_to_hook_ratio": round(feat_j.down_to_hook_ratio, 3),
                "turn_angle_deg": round(feat_j.turn_angle_deg, 1),
                "hook_direction": feat_j.hook_direction,
                "pinky_extension": round(feat_j.pinky_extension, 3),
                "handshape_dominance": round(feat_j.handshape_dominance, 3),
                "composite_score": round(feat_j.composite_score, 3),
            },
            "z": {
                "aspect_ratio": round(feat_z.aspect_ratio, 3),
                "min_stroke_span": round(min(abs(feat_z.dx_stroke1), abs(feat_z.dx_stroke2), abs(feat_z.dx_stroke3)), 3),
                "diagonal_downward": round(feat_z.diagonal_downward, 3),
                "turn1_angle_deg": round(feat_z.turn1_angle_deg, 1),
                "turn2_angle_deg": round(feat_z.turn2_angle_deg, 1),
                "stroke_ratios": [round(r, 3) for r in feat_z.stroke_length_ratios],
                "reversals_valid": feat_z.reversals_valid,
                "index_extension": round(feat_z.index_extension, 3),
                "handshape_dominance": round(feat_z.handshape_dominance, 3),
                "composite_score": round(feat_z.composite_score, 3),
            },
        }

        # 4. Multi-Criteria Decision Gating
        # A. Minimum confidence score check
        if best_score < self.min_confidence:
            return RatioClassificationResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                ratios=ratios_dict,
                features_j=feat_j,
                features_z=feat_z,
                status="REJECTED",
                rejection_reason=f"INSUFFICIENT_RATIO_SCORE ({best_score:.3f} < {self.min_confidence:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # B. Separation margin check between J and Z
        if margin < self.min_margin:
            return RatioClassificationResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                ratios=ratios_dict,
                features_j=feat_j,
                features_z=feat_z,
                status="REJECTED",
                rejection_reason=f"AMBIGUOUS_SCORE_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # C. Mandatory geometric sanity checks for winning class
        if score_j > score_z:
            if feat_j.is_circular_loop:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason="REJECTED_CIRCULAR_LOOP_MOTION",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_j.downward_drop < self.min_j_drop:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_J_DOWNWARD_DROP ({feat_j.downward_drop:.3f} < {self.min_j_drop:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_j.hook_lateral < self.min_j_hook:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_J_HOOK_LATERAL ({feat_j.hook_lateral:.3f} < {self.min_j_hook:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_j.pinky_extension < self.min_pinky_ext or feat_j.pinky_vs_index < -0.05:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"PINKY_HANDSHAPE_MISMATCH (ext={feat_j.pinky_extension:.3f}, vs_idx={feat_j.pinky_vs_index:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            pred = "J"
            conf = score_j

        else:
            if not feat_z.reversals_valid:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason="Z_DIRECTION_REVERSALS_INVALID",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_z.diagonal_downward < self.min_z_diag_drop:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_Z_DIAGONAL_DROP ({feat_z.diagonal_downward:.3f} < {self.min_z_diag_drop:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            min_z_span_actual = min(abs(feat_z.dx_stroke1), abs(feat_z.dx_stroke2), abs(feat_z.dx_stroke3))
            if min_z_span_actual < self.min_z_span:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INSUFFICIENT_Z_STROKE_SPAN ({min_z_span_actual:.3f} < {self.min_z_span:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            if feat_z.index_extension < self.min_index_ext or feat_z.index_vs_pinky < -0.05:
                return RatioClassificationResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    ratios=ratios_dict,
                    features_j=feat_j,
                    features_z=feat_z,
                    status="REJECTED",
                    rejection_reason=f"INDEX_HANDSHAPE_MISMATCH (ext={feat_z.index_extension:.3f}, vs_pnk={feat_z.index_vs_pinky:.3f})",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
            pred = "Z"
            conf = score_z

        return RatioClassificationResult(
            prediction=pred,
            confidence=conf,
            score_j=score_j,
            score_z=score_z,
            margin=margin,
            ratios=ratios_dict,
            features_j=feat_j,
            features_z=feat_z,
            status="OK",
            rejection_reason="",
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )
