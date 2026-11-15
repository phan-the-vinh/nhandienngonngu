"""Handshape + Trajectory Fusion Classifier for Dynamic J/Z Gestures (Stage 7).

Integrates two independent, orthogonal streams of physical evidence:
1. Handshape Branch: Multi-frame geometric analysis of 21 MediaPipe landmarks.
   - Pinky vs Index tip extensions from wrist normalized by palm scale.
   - Curled status of other fingers (middle, ring, thumb).
   - Distance from fingertips to respective MCP joints.
   - Handshape stability/consistency across the 16-frame window.
   - Independent HandshapeResult ('J', 'Z', 'UNKNOWN').

2. Trajectory Branch: Kinematic and morphological path geometry over time.
   - Pinky tip (Landmark 20) path for J: downward drop, horizontal hook, turn angle, loop closure.
   - Index tip (Landmark 8) path for Z: 3-stroke partition, 2 reversals, diagonal drop, turn angles.
   - Cumulative path length (distinguishing genuine motion from static poses).
   - Independent TrajectoryResult ('J', 'Z', 'UNKNOWN').

3. Fusion Logic:
   - Handshape J + Trajectory J -> Candidate J
   - Handshape Z + Trajectory Z -> Candidate Z
   - Handshape and Trajectory conflict -> UNKNOWN
   - Valid Handshape but no motion (static pose) -> UNKNOWN (Rejects static poses that fool LR)
   - Valid Trajectory but wrong handshape -> UNKNOWN (Rejects whole-hand/fist motions)
   - Incomplete / Out-of-Distribution motion -> UNKNOWN
   - Detailed per-attempt attribution of rejection reason (Handshape vs Trajectory vs Both).
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
    MIDDLE_TIP_IDX,
    PALM_SCALE_LANDMARKS,
    PINKY_TIP_IDX,
    RING_TIP_IDX,
    SEQUENCE_LENGTH,
    THUMB_TIP_IDX,
)


def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, val)))


# ==============================================================================
# DATA CONTRACTS
# ==============================================================================

@dataclass(frozen=True)
class HandshapeMetrics:
    """Detailed continuous handshape measurements across 16 frames."""
    pinky_extension_mean: float         # ||p20 - p0|| / s_palm (mean over 16 frames)
    index_extension_mean: float         # ||p8 - p0|| / s_palm
    middle_extension_mean: float        # ||p12 - p0|| / s_palm
    ring_extension_mean: float          # ||p16 - p0|| / s_palm
    thumb_extension_mean: float         # ||p4 - p0|| / s_palm
    pinky_curl_mean: float              # ||p20 - p17|| / s_palm
    index_curl_mean: float              # ||p8 - p5|| / s_palm
    pinky_vs_index: float               # pinky_ext - index_ext
    index_vs_pinky: float               # index_ext - pinky_ext
    pinky_vs_curled_contrast: float     # pinky_ext - max(index, middle, ring)
    index_vs_curled_contrast: float     # index_ext - max(middle, ring, pinky)
    pinky_stability_std: float          # std of pinky extension across 16 frames
    index_stability_std: float          # std of index extension across 16 frames
    palm_scale_mean: float              # mean palm scale
    palm_scale_std: float               # std of palm scale (hand stability)


@dataclass(frozen=True)
class HandshapeResult:
    """Output of the independent Handshape Branch."""
    prediction: str                     # 'J', 'Z', or 'UNKNOWN'
    confidence: float                   # 0.0 .. 1.0
    score_j: float                      # 0.0 .. 1.0
    score_z: float                      # 0.0 .. 1.0
    margin: float                       # |score_j - score_z|
    metrics: HandshapeMetrics | None
    rejection_reason: str = ""
    status: str = "OK"                  # 'OK', 'REJECTED', 'INVALID'


@dataclass(frozen=True)
class TrajectoryMetricsJ:
    """Trajectory morphological measurements for J (Pinky, Landmark 20)."""
    downward_drop: float                # Vertical drop from apex to nadir
    hook_lateral: float                 # Max horizontal excursion post-nadir
    down_to_hook_ratio: float           # drop / hook
    turn_angle_deg: float               # Turn angle at nadir
    hook_direction: str                 # 'RIGHT' or 'LEFT'
    cumulative_path_len: float          # Total path length of pinky
    is_circular_loop: bool              # Closed loop rejection flag
    drop_linearity: float               # Straightness of downward plunge


@dataclass(frozen=True)
class TrajectoryMetricsZ:
    """Trajectory morphological measurements for Z (Index, Landmark 8)."""
    corner1_idx: int                    # Frame of first corner
    corner2_idx: int                    # Frame of second corner
    dx_strokes: tuple[float, float, float]  # (dx1, dx2, dx3)
    dy_strokes: tuple[float, float, float]  # (dy1, dy2, dy3)
    diagonal_downward: float            # dy2 (must be positive)
    reversals_count: int                # Number of horizontal reversals (0, 1, 2)
    reversals_valid: bool               # True if 2 valid reversals
    turn1_angle_deg: float              # Turn angle at corner 1
    turn2_angle_deg: float              # Turn angle at corner 2
    stroke_length_ratios: tuple[float, float, float]  # (r1, r2, r3)
    cumulative_path_len: float          # Total path length of index
    piecewise_similarity: float         # Fit to 3-segment Z template


@dataclass(frozen=True)
class TrajectoryResult:
    """Output of the independent Trajectory Branch."""
    prediction: str                     # 'J', 'Z', or 'UNKNOWN'
    confidence: float                   # 0.0 .. 1.0
    score_j: float                      # 0.0 .. 1.0
    score_z: float                      # 0.0 .. 1.0
    margin: float                       # |score_j - score_z|
    metrics_j: TrajectoryMetricsJ | None
    metrics_z: TrajectoryMetricsZ | None
    rejection_reason: str = ""
    status: str = "OK"                  # 'OK', 'REJECTED', 'STATIC_POSE', 'INVALID'


@dataclass(frozen=True)
class FusionClassificationResult:
    """Final unified decision of the Handshape + Trajectory Fusion Classifier."""
    prediction: str                     # 'J', 'Z', or 'UNKNOWN'
    confidence: float                   # Unified fusion confidence (0.0 .. 1.0)
    decision_path: str                  # e.g., 'AGREED_J', 'AGREED_Z', 'CONFLICT_REJECTED', 'STATIC_POSE_REJECTED'
    primary_rejection_source: str       # 'NONE', 'HANDSHAPE', 'TRAJECTORY', 'CONFLICT', 'BOTH'
    rejection_reason: str
    handshape_result: HandshapeResult
    trajectory_result: TrajectoryResult
    latency_ms: float = 0.0


# ==============================================================================
# HANDSHAPE BRANCH IMPLEMENTATION
# ==============================================================================

class HandshapeBranch:
    """Evaluates static and multi-frame finger configuration across all 21 landmarks."""

    def __init__(
        self,
        min_confidence: float = 0.40,
        min_margin: float = 0.08,
        min_pinky_ext: float = 0.95,
        min_index_ext: float = 1.00,
        min_pinky_contrast: float = 0.10,
        min_index_contrast: float = 0.20,
        max_stability_std: float = 0.35,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.min_pinky_ext = float(min_pinky_ext)
        self.min_index_ext = float(min_index_ext)
        self.min_pinky_contrast = float(min_pinky_contrast)
        self.min_index_contrast = float(min_index_contrast)
        self.max_stability_std = float(max_stability_std)

    def evaluate(
        self,
        landmarks_3d: np.ndarray,
        palm_scales: np.ndarray,
    ) -> HandshapeResult:
        """Evaluate handshape evidence across all 16 frames.

        Parameters
        ----------
        landmarks_3d : np.ndarray of shape (16, 21, 3)
        palm_scales : np.ndarray of shape (16,)
        """
        safe_palm = np.where(palm_scales > 1e-6, palm_scales, 1.0)

        # 1. Extensions from wrist (Landmark 0) per frame normalized by palm scale
        ext_thumb = np.linalg.norm(landmarks_3d[:, THUMB_TIP_IDX, :], axis=1) / safe_palm
        ext_index = np.linalg.norm(landmarks_3d[:, INDEX_TIP_IDX, :], axis=1) / safe_palm
        ext_middle = np.linalg.norm(landmarks_3d[:, MIDDLE_TIP_IDX, :], axis=1) / safe_palm
        ext_ring = np.linalg.norm(landmarks_3d[:, RING_TIP_IDX, :], axis=1) / safe_palm
        ext_pinky = np.linalg.norm(landmarks_3d[:, PINKY_TIP_IDX, :], axis=1) / safe_palm

        # 2. Tip-to-MCP distances (curl indicators)
        curl_index = np.linalg.norm(landmarks_3d[:, 8, :] - landmarks_3d[:, 5, :], axis=1) / safe_palm
        curl_pinky = np.linalg.norm(landmarks_3d[:, 20, :] - landmarks_3d[:, 17, :], axis=1) / safe_palm

        # 3. Aggregates across 16 frames
        m_pinky = float(np.mean(ext_pinky))
        m_index = float(np.mean(ext_index))
        m_middle = float(np.mean(ext_middle))
        m_ring = float(np.mean(ext_ring))
        m_thumb = float(np.mean(ext_thumb))

        std_pinky = float(np.std(ext_pinky))
        std_index = float(np.std(ext_index))

        # Check for synthetic or zero-landmark inputs (OOD safeguard)
        all_exts = np.array([m_pinky, m_index, m_middle, m_ring, m_thumb])
        if float(np.ptp(all_exts)) < 0.05 or np.all(all_exts < 0.30):
            return HandshapeResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                metrics=None,
                rejection_reason="UNIFORM_OR_DEGENERATE_HANDSHAPE_OOD",
                status="REJECTED",
            )

        pinky_vs_index = float(m_pinky - m_index)
        index_vs_pinky = float(m_index - m_pinky)

        pinky_vs_curled = float(m_pinky - max(m_index, m_middle, m_ring))
        index_vs_curled = float(m_index - max(m_middle, m_ring, m_pinky))

        metrics = HandshapeMetrics(
            pinky_extension_mean=m_pinky,
            index_extension_mean=m_index,
            middle_extension_mean=m_middle,
            ring_extension_mean=m_ring,
            thumb_extension_mean=m_thumb,
            pinky_curl_mean=float(np.mean(curl_pinky)),
            index_curl_mean=float(np.mean(curl_index)),
            pinky_vs_index=pinky_vs_index,
            index_vs_pinky=index_vs_pinky,
            pinky_vs_curled_contrast=pinky_vs_curled,
            index_vs_curled_contrast=index_vs_curled,
            pinky_stability_std=std_pinky,
            index_stability_std=std_index,
            palm_scale_mean=float(np.mean(palm_scales)),
            palm_scale_std=float(np.std(palm_scales)),
        )

        # 4. Continuous Scoring for J Handshape (Pinky dominant, others curled)
        s_j_ext = _clamp((m_pinky - 0.90) / 0.60)
        s_j_contrast = _clamp((pinky_vs_curled + 0.15) / 0.45)
        s_j_stable = _clamp(1.0 - (std_pinky / self.max_stability_std))
        score_j = float(0.40 * s_j_ext + 0.45 * s_j_contrast + 0.15 * s_j_stable)

        # 5. Continuous Scoring for Z Handshape (Index dominant, others curled)
        s_z_ext = _clamp((m_index - 0.95) / 0.65)
        s_z_contrast = _clamp((index_vs_curled - 0.05) / 0.55)
        s_z_stable = _clamp(1.0 - (std_index / self.max_stability_std))
        score_z = float(0.40 * s_z_ext + 0.45 * s_z_contrast + 0.15 * s_z_stable)

        margin = float(abs(score_j - score_z))
        best_score = max(score_j, score_z)

        # 6. Handshape Decision
        if best_score < self.min_confidence:
            return HandshapeResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics=metrics,
                rejection_reason=f"LOW_HANDSHAPE_SCORE ({best_score:.3f} < {self.min_confidence:.3f})",
                status="REJECTED",
            )

        if margin < self.min_margin:
            return HandshapeResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics=metrics,
                rejection_reason=f"AMBIGUOUS_HANDSHAPE_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                status="REJECTED",
            )

        if score_j > score_z:
            if m_pinky < self.min_pinky_ext or pinky_vs_curled < self.min_pinky_contrast:
                return HandshapeResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics=metrics,
                    rejection_reason=f"INSUFFICIENT_PINKY_CONTRAST (ext={m_pinky:.2f}, contrast={pinky_vs_curled:.2f})",
                    status="REJECTED",
                )
            return HandshapeResult(
                prediction="J",
                confidence=score_j,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics=metrics,
                status="OK",
            )
        else:
            if m_index < self.min_index_ext or index_vs_curled < self.min_index_contrast:
                return HandshapeResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics=metrics,
                    rejection_reason=f"INSUFFICIENT_INDEX_CONTRAST (ext={m_index:.2f}, contrast={index_vs_curled:.2f})",
                    status="REJECTED",
                )
            return HandshapeResult(
                prediction="Z",
                confidence=score_z,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics=metrics,
                status="OK",
            )


# ==============================================================================
# TRAJECTORY BRANCH IMPLEMENTATION
# ==============================================================================

class TrajectoryBranch:
    """Evaluates motion trajectory morphology over time for Landmark 20 (J) and Landmark 8 (Z)."""

    def __init__(
        self,
        min_confidence: float = 0.38,
        min_margin: float = 0.05,
        min_motion_len: float = 0.20,
        min_j_drop: float = 0.18,
        min_j_hook: float = 0.08,
        min_z_span: float = 0.05,
        min_z_diag_drop: float = 0.04,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.min_motion_len = float(min_motion_len)
        self.min_j_drop = float(min_j_drop)
        self.min_j_hook = float(min_j_hook)
        self.min_z_span = float(min_z_span)
        self.min_z_diag_drop = float(min_z_diag_drop)

    def evaluate_j_trajectory(self, path_pinky: np.ndarray) -> tuple[float, TrajectoryMetricsJ]:
        """Compute morphological score and metrics for J (Pinky, Landmark 20)."""
        x = path_pinky[:, 0]
        y = path_pinky[:, 1]  # positive Y downward

        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        step_diffs = np.linalg.norm(np.diff(path_pinky, axis=0), axis=1)
        path_len = float(np.sum(step_diffs))

        # Nadir (lowest point) and Apex (highest point prior to nadir)
        t_nadir = int(np.argmax(y[:15]))
        t_apex = int(np.argmin(y[: t_nadir + 1]))

        downward_drop = float(y[t_nadir] - y[t_apex])

        # Lateral hook excursion
        lateral_offsets = np.abs(x[t_nadir:] - x[t_nadir])
        if len(lateral_offsets) > 0:
            hook_lateral = float(np.max(lateral_offsets))
            t_hook_peak = t_nadir + int(np.argmax(lateral_offsets))
            hook_direction = "RIGHT" if (x[t_hook_peak] - x[t_nadir]) > 0 else "LEFT"
        else:
            hook_lateral = 0.0
            t_hook_peak = t_nadir
            hook_direction = "NONE"

        down_to_hook_ratio = float(downward_drop / max(hook_lateral, 1e-6))

        # Drop straightness & loop closure
        drop_x_span = float(np.ptp(x[t_apex : t_nadir + 1])) if t_nadir > t_apex else 0.0
        drop_linearity = float(drop_x_span / max(downward_drop, 1e-6))

        net_dist = float(np.linalg.norm(path_pinky[-1] - path_pinky[0]))
        is_circular = bool(net_dist < 0.35 * max(y_span, 1e-6) and drop_linearity > 0.65 and x_span > 0.50 * y_span)

        # Turn angle at nadir
        v_in = path_pinky[t_nadir] - path_pinky[t_apex]
        v_out = path_pinky[t_hook_peak] - path_pinky[t_nadir]
        n_in, n_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
        if n_in > 1e-4 and n_out > 1e-4:
            cos_turn = float(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))
            turn_angle_deg = float(np.degrees(np.arccos(cos_turn)))
        else:
            turn_angle_deg = 90.0

        # Continuous scoring terms
        s_path = _clamp((path_len - 0.15) / 0.50)
        s_drop = _clamp((downward_drop - 0.15) / 0.55)
        s_hook = _clamp((hook_lateral - 0.06) / 0.25)
        s_ratio = _clamp(1.0 - abs(down_to_hook_ratio - 2.5) / 3.5) if down_to_hook_ratio > 0.4 else 0.0
        s_turn = _clamp(1.0 - abs(turn_angle_deg - 105.0) / 60.0)
        s_circ = 0.0 if is_circular else 1.0

        score_j = float((0.30 * s_drop + 0.25 * s_hook + 0.15 * s_ratio + 0.15 * s_turn + 0.15 * s_path) * s_circ)

        metrics_j = TrajectoryMetricsJ(
            downward_drop=downward_drop,
            hook_lateral=hook_lateral,
            down_to_hook_ratio=down_to_hook_ratio,
            turn_angle_deg=turn_angle_deg,
            hook_direction=hook_direction,
            cumulative_path_len=path_len,
            is_circular_loop=is_circular,
            drop_linearity=drop_linearity,
        )
        return score_j, metrics_j

    def evaluate_z_trajectory(self, path_index: np.ndarray) -> tuple[float, TrajectoryMetricsZ]:
        """Compute morphological score and metrics for Z (Index, Landmark 8)."""
        x = path_index[:, 0]
        y = path_index[:, 1]

        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        step_diffs = np.linalg.norm(np.diff(path_index, axis=0), axis=1)
        path_len = float(np.sum(step_diffs))

        # Search for optimal 3-stroke partition
        best_score = -1e9
        best_t1, best_t2 = 4, 10
        best_dx = (0.0, 0.0, 0.0)
        best_dy = (0.0, 0.0, 0.0)
        best_angles = (0.0, 0.0)
        best_rev = False

        for t1 in range(1, 9):
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

                v1 = np.array([dx1, dy1])
                v2 = np.array([dx2, dy2])
                v3 = np.array([dx3, dy3])
                n1, n2, n3 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2)), float(np.linalg.norm(v3))

                ang1 = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / max(n1 * n2, 1e-6), -1.0, 1.0)))) if n1 > 1e-4 and n2 > 1e-4 else 0.0
                ang2 = float(np.degrees(np.arccos(np.clip(np.dot(v2, v3) / max(n2 * n3, 1e-6), -1.0, 1.0)))) if n2 > 1e-4 and n3 > 1e-4 else 0.0

                search_score = (abs(dx1) + abs(dx2) + abs(dx3)) + 1.2 * dy2
                if is_valid_rev:
                    search_score += 2.5
                if dy2 > 0:
                    search_score += 1.0

                if search_score > best_score:
                    best_score = search_score
                    best_t1, best_t2 = t1, t2
                    best_dx = (dx1, dx2, dx3)
                    best_dy = (dy1, dy2, dy3)
                    best_angles = (ang1, ang2)
                    best_rev = is_valid_rev

        dx1, dx2, dx3 = best_dx
        dy1, dy2, dy3 = best_dy
        ang1, ang2 = best_angles

        l1, l2, l3 = float(np.hypot(dx1, dy1)), float(np.hypot(dx2, dy2)), float(np.hypot(dx3, dy3))
        tot_l = max(l1 + l2 + l3, 1e-6)
        r1, r2, r3 = l1 / tot_l, l2 / tot_l, l3 / tot_l

        reversals_count = (1 if dx1 * dx2 < 0 else 0) + (1 if dx2 * dx3 < 0 else 0)
        min_span = min(abs(dx1), abs(dx2), abs(dx3))

        # Piecewise 3-line similarity
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
                d = float(abs(seg_vec[0] * (pt[1] - p_start[1]) - seg_vec[1] * (pt[0] - p_start[0])) / seg_len)
            else:
                d = float(np.linalg.norm(pt - p_start))
            devs.append(d)
        piecewise_sim = _clamp(float(np.exp(-float(np.mean(devs)) / 0.18)))

        # Continuous scoring terms
        s_rev = 1.0 if best_rev else (0.50 if reversals_count == 1 and dy2 > 0.05 else 0.0)
        s_span = _clamp((min_span - 0.04) / 0.18)
        s_diag = _clamp((dy2 - 0.03) / 0.22) if dy2 > 0 else 0.0
        s_ang = 0.5 * _clamp(1.0 - abs(ang1 - 135.0) / 60.0) + 0.5 * _clamp(1.0 - abs(ang2 - 135.0) / 60.0)
        s_path = _clamp((path_len - 0.18) / 0.55)

        # Rejection multipliers (smooth penalties instead of brittle binary 0)
        is_flat_horiz = bool(dy2 < 0.02 and y_span < 0.08)
        is_pure_diag = bool(min_span < 0.03 and reversals_count == 0)
        mult = 0.0 if (is_flat_horiz or is_pure_diag) else 1.0

        score_z = float((0.30 * s_rev + 0.20 * s_span + 0.20 * s_diag + 0.15 * s_ang + 0.15 * s_path) * mult)

        metrics_z = TrajectoryMetricsZ(
            corner1_idx=best_t1,
            corner2_idx=best_t2,
            dx_strokes=(dx1, dx2, dx3),
            dy_strokes=(dy1, dy2, dy3),
            diagonal_downward=dy2,
            reversals_count=reversals_count,
            reversals_valid=best_rev,
            turn1_angle_deg=ang1,
            turn2_angle_deg=ang2,
            stroke_length_ratios=(r1, r2, r3),
            cumulative_path_len=path_len,
            piecewise_similarity=piecewise_sim,
        )
        return score_z, metrics_z

    def evaluate(
        self,
        path_index: np.ndarray,
        path_pinky: np.ndarray,
    ) -> TrajectoryResult:
        """Evaluate trajectory evidence independently for J and Z paths."""
        p_len_pinky = float(np.sum(np.linalg.norm(np.diff(path_pinky, axis=0), axis=1)))
        p_len_index = float(np.sum(np.linalg.norm(np.diff(path_index, axis=0), axis=1)))
        max_motion = max(p_len_pinky, p_len_index)

        # Check for static pose or insufficient motion
        if max_motion < self.min_motion_len:
            return TrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                metrics_j=None,
                metrics_z=None,
                rejection_reason=f"STATIC_POSE_INSUFFICIENT_MOTION ({max_motion:.3f} < {self.min_motion_len:.3f})",
                status="STATIC_POSE",
            )

        score_j, metrics_j = self.evaluate_j_trajectory(path_pinky)
        score_z, metrics_z = self.evaluate_z_trajectory(path_index)

        margin = float(abs(score_j - score_z))
        best_score = max(score_j, score_z)

        if best_score < self.min_confidence:
            return TrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics_j=metrics_j,
                metrics_z=metrics_z,
                rejection_reason=f"LOW_TRAJECTORY_SCORE ({best_score:.3f} < {self.min_confidence:.3f})",
                status="REJECTED",
            )

        if margin < self.min_margin:
            return TrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics_j=metrics_j,
                metrics_z=metrics_z,
                rejection_reason=f"AMBIGUOUS_TRAJECTORY_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                status="REJECTED",
            )

        if score_j > score_z:
            if metrics_j.is_circular_loop:
                return TrajectoryResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics_j=metrics_j,
                    metrics_z=metrics_z,
                    rejection_reason="REJECTED_CIRCULAR_LOOP",
                    status="REJECTED",
                )
            if metrics_j.downward_drop < self.min_j_drop:
                return TrajectoryResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics_j=metrics_j,
                    metrics_z=metrics_z,
                    rejection_reason=f"INSUFFICIENT_J_DOWNWARD_DROP ({metrics_j.downward_drop:.3f} < {self.min_j_drop:.3f})",
                    status="REJECTED",
                )
            if metrics_j.hook_lateral < self.min_j_hook:
                return TrajectoryResult(
                    prediction="UNKNOWN",
                    confidence=score_j,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics_j=metrics_j,
                    metrics_z=metrics_z,
                    rejection_reason=f"INSUFFICIENT_J_HOOK_LATERAL ({metrics_j.hook_lateral:.3f} < {self.min_j_hook:.3f})",
                    status="REJECTED",
                )
            return TrajectoryResult(
                prediction="J",
                confidence=score_j,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics_j=metrics_j,
                metrics_z=metrics_z,
                status="OK",
            )
        else:
            if metrics_z.diagonal_downward < self.min_z_diag_drop:
                return TrajectoryResult(
                    prediction="UNKNOWN",
                    confidence=score_z,
                    score_j=score_j,
                    score_z=score_z,
                    margin=margin,
                    metrics_j=metrics_j,
                    metrics_z=metrics_z,
                    rejection_reason=f"INSUFFICIENT_Z_DIAGONAL_DROP ({metrics_z.diagonal_downward:.3f} < {self.min_z_diag_drop:.3f})",
                    status="REJECTED",
                )
            return TrajectoryResult(
                prediction="Z",
                confidence=score_z,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                metrics_j=metrics_j,
                metrics_z=metrics_z,
                status="OK",
            )


# ==============================================================================
# UNIFIED FUSION CLASSIFIER
# ==============================================================================

class HandshapeTrajectoryFusionClassifier:
    """Unified J/Z Classifier fusing Handshape Branch and Trajectory Branch."""

    def __init__(
        self,
        handshape_branch: HandshapeBranch | None = None,
        trajectory_branch: TrajectoryBranch | None = None,
        weight_handshape: float = 0.45,
        weight_trajectory: float = 0.55,
        min_fused_confidence: float = 0.42,
    ) -> None:
        self.handshape_branch = handshape_branch or HandshapeBranch()
        self.trajectory_branch = trajectory_branch or TrajectoryBranch()
        self.w_hs = float(weight_handshape)
        self.w_tr = float(weight_trajectory)
        self.min_fused_confidence = float(min_fused_confidence)

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> FusionClassificationResult:
        """Classify a 16x69 dynamic sequence via Handshape + Trajectory fusion."""
        t_start = time.perf_counter()

        # 1. Defensive input validation
        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                dummy_hs = HandshapeResult("UNKNOWN", 0.0, 0.0, 0.0, 0.0, None, "INVALID_INPUT", "INVALID")
                dummy_tr = TrajectoryResult("UNKNOWN", 0.0, 0.0, 0.0, 0.0, None, None, "INVALID_INPUT", "INVALID")
                return FusionClassificationResult(
                    prediction="UNKNOWN",
                    confidence=0.0,
                    decision_path="INVALID_INPUT",
                    primary_rejection_source="BOTH",
                    rejection_reason="INVALID_INPUT_SHAPE_OR_NON_FINITE",
                    handshape_result=dummy_hs,
                    trajectory_result=dummy_tr,
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            dummy_hs = HandshapeResult("UNKNOWN", 0.0, 0.0, 0.0, 0.0, None, f"ERROR: {exc}", "INVALID")
            dummy_tr = TrajectoryResult("UNKNOWN", 0.0, 0.0, 0.0, 0.0, None, None, f"ERROR: {exc}", "INVALID")
            return FusionClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                decision_path="ERROR",
                primary_rejection_source="BOTH",
                rejection_reason=f"INPUT_CONVERSION_ERROR: {exc}",
                handshape_result=dummy_hs,
                trajectory_result=dummy_tr,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 2. Extract landmark representation and palm scale
        landmarks_3d = arr[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_pts = landmarks_3d[:, PALM_SCALE_LANDMARKS, :]
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        # 3. Extract trajectories (palm-scale normalized columns 65..68)
        path_index = arr[:, 65:67]
        path_pinky = arr[:, 67:69]

        # 4. Evaluate Handshape and Trajectory branches
        hs_res = self.handshape_branch.evaluate(landmarks_3d, palm_scales)
        tr_res = self.trajectory_branch.evaluate(path_index, path_pinky)

        # 5. Core Fusion Logic
        p_hs = hs_res.prediction
        p_tr = tr_res.prediction

        # Case A: Both branches agree on positive gesture
        if p_hs in ("J", "Z") and p_hs == p_tr:
            fused_conf = self.w_hs * hs_res.confidence + self.w_tr * tr_res.confidence
            decision_path = f"AGREED_{p_hs}"
            return FusionClassificationResult(
                prediction=p_hs,
                confidence=float(fused_conf),
                decision_path=decision_path,
                primary_rejection_source="NONE",
                rejection_reason="",
                handshape_result=hs_res,
                trajectory_result=tr_res,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Case B: Handshape and Trajectory conflict (e.g. Handshape J, Trajectory Z)
        if p_hs in ("J", "Z") and p_tr in ("J", "Z") and p_hs != p_tr:
            return FusionClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                decision_path="CONFLICT_REJECTED",
                primary_rejection_source="CONFLICT",
                rejection_reason=f"HANDSHAPE_TRAJECTORY_CONFLICT (Handshape={p_hs}, Trajectory={p_tr})",
                handshape_result=hs_res,
                trajectory_result=tr_res,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Case C: Handshape valid, but Trajectory is UNKNOWN due to static pose (Holding hand still)
        if p_hs in ("J", "Z") and tr_res.status == "STATIC_POSE":
            return FusionClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                decision_path="STATIC_POSE_REJECTED",
                primary_rejection_source="TRAJECTORY",
                rejection_reason=f"STATIC_POSE_REJECTED: Handshape is {p_hs} but no physical motion occurred ({tr_res.rejection_reason})",
                handshape_result=hs_res,
                trajectory_result=tr_res,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Case D: Handshape valid, but Trajectory failed geometric criteria (incomplete / wrong shape)
        if p_hs in ("J", "Z") and p_tr == "UNKNOWN":
            return FusionClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                decision_path="TRAJECTORY_REJECTED",
                primary_rejection_source="TRAJECTORY",
                rejection_reason=f"TRAJECTORY_REJECTED: Handshape={p_hs} but trajectory invalid ({tr_res.rejection_reason})",
                handshape_result=hs_res,
                trajectory_result=tr_res,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Case E: Trajectory valid, but Handshape is UNKNOWN (non-finger motion or open hand)
        if p_tr in ("J", "Z") and p_hs == "UNKNOWN":
            return FusionClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                decision_path="HANDSHAPE_REJECTED",
                primary_rejection_source="HANDSHAPE",
                rejection_reason=f"HANDSHAPE_REJECTED: Trajectory={p_tr} but handshape unverified ({hs_res.rejection_reason})",
                handshape_result=hs_res,
                trajectory_result=tr_res,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Case F: Both branches rejected / UNKNOWN
        return FusionClassificationResult(
            prediction="UNKNOWN",
            confidence=0.0,
            decision_path="BOTH_REJECTED",
            primary_rejection_source="BOTH",
            rejection_reason=f"BOTH_REJECTED: HS=({hs_res.rejection_reason}), TR=({tr_res.rejection_reason})",
            handshape_result=hs_res,
            trajectory_result=tr_res,
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )
