"""Target Finger Trajectory Recognition Engine for Dynamic J and Z Gestures (Stage 9).

Focuses specifically on the morphological dynamics of the target drawing fingertip:
- J: Landmark 20 (pinky tip) trajectory: downward plunge, inflection point,
     lateral hook excursion, turn angle, curvature continuity, and anti-1D drop guard.
- Z: Landmark 8 (index tip) trajectory: 3-stroke sequential partition,
     horizontal top stroke, downward diagonal with reversal 1, final horizontal
     stroke with reversal 2, stroke length ratios, turn angles, and anti-tremor guard.

Provides three decoupled recognition interfaces for strict ablation:
1. Trajectory-Only Recognizer: Classifies purely from fingertip motion morphology.
2. Handshape-Only Recognizer: Classifies purely from 21-landmark hand configuration.
3. Target Finger Trajectory Recognizer (Combined): Fuses target trajectory morphology
   with soft handshape guidance, anti-static guard, and OOD safeguards.
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
from app.inference.dynamic_jz.fusion_classifier import (
    HandshapeBranch,
    HandshapeMetrics,
    HandshapeResult,
)


def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, val)))


# ==============================================================================
# TELEMETRY AND RESULT DATACLASSES
# ==============================================================================

@dataclass(frozen=True)
class TargetPinkyJMetrics:
    """Detailed morphological telemetry for Landmark 20 (pinky tip) J trajectory."""
    path_length: float
    downward_drop: float                  # y_nadir - y_apex
    hook_lateral: float                   # max |x - x_nadir| after nadir
    hook_direction: str                   # 'LEFT', 'RIGHT', 'NONE'
    down_to_hook_ratio: float             # downward_drop / hook_lateral
    turn_angle_deg: float                 # angle at nadir between plunge and hook vectors
    drop_linearity: float                 # x_span during drop / downward_drop
    is_circular_loop: bool                # closed loop rejection
    is_1d_downward: bool                  # pure 1D plunge without lateral hook
    is_static: bool                       # path length < static threshold
    temporal_stages: dict[str, int]       # {'apex_t': int, 'nadir_t': int, 'hook_peak_t': int}
    curvature_smoothness: float           # smooth direction transition score


@dataclass(frozen=True)
class TargetIndexZMetrics:
    """Detailed morphological telemetry for Landmark 8 (index tip) Z trajectory."""
    path_length: float
    corner1_idx: int                      # t1 partition index
    corner2_idx: int                      # t2 partition index
    dx_strokes: tuple[float, float, float]
    dy_strokes: tuple[float, float, float]
    stroke_lengths: tuple[float, float, float]
    stroke_ratios: tuple[float, float, float]
    diagonal_downward: float              # dy2 (stroke 2 downward drop)
    reversals_count: int                  # number of x-direction sign changes (0, 1, 2)
    reversals_valid: bool                 # true if dx1*dx2 < 0 and dx2*dx3 < 0
    turn1_angle_deg: float                # angle at corner 1 between stroke 1 and 2
    turn2_angle_deg: float                # angle at corner 2 between stroke 2 and 3
    is_flat_horizontal: bool              # pure horizontal swipe without vertical diagonal
    is_single_diagonal: bool              # single diagonal slash without reversals
    is_tremor: bool                       # high frequency jitter with tiny stroke spans
    is_static: bool                       # path length < static threshold
    piecewise_fit_error: float            # mean orthogonal error from 3-stroke segments


@dataclass(frozen=True)
class TargetTrajectoryResult:
    """Standardized output contract for Stage 9 Target Finger Trajectory Recognizer."""
    prediction: str                       # 'J', 'Z', or 'UNKNOWN'
    confidence: float                     # 0.0 .. 1.0
    score_j: float                        # Confidence / score for J
    score_z: float                        # Confidence / score for Z
    margin: float                         # |score_j - score_z|
    decision_path: str                    # e.g., 'TARGET_J', 'TARGET_Z', 'STATIC_REJECTED'
    rejection_reason: str
    pinky_j_metrics: TargetPinkyJMetrics | None = None
    index_z_metrics: TargetIndexZMetrics | None = None
    handshape_metrics: HandshapeMetrics | None = None
    latency_ms: float = 0.0


# ==============================================================================
# TARGET FINGER ANALYZERS
# ==============================================================================

class TargetPinkyJAnalyzer:
    """Analyzes landmark 20 (pinky tip) trajectory morphology for J gestures."""

    def __init__(
        self,
        min_path_len: float = 0.15,
        min_downward_drop: float = 0.12,
        min_hook_lateral: float = 0.05,
        max_ratio_drop_to_hook: float = 6.5,
    ) -> None:
        self.min_path_len = float(min_path_len)
        self.min_downward_drop = float(min_downward_drop)
        self.min_hook_lateral = float(min_hook_lateral)
        self.max_ratio_drop_to_hook = float(max_ratio_drop_to_hook)

    def analyze(self, path_pinky: np.ndarray) -> tuple[float, TargetPinkyJMetrics]:
        """Evaluate landmark 20 (pinky tip) 2D trajectory.

        Parameters
        ----------
        path_pinky : np.ndarray of shape (16, 2)
            Displacement path of landmark 20 normalized by palm scale.
            x: horizontal axis (right positive), y: vertical axis (downward positive).
        """
        x = path_pinky[:, 0]
        y = path_pinky[:, 1]

        step_diffs = np.linalg.norm(np.diff(path_pinky, axis=0), axis=1)
        path_len = float(np.sum(step_diffs))
        is_static = bool(path_len < self.min_path_len)

        # 1. Identify Nadir (lowest point) and Apex (highest point preceding nadir)
        t_nadir = int(np.argmax(y[:15]))
        t_apex = int(np.argmin(y[: t_nadir + 1]))

        downward_drop = float(y[t_nadir] - y[t_apex])

        # 2. Identify Lateral Hook Excursion after Nadir
        lateral_offsets = np.abs(x[t_nadir:] - x[t_nadir])
        if len(lateral_offsets) > 0:
            hook_lateral = float(np.max(lateral_offsets))
            t_hook_peak = t_nadir + int(np.argmax(lateral_offsets))
            dx_hook = float(x[t_hook_peak] - x[t_nadir])
            hook_direction = "RIGHT" if dx_hook > 0 else ("LEFT" if dx_hook < 0 else "NONE")
        else:
            hook_lateral = 0.0
            t_hook_peak = t_nadir
            hook_direction = "NONE"

        down_to_hook_ratio = float(downward_drop / max(hook_lateral, 1e-6))

        # 3. Drop Linearity & Anti-1D Drop Guard
        drop_x_span = float(np.ptp(x[t_apex : t_nadir + 1])) if t_nadir > t_apex else 0.0
        drop_linearity = float(drop_x_span / max(downward_drop, 1e-6))

        is_1d_downward = bool(
            downward_drop >= self.min_downward_drop
            and (hook_lateral < self.min_hook_lateral or down_to_hook_ratio > self.max_ratio_drop_to_hook)
        )

        # 4. Loop Closure & Anti-Circular Loop Guard
        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        net_dist = float(np.linalg.norm(path_pinky[-1] - path_pinky[0]))
        is_circular = bool(
            net_dist < 0.35 * max(y_span, 1e-6)
            and drop_linearity > 0.65
            and x_span > 0.50 * y_span
        )

        # 5. Turning Angle at Inflection Point
        v_in = path_pinky[t_nadir] - path_pinky[t_apex]
        v_out = path_pinky[t_hook_peak] - path_pinky[t_nadir]
        n_in, n_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
        if n_in > 1e-4 and n_out > 1e-4:
            cos_turn = float(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))
            turn_angle_deg = float(np.degrees(np.arccos(cos_turn)))
        else:
            turn_angle_deg = 90.0

        # Curvature smoothness
        angles_along_path = []
        for i in range(1, len(path_pinky) - 1):
            va = path_pinky[i] - path_pinky[i - 1]
            vb = path_pinky[i + 1] - path_pinky[i]
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if na > 1e-4 and nb > 1e-4:
                ct = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
                angles_along_path.append(float(np.degrees(np.arccos(ct))))
        mean_step_turn = float(np.mean(angles_along_path)) if angles_along_path else 0.0
        curvature_smoothness = _clamp(1.0 - (mean_step_turn / 90.0))

        # 6. Continuous Scoring
        s_drop = _clamp((downward_drop - 0.12) / 0.55)
        s_hook = _clamp((hook_lateral - 0.04) / 0.25)
        s_ratio = _clamp(1.0 - abs(down_to_hook_ratio - 2.5) / 3.5) if down_to_hook_ratio > 0.3 else 0.0
        s_turn = _clamp(1.0 - abs(turn_angle_deg - 105.0) / 60.0)
        s_path = _clamp((path_len - 0.15) / 0.50)

        # Multiplicative vetoes for degenerate or invalid shapes
        veto_mult = 1.0
        if is_static or is_1d_downward or is_circular:
            veto_mult = 0.0

        score_j = float(
            (0.30 * s_drop + 0.25 * s_hook + 0.15 * s_ratio + 0.15 * s_turn + 0.15 * s_path) * veto_mult
        )

        metrics = TargetPinkyJMetrics(
            path_length=path_len,
            downward_drop=downward_drop,
            hook_lateral=hook_lateral,
            hook_direction=hook_direction,
            down_to_hook_ratio=down_to_hook_ratio,
            turn_angle_deg=turn_angle_deg,
            drop_linearity=drop_linearity,
            is_circular_loop=is_circular,
            is_1d_downward=is_1d_downward,
            is_static=is_static,
            temporal_stages={"apex_t": t_apex, "nadir_t": t_nadir, "hook_peak_t": t_hook_peak},
            curvature_smoothness=curvature_smoothness,
        )
        return score_j, metrics


class TargetIndexZAnalyzer:
    """Analyzes landmark 8 (index tip) trajectory morphology for Z gestures."""

    def __init__(
        self,
        min_path_len: float = 0.15,
        min_horizontal_span: float = 0.04,
        min_diagonal_drop: float = 0.015,
        min_stroke_length: float = 0.035,
    ) -> None:
        self.min_path_len = float(min_path_len)
        self.min_horizontal_span = float(min_horizontal_span)
        self.min_diagonal_drop = float(min_diagonal_drop)
        self.min_stroke_length = float(min_stroke_length)

    def analyze(self, path_index: np.ndarray) -> tuple[float, TargetIndexZMetrics]:
        """Evaluate landmark 8 (index tip) 2D trajectory.

        Parameters
        ----------
        path_index : np.ndarray of shape (16, 2)
            Displacement path of landmark 8 normalized by palm scale.
        """
        x = path_index[:, 0]
        y = path_index[:, 1]

        step_diffs = np.linalg.norm(np.diff(path_index, axis=0), axis=1)
        path_len = float(np.sum(step_diffs))
        is_static = bool(path_len < self.min_path_len)

        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))

        # 1. Search for optimal 3-stroke partition (0 < t1 < t2 < 15)
        best_search_score = -1e9
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

                ang1 = (
                    float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / max(n1 * n2, 1e-6), -1.0, 1.0))))
                    if n1 > 1e-4 and n2 > 1e-4 else 0.0
                )
                ang2 = (
                    float(np.degrees(np.arccos(np.clip(np.dot(v2, v3) / max(n2 * n3, 1e-6), -1.0, 1.0))))
                    if n2 > 1e-4 and n3 > 1e-4 else 0.0
                )

                search_score = (abs(dx1) + abs(dx2) + abs(dx3)) + 1.2 * dy2
                if is_valid_rev:
                    search_score += 2.5
                if dy2 > 0:
                    search_score += 1.0

                if search_score > best_search_score:
                    best_search_score = search_score
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
        min_stroke_span = min(abs(dx1), abs(dx2), abs(dx3))
        min_len = min(l1, l2, l3)

        # 2. Piecewise fit error
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
        mean_fit_error = float(np.mean(devs))

        # 3. Guards
        is_flat_horizontal = bool(dy2 < self.min_diagonal_drop and y_span < 0.08)
        is_single_diagonal = bool(reversals_count < 2 or not best_rev or min_stroke_span < self.min_horizontal_span)
        is_tremor = bool(min_len < self.min_stroke_length and reversals_count >= 1)

        # 4. Continuous Scoring Terms
        s_rev = 1.0 if best_rev else 0.0
        s_span = _clamp((min_stroke_span - 0.04) / 0.18)
        s_diag = _clamp((dy2 - self.min_diagonal_drop) / 0.22) if dy2 > 0 else 0.0
        s_ang = 0.5 * _clamp(1.0 - abs(ang1 - 135.0) / 60.0) + 0.5 * _clamp(1.0 - abs(ang2 - 135.0) / 60.0)
        s_path = _clamp((path_len - 0.18) / 0.55)

        veto_mult = 1.0
        if is_static or is_flat_horizontal or is_single_diagonal or is_tremor:
            veto_mult = 0.0

        score_z = float(
            (0.30 * s_rev + 0.20 * s_span + 0.20 * s_diag + 0.15 * s_ang + 0.15 * s_path) * veto_mult
        )

        metrics = TargetIndexZMetrics(
            path_length=path_len,
            corner1_idx=best_t1,
            corner2_idx=best_t2,
            dx_strokes=(dx1, dx2, dx3),
            dy_strokes=(dy1, dy2, dy3),
            stroke_lengths=(l1, l2, l3),
            stroke_ratios=(r1, r2, r3),
            diagonal_downward=dy2,
            reversals_count=reversals_count,
            reversals_valid=best_rev,
            turn1_angle_deg=ang1,
            turn2_angle_deg=ang2,
            is_flat_horizontal=is_flat_horizontal,
            is_single_diagonal=is_single_diagonal,
            is_tremor=is_tremor,
            is_static=is_static,
            piecewise_fit_error=mean_fit_error,
        )
        return score_z, metrics


# ==============================================================================
# INDEPENDENT RECOGNIZERS FOR ABLATION
# ==============================================================================

class TrajectoryOnlyJZRecognizer:
    """Classifies dynamic J/Z gestures solely from fingertip motion morphology."""

    def __init__(
        self,
        min_confidence: float = 0.30,
        min_margin: float = 0.02,
        min_motion_len: float = 0.15,
        min_j_drop: float = 0.12,
        min_j_hook: float = 0.05,
        min_z_span: float = 0.04,
        min_z_diag_drop: float = 0.015,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.min_motion_len = float(min_motion_len)
        self.pinky_analyzer = TargetPinkyJAnalyzer(
            min_path_len=min_motion_len,
            min_downward_drop=min_j_drop,
            min_hook_lateral=min_j_hook,
        )
        self.index_analyzer = TargetIndexZAnalyzer(
            min_path_len=min_motion_len,
            min_horizontal_span=min_z_span,
            min_diagonal_drop=min_z_diag_drop,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> TargetTrajectoryResult:
        """Classify sequence (16, 69) using only Landmark 20 and Landmark 8 trajectories."""
        t_start = time.perf_counter()
        seq = np.asarray(sequence, dtype=np.float64)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="INVALID_SHAPE",
                rejection_reason=f"Invalid shape {seq.shape}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        path_idx = seq[:, 65:67]
        path_pky = seq[:, 67:69]

        score_j, metrics_j = self.pinky_analyzer.analyze(path_pky)
        score_z, metrics_z = self.index_analyzer.analyze(path_idx)

        best_score = max(score_j, score_z)
        margin = float(abs(score_j - score_z))

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if metrics_j.is_static and metrics_z.is_static:
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="STATIC_POSE_REJECTED",
                rejection_reason=f"Static pose (path_pky={metrics_j.path_length:.3f}, path_idx={metrics_z.path_length:.3f})",
                pinky_j_metrics=metrics_j,
                index_z_metrics=metrics_z,
                latency_ms=latency_ms,
            )

        if best_score < self.min_confidence:
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                decision_path="LOW_TRAJECTORY_SCORE",
                rejection_reason=f"Low trajectory score ({best_score:.3f} < {self.min_confidence:.3f})",
                pinky_j_metrics=metrics_j,
                index_z_metrics=metrics_z,
                latency_ms=latency_ms,
            )

        if margin < self.min_margin:
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                decision_path="AMBIGUOUS_TRAJECTORY_MARGIN",
                rejection_reason=f"Ambiguous margin ({margin:.3f} < {self.min_margin:.3f})",
                pinky_j_metrics=metrics_j,
                index_z_metrics=metrics_z,
                latency_ms=latency_ms,
            )

        if score_j > score_z:
            return TargetTrajectoryResult(
                prediction="J",
                confidence=score_j,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                decision_path="TRAJECTORY_ONLY_J",
                rejection_reason="",
                pinky_j_metrics=metrics_j,
                index_z_metrics=metrics_z,
                latency_ms=latency_ms,
            )
        else:
            return TargetTrajectoryResult(
                prediction="Z",
                confidence=score_z,
                score_j=score_j,
                score_z=score_z,
                margin=margin,
                decision_path="TRAJECTORY_ONLY_Z",
                rejection_reason="",
                pinky_j_metrics=metrics_j,
                index_z_metrics=metrics_z,
                latency_ms=latency_ms,
            )


class HandshapeOnlyJZRecognizer:
    """Classifies dynamic J/Z gestures solely from 21-landmark handshape geometry."""

    def __init__(
        self,
        min_confidence: float = 0.35,
        min_margin: float = 0.05,
        min_pinky_ext: float = 0.90,
        min_index_ext: float = 0.95,
        min_pinky_contrast: float = 0.08,
        min_index_contrast: float = 0.15,
        max_stability_std: float = 0.38,
    ) -> None:
        self.branch = HandshapeBranch(
            min_confidence=min_confidence,
            min_margin=min_margin,
            min_pinky_ext=min_pinky_ext,
            min_index_ext=min_index_ext,
            min_pinky_contrast=min_pinky_contrast,
            min_index_contrast=min_index_contrast,
            max_stability_std=max_stability_std,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> TargetTrajectoryResult:
        """Classify sequence (16, 69) using only columns 0:63 handshape."""
        t_start = time.perf_counter()
        seq = np.asarray(sequence, dtype=np.float64)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="INVALID_SHAPE",
                rejection_reason=f"Invalid shape {seq.shape}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        lms = seq[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_dists = np.linalg.norm(lms[:, PALM_SCALE_LANDMARKS, :], axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        hs_res = self.branch.evaluate(lms, palm_scales)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return TargetTrajectoryResult(
            prediction=hs_res.prediction,
            confidence=hs_res.confidence,
            score_j=hs_res.score_j,
            score_z=hs_res.score_z,
            margin=hs_res.margin,
            decision_path=f"HANDSHAPE_ONLY_{hs_res.prediction}",
            rejection_reason=hs_res.rejection_reason,
            handshape_metrics=hs_res.metrics,
            latency_ms=latency_ms,
        )


class TargetFingerTrajectoryRecognizer:
    """Comprehensive Target Finger Trajectory Recognizer fusing trajectory and handshape.

    Combines:
    1. Landmark 20 (pinky tip) trajectory morphology for J.
    2. Landmark 8 (index tip) trajectory morphology for Z.
    3. Handshape branch for target finger confirmation.
    4. Anti-static guard against resting hands.
    5. Soft handshape veto (threshold = 0.15) protecting against subtle foreshortening
       while strictly vetoing curled-finger cross-confusions.
    6. OOD guard rejecting degenerate inputs.
    """

    def __init__(
        self,
        min_motion_len: float = 0.15,
        min_joint_score: float = 0.35,
        min_margin: float = 0.05,
        weight_handshape: float = 0.40,
        weight_trajectory: float = 0.60,
        agreement_bonus_weight: float = 0.15,
        handshape_veto_threshold: float = 0.15,
        min_j_drop: float = 0.12,
        min_j_hook: float = 0.05,
        min_z_span: float = 0.04,
        min_z_diag_drop: float = 0.015,
    ) -> None:
        self.min_motion_len = float(min_motion_len)
        self.min_joint_score = float(min_joint_score)
        self.min_margin = float(min_margin)
        self.w_hs = float(weight_handshape)
        self.w_tr = float(weight_trajectory)
        self.w_bonus = float(agreement_bonus_weight)
        self.hs_veto = float(handshape_veto_threshold)

        self.pinky_analyzer = TargetPinkyJAnalyzer(
            min_path_len=min_motion_len,
            min_downward_drop=min_j_drop,
            min_hook_lateral=min_j_hook,
        )
        self.index_analyzer = TargetIndexZAnalyzer(
            min_path_len=min_motion_len,
            min_horizontal_span=min_z_span,
            min_diagonal_drop=min_z_diag_drop,
        )
        self.handshape_branch = HandshapeBranch(
            min_confidence=0.35,
            min_margin=0.05,
            min_pinky_ext=0.90,
            min_index_ext=0.95,
            min_pinky_contrast=0.08,
            min_index_contrast=0.15,
            max_stability_std=0.38,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> TargetTrajectoryResult:
        """Classify sequence (16, 69) fusing target trajectory morphology with handshape."""
        t_start = time.perf_counter()
        seq = np.asarray(sequence, dtype=np.float64)
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="INVALID_SHAPE",
                rejection_reason=f"Invalid shape {seq.shape}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 1. Anti-static guard on motion paths
        path_idx = seq[:, 65:67]
        path_pky = seq[:, 67:69]
        p_len_pinky = float(np.sum(np.linalg.norm(np.diff(path_pky, axis=0), axis=1)))
        p_len_index = float(np.sum(np.linalg.norm(np.diff(path_idx, axis=0), axis=1)))
        max_motion = max(p_len_pinky, p_len_index)

        if max_motion < self.min_motion_len:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="STATIC_POSE_REJECTED",
                rejection_reason=f"STATIC_POSE_INSUFFICIENT_MOTION ({max_motion:.3f} < {self.min_motion_len:.3f})",
                latency_ms=latency_ms,
            )

        # 2. Handshape evidence
        lms = seq[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_dists = np.linalg.norm(lms[:, PALM_SCALE_LANDMARKS, :], axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        hs_res = self.handshape_branch.evaluate(lms, palm_scales)
        if hs_res.status == "REJECTED" and "OOD" in hs_res.rejection_reason:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="OOD_REJECTED",
                rejection_reason=f"OOD_DEGENERATE_HANDSHAPE ({hs_res.rejection_reason})",
                handshape_metrics=hs_res.metrics,
                latency_ms=latency_ms,
            )

        s_hs_j = hs_res.score_j
        s_hs_z = hs_res.score_z

        # 3. Target finger trajectory evidence
        s_tr_j, m_j = self.pinky_analyzer.analyze(path_pky)
        s_tr_z, m_z = self.index_analyzer.analyze(path_idx)

        # 4. Joint score computation with agreement bonus
        bonus_j = self.w_bonus * s_hs_j * s_tr_j
        bonus_z = self.w_bonus * s_hs_z * s_tr_z

        joint_j = self.w_hs * s_hs_j + self.w_tr * s_tr_j + bonus_j
        joint_z = self.w_hs * s_hs_z + self.w_tr * s_tr_z + bonus_z

        # 5. Target finger handshape vetoes
        # If target finger is curled (negative contrast or low score), veto joint score
        is_curled_pinky = bool(hs_res.metrics is not None and hs_res.metrics.pinky_vs_curled_contrast < -0.05)
        is_curled_index = bool(hs_res.metrics is not None and hs_res.metrics.index_vs_curled_contrast < -0.05)

        if s_tr_j < 0.20 or s_hs_j <= self.hs_veto or is_curled_pinky:
            joint_j = 0.0
        if s_tr_z < 0.20 or s_hs_z <= self.hs_veto or is_curled_index or m_z.diagonal_downward < 0.003:
            joint_z = 0.0

        best_score = max(joint_j, joint_z)
        margin = float(abs(joint_j - joint_z))
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if best_score < self.min_joint_score:
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="INSUFFICIENT_JOINT_SCORE",
                rejection_reason=f"INSUFFICIENT_JOINT_SCORE ({best_score:.3f} < {self.min_joint_score:.3f})",
                pinky_j_metrics=m_j,
                index_z_metrics=m_z,
                handshape_metrics=hs_res.metrics,
                latency_ms=latency_ms,
            )

        if margin < self.min_margin:
            return TargetTrajectoryResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="AMBIGUOUS_JOINT_MARGIN",
                rejection_reason=f"AMBIGUOUS_JOINT_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                pinky_j_metrics=m_j,
                index_z_metrics=m_z,
                handshape_metrics=hs_res.metrics,
                latency_ms=latency_ms,
            )

        if joint_j > joint_z:
            return TargetTrajectoryResult(
                prediction="J",
                confidence=_clamp(joint_j),
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="TARGET_FINGER_J",
                rejection_reason="",
                pinky_j_metrics=m_j,
                index_z_metrics=m_z,
                handshape_metrics=hs_res.metrics,
                latency_ms=latency_ms,
            )
        else:
            return TargetTrajectoryResult(
                prediction="Z",
                confidence=_clamp(joint_z),
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="TARGET_FINGER_Z",
                rejection_reason="",
                pinky_j_metrics=m_j,
                index_z_metrics=m_z,
                handshape_metrics=hs_res.metrics,
                latency_ms=latency_ms,
            )
