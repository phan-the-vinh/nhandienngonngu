"""Adaptive Motion Recognition Engine for Dynamic J and Z Gestures (Stage 8).

Replaces rigid boolean "Handshape AND Trajectory" gating with an adaptive,
continuous-evidence integration engine:

1. Handshape Branch: Continuous confidence scoring across all 21 landmarks.
   - Pinky tip extension & contrast vs curled fingers for J.
   - Index tip extension & contrast vs curled fingers for Z.
   - Continuous degree-of-belief scoring [0, 1] without brittle binary hard cutoffs.

2. Trajectory Branch: Independent tracking of active fingertips.
   - Pinky tip (Landmark 20) for J downward plunge and lateral hook.
   - Index tip (Landmark 8) for Z 3-stroke partition, reversals, and diagonal drop.
   - Dynamic motion amplitude scaling (tolerating small vs large stroke sizes).
   - Discrimination against jitter, tracking noise, and static poses.

3. Adaptive Decision Arbiter:
   - Evaluates joint posterior belief S_joint(J) and S_joint(Z).
   - Soft agreement bonus when both handshape and trajectory provide positive evidence.
   - Target-guided trajectory evaluation prevents curled inactive finger drift
     from corrupting active finger trajectory margins.
   - Anti-static guard: Sequences with cumulative path length < 0.15 palm scales
     are classified as STATIC_POSE (returning UNKNOWN), preventing static hand
     holds from triggering false alarms (the core flaw of Baseline LR).
   - OOD guard: Non-hand or degenerate inputs are rejected as UNKNOWN.
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
    MIDDLE_TIP_IDX,
    PALM_SCALE_LANDMARKS,
    PINKY_TIP_IDX,
    RING_TIP_IDX,
    SEQUENCE_LENGTH,
    THUMB_TIP_IDX,
)
from app.inference.dynamic_jz.fusion_classifier import HandshapeBranch, TrajectoryBranch


def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, val)))


@dataclass(frozen=True)
class AdaptiveMotionMetrics:
    """Continuous evidence telemetry for adaptive J/Z classification."""
    handshape_score_j: float
    handshape_score_z: float
    trajectory_score_j: float
    trajectory_score_z: float
    joint_score_j: float
    joint_score_z: float
    motion_length_pinky: float
    motion_length_index: float
    max_motion_length: float
    is_static_pose: bool
    agreement_bonus_j: float
    agreement_bonus_z: float


@dataclass(frozen=True)
class AdaptiveClassificationResult:
    """Output contract of the Adaptive Motion Recognition Classifier."""
    prediction: str                     # 'J', 'Z', or 'UNKNOWN'
    confidence: float                   # 0.0 .. 1.0
    score_j: float                      # Joint score for J
    score_z: float                      # Joint score for Z
    margin: float                       # |score_j - score_z|
    decision_path: str                  # e.g., 'ADAPTIVE_J', 'ADAPTIVE_Z', 'STATIC_POSE_REJECTED'
    rejection_reason: str
    metrics: AdaptiveMotionMetrics | None
    latency_ms: float = 0.0


class AdaptiveMotionJZClassifier:
    """Adaptive Motion Recognition Classifier for Dynamic ASL J/Z gestures."""

    def __init__(
        self,
        min_motion_len: float = 0.15,
        min_adaptive_score: float = 0.38,
        min_margin: float = 0.06,
        weight_handshape: float = 0.45,
        weight_trajectory: float = 0.55,
        agreement_bonus_weight: float = 0.15,
        handshape_veto_threshold: float = 0.25,
        z_diagonal_threshold: float = 0.015,
    ) -> None:
        self.min_motion_len = float(min_motion_len)
        self.min_adaptive_score = float(min_adaptive_score)
        self.min_margin = float(min_margin)
        self.w_hs = float(weight_handshape)
        self.w_tr = float(weight_trajectory)
        self.w_bonus = float(agreement_bonus_weight)
        self.hs_veto = float(handshape_veto_threshold)
        self.z_diag_thresh = float(z_diagonal_threshold)

        # Reusable constituent branches
        self.handshape_branch = HandshapeBranch(
            min_confidence=0.35,
            min_margin=0.05,
            min_pinky_ext=0.90,
            min_index_ext=0.95,
            min_pinky_contrast=0.08,
            min_index_contrast=0.15,
            max_stability_std=0.38,
        )
        self.trajectory_branch = TrajectoryBranch(
            min_confidence=0.30,
            min_margin=0.02,
            min_motion_len=0.15,
            min_j_drop=0.10,
            min_j_hook=0.03,
            min_z_span=0.03,
            min_z_diag_drop=0.02,
        )

    def classify(self, sequence: Sequence[Sequence[float]] | np.ndarray) -> AdaptiveClassificationResult:
        """Classify a 16x69 dynamic sequence using adaptive joint motion recognition."""
        t_start = time.perf_counter()

        # 1. Defensive numerical and dimension validation
        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                return AdaptiveClassificationResult(
                    prediction="UNKNOWN",
                    confidence=0.0,
                    score_j=0.0,
                    score_z=0.0,
                    margin=0.0,
                    decision_path="INVALID_INPUT",
                    rejection_reason="INVALID_INPUT_SHAPE_OR_NON_FINITE",
                    metrics=None,
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="ERROR",
                rejection_reason=f"INPUT_CONVERSION_ERROR: {exc}",
                metrics=None,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 2. Extract landmark representation and palm scale
        landmarks_3d = arr[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))
        mcp_pts = landmarks_3d[:, PALM_SCALE_LANDMARKS, :]
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)
        palm_scales = np.median(mcp_dists, axis=1)

        # 3. Extract trajectories (cols 65:67 for index, 67:69 for pinky)
        path_index = arr[:, 65:67]
        path_pinky = arr[:, 67:69]

        p_len_pinky = float(np.sum(np.linalg.norm(np.diff(path_pinky, axis=0), axis=1)))
        p_len_index = float(np.sum(np.linalg.norm(np.diff(path_index, axis=0), axis=1)))
        max_motion = max(p_len_pinky, p_len_index)

        # 4. Anti-static pose safeguard
        # Reject if neither active fingertip moved with sufficient amplitude
        if max_motion < self.min_motion_len:
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="STATIC_POSE_REJECTED",
                rejection_reason=f"STATIC_POSE_INSUFFICIENT_MOTION ({max_motion:.3f} < {self.min_motion_len:.3f})",
                metrics=AdaptiveMotionMetrics(
                    handshape_score_j=0.0,
                    handshape_score_z=0.0,
                    trajectory_score_j=0.0,
                    trajectory_score_z=0.0,
                    joint_score_j=0.0,
                    joint_score_z=0.0,
                    motion_length_pinky=p_len_pinky,
                    motion_length_index=p_len_index,
                    max_motion_length=max_motion,
                    is_static_pose=True,
                    agreement_bonus_j=0.0,
                    agreement_bonus_z=0.0,
                ),
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 5. Handshape evidence evaluation
        hs_res = self.handshape_branch.evaluate(landmarks_3d, palm_scales)
        if hs_res.metrics is None:
            # Degenerate or uniform non-hand landmarks (e.g. OOD synthetic input)
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=0.0,
                score_z=0.0,
                margin=0.0,
                decision_path="OOD_REJECTED",
                rejection_reason=f"OOD_DEGENERATE_HANDSHAPE ({hs_res.rejection_reason})",
                metrics=None,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        s_hs_j = hs_res.score_j
        s_hs_z = hs_res.score_z

        # 6. Target-guided Trajectory evaluation
        # For J: evaluate pinky path
        s_tr_j, m_j = self.trajectory_branch.evaluate_j_trajectory(path_pinky)
        # For Z: evaluate index path
        s_tr_z, m_z = self.trajectory_branch.evaluate_z_trajectory(path_index)

        # 7. Adaptive Evidence Integration
        # A. Agreement bonuses: when both handshape and trajectory show positive evidence
        bonus_j = self.w_bonus * min(s_hs_j, s_tr_j) if (s_hs_j > 0.40 and s_tr_j > 0.30) else 0.0
        bonus_z = self.w_bonus * min(s_hs_z, s_tr_z) if (s_hs_z > 0.40 and s_tr_z > 0.30) else 0.0

        # B. Handshape veto: if handshape strongly opposes the class, clamp to zero
        mult_hs_j = 0.0 if s_hs_j < self.hs_veto else 1.0
        mult_hs_z = 0.0 if s_hs_z < self.hs_veto else 1.0

        # C. Trajectory essential morphology guards (preventing lines down or diagonal swipes from being accepted)
        has_j_morphology = bool(
            m_j.hook_lateral >= 0.03
            and not m_j.is_circular_loop
        )
        mult_tr_j = 1.0 if has_j_morphology else 0.0

        has_z_morphology = bool(
            m_z.reversals_count >= 1
            and (m_z.diagonal_downward >= self.z_diag_thresh or abs(m_z.dy_strokes[1]) >= self.z_diag_thresh)
        )
        mult_tr_z = 1.0 if has_z_morphology else 0.0

        # D. Joint scores
        joint_j = float((self.w_hs * s_hs_j + self.w_tr * s_tr_j + bonus_j) * mult_hs_j * mult_tr_j)
        joint_z = float((self.w_hs * s_hs_z + self.w_tr * s_tr_z + bonus_z) * mult_hs_z * mult_tr_z)

        margin = float(abs(joint_j - joint_z))
        best_score = max(joint_j, joint_z)

        metrics = AdaptiveMotionMetrics(
            handshape_score_j=s_hs_j,
            handshape_score_z=s_hs_z,
            trajectory_score_j=s_tr_j,
            trajectory_score_z=s_tr_z,
            joint_score_j=joint_j,
            joint_score_z=joint_z,
            motion_length_pinky=p_len_pinky,
            motion_length_index=p_len_index,
            max_motion_length=max_motion,
            is_static_pose=False,
            agreement_bonus_j=bonus_j,
            agreement_bonus_z=bonus_z,
        )

        # Cross-finger conflict detection:
        # e.g. Handshape says J but Trajectory clearly says Z, or Handshape says Z but Trajectory clearly says J
        hs_favors_j = s_hs_j > s_hs_z + 0.20
        hs_favors_z = s_hs_z > s_hs_j + 0.20
        tr_favors_j = s_tr_j > s_tr_z + 0.15 and has_j_morphology
        tr_favors_z = s_tr_z > s_tr_j + 0.15 and has_z_morphology

        if (hs_favors_j and tr_favors_z) or (hs_favors_z and tr_favors_j):
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=0.0,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="CONFLICT_REJECTED",
                rejection_reason=f"HANDSHAPE_TRAJECTORY_CONFLICT (HS_J={s_hs_j:.2f}, HS_Z={s_hs_z:.2f}, TR_J={s_tr_j:.2f}, TR_Z={s_tr_z:.2f})",
                metrics=metrics,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 8. Adaptive Decision Gating
        if best_score < self.min_adaptive_score:
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="INSUFFICIENT_ADAPTIVE_SCORE",
                rejection_reason=f"INSUFFICIENT_ADAPTIVE_SCORE ({best_score:.3f} < {self.min_adaptive_score:.3f})",
                metrics=metrics,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        if margin < self.min_margin:
            return AdaptiveClassificationResult(
                prediction="UNKNOWN",
                confidence=best_score,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="AMBIGUOUS_ADAPTIVE_MARGIN",
                rejection_reason=f"AMBIGUOUS_ADAPTIVE_MARGIN ({margin:.3f} < {self.min_margin:.3f})",
                metrics=metrics,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        if joint_j > joint_z:
            return AdaptiveClassificationResult(
                prediction="J",
                confidence=joint_j,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="ADAPTIVE_J",
                rejection_reason="",
                metrics=metrics,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )
        else:
            return AdaptiveClassificationResult(
                prediction="Z",
                confidence=joint_z,
                score_j=joint_j,
                score_z=joint_z,
                margin=margin,
                decision_path="ADAPTIVE_Z",
                rejection_reason="",
                metrics=metrics,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )
