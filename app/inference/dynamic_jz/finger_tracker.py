"""Finger Tracking Engine for Dynamic J and Z Gestures.

Provides standalone feature extraction for 16x69 dynamic sequence data:
- Global fingertip and wrist displacement trajectories (camera-plane motion).
- Wrist-relative fingertip trajectories (local articulation).
- Continuous per-frame and endpoint handshape extension ratios and inter-finger distances.
- Frame-level displacement deltas (clearly distinguished from real-time velocity/second).
- Comprehensive tracking quality metrics without synthetic timestamps or double normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


FEATURE_VERSION = "finger_tracking_v1"
SEQUENCE_LENGTH = 16
FEATURE_DIM = 69
LANDMARK_COUNT = 21

# Landmark indices defined by MediaPipe Hand landmark topology
WRIST_IDX = 0
THUMB_TIP_IDX = 4
INDEX_MCP_IDX = 5
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9
MIDDLE_TIP_IDX = 12
RING_MCP_IDX = 13
RING_TIP_IDX = 16
PINKY_MCP_IDX = 17
PINKY_TIP_IDX = 20

PALM_SCALE_LANDMARKS = (INDEX_MCP_IDX, MIDDLE_MCP_IDX, PINKY_MCP_IDX)  # (5, 9, 17)
FINGERTIP_INDICES = (THUMB_TIP_IDX, INDEX_TIP_IDX, MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX)

# Global trajectory column mappings in baseline-b 69-feature format:
# Columns 63:65 = Wrist (Landmark 0) displacement / S_palm
# Columns 65:67 = Index tip (Landmark 8) displacement / S_palm
# Columns 67:69 = Pinky tip (Landmark 20) displacement / S_palm
GLOBAL_COLUMNS = {
    "wrist": (63, 65),
    "index": (65, 67),
    "pinky": (67, 69),
}

# Letter-specific active fingertip mappings
TARGET_ACTIVE_FINGERTIP = {
    "J": {
        "landmark_id": PINKY_TIP_IDX,  # 20
        "global_cols": GLOBAL_COLUMNS["pinky"],  # (67, 69)
        "other_finger_landmarks": (INDEX_TIP_IDX, MIDDLE_TIP_IDX, RING_TIP_IDX),  # (8, 12, 16)
    },
    "Z": {
        "landmark_id": INDEX_TIP_IDX,  # 8
        "global_cols": GLOBAL_COLUMNS["index"],  # (65, 67)
        "other_finger_landmarks": (MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX),  # (12, 16, 20)
    },
}


class FingerTrackingError(ValueError):
    """Raised when sequence shape, structure, or target configuration is invalid."""


@dataclass(frozen=True)
class HandshapeFeatures:
    """Continuous handshape metrics computed per-frame without hard thresholds."""
    active_extension: np.ndarray          # (16,) Extension ratio of active finger relative to palm scale
    other_extensions: dict[str, np.ndarray]  # (16,) per other finger: 'index', 'middle', 'ring', 'pinky'
    curled_extension_mean: np.ndarray     # (16,) Mean extension ratio of non-active fingers
    thumb_extension: np.ndarray           # (16,) Extension ratio of thumb tip
    inter_fingertip_distances: dict[str, np.ndarray]  # e.g., 'thumb_to_index', 'index_to_pinky'
    endpoint_handshape: dict[str, Any]  # Frame 15 snapshot of key handshape metrics


@dataclass(frozen=True)
class VelocityProfile:
    """Frame-to-frame displacement deltas (clearly distinguished from velocity/second)."""
    delta_displacement: np.ndarray        # (16, 2) Delta between consecutive frames in S_palm units
    speed_per_frame: np.ndarray           # (16,) Euclidean norm of delta_displacement per frame
    cumulative_path_length: float         # Total sum of frame-to-frame step lengths


@dataclass(frozen=True)
class TrackingQuality:
    """Evaluates sample integrity, missing data, and tracking stability."""
    valid_frames_count: int
    missing_landmark_count: int
    duplicate_frame_count: int
    outlier_displacement_count: int
    has_nan_inf: bool
    frame_indices_monotonic: str          # "PASS", "FAIL", or "UNKNOWN"
    frame_indices_gaps: list[int] | str   # List of frame index gaps, or "UNKNOWN"
    handshape_consistency_std: float      # Standard deviation of palm scale across frames


@dataclass(frozen=True)
class FingerTrackingResult:
    """Complete output contract of the Finger Tracking Engine."""
    letter_target: str
    global_trajectory: np.ndarray         # (16, 2) Active fingertip displacement from frame 0 / S_palm
    wrist_motion: np.ndarray              # (16, 2) Wrist displacement from frame 0 / S_palm
    wrist_relative_trajectory: np.ndarray # (16, 3) Active fingertip relative to wrist (D_max normalized)
    handshape_features: HandshapeFeatures
    velocity_profile: VelocityProfile
    tracking_quality: TrackingQuality
    frame_indices: list[int] | None
    feature_version: str = FEATURE_VERSION
    validity_status: str = "VALID"        # "VALID" | "SUSPICIOUS" | "INVALID"
    invalidation_reason: str = ""


class FingerTracker:
    """Extracts separated finger tracking and handshape features for dynamic J/Z gestures."""

    def __init__(self) -> None:
        pass

    def extract(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        letter_target: str,
        selected_original_frame_indices: list[int] | None = None,
    ) -> FingerTrackingResult:
        """Extract separated tracking, handshape, and quality features from a (16, 69) sequence.

        Parameters
        ----------
        sequence : array_like of shape (16, 69)
            Chronological 16-frame sequence in project baseline-b format.
            Cols 0..62: 21 relative 3D landmarks normalized by frame D_max.
            Cols 63..68: 2D displacement from frame 0 normalized by sequence S_palm.
        letter_target : str
            Either 'J' (uses pinky tip landmark 20) or 'Z' (uses index tip landmark 8).
        selected_original_frame_indices : list[int], optional
            Original video frame numbers from which the 16 frames were sampled.
            Used solely for temporal spacing validation; no fake timestamps are invented.

        Returns
        -------
        FingerTrackingResult
            Dataclass containing isolated feature groups suitable for independent DTW weighting.
        """
        # 1. Target validation
        target = str(letter_target).strip().upper()
        if target not in TARGET_ACTIVE_FINGERTIP:
            raise FingerTrackingError(
                f"Invalid letter_target '{letter_target}'. Must be 'J' or 'Z'."
            )
        cfg = TARGET_ACTIVE_FINGERTIP[target]
        active_lm_id = cfg["landmark_id"]
        global_col_start, global_col_end = cfg["global_cols"]

        # 2. Defensive copy and shape verification
        try:
            seq = np.asarray(sequence, dtype=np.float64).copy()
        except Exception as exc:
            raise FingerTrackingError(f"Cannot convert sequence to float64 array: {exc}") from exc

        if seq.ndim != 2:
            raise FingerTrackingError(
                f"Expected 2D sequence array, got shape {seq.shape} (ndim={seq.ndim})"
            )
        if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise FingerTrackingError(
                f"Invalid sequence shape {seq.shape}. Expected ({SEQUENCE_LENGTH}, {FEATURE_DIM})."
            )

        # 3. Basic numerical checks
        has_nan_inf = bool(not np.isfinite(seq).all())
        validity_status = "VALID"
        invalidation_reasons: list[str] = []

        if has_nan_inf:
            validity_status = "INVALID"
            invalidation_reasons.append("NON_FINITE_VALUES")

        # 4. Extract Trajectories
        # A. Global Trajectories (from columns 63..68, normalized by sequence S_palm)
        # Wrist displacement: cols 63:65
        wrist_motion = seq[:, GLOBAL_COLUMNS["wrist"][0]:GLOBAL_COLUMNS["wrist"][1]].copy()
        # Active fingertip displacement: cols 65:67 (for Z) or 67:69 (for J)
        global_trajectory = seq[:, global_col_start:global_col_end].copy()

        # B. Wrist-relative Trajectory (from columns 0..62, normalized by frame D_max)
        # Landmark i has coordinates at slice (3*i, 3*i + 3)
        wrist_rel_active = seq[:, active_lm_id * 3: active_lm_id * 3 + 3].copy()

        # 5. Extract Handshape Features (per-frame from columns 0..62)
        # Reconstruct (16, 21, 3) relative coordinates
        landmarks_3d = seq[:, :63].reshape((SEQUENCE_LENGTH, LANDMARK_COUNT, 3))

        # Check for missing frames (where all coordinates 3..62 are zero)
        missing_frames_mask = np.all(np.abs(landmarks_3d[:, 1:, :]) < 1e-7, axis=(1, 2))
        missing_count = int(np.sum(missing_frames_mask))

        # Compute per-frame palm scale: median distance from wrist (0) to MCPs (5, 9, 17)
        mcp_pts = landmarks_3d[:, PALM_SCALE_LANDMARKS, :]  # (16, 3, 3)
        mcp_dists = np.linalg.norm(mcp_pts, axis=2)          # (16, 3)
        palm_scales = np.median(mcp_dists, axis=1)          # (16,)

        # Guard against zero or degenerate palm scale
        safe_palm_scales = np.where(palm_scales > 1e-6, palm_scales, 1.0)
        handshape_consistency = float(np.std(palm_scales))

        # Extension ratios: ||p_tip(t) - p_wrist(t)|| / S_palm(t)
        active_tip_pts = landmarks_3d[:, active_lm_id, :]
        active_extension = np.linalg.norm(active_tip_pts, axis=1) / safe_palm_scales

        # Other finger extensions
        finger_names = {
            INDEX_TIP_IDX: "index",
            MIDDLE_TIP_IDX: "middle",
            RING_TIP_IDX: "ring",
            PINKY_TIP_IDX: "pinky",
        }
        other_extensions: dict[str, np.ndarray] = {}
        curled_ext_list = []
        for lm_id in cfg["other_finger_landmarks"]:
            tip_pts = landmarks_3d[:, lm_id, :]
            ext = np.linalg.norm(tip_pts, axis=1) / safe_palm_scales
            name = finger_names.get(lm_id, f"finger_{lm_id}")
            other_extensions[name] = ext
            curled_ext_list.append(ext)

        curled_extension_mean = np.mean(np.stack(curled_ext_list, axis=0), axis=0)

        # Thumb extension
        thumb_tip_pts = landmarks_3d[:, THUMB_TIP_IDX, :]
        thumb_extension = np.linalg.norm(thumb_tip_pts, axis=1) / safe_palm_scales

        # Inter-fingertip distances normalized by palm scale
        thumb_to_index = np.linalg.norm(landmarks_3d[:, THUMB_TIP_IDX] - landmarks_3d[:, INDEX_TIP_IDX], axis=1) / safe_palm_scales
        thumb_to_middle = np.linalg.norm(landmarks_3d[:, THUMB_TIP_IDX] - landmarks_3d[:, MIDDLE_TIP_IDX], axis=1) / safe_palm_scales
        thumb_to_pinky = np.linalg.norm(landmarks_3d[:, THUMB_TIP_IDX] - landmarks_3d[:, PINKY_TIP_IDX], axis=1) / safe_palm_scales
        index_to_pinky = np.linalg.norm(landmarks_3d[:, INDEX_TIP_IDX] - landmarks_3d[:, PINKY_TIP_IDX], axis=1) / safe_palm_scales

        inter_distances = {
            "thumb_to_index": thumb_to_index,
            "thumb_to_middle": thumb_to_middle,
            "thumb_to_pinky": thumb_to_pinky,
            "index_to_pinky": index_to_pinky,
        }

        # Endpoint handshape (frame 15)
        endpoint_handshape: dict[str, float | str] = {
            "active_finger": "pinky" if target == "J" else "index",
            "active_extension": float(active_extension[-1]),
            "curled_extension_mean": float(curled_extension_mean[-1]),
            "thumb_extension": float(thumb_extension[-1]),
            "index_to_pinky_distance": float(index_to_pinky[-1]),
            "palm_scale": float(palm_scales[-1]),
        }

        handshape = HandshapeFeatures(
            active_extension=active_extension,
            other_extensions=other_extensions,
            curled_extension_mean=curled_extension_mean,
            thumb_extension=thumb_extension,
            inter_fingertip_distances=inter_distances,
            endpoint_handshape=endpoint_handshape,
        )

        # 6. Displacement deltas per frame (NOT velocity/second)
        # delta(0) is [0.0, 0.0] by convention
        delta_disp = np.zeros_like(global_trajectory)
        delta_disp[1:] = global_trajectory[1:] - global_trajectory[:-1]
        speeds = np.linalg.norm(delta_disp, axis=1)
        path_len = float(np.sum(speeds))

        velocity_prof = VelocityProfile(
            delta_displacement=delta_disp,
            speed_per_frame=speeds,
            cumulative_path_length=path_len,
        )

        # 7. Quality metrics and anomaly detection
        # Consecutive duplicate frame check
        diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=1)
        duplicate_count = int(np.sum(diffs < 1e-7))

        # Outlier displacement check: single frame jump > 1.5 palm scales
        outlier_count = int(np.sum(speeds > 1.5))

        # Frame index validation
        monotonic_status = "UNKNOWN"
        gaps_result: list[int] | str = "UNKNOWN"
        if selected_original_frame_indices is not None:
            indices = list(selected_original_frame_indices)
            if len(indices) != SEQUENCE_LENGTH:
                monotonic_status = "FAIL"
                invalidation_reasons.append("INVALID_FRAME_INDICES_LENGTH")
            elif any(not isinstance(idx, int) or idx < 0 for idx in indices):
                monotonic_status = "FAIL"
                invalidation_reasons.append("NON_INTEGER_OR_NEGATIVE_FRAME_INDICES")
            elif any(b <= a for a, b in zip(indices, indices[1:])):
                monotonic_status = "FAIL"
                invalidation_reasons.append("NON_MONOTONIC_FRAME_INDICES")
            else:
                monotonic_status = "PASS"
                gaps_result = [b - a for a, b in zip(indices, indices[1:])]

        # Overall validity assessment
        if duplicate_count > 4:
            validity_status = "SUSPICIOUS"
            invalidation_reasons.append(f"EXCESSIVE_DUPLICATES_{duplicate_count}")
        if outlier_count > 0:
            validity_status = "SUSPICIOUS"
            invalidation_reasons.append(f"TRAJECTORY_SPIKES_{outlier_count}")
        if missing_count > 0:
            validity_status = "INVALID"
            invalidation_reasons.append(f"MISSING_LANDMARKS_{missing_count}")
        if monotonic_status == "FAIL":
            validity_status = "INVALID"

        quality = TrackingQuality(
            valid_frames_count=SEQUENCE_LENGTH - missing_count,
            missing_landmark_count=missing_count,
            duplicate_frame_count=duplicate_count,
            outlier_displacement_count=outlier_count,
            has_nan_inf=has_nan_inf,
            frame_indices_monotonic=monotonic_status,
            frame_indices_gaps=gaps_result,
            handshape_consistency_std=handshape_consistency,
        )

        return FingerTrackingResult(
            letter_target=target,
            global_trajectory=global_trajectory,
            wrist_motion=wrist_motion,
            wrist_relative_trajectory=wrist_rel_active,
            handshape_features=handshape,
            velocity_profile=velocity_prof,
            tracking_quality=quality,
            frame_indices=selected_original_frame_indices,
            feature_version=FEATURE_VERSION,
            validity_status=validity_status,
            invalidation_reason="; ".join(invalidation_reasons),
        )


def extract_finger_tracking(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    letter_target: str,
    selected_original_frame_indices: list[int] | None = None,
) -> FingerTrackingResult:
    """Functional convenience wrapper for FingerTracker().extract()."""
    tracker = FingerTracker()
    return tracker.extract(
        sequence=sequence,
        letter_target=letter_target,
        selected_original_frame_indices=selected_original_frame_indices,
    )
