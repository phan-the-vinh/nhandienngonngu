"""Baseline-B features for one 16-frame dynamic hand-gesture sequence.

The first 63 values at each time step come from the project's canonical
normalize_landmarks() function. The final six preserve 2D motion that its
per-frame wrist subtraction would otherwise discard.
"""

from __future__ import annotations

import numpy as np

from ai.preprocessing.landmark_utils import normalize_landmarks


SEQUENCE_LENGTH = 16
FEATURE_DIM = 69
TRAJECTORY_LANDMARKS = (0, 8, 20)  # wrist, index tip, little-finger tip
SCALE_LANDMARKS = (5, 9, 17)  # palm MCPs
SCALE_EPSILON_PX = 1e-6


class DynamicSequenceError(ValueError):
    """A sequence cannot be represented without inventing landmark data."""


def sample_frame_indices(decoded_frame_count: int, length: int = SEQUENCE_LENGTH) -> list[int]:
    """Choose unique, ordered indices spanning the actual decoded video."""
    if not isinstance(decoded_frame_count, int) or decoded_frame_count < length or length < 2:
        raise DynamicSequenceError("TOO_FEW_DECODED_FRAMES")
    indices = np.rint(np.linspace(0, decoded_frame_count - 1, length)).astype(int).tolist()
    if len(set(indices)) != length or indices[0] != 0 or indices[-1] != decoded_frame_count - 1:
        raise DynamicSequenceError("INVALID_TEMPORAL_SAMPLE_INDICES")
    return indices


def sample_valid_frame_indices(valid_original_indices: list[int], length: int = SEQUENCE_LENGTH) -> list[int]:
    """Evenly sample chronological original indices of frames with valid landmarks.

    This samples positions in the *valid-frame list*, not the video timeline.
    Missing frames are never filled, interpolated, or selected.
    """
    if (not isinstance(valid_original_indices, list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in valid_original_indices)
            or any(right <= left for left, right in zip(valid_original_indices, valid_original_indices[1:]))):
        raise DynamicSequenceError("INVALID_VALID_FRAME_INDICES")
    if len(valid_original_indices) < length:
        raise DynamicSequenceError("INSUFFICIENT_VALID_LANDMARK_FRAMES")
    return [valid_original_indices[position]
            for position in sample_frame_indices(len(valid_original_indices), length)]


def sample_interval_indices(start: int, end: int, length: int = SEQUENCE_LENGTH) -> list[int]:
    """Use the exact whole-clip sampler, restricted to an inclusive interval."""
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise DynamicSequenceError("INVALID_ACTIVE_INTERVAL")
    return [start + index for index in sample_frame_indices(end - start + 1, length)]


def _smooth_finite(values: np.ndarray, width: int, reducer) -> np.ndarray:
    """Centered smoothing; missing measurements remain missing at their index."""
    radius = width // 2
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        window = values[max(0, index - radius):min(len(values), index + radius + 1)]
        finite = window[np.isfinite(window)]
        if finite.size >= min(2, len(window)):
            result[index] = reducer(finite)
    return result


def detect_active_interval(pixel_landmarks: list[np.ndarray | None], params: dict) -> dict:
    """Conservative, label-agnostic motion interval proposal for *preview only*.

    Valid entries are (21, 2) pixel-XY arrays; None means MediaPipe did not
    yield 21 finite landmarks. No missing coordinates are interpolated. Scores
    at t>=1 are max_k ||p[t,k]-p[t-1,k]|| / S for k=(0,8,20), where S is one
    median wrist-to-MCP pixel distance across valid frames and MCPs 5,9,17.
    """
    count = len(pixel_landmarks)
    result = {"status": "REVIEW_REQUIRED", "failure_reason": "", "frame_count": count,
              "active_start": None, "active_end": None,
              "margin_start": None, "margin_end": None,
              "sampled_frame_indices": [], "hand_scale_px": None,
              "noise_floor": None, "threshold": None, "peak_score": None,
              "raw_motion_scores": [], "smoothed_motion_scores": []}

    def review(reason: str) -> dict:
        result["failure_reason"] = reason
        return result

    if count < SEQUENCE_LENGTH:
        return review("TOO_FEW_DECODED_FRAMES")
    valid = np.zeros(count, dtype=bool)
    points = np.full((count, 21, 2), np.nan, dtype=np.float64)
    for index, frame in enumerate(pixel_landmarks):
        if frame is None:
            continue
        array = np.asarray(frame, dtype=np.float64)
        if array.shape != (21, 2) or not np.isfinite(array).all():
            continue
        valid[index] = True
        points[index] = array
    result["valid_landmark_frames"] = int(valid.sum())
    if valid.mean() < params["min_valid_fraction"]:
        return review("INSUFFICIENT_LANDMARK_COVERAGE")

    palm_distances = np.linalg.norm(
        points[valid][:, SCALE_LANDMARKS, :] - points[valid][:, 0:1, :], axis=2
    )
    scale = float(np.median(palm_distances))
    if not np.isfinite(scale) or scale <= SCALE_EPSILON_PX:
        return review("DEGENERATE_HAND_SCALE")
    result["hand_scale_px"] = scale

    scores = np.full(count, np.nan, dtype=np.float64)
    for index in range(1, count):
        if valid[index - 1] and valid[index]:
            displacement = points[index, TRAJECTORY_LANDMARKS, :] - points[index - 1, TRAJECTORY_LANDMARKS, :]
            scores[index] = float(np.max(np.linalg.norm(displacement, axis=1)) / scale)
    median_scores = _smooth_finite(scores, params["median_window"], np.median)
    smoothed = _smooth_finite(median_scores, params["mean_window"], np.mean)
    result["raw_motion_scores"] = [float(x) if np.isfinite(x) else None for x in scores]
    result["smoothed_motion_scores"] = [float(x) if np.isfinite(x) else None for x in smoothed]
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size < SEQUENCE_LENGTH - 1:
        return review("INSUFFICIENT_VALID_MOTION_EDGES")
    peak = float(np.max(finite))
    result["peak_score"] = peak
    lower_half = np.sort(finite)[:max(1, finite.size // 2)]
    noise_floor = float(np.median(lower_half))
    robust_sigma = float(1.4826 * np.median(np.abs(lower_half - noise_floor)))
    result["noise_floor"] = noise_floor
    if peak < params["min_peak_score"] or peak < params["min_peak_noise_ratio"] * max(noise_floor, 1e-12):
        return review("NO_CONFIDENT_MOTION_PEAK")
    threshold = max(noise_floor + params["noise_mad_multiplier"] * robust_sigma,
                    params["relative_peak_fraction"] * peak)
    result["threshold"] = float(threshold)
    active = np.flatnonzero(np.isfinite(smoothed) & (smoothed >= threshold))
    if active.size < params["min_active_edges"]:
        return review("TOO_FEW_ACTIVE_EDGES")

    clusters: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        if int(index) - clusters[-1][-1] - 1 <= params["max_gap_frames"]:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])
    clusters = [cluster for cluster in clusters if len(cluster) >= params["min_active_edges"]]
    if not clusters:
        return review("NO_SUSTAINED_MOTION_CLUSTER")
    masses = [float(np.sum(smoothed[cluster])) for cluster in clusters]
    main_index = int(np.argmax(masses))
    if any(mass >= params["secondary_mass_ratio"] * masses[main_index]
           for index, mass in enumerate(masses) if index != main_index):
        return review("MULTIPLE_SEPARATED_MOTION_CLUSTERS")

    main = clusters[main_index]
    active_start = max(0, main[0] - 1)
    active_end = main[-1]
    margin_start = max(0, active_start - params["margin_frames"])
    margin_end = min(count - 1, active_end + params["margin_frames"])
    result.update(active_start=active_start, active_end=active_end,
                  margin_start=margin_start, margin_end=margin_end)
    if not valid[margin_start:margin_end + 1].all():
        return review("MISSING_LANDMARK_IN_PROPOSED_INTERVAL")
    if margin_end - margin_start + 1 < SEQUENCE_LENGTH:
        return review("ACTIVE_INTERVAL_SHORTER_THAN_16")
    result["sampled_frame_indices"] = sample_interval_indices(margin_start, margin_end)
    result["status"] = "OK"
    return result


def build_sequence_features(
    landmark_frames: list[list[object]],
    frame_sizes: list[tuple[int, int]],
    length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, float]:
    """Return (length, 69) features and one fixed hand scale in pixels.

    Each trajectory XY uses MediaPipe's image-normalized XY multiplied by the
    actual width/height. For k in (0, 8, 20), displacement is
    (pixel_xy[t, k] - pixel_xy[0, k]) / hand_scale_px. The *single* scale is
    the median of wrist-to-MCP pixel distances for MCPs 5, 9, 17 across all
    selected frames. No frame-specific trajectory normalization is applied.
    """
    if len(landmark_frames) != length or len(frame_sizes) != length:
        raise DynamicSequenceError("INVALID_SEQUENCE_LENGTH")
    if len(set(frame_sizes)) != 1:
        raise DynamicSequenceError("INCONSISTENT_FRAME_SIZE")

    shapes = []
    pixel_points = []
    for hand, (width, height) in zip(landmark_frames, frame_sizes):
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise DynamicSequenceError("INVALID_FRAME_SIZE")
        if len(hand) != 21:
            raise DynamicSequenceError("INVALID_LANDMARK_COUNT")
        try:
            raw = np.asarray([(float(p.x), float(p.y), float(p.z)) for p in hand], dtype=np.float64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicSequenceError("INVALID_LANDMARK_VALUES") from exc
        if raw.shape != (21, 3) or not np.isfinite(raw).all():
            raise DynamicSequenceError("NONFINITE_LANDMARK")
        shape = np.asarray(normalize_landmarks(hand), dtype=np.float64)
        if shape.shape != (63,) or not np.isfinite(shape).all():
            raise DynamicSequenceError("INVALID_SHAPE_FEATURES")
        shapes.append(shape)
        pixel_points.append(raw[:, :2] * np.array([width, height], dtype=np.float64))

    points = np.stack(pixel_points)
    palm_distances = np.linalg.norm(
        points[:, SCALE_LANDMARKS, :] - points[:, 0:1, :], axis=2
    )
    hand_scale_px = float(np.median(palm_distances))
    if not np.isfinite(hand_scale_px) or hand_scale_px <= SCALE_EPSILON_PX:
        raise DynamicSequenceError("DEGENERATE_HAND_SCALE")

    selected = points[:, TRAJECTORY_LANDMARKS, :]
    trajectory = (selected - selected[0:1, :, :]) / hand_scale_px
    features = np.concatenate((np.stack(shapes), trajectory.reshape(length, 6)), axis=1)
    if features.shape != (length, FEATURE_DIM) or not np.isfinite(features).all():
        raise DynamicSequenceError("INVALID_SEQUENCE_FEATURES")
    return features, hand_scale_px
