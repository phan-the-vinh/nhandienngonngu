"""Feature Sequence Builder for Dynamic Time Warping (DTW).

Constructs structured, dimensionless multivariate feature sequences from FingerTrackingResult
under four distinct configurations:
- CONFIG_A: Global trajectory + handshape (6D)
- CONFIG_B: Wrist-relative trajectory + handshape (5D)
- CONFIG_C: Global trajectory + wrist-relative trajectory + handshape (9D)
- CONFIG_D: Global trajectory + wrist motion + handshape (10D)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from app.inference.dynamic_jz.finger_tracker import FingerTrackingResult, FingerTrackingError


FEATURE_CONFIGURATIONS = {
    "CONFIG_A": {
        "description": "Global active fingertip displacement + frame delta + continuous handshape",
        "dimension": 6,
        "feature_names": ["x_global", "y_global", "dx_global", "dy_global", "active_ext", "curled_ext"],
        "default_weights": [1.0, 1.0, 0.5, 0.5, 0.8, 0.8],
    },
    "CONFIG_B": {
        "description": "Wrist-relative active fingertip trajectory + continuous handshape",
        "dimension": 5,
        "feature_names": ["x_wrist_rel", "y_wrist_rel", "z_wrist_rel", "active_ext", "curled_ext"],
        "default_weights": [1.0, 1.0, 0.5, 0.8, 0.8],
    },
    "CONFIG_C": {
        "description": "Global trajectory + wrist-relative trajectory + continuous handshape",
        "dimension": 9,
        "feature_names": [
            "x_global", "y_global", "dx_global", "dy_global",
            "x_wrist_rel", "y_wrist_rel", "z_wrist_rel",
            "active_ext", "curled_ext",
        ],
        "default_weights": [1.0, 1.0, 0.5, 0.5, 0.8, 0.8, 0.4, 0.8, 0.8],
    },
    "CONFIG_D": {
        "description": "Global active fingertip + global wrist motion + continuous handshape",
        "dimension": 10,
        "feature_names": [
            "x_global", "y_global", "dx_global", "dy_global",
            "x_wrist", "y_wrist", "dx_wrist", "dy_wrist",
            "active_ext", "curled_ext",
        ],
        "default_weights": [1.0, 1.0, 0.5, 0.5, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8],
    },
    "CONFIG_E": {
        "description": "Enhanced: Trajectory + Delta + Speed + Unit Direction + Turn Curvature + Wrist-rel + Handshape + Dominance",
        "dimension": 14,
        "feature_names": [
            "x_global", "y_global",
            "dx_global", "dy_global",
            "speed",
            "dir_cos", "dir_sin",
            "turn_angle",
            "x_wrist_rel", "y_wrist_rel", "z_wrist_rel",
            "active_ext", "curled_ext", "dominance",
        ],
        "default_weights": [1.0, 1.0, 0.6, 0.6, 0.5, 1.2, 1.2, 1.5, 0.5, 0.5, 0.3, 0.8, 0.8, 1.2],
    },
}


def build_dtw_features(
    result: FingerTrackingResult,
    config_name: str = "CONFIG_A",
    custom_weights: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Build a (T, D) feature sequence and associated weights from a FingerTrackingResult.

    Parameters
    ----------
    result : FingerTrackingResult
        Extracted finger tracking result from Stage 1.
    config_name : str
        One of 'CONFIG_A', 'CONFIG_B', 'CONFIG_C', 'CONFIG_D'.
    custom_weights : list[float], optional
        User-supplied custom weights for feature dimensions.

    Returns
    -------
    features : np.ndarray of shape (T, D)
        The multivariate feature sequence.
    weights : list[float]
        Dimension weights matching feature order.
    """
    config_key = str(config_name).strip().upper()
    if config_key not in FEATURE_CONFIGURATIONS:
        raise FingerTrackingError(
            f"Unknown feature config '{config_name}'. Must be one of {list(FEATURE_CONFIGURATIONS.keys())}."
        )
    cfg = FEATURE_CONFIGURATIONS[config_key]
    T = len(result.global_trajectory)

    # Base components
    # 1. Global active trajectory: (T, 2)
    glob = result.global_trajectory
    # 2. Frame-level delta displacement: (T, 2)
    delta_glob = result.velocity_profile.delta_displacement
    # 3. Handshape features: (T, 1) each
    act_ext = result.handshape_features.active_extension[:, None]
    curl_ext = result.handshape_features.curled_extension_mean[:, None]

    if config_key == "CONFIG_A":
        # Shape: (T, 6) = [glob(2), delta_glob(2), act_ext(1), curl_ext(1)]
        feat = np.concatenate([glob, delta_glob, act_ext, curl_ext], axis=1)

    elif config_key == "CONFIG_B":
        # Wrist-relative trajectory: (T, 3)
        wrist_rel = result.wrist_relative_trajectory
        # Shape: (T, 5) = [wrist_rel(3), act_ext(1), curl_ext(1)]
        feat = np.concatenate([wrist_rel, act_ext, curl_ext], axis=1)

    elif config_key == "CONFIG_C":
        wrist_rel = result.wrist_relative_trajectory
        # Shape: (T, 9) = [glob(2), delta_glob(2), wrist_rel(3), act_ext(1), curl_ext(1)]
        feat = np.concatenate([glob, delta_glob, wrist_rel, act_ext, curl_ext], axis=1)

    elif config_key == "CONFIG_D":
        wrist_glob = result.wrist_motion
        delta_wrist = np.zeros_like(wrist_glob)
        delta_wrist[1:] = wrist_glob[1:] - wrist_glob[:-1]
        # Shape: (T, 10) = [glob(2), delta_glob(2), wrist_glob(2), delta_wrist(2), act_ext(1), curl_ext(1)]
        feat = np.concatenate([glob, delta_glob, wrist_glob, delta_wrist, act_ext, curl_ext], axis=1)

    elif config_key == "CONFIG_E":
        # Kinematic and directional features:
        speed = result.velocity_profile.speed_per_frame[:, None]
        safe_speed = np.where(speed > 1e-6, speed, 1.0)
        dir_cos = np.where(speed > 1e-6, delta_glob[:, :1] / safe_speed, 0.0)
        dir_sin = np.where(speed > 1e-6, delta_glob[:, 1:2] / safe_speed, 0.0)

        headings = np.arctan2(delta_glob[:, 1], delta_glob[:, 0])
        turn_angle = np.zeros((T, 1), dtype=np.float64)
        if T >= 3:
            d_theta = headings[2:] - headings[1:-1]
            d_theta_wrapped = np.arctan2(np.sin(d_theta), np.cos(d_theta)) / np.pi
            valid_turns = (speed[2:, 0] > 0.02) & (speed[1:-1, 0] > 0.02)
            turn_angle[2:, 0] = np.where(valid_turns, d_theta_wrapped, 0.0)

        wrist_rel = result.wrist_relative_trajectory
        dominance = act_ext - curl_ext

        # Shape: (T, 14) = [glob(2), delta(2), speed(1), dir(2), turn(1), wrist_rel(3), act_ext(1), curl_ext(1), dom(1)]
        feat = np.concatenate([
            glob, delta_glob, speed, dir_cos, dir_sin, turn_angle,
            wrist_rel, act_ext, curl_ext, dominance
        ], axis=1)

    else:
        raise FingerTrackingError(f"Unhandled configuration {config_key}")

    weights = custom_weights if custom_weights is not None else list(cfg["default_weights"])
    if len(weights) != cfg["dimension"]:
        raise FingerTrackingError(
            f"Weights length {len(weights)} does not match {config_key} dimension {cfg['dimension']}."
        )

    return feat, weights
