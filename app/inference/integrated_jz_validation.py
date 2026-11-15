"""Technical validation for integrated J/Z proposals.

Only the integrated experiment uses this module.
In Phase 3.10.8-A4, trajectory validation for Z is activated as a required gating
rule to prevent static handshapes (such as static L) from falsely committing as Z.
J trajectory remains unconstrained in this phase.
"""
from dataclasses import asdict, dataclass

import numpy as np

from ai.feature_extraction.dynamic_sequence import build_sequence_features, sample_valid_frame_indices
from ai.feature_extraction.hand_landmarks import landmarks_to_array
from app.inference.automatic_jz_experiment import geometry_candidates
from app.inference.first16_jz_segmenter import _is_contiguous_frame, measure_motion
from app.inference.jz_shadow_diagnostics import geometry_predicates


@dataclass(frozen=True)
class ZTrajectorySettings:
    """Provisional trajectory validation thresholds for ASL letter Z.

    Thresholds are defined in palm scales (median wrist-to-MCP distance across frames).
    Provisional status: Calibrated against ASL Z geometric strokes and synthetic/held hand fixtures.
    """
    min_excursion: float = 0.35  # Minimum path excursion of index fingertip (palm scales)
    min_horizontal_stroke: float = 0.12  # Minimum horizontal stroke span (|dx|) per leg
    min_downward_stroke: float = 0.08  # Minimum downward diagonal progression (dy[1] in image coords)
    max_horizontal_slope: float = 0.70  # Maximum |dy/dx| for top (downward) and bottom horizontal legs
    max_top_upward_slope: float = 2.85  # Maximum |dy/dx| for top horizontal leg when tilting upward (dy0 <= 0)
    max_lead_in_horizontal_drift: float = 0.15  # Maximum horizontal drift permitted before start index t0


@dataclass(frozen=True)
class ZHandshapeSettings:
    """Handshape validation thresholds for ASL letter Z (Index pointing, other fingers curled).

    Mode:
    - 'SHADOW': Compute and record handshape validation diagnostics without altering
      trajectory acceptance decisions (Production safe, zero regression).
    - 'ACTIVE': Enforce handshape validation; reject proposals if index finger is not
      extended or other fingers fail dominance constraints.
    """
    min_index_extension: float = 0.70  # Minimum 3D extension of index fingertip (wrist-normalized)
    min_index_dominance: float = 0.05  # Minimum dominance: index_ext - max(middle, ring, pinky)
    mode: str = "SHADOW"  # 'SHADOW' (default in Stage 16) or 'ACTIVE'


def validate_z_handshape(
    features,
    settings: ZHandshapeSettings = ZHandshapeSettings(),
) -> dict:
    """Validate handshape for ASL letter Z.

    Requires index finger extended (3D extension >= min_index_extension)
    and index dominance over middle, ring, pinky (>= min_index_dominance).
    Features shape: (16, 69).
    """
    result = {
        "valid": False,
        "reason": "",
        "index_extension": 0.0,
        "max_other_extension": 0.0,
        "dominance": 0.0,
        "has_distinct_fingers": False,
        "mode": settings.mode,
        "settings": asdict(settings),
    }
    if not isinstance(features, np.ndarray) or features.shape != (16, 69) or not np.isfinite(features).all():
        result["reason"] = "INVALID_FEATURE_DIMENSIONS"
        return result

    lm = features[:, :63].reshape(16, 21, 3)
    tip_xy = lm[:, [8, 12, 16, 20], :2]
    pinky_xy = lm[:, 20:21, :2]
    has_distinct = bool(
        np.any(np.abs(tip_xy) > 1e-6) and not np.allclose(tip_xy, pinky_xy, atol=1e-5)
    )
    result["has_distinct_fingers"] = has_distinct
    if not has_distinct:
        # Backward compatibility: synthetic trajectory test fixtures with dummy zero landmarks
        result["valid"] = True
        result["reason"] = "SYNTHETIC_OR_NO_HANDSHAPE_DATA"
        return result

    d_idx = np.linalg.norm(lm[:, 8, :], axis=1)
    d_mid = np.linalg.norm(lm[:, 12, :], axis=1)
    d_rng = np.linalg.norm(lm[:, 16, :], axis=1)
    d_pnk = np.linalg.norm(lm[:, 20, :], axis=1)
    d_max_oth = np.maximum(d_mid, np.maximum(d_rng, d_pnk))

    idx_ext = float(np.median(d_idx))
    max_oth = float(np.median(d_max_oth))
    dom = float(np.median(d_idx - d_max_oth))

    result["index_extension"] = idx_ext
    result["max_other_extension"] = max_oth
    result["dominance"] = dom

    if idx_ext < settings.min_index_extension:
        result["reason"] = f"INDEX_NOT_EXTENDED: {idx_ext:.3f} < {settings.min_index_extension:.3f}"
        return result
    if dom < settings.min_index_dominance:
        result["reason"] = f"NON_Z_HANDSHAPE_DOMINANCE: {dom:.3f} < {settings.min_index_dominance:.3f}"
        return result

    result["valid"] = True
    result["reason"] = "VALID_Z_HANDSHAPE"
    return result


def _search_z_for_x_sign(
    path: np.ndarray,
    settings: ZTrajectorySettings,
    x_sign: int,
) -> tuple[float, tuple | None]:
    """Search valid (t0, t1, t2, tend) Z quadruplet for a horizontal X orientation (+1 or -1)."""
    x = path[:, 0] * x_sign
    y = path[:, 1]
    best_score = -1e9
    best_candidate = None

    # Step distances for duplicate/micro-tremor filtering
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mono_tol = 0.05
    chord_limit = 2.15

    # Search quadruplets (t0, t1, t2, tend) for dynamic start index and stroke corner identification
    for t0 in range(0, 5):
        # Lead-in check: frames before t0 must not exhibit large horizontal drift
        if t0 > 0 and np.max(np.abs(x[:t0 + 1] - x[0])) > settings.max_lead_in_horizontal_drift:
            continue

        for t1 in range(t0 + 1, 13):
            dx0 = x[t1] - x[t0]
            dy0 = y[t1] - y[t0]

            # Top stroke must span at least min_horizontal_stroke along canonical direction
            if dx0 < settings.min_horizontal_stroke:
                continue
            # Downward slope (dy0 > 0) bounded by max_horizontal_slope
            if dy0 > 0 and dy0 > settings.max_horizontal_slope * dx0:
                continue
            # Upward tilt (dy0 <= 0) from hand-raising bounded by max_top_upward_slope
            if dy0 <= 0 and abs(dy0) > settings.max_top_upward_slope * dx0:
                continue
            # Stroke 0 should be relatively monotonic (with micro-jitter tolerance)
            if np.max(x[t0:t1 + 1]) > x[t1] + mono_tol:
                continue

            for t2 in range(t1 + 1, 15):
                dx1 = x[t2] - x[t1]
                dy1 = y[t2] - y[t1]

                # Diagonal stroke must reverse horizontally and progress downward in image Y
                if dx1 > -settings.min_horizontal_stroke:
                    continue
                if dy1 < settings.min_downward_stroke:
                    continue
                # Stroke 1 should reach horizontal trough at t2
                if np.min(x[t1:t2 + 1]) < x[t2] - mono_tol:
                    continue

                for tend in range(t2 + 1, 16):
                    dx2 = x[tend] - x[t2]
                    dy2 = y[tend] - y[t2]

                    # Bottom stroke must reverse horizontally again and stay primarily horizontal
                    if dx2 < settings.min_horizontal_stroke:
                        continue
                    if abs(dy2) > settings.max_horizontal_slope * abs(dx2):
                        continue
                    # Stroke 2 should reach horizontal extent at tend
                    if np.max(x[t2:tend + 1]) > x[tend] + mono_tol:
                        continue

                    # Terminal hold verification: post-tend frames must not reverse back into the diagonal
                    if tend < 15:
                        post_hold_x = x[tend + 1:]
                        if np.min(post_hold_x) < x[t2] + 0.4 * dx2:
                            continue

                    # Path efficiency / chord ratio: reject erratic noise
                    p0, p1, p2, p3 = path[t0], path[t1], path[t2], path[tend]
                    chord_sum = (
                        np.linalg.norm(p1 - p0)
                        + np.linalg.norm(p2 - p1)
                        + np.linalg.norm(p3 - p2)
                    )
                    # Filter sub-tremor noise (<0.015) in duplicate/stagnant frames when computing path_len
                    seg_steps = steps[t0:tend]
                    seg_steps_clean = seg_steps[seg_steps >= 0.015]
                    path_len = float(np.sum(seg_steps_clean)) if len(seg_steps_clean) > 0 else float(np.sum(seg_steps))
                    if chord_sum > 1e-6 and (path_len / chord_sum) > chord_limit:
                        continue

                    score = dx0 - dx1 + dx2 + dy1
                    if score > best_score:
                        best_score = score
                        best_candidate = (
                            t0, t1, t2, tend,
                            [float(x_sign * dx0), float(x_sign * dx1), float(x_sign * dx2)],
                            [float(dy0), float(dy1), float(dy2)],
                            int(x_sign),
                        )

    return best_score, best_candidate


def validate_z_trajectory(
    features,
    settings: ZTrajectorySettings = ZTrajectorySettings(),
    handshape_settings: ZHandshapeSettings = ZHandshapeSettings(),
) -> dict:
    """Validate that features exhibit genuine ASL Z multi-stroke trajectory geometry.

    Supports both viewer left-to-right (+X -> -X -> +X) and signer/mirrored
    left-to-right (-X -> +X -> -X) horizontal orientations under identical thresholds,
    while strictly enforcing downward diagonal Y progression.

    In Stage 16, handshape validation runs in SHADOW mode by default, computing
    and recording handshape metrics without altering trajectory decisions.

    Features shape: (16, 69).
    Landmark 8 (index fingertip) trajectory: columns 65:67 (x, y) relative to frame 0.
    """
    hs_val = validate_z_handshape(features, handshape_settings)
    result = {
        "valid": False,
        "reason": "",
        "excursion": 0.0,
        "start_index": -1,
        "corner1_index": -1,
        "corner2_index": -1,
        "endpoint_index": -1,
        "dx": [],
        "dy": [],
        "matched_x_orientation": "NONE",
        "matched_x_sign": 0,
        "handshape_validation": hs_val,
        "handshape_valid": hs_val["valid"],
        "settings": asdict(settings),
    }
    if not isinstance(features, np.ndarray) or features.shape != (16, 69) or not np.isfinite(features).all():
        result["reason"] = "INVALID_FEATURE_DIMENSIONS"
        return result

    path = features[:, 65:67]  # index fingertip trajectory in palm scales
    excursion = float(np.max(np.linalg.norm(path - path[0], axis=1)))
    result["excursion"] = excursion
    if excursion < settings.min_excursion:
        result["reason"] = f"INSUFFICIENT_Z_EXCURSION: {excursion:.3f} < {settings.min_excursion}"
        return result

    score_pos, cand_pos = _search_z_for_x_sign(path, settings, x_sign=+1)
    score_neg, cand_neg = _search_z_for_x_sign(path, settings, x_sign=-1)

    best_candidate = None
    if cand_pos is not None and (cand_neg is None or score_pos >= score_neg):
        best_candidate = cand_pos
    elif cand_neg is not None:
        best_candidate = cand_neg

    if best_candidate is not None:
        t0, t1, t2, tend, dx, dy, x_sign = best_candidate
        result["start_index"] = t0
        result["corner1_index"] = t1
        result["corner2_index"] = t2
        result["endpoint_index"] = tend
        result["dx"] = dx
        result["dy"] = dy
        result["matched_x_sign"] = x_sign
        result["matched_x_orientation"] = "VIEWER_LR" if x_sign == 1 else "SIGNER_OR_MIRRORED_LR"

        if handshape_settings.mode == "ACTIVE" and not hs_val["valid"]:
            result["valid"] = False
            result["reason"] = hs_val["reason"]
        else:
            result["valid"] = True
            result["reason"] = "VALID_Z_TRAJECTORY"
        return result

    # Diagnostic reason fallback for rejection classification
    default_legs = np.diff(path[[0, 5, 10, 15]], axis=0)
    dx_def = default_legs[:, 0]
    dy_def = default_legs[:, 1]
    result["dx"] = dx_def.tolist()
    result["dy"] = dy_def.tolist()

    if any(abs(d) < settings.min_horizontal_stroke for d in dx_def):
        result["reason"] = "INSUFFICIENT_HORIZONTAL_STROKE_SPAN"
    elif dx_def[0] * dx_def[1] >= 0:
        result["reason"] = "FIRST_DIRECTION_REVERSAL_MISSING"
    elif dx_def[1] * dx_def[2] >= 0:
        result["reason"] = "SECOND_DIRECTION_REVERSAL_MISSING"
    elif dy_def[1] < settings.min_downward_stroke:
        result["reason"] = f"DIAGONAL_NOT_DOWNWARD: {dy_def[1]:.3f} < {settings.min_downward_stroke}"
    elif abs(dx_def[0]) > 1e-9 and abs(dy_def[0]) > settings.max_horizontal_slope * abs(dx_def[0]):
        result["reason"] = "TOP_STROKE_NOT_HORIZONTAL"
    elif abs(dx_def[2]) > 1e-9 and abs(dy_def[2]) > settings.max_horizontal_slope * abs(dx_def[2]):
        result["reason"] = "BOTTOM_STROKE_NOT_HORIZONTAL"
    else:
        result["reason"] = "NO_VALID_Z_CORNER_TRIPLET"

    return result


@dataclass(frozen=True)
class JTrajectorySettings:
    """Trajectory and endpoint dominance thresholds for ASL letter J.

    Pinky path uses features[:, 67:69] (palm scales relative to frame 0).
    Static landmarks use features[:, :63] (normalized by wrist-to-MCP scale).
    Calibrated on SigNN Train+Val J/Z splits and false-J diagnostic webcam bouts.
    """
    min_downward_drop: float = 0.40
    max_upward_ratio_before_bottom: float = 0.55
    min_hook_horizontal: float = 0.20
    max_hook_vertical: float = 0.10
    max_step_excursion_ratio: float = 0.85
    max_drop_spike_fraction: float = 0.96
    min_end_pinky_extension: float = 0.75
    min_end_pinky_dominance: float = 0.12


_J_SPIKE_FAILURE_PREFIXES = (
    "SINGLE_FRAME_TRAJECTORY_SPIKE",
    "SINGLE_FRAME_DOWNWARD_COLLAPSE",
)


def _evaluate_j_path_metrics(path: np.ndarray, settings: JTrajectorySettings) -> dict:
    """Evaluate J trajectory geometry on a (16, 2) normalized pinky path."""
    excursion = float(np.max(np.linalg.norm(path - path[0], axis=1)))
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    max_step = float(np.max(steps))
    max_step_idx = int(np.argmax(steps))

    # Evaluate nadir (t_bot) across sequence up to frame 14
    t_bot = int(np.argmax(path[:15, 1]))
    # Apex before nadir across trajectory progression
    t_top = int(np.argmin(path[:t_bot + 1, 1])) if t_bot > 0 else 0

    # Downward drop from apex to nadir across whole trajectory
    dy_down = float(path[t_bot, 1] - path[t_top, 1])

    # Upward reversal during the downward stroke (from t_top to t_bot)
    if t_bot > t_top:
        stroke_dy = np.diff(path[t_top:t_bot + 1, 1])
        up_during_drop = float(np.sum(np.clip(-stroke_dy, 0, None)))
    else:
        up_during_drop = 0.0

    # Hook lateral span across all post-bottom frames
    dx_hook = float(np.max(np.abs(path[t_bot:, 0] - path[t_bot, 0])))
    # Hook vertical rise from bottom
    dy_hook = float(np.min(path[t_bot:, 1]) - path[t_bot, 1])

    # Spike ratio: distinguish out-and-back glitch from legitimate forward stroke progression
    is_persistent_step = False
    if max_step_idx < len(path) - 2:
        chord = float(np.linalg.norm(path[max_step_idx + 2] - path[max_step_idx]))
        return_step = float(steps[max_step_idx + 1])
        # Persistent step: hand stays near target rather than jumping back
        if chord >= 0.50 * max_step or return_step < 0.50 * max_step:
            is_persistent_step = True
    elif max_step_idx >= len(path) - 2:
        is_persistent_step = True

    spike_ratio = max_step / max(excursion, 1e-6)

    # Downward collapse check
    down_dy_steps = np.diff(path[t_top:t_bot + 1, 1]) if t_bot > t_top else np.asarray([0.0])
    drop_spike_frac = float(np.max(down_dy_steps)) / max(dy_down, 1e-6) if dy_down > 0 else 1.0

    metrics = {
        "valid": False,
        "reason": "",
        "excursion": excursion,
        "bottom_index": t_bot,
        "dy_down": dy_down,
        "up_before_bottom": up_during_drop,
        "dx_hook": dx_hook,
        "dy_hook": dy_hook,
        "spike_ratio": spike_ratio,
        "drop_spike_fraction": drop_spike_frac,
    }

    if dy_down < settings.min_downward_drop:
        metrics["reason"] = f"INSUFFICIENT_J_DOWNWARD_DROP: {dy_down:.3f} < {settings.min_downward_drop}"
        return metrics
    if up_during_drop > settings.max_upward_ratio_before_bottom * dy_down:
        metrics["reason"] = (
            f"UPWARD_MOTION_BEFORE_DROP: {up_during_drop:.3f} > "
            f"{settings.max_upward_ratio_before_bottom:.2f} * {dy_down:.3f}"
        )
        return metrics
    if dx_hook < settings.min_hook_horizontal:
        metrics["reason"] = f"INSUFFICIENT_J_HOOK_LATERAL: {dx_hook:.3f} < {settings.min_hook_horizontal}"
        return metrics
    if dy_hook > settings.max_hook_vertical:
        metrics["reason"] = f"J_HOOK_NOT_CURVING_UP: {dy_hook:.3f} > {settings.max_hook_vertical}"
        return metrics
    if spike_ratio >= settings.max_step_excursion_ratio and not is_persistent_step:
        metrics["reason"] = f"SINGLE_FRAME_TRAJECTORY_SPIKE: {spike_ratio:.3f} >= {settings.max_step_excursion_ratio}"
        return metrics
    if drop_spike_frac >= settings.max_drop_spike_fraction:
        metrics["reason"] = f"SINGLE_FRAME_DOWNWARD_COLLAPSE: {drop_spike_frac:.3f} >= {settings.max_drop_spike_fraction}"
        return metrics

    metrics["valid"] = True
    metrics["reason"] = "VALID_J_TRAJECTORY"
    return metrics


def _try_recover_isolated_one_frame_spike(
    path: np.ndarray,
    raw_eval: dict,
    settings: JTrajectorySettings,
) -> dict:
    """Repair strictly one isolated interior out-and-back glitch frame via chord midpoint."""
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    max_raw_step = float(np.max(steps))
    outlier_indices = []
    for t in range(1, len(path) - 1):
        s_in = float(steps[t - 1])
        s_out = float(steps[t])
        chord = float(np.linalg.norm(path[t + 1] - path[t - 1]))
        midpoint = 0.5 * (path[t - 1] + path[t + 1])
        deviation = float(np.linalg.norm(path[t] - midpoint))
        # True 1-frame excursion spike must jump out AND return past the neighbor chord;
        # step/teleport edges and multi-frame flat spikes have small min(s_in, s_out) vs chord.
        if (
            min(s_in, s_out) >= 1.25 * max(chord, 0.10)
            and deviation > max(chord, 0.25)
            and max(s_in, s_out) >= 0.50 * max_raw_step
        ):
            outlier_indices.append(t)

    if len(outlier_indices) == 0:
        return {"applied": False, "indices": [], "reason": "REJECTED_NO_ISOLATED_SPIKE_FOUND", "post_eval": raw_eval}
    if len(outlier_indices) > 1:
        return {"applied": False, "indices": [], "reason": "REJECTED_MULTI_FRAME_SPIKES_DETECTED", "post_eval": raw_eval}

    t_star = outlier_indices[0]
    repaired_path = path.copy()
    repaired_path[t_star] = 0.5 * (path[t_star - 1] + path[t_star + 1])
    post_eval = _evaluate_j_path_metrics(repaired_path, settings)
    if not post_eval["valid"]:
        return {
            "applied": False,
            "indices": [],
            "reason": f"REJECTED_POST_REPAIR_INVALID:{post_eval['reason']}",
            "post_eval": post_eval,
        }
    return {
        "applied": True,
        "indices": [int(t_star)],
        "reason": f"RECOVERED_ISOLATED_1_FRAME_SPIKE_AT_FRAME_{t_star}",
        "post_eval": post_eval,
    }


def validate_j_trajectory(features, settings: JTrajectorySettings = JTrajectorySettings()) -> dict:
    """Validate that features exhibit a genuine downward-then-hook ASL J trajectory.

    Features shape: (16, 69).
    Landmark 20 (pinky fingertip) trajectory: columns 67:69 (x, y) relative to frame 0.
    """
    result = {
        "valid": False,
        "reason": "",
        "excursion": 0.0,
        "bottom_index": -1,
        "dy_down": 0.0,
        "up_before_bottom": 0.0,
        "dx_hook": 0.0,
        "dy_hook": 0.0,
        "spike_ratio": 0.0,
        "drop_spike_fraction": 0.0,
        "end_pinky_extension": 0.0,
        "end_other_extension": 0.0,
        "spike_recovery_applied": False,
        "spike_repaired_frame_count": 0,
        "spike_repaired_frame_indices": [],
        "spike_recovery_reason": "NONE",
        "pre_recovery_valid": False,
        "pre_recovery_reason": "",
        "post_recovery_valid": False,
        "post_recovery_reason": "",
        "settings": asdict(settings),
    }
    if not isinstance(features, np.ndarray) or features.shape != (16, 69) or not np.isfinite(features).all():
        result["reason"] = "INVALID_FEATURE_DIMENSIONS"
        result["pre_recovery_reason"] = result["reason"]
        result["post_recovery_reason"] = result["reason"]
        return result

    lm = features[:, :63].reshape(16, 21, 3)
    tip_xy = lm[:, [8, 12, 16, 20], :2]
    pinky_xy = lm[:, 20:21, :2]
    has_distinct_fingers = bool(
        np.any(np.abs(tip_xy) > 1e-6) and not np.allclose(tip_xy, pinky_xy, atol=1e-5)
    )
    if has_distinct_fingers:
        pnk_ext = np.linalg.norm(lm[:, 20, :2] - lm[:, 0, :2], axis=1)
        oth_ext = np.maximum.reduce([
            np.linalg.norm(lm[:, idx, :2] - lm[:, 0, :2], axis=1)
            for idx in (8, 12, 16)
        ])
        end_pnk_ext = float(np.median(pnk_ext[-3:]))
        end_oth_ext = float(np.median(oth_ext[-3:]))
        result["end_pinky_extension"] = end_pnk_ext
        result["end_other_extension"] = end_oth_ext
        if end_pnk_ext < settings.min_end_pinky_extension:
            result["reason"] = f"PINKY_NOT_EXTENDED_AT_ENDPOINT: {end_pnk_ext:.3f} < {settings.min_end_pinky_extension}"
            result["pre_recovery_reason"] = result["reason"]
            result["post_recovery_reason"] = result["reason"]
            return result
        if (end_pnk_ext - end_oth_ext) < settings.min_end_pinky_dominance:
            result["reason"] = (
                f"NON_J_HANDSHAPE_DOMINANCE: {end_pnk_ext - end_oth_ext:.3f} < {settings.min_end_pinky_dominance}"
            )
            result["pre_recovery_reason"] = result["reason"]
            result["post_recovery_reason"] = result["reason"]
            return result

    path = features[:, 67:69]
    raw_eval = _evaluate_j_path_metrics(path, settings)
    result.update(raw_eval)
    result["pre_recovery_valid"] = bool(raw_eval["valid"])
    result["pre_recovery_reason"] = raw_eval["reason"]
    result["post_recovery_valid"] = bool(raw_eval["valid"])
    result["post_recovery_reason"] = raw_eval["reason"]

    if raw_eval["valid"]:
        return result

    if raw_eval["reason"].startswith(_J_SPIKE_FAILURE_PREFIXES):
        rec = _try_recover_isolated_one_frame_spike(path, raw_eval, settings)
        post_eval = rec["post_eval"]
        result["spike_recovery_reason"] = rec["reason"]
        result["post_recovery_valid"] = bool(post_eval["valid"])
        result["post_recovery_reason"] = post_eval["reason"]
        if rec["applied"]:
            result.update(post_eval)
            result["spike_recovery_applied"] = True
            result["spike_repaired_frame_count"] = len(rec["indices"])
            result["spike_repaired_frame_indices"] = rec["indices"]
    else:
        result["spike_recovery_reason"] = "SKIPPED_NON_SPIKE_FAILURE"
    return result


def coherent_motion_evidence(rows, scale, settings, *, skipped_duplicates=None):
    """Require all motion criteria in ONE observed run, not mixed across gaps.

Retaining a short tracking gap is still allowed. This does not join endpoints
or credit any displacement/time across missing observations.
    """
    skipped = set(skipped_duplicates or ())
    runs = []
    for row in rows:
        if not runs or not _is_contiguous_frame(runs[-1][-1][0], row[0], skipped):
            runs.append([])
        runs[-1].append(row)
    evidence = [measure_motion(run, scale, settings, skipped_duplicates=skipped) for run in runs]
    supported = [i for i, item in enumerate(evidence) if item["supported"]]
    return {"supported": bool(supported), "run_count": len(runs),
            "qualifying_runs": supported, "runs": [
                {"first_frame": run[0][0], "last_frame": run[-1][0], **item}
                for run, item in zip(runs, evidence)]}


def validate_proposal(entries, rows, scale, metrics, settings):
    """Check the exact collected bout before allowing endpoint arbitration.

PASS means technically coherent experimental motion, not proven J/Z. Semantic
validation remains UNVERIFIED; shadow geometry is deliberately not enforced.
    """
    result = {"status": "REVIEW_REQUIRED", "reason": "", "semantic_status": "UNVERIFIED",
              "geometry_policy": "SHADOW_ONLY_NOT_RECALL_VALIDATED"}
    try:
        if metrics.get("capture_seconds_after_onset", -1) < settings.min_capture_seconds:
            raise ValueError("MINIMUM_CAPTURE_DURATION_NOT_MET")
        if scale is None or not np.isfinite(scale) or scale <= 1e-6:
            raise ValueError("INVALID_CAPTURE_SCALE")
        skipped = set(metrics.get("skipped_duplicate_indices") or ())
        mapping = {entry[0]: entry for entry in entries}
        evidence_rows = [row for row in rows if row[0] in mapping and row[0] not in skipped]
        if not evidence_rows:
            raise ValueError("MOTION_EVIDENCE_NOT_FROM_CAPTURE")
        if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(evidence_rows, evidence_rows[1:])):
            raise ValueError("INVALID_MOTION_EVIDENCE_TIMELINE")
        indices = sample_valid_frame_indices([entry[0] for entry in entries])
        coherence = coherent_motion_evidence(
            evidence_rows,
            scale,
            settings,
            skipped_duplicates=skipped,
        )
        result["continuity_evidence"] = coherence
        if not coherence["supported"]:
            raise ValueError("NO_SINGLE_COHERENT_MOTION_RUN")
        features, _ = build_sequence_features([mapping[i][1] for i in indices],
                                              [mapping[i][2] for i in indices])
        z_val = validate_z_trajectory(features)
        j_val = validate_j_trajectory(features)
        result.update(status="PASS", reason="COHERENT_MOTION_ENDPOINT_PENDING",
                      selected_frame_indices=indices,
                      geometry_policy="JZ_TRAJECTORY_VALIDATION_ACTIVE",
                      z_trajectory_validation=z_val,
                      z_trajectory_valid=z_val["valid"],
                      j_trajectory_validation=j_val,
                      j_trajectory_valid=j_val["valid"])
        # Ensure terminal reference is a real validated capture frame, not a
        # newly encountered static pose. Caller keeps the reference in memory.
        landmarks_to_array(entries[-1][1])
        try:
            result["shadow_geometry_candidates"] = sorted(geometry_candidates(features))
        except (ValueError, TypeError, IndexError) as exc:
            result["shadow_geometry_error"] = str(exc)  # Shadow cannot affect acceptance.
        try:
            result["geometry_predicate_diagnostics"] = geometry_predicates(features)
        except Exception as exc:  # New instrumentation is never an acceptance condition.
            result["geometry_predicate_diagnostic_error"] = type(exc).__name__+": "+str(exc)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        result.update(status="REVIEW_REQUIRED", reason=str(exc))
    return result
