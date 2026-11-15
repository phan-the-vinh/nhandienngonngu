"""DTW Matcher for Dynamic J/Z Gesture Classification.

Evaluates an input sequence against stored representative templates:
- Computes distances to all candidate templates per class.
- Evaluates best class match, class distances, and margin separation.
- Enforces rejection thresholds and ambiguity margins to safely return UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from app.inference.dynamic_jz.dtw_engine import dtw_multivariate, DTWError
from app.inference.dynamic_jz.feature_builder import build_dtw_features
from app.inference.dynamic_jz.finger_tracker import FingerTrackingResult
from app.inference.dynamic_jz.template_store import TemplateStore, DTWTemplate


@dataclass(frozen=True)
class DTWMatchResult:
    """Classification result from the DTW Matcher."""
    predicted_label: str                  # 'J', 'Z', or 'UNKNOWN'
    distance_j: float                     # Minimum distance to any J template
    distance_z: float                     # Minimum distance to any Z template
    best_template_id: str                 # Identifier of the closest matching template
    normalized_distance: float            # Distance to the winning template
    distance_margin: float                # Difference |distance_j - distance_z|
    alignment_path_length: int            # Warping path length to the best template
    tracking_quality: str                 # 'VALID', 'SUSPICIOUS', or 'INVALID'
    rejection_reason: str = ""            # Reason if predicted_label is 'UNKNOWN'


class DTWMatcher:
    """Matches an input finger tracking sequence against stored J and Z templates."""

    def __init__(
        self,
        store: TemplateStore,
        config_name: str = "CONFIG_A",
        radius: int = 3,
        threshold_j: float = 2.0,
        threshold_z: float = 2.0,
        min_margin: float = 0.10,
    ) -> None:
        self.store = store
        self.config_name = str(config_name).strip().upper()
        self.radius = int(radius)
        self.threshold_j = float(threshold_j)
        self.threshold_z = float(threshold_z)
        self.min_margin = float(min_margin)

        self.j_templates = self.store.get_by_label("J")
        self.z_templates = self.store.get_by_label("Z")
        if not self.j_templates or not self.z_templates:
            raise ValueError("TemplateStore must contain at least one J and one Z template.")

    def match(self, tracking_result: FingerTrackingResult) -> DTWMatchResult:
        """Classify a FingerTrackingResult as 'J', 'Z', or 'UNKNOWN'.

        Parameters
        ----------
        tracking_result : FingerTrackingResult
            Extracted finger tracking result from Stage 1.

        Returns
        -------
        DTWMatchResult
            Classification details, distances, margin, and confidence status.
        """
        quality = tracking_result.validity_status

        # 1. Quality gate
        if quality == "INVALID":
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_template_id="",
                normalized_distance=float("inf"),
                distance_margin=0.0,
                alignment_path_length=0,
                tracking_quality=quality,
                rejection_reason=f"INVALID_TRACKING_QUALITY: {tracking_result.invalidation_reason}",
            )

        seq_len = len(tracking_result.global_trajectory)
        if seq_len < 8:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_template_id="",
                normalized_distance=float("inf"),
                distance_margin=0.0,
                alignment_path_length=0,
                tracking_quality=quality,
                rejection_reason=f"SEQUENCE_TOO_SHORT: length={seq_len} < 8",
            )

        # 2. Extract feature sequence
        input_seq, weights = build_dtw_features(tracking_result, config_name=self.config_name)

        # 3. Match against all J templates
        min_d_j = float("inf")
        best_t_j: DTWTemplate | None = None
        best_path_j = 0
        for t in self.j_templates:
            t_seq = t.get_array()
            try:
                d, path_len, _ = dtw_multivariate(
                    input_seq, t_seq, weights=weights, radius=self.radius
                )
                if d < min_d_j:
                    min_d_j = d
                    best_t_j = t
                    best_path_j = path_len
            except DTWError:
                continue

        # 4. Match against all Z templates
        min_d_z = float("inf")
        best_t_z: DTWTemplate | None = None
        best_path_z = 0
        for t in self.z_templates:
            t_seq = t.get_array()
            try:
                d, path_len, _ = dtw_multivariate(
                    input_seq, t_seq, weights=weights, radius=self.radius
                )
                if d < min_d_z:
                    min_d_z = d
                    best_t_z = t
                    best_path_z = path_len
            except DTWError:
                continue

        # If matching failed against all templates of either class
        if best_t_j is None or best_t_z is None:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id="",
                normalized_distance=float("inf"),
                distance_margin=0.0,
                alignment_path_length=0,
                tracking_quality=quality,
                rejection_reason="ALIGNMENT_FAILED",
            )

        # 5. Evaluate candidate class and confidence margin
        if min_d_j < min_d_z:
            candidate = "J"
            best_dist = min_d_j
            other_dist = min_d_z
            best_template = best_t_j
            best_path_len = best_path_j
            thresh = self.threshold_j
        else:
            candidate = "Z"
            best_dist = min_d_z
            other_dist = min_d_j
            best_template = best_t_z
            best_path_len = best_path_z
            thresh = self.threshold_z

        margin = float(other_dist - best_dist)

        # 6. Rejection logic
        if best_dist > thresh:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id=best_template.template_id,
                normalized_distance=best_dist,
                distance_margin=margin,
                alignment_path_length=best_path_len,
                tracking_quality=quality,
                rejection_reason=f"DISTANCE_EXCEEDS_THRESHOLD ({best_dist:.3f} > {thresh:.3f})",
            )

        if margin < self.min_margin:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id=best_template.template_id,
                normalized_distance=best_dist,
                distance_margin=margin,
                alignment_path_length=best_path_len,
                tracking_quality=quality,
                rejection_reason=f"AMBIGUOUS_MARGIN_TOO_SMALL ({margin:.3f} < {self.min_margin:.3f})",
            )

        return DTWMatchResult(
            predicted_label=candidate,
            distance_j=min_d_j,
            distance_z=min_d_z,
            best_template_id=best_template.template_id,
            normalized_distance=best_dist,
            distance_margin=margin,
            alignment_path_length=best_path_len,
            tracking_quality=quality,
            rejection_reason="",
        )

    def match_dual(
        self,
        result_j: FingerTrackingResult,
        result_z: FingerTrackingResult,
    ) -> DTWMatchResult:
        """Classify by testing hypothesis J with result_j against J templates and
        hypothesis Z with result_z against Z templates.

        Parameters
        ----------
        result_j : FingerTrackingResult
            Extracted with letter_target='J' (pinky tracking).
        result_z : FingerTrackingResult
            Extracted with letter_target='Z' (index tracking).

        Returns
        -------
        DTWMatchResult
        """
        # Quality check: if either is INVALID
        if result_j.validity_status == "INVALID" or result_z.validity_status == "INVALID":
            reason = result_j.invalidation_reason or result_z.invalidation_reason
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_template_id="",
                normalized_distance=float("inf"),
                distance_margin=0.0,
                alignment_path_length=0,
                tracking_quality="INVALID",
                rejection_reason=f"INVALID_TRACKING_QUALITY: {reason}",
            )

        # Build features
        feat_j, weights_j = build_dtw_features(result_j, config_name=self.config_name)
        feat_z, weights_z = build_dtw_features(result_z, config_name=self.config_name)

        # Match against J templates using feat_j
        min_d_j = float("inf")
        best_t_j: DTWTemplate | None = None
        best_path_j = 0
        for t in self.j_templates:
            t_seq = t.get_array()
            try:
                d, path_len, _ = dtw_multivariate(
                    feat_j, t_seq, weights=weights_j, radius=self.radius
                )
                if d < min_d_j:
                    min_d_j = d
                    best_t_j = t
                    best_path_j = path_len
            except DTWError:
                continue

        # Match against Z templates using feat_z
        min_d_z = float("inf")
        best_t_z: DTWTemplate | None = None
        best_path_z = 0
        for t in self.z_templates:
            t_seq = t.get_array()
            try:
                d, path_len, _ = dtw_multivariate(
                    feat_z, t_seq, weights=weights_z, radius=self.radius
                )
                if d < min_d_z:
                    min_d_z = d
                    best_t_z = t
                    best_path_z = path_len
            except DTWError:
                continue

        if best_t_j is None or best_t_z is None:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id="",
                normalized_distance=float("inf"),
                distance_margin=0.0,
                alignment_path_length=0,
                tracking_quality=result_j.validity_status,
                rejection_reason="ALIGNMENT_FAILED",
            )

        if min_d_j < min_d_z:
            candidate = "J"
            best_dist = min_d_j
            other_dist = min_d_z
            best_template = best_t_j
            best_path_len = best_path_j
            thresh = self.threshold_j
            quality = result_j.validity_status
        else:
            candidate = "Z"
            best_dist = min_d_z
            other_dist = min_d_j
            best_template = best_t_z
            best_path_len = best_path_z
            thresh = self.threshold_z
            quality = result_z.validity_status

        margin = float(other_dist - best_dist)

        if best_dist > thresh:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id=best_template.template_id,
                normalized_distance=best_dist,
                distance_margin=margin,
                alignment_path_length=best_path_len,
                tracking_quality=quality,
                rejection_reason=f"DISTANCE_EXCEEDS_THRESHOLD ({best_dist:.3f} > {thresh:.3f})",
            )

        if margin < self.min_margin:
            return DTWMatchResult(
                predicted_label="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_template_id=best_template.template_id,
                normalized_distance=best_dist,
                distance_margin=margin,
                alignment_path_length=best_path_len,
                tracking_quality=quality,
                rejection_reason=f"AMBIGUOUS_MARGIN_TOO_SMALL ({margin:.3f} < {self.min_margin:.3f})",
            )

        return DTWMatchResult(
            predicted_label=candidate,
            distance_j=min_d_j,
            distance_z=min_d_z,
            best_template_id=best_template.template_id,
            normalized_distance=best_dist,
            distance_margin=margin,
            alignment_path_length=best_path_len,
            tracking_quality=quality,
            rejection_reason="",
        )

    def match_sequence(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        tracker: Any | None = None,
        selected_original_frame_indices: list[int] | None = None,
    ) -> DTWMatchResult:
        """Convenience method to classify a raw (16, 69) sequence using dual tracking."""
        if tracker is None:
            from app.inference.dynamic_jz.finger_tracker import FingerTracker
            tracker = FingerTracker()

        res_j = tracker.extract(
            sequence, letter_target="J", selected_original_frame_indices=selected_original_frame_indices
        )
        res_z = tracker.extract(
            sequence, letter_target="Z", selected_original_frame_indices=selected_original_frame_indices
        )
        return self.match_dual(res_j, res_z)
