"""Enhanced Multivariate DTW Engine for Dynamic J/Z Gestures (Method 1).

Enhancements over Stage 2/3 Baseline DTW:
- Employs CONFIG_E (14D) multivariate feature representation:
  * Global active fingertip trajectory & frame deltas (x, y, dx, dy).
  * Frame speed / step magnitude.
  * Unit direction vectors (cos theta, sin theta).
  * Angular turn change / curvature (delta theta) between consecutive motion vectors.
  * Wrist-relative active fingertip 3D coordinates.
  * Continuous handshape extension & finger dominance (active_ext - curled_ext).
- Enhanced out-of-distribution (OOD) rejection:
  * Minimum classification margin threshold.
  * Handshape dominance sanity check (active finger must exceed curled finger extension).
  * Maximum DTW warping distance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from app.inference.dynamic_jz.dtw_engine import DTWError, dtw_multivariate
from app.inference.dynamic_jz.feature_builder import build_dtw_features
from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    FingerTracker,
    FingerTrackingError,
    FingerTrackingResult,
    SEQUENCE_LENGTH,
)
from app.inference.dynamic_jz.template_store import DTWTemplate, TemplateStore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENHANCED_TEMPLATE_PATH = ROOT / "models/dynamic_jz/dtw_templates/templates_config_e_valid_only_k2.json"
EXPECTED_ENHANCED_TEMPLATE_SHA256 = "03d889fa54ddce22ba29fda2b0172185a547d7ea7ef8a8859fa6a8489b3a7e95"


@dataclass(frozen=True)
class EnhancedDTWResult:
    """Output contract for Enhanced DTW evaluation."""
    prediction: str                      # 'J', 'Z', or 'UNKNOWN'
    distance_j: float                    # DTW distance to closest J template
    distance_z: float                    # DTW distance to closest Z template
    best_distance: float                 # min(distance_j, distance_z)
    margin: float                        # |distance_j - distance_z|
    best_template_id: str                # Template ID of winning match
    tracking_quality: str                # 'VALID', 'SUSPICIOUS', 'INVALID'
    handshape_dominance_j: float         # active_ext - curled_ext for J
    handshape_dominance_z: float         # active_ext - curled_ext for Z
    status: str                          # 'OK', 'REJECTED', 'INVALID_INPUT', 'ERROR', 'UNAVAILABLE', 'DISABLED'
    rejection_reason: str = ""
    latency_ms: float = 0.0


class EnhancedDTWClassifier:
    """Enhanced DTW Classifier using CONFIG_E (14D) features and dual hypothesis matching."""

    def __init__(
        self,
        template_path: Path | str | None = None,
        radius: int = 3,
        threshold: float = 2.65,
        min_margin: float = 0.15,
        min_dominance: float = -0.05,
        verify_sha256: bool = False,
    ) -> None:
        self.template_path = Path(template_path or DEFAULT_ENHANCED_TEMPLATE_PATH)
        self.radius = int(radius)
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.min_dominance = float(min_dominance)
        self.verify_sha256 = bool(verify_sha256)

        self.tracker = FingerTracker()
        self.store: TemplateStore | None = None
        self.j_templates: list[DTWTemplate] = []
        self.z_templates: list[DTWTemplate] = []
        self._init_error = ""

        self._load_templates()

    def _load_templates(self) -> None:
        if not self.template_path.exists():
            self._init_error = f"Template file not found: {self.template_path}"
            return

        if self.verify_sha256:
            data = self.template_path.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != EXPECTED_ENHANCED_TEMPLATE_SHA256:
                self._init_error = (
                    f"Template checksum mismatch: expected {EXPECTED_ENHANCED_TEMPLATE_SHA256}, got {actual_sha}"
                )
                return

        try:
            self.store = TemplateStore.load_json(self.template_path)
            self.j_templates = self.store.get_by_label("J")
            self.z_templates = self.store.get_by_label("Z")
            if not self.j_templates or not self.z_templates:
                self._init_error = "Template store must contain both J and Z templates."
        except Exception as exc:
            self._init_error = f"Failed to load templates: {exc}"

    @property
    def is_available(self) -> bool:
        return self._init_error == "" and len(self.j_templates) > 0 and len(self.z_templates) > 0

    def classify(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        selected_original_frame_indices: list[int] | None = None,
    ) -> EnhancedDTWResult:
        """Classify a (16, 69) sequence using dual hypothesis matching under CONFIG_E."""
        t_start = time.perf_counter()

        if not self.is_available:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_distance=float("inf"),
                margin=0.0,
                best_template_id="",
                tracking_quality="INVALID",
                handshape_dominance_j=0.0,
                handshape_dominance_z=0.0,
                status="UNAVAILABLE",
                rejection_reason=f"INITIALIZATION_ERROR: {self._init_error}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Defensive copy & shape check
        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                return EnhancedDTWResult(
                    prediction="UNKNOWN",
                    distance_j=float("inf"),
                    distance_z=float("inf"),
                    best_distance=float("inf"),
                    margin=0.0,
                    best_template_id="",
                    tracking_quality="INVALID",
                    handshape_dominance_j=0.0,
                    handshape_dominance_z=0.0,
                    status="INVALID_INPUT",
                    rejection_reason="INVALID_SHAPE_OR_NON_FINITE",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_distance=float("inf"),
                margin=0.0,
                best_template_id="",
                tracking_quality="INVALID",
                handshape_dominance_j=0.0,
                handshape_dominance_z=0.0,
                status="ERROR",
                rejection_reason=f"INPUT_CONVERSION_ERROR: {exc}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 1. Extract dual hypothesis tracking
        try:
            res_j = self.tracker.extract(
                arr, letter_target="J", selected_original_frame_indices=selected_original_frame_indices
            )
            res_z = self.tracker.extract(
                arr, letter_target="Z", selected_original_frame_indices=selected_original_frame_indices
            )
        except Exception as exc:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_distance=float("inf"),
                margin=0.0,
                best_template_id="",
                tracking_quality="INVALID",
                handshape_dominance_j=0.0,
                handshape_dominance_z=0.0,
                status="ERROR",
                rejection_reason=f"FEATURE_EXTRACTION_ERROR: {exc}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # Quality check
        quality = "VALID"
        if res_j.validity_status == "INVALID" or res_z.validity_status == "INVALID":
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=float("inf"),
                distance_z=float("inf"),
                best_distance=float("inf"),
                margin=0.0,
                best_template_id="",
                tracking_quality="INVALID",
                handshape_dominance_j=0.0,
                handshape_dominance_z=0.0,
                status="REJECTED",
                rejection_reason=f"INVALID_TRACKING: {res_j.invalidation_reason or res_z.invalidation_reason}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )
        if res_j.validity_status == "SUSPICIOUS" or res_z.validity_status == "SUSPICIOUS":
            quality = "SUSPICIOUS"

        # Compute handshape dominance metrics
        dom_j = float(res_j.handshape_features.active_extension[-1] - res_j.handshape_features.curled_extension_mean[-1])
        dom_z = float(res_z.handshape_features.active_extension[-1] - res_z.handshape_features.curled_extension_mean[-1])

        # 2. Build CONFIG_E features for both hypotheses
        feat_j, weights = build_dtw_features(res_j, config_name="CONFIG_E")
        feat_z, _ = build_dtw_features(res_z, config_name="CONFIG_E")

        # 3. Match J templates
        min_d_j = float("inf")
        best_t_j = ""
        for t in self.j_templates:
            try:
                d, _, _ = dtw_multivariate(feat_j, t.get_array(), weights=weights, radius=self.radius)
                if d < min_d_j:
                    min_d_j = d
                    best_t_j = t.template_id
            except DTWError:
                continue

        # 4. Match Z templates
        min_d_z = float("inf")
        best_t_z = ""
        for t in self.z_templates:
            try:
                d, _, _ = dtw_multivariate(feat_z, t.get_array(), weights=weights, radius=self.radius)
                if d < min_d_z:
                    min_d_z = d
                    best_t_z = t.template_id
            except DTWError:
                continue

        if min_d_j == float("inf") or min_d_z == float("inf"):
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_distance=min(min_d_j, min_d_z),
                margin=0.0,
                best_template_id="",
                tracking_quality=quality,
                handshape_dominance_j=dom_j,
                handshape_dominance_z=dom_z,
                status="ERROR",
                rejection_reason="DTW_ALIGNMENT_FAILED",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        margin = float(abs(min_d_j - min_d_z))
        best_dist = min(min_d_j, min_d_z)
        cand = "J" if min_d_j < min_d_z else "Z"
        best_tid = best_t_j if cand == "J" else best_t_z

        # 5. OOD and rejection criteria
        # A. Max distance threshold
        if best_dist > self.threshold:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_distance=best_dist,
                margin=margin,
                best_template_id=best_tid,
                tracking_quality=quality,
                handshape_dominance_j=dom_j,
                handshape_dominance_z=dom_z,
                status="REJECTED",
                rejection_reason=f"DISTANCE_EXCEEDS_THRESHOLD ({best_dist:.3f} > {self.threshold:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # B. Separation margin threshold
        if margin < self.min_margin:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_distance=best_dist,
                margin=margin,
                best_template_id=best_tid,
                tracking_quality=quality,
                handshape_dominance_j=dom_j,
                handshape_dominance_z=dom_z,
                status="REJECTED",
                rejection_reason=f"AMBIGUOUS_MARGIN_TOO_SMALL ({margin:.3f} < {self.min_margin:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # C. Handshape dominance gate: candidate's active finger must be dominant
        cand_dom = dom_j if cand == "J" else dom_z
        if cand_dom < self.min_dominance:
            return EnhancedDTWResult(
                prediction="UNKNOWN",
                distance_j=min_d_j,
                distance_z=min_d_z,
                best_distance=best_dist,
                margin=margin,
                best_template_id=best_tid,
                tracking_quality=quality,
                handshape_dominance_j=dom_j,
                handshape_dominance_z=dom_z,
                status="REJECTED",
                rejection_reason=f"INSUFFICIENT_HANDSHAPE_DOMINANCE ({cand_dom:.3f} < {self.min_dominance:.3f})",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        return EnhancedDTWResult(
            prediction=cand,
            distance_j=min_d_j,
            distance_z=min_d_z,
            best_distance=best_dist,
            margin=margin,
            best_template_id=best_tid,
            tracking_quality=quality,
            handshape_dominance_j=dom_j,
            handshape_dominance_z=dom_z,
            status="OK",
            rejection_reason="",
            latency_ms=(time.perf_counter() - t_start) * 1000.0,
        )
