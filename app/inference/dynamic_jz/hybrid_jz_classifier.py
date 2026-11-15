"""Hybrid Decision Classifier for Dynamic J/Z Gestures.

Combines three complementary recognition mechanisms into an authoritative consensus:
1. Baseline Logistic Regression Pipeline (Statistical 1104-D discriminative boundary).
2. Enhanced Multivariate DTW Engine (14-D CONFIG_E, exemplar trajectory & kinematic alignment).
3. Geometric & Ratio Classifier (Multi-criteria stroke geometry, angles, and ratios).

Decision Architecture:
- Consensus Agreement: When LR and Enhanced DTW agree, confidence is reinforced.
- Conflict Arbitration: Resolves disagreements using DTW distance, margin, and geometric scores.
- OOD & Noise Rejection: When both DTW and Geometric reject an attempt (even if binary LR
  forces a prediction), Hybrid safely outputs UNKNOWN.
- Non-Standard J Hook Tolerance: Does not reject valid J gestures merely because the hook
  differs from canonical curvature, provided DTW and downward progression validate it.
- Never relies on a single geometric ratio; never hardcodes thresholds on Test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import joblib
import numpy as np

from app.inference.dynamic_jz.enhanced_dtw import (
    DEFAULT_ENHANCED_TEMPLATE_PATH,
    EnhancedDTWClassifier,
    EnhancedDTWResult,
)
from app.inference.dynamic_jz.finger_tracker import FEATURE_DIM, SEQUENCE_LENGTH
from app.inference.dynamic_jz.geometric_classifier import (
    GeometricClassificationResult,
    GeometricClassifier,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LR_MODEL_PATH = ROOT / "data/experiments/dynamic_jz_binary_baseline_v1/model_pipeline.joblib"


@dataclass(frozen=True)
class HybridEvidence:
    """Detailed evidence breakdown from each constituent classifier."""
    lr_prediction: str                  # 'J' or 'Z'
    lr_confidence: float                # max(P(J), P(Z))
    lr_prob_j: float                    # P(J)
    lr_prob_z: float                    # P(Z)
    dtw_prediction: str                 # 'J', 'Z', or 'UNKNOWN'
    dtw_best_distance: float
    dtw_margin: float
    dtw_status: str
    geom_prediction: str                # 'J', 'Z', or 'UNKNOWN'
    geom_score_j: float
    geom_score_z: float
    geom_margin: float
    geom_status: str


@dataclass(frozen=True)
class HybridDecisionResult:
    """Output contract of the Hybrid J/Z Decision Classifier."""
    prediction: str                     # 'J', 'Z', or 'UNKNOWN'
    confidence: float                   # Calibrated composite confidence (0.0 to 1.0)
    decision_path: str                  # Diagnostic decision rule tag
    agreement_count: int                # Number of models agreeing on the winning label (1..3)
    evidence: HybridEvidence
    rejection_reason: str = ""
    latency_ms: float = 0.0


class HybridJzClassifier:
    """Decision fusion classifier orchestrating Logistic Regression, DTW, and Geometric analysis."""

    def __init__(
        self,
        lr_model_path: Path | str | None = None,
        dtw_template_path: Path | str | None = None,
        dtw_threshold: float = 2.65,
        dtw_min_margin: float = 0.15,
        dtw_min_dominance: float = -0.05,
        geom_min_score: float = 0.45,
        geom_min_margin: float = 0.08,
    ) -> None:
        self.lr_model_path = Path(lr_model_path or DEFAULT_LR_MODEL_PATH)
        self.lr_pipeline = None
        self._load_lr_model()

        self.enhanced_dtw = EnhancedDTWClassifier(
            template_path=dtw_template_path or DEFAULT_ENHANCED_TEMPLATE_PATH,
            threshold=dtw_threshold,
            min_margin=dtw_min_margin,
            min_dominance=dtw_min_dominance,
        )

        self.geom_classifier = GeometricClassifier(
            min_score=geom_min_score,
            min_margin=geom_min_margin,
        )

    def _load_lr_model(self) -> None:
        if self.lr_model_path.exists():
            try:
                self.lr_pipeline = joblib.load(self.lr_model_path)
            except Exception as exc:
                self.lr_pipeline = None

    @property
    def is_available(self) -> bool:
        return self.lr_pipeline is not None and self.enhanced_dtw.is_available

    def classify(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        selected_original_frame_indices: list[int] | None = None,
    ) -> HybridDecisionResult:
        """Classify a 16x69 dynamic sequence using the hybrid fusion policy."""
        t_start = time.perf_counter()

        # 1. Defensive input validation
        try:
            arr = np.asarray(sequence, dtype=np.float64)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(arr).all():
                dummy_ev = HybridEvidence(
                    lr_prediction="UNKNOWN", lr_confidence=0.0, lr_prob_j=0.0, lr_prob_z=0.0,
                    dtw_prediction="UNKNOWN", dtw_best_distance=float("inf"), dtw_margin=0.0, dtw_status="INVALID_INPUT",
                    geom_prediction="UNKNOWN", geom_score_j=0.0, geom_score_z=0.0, geom_margin=0.0, geom_status="INVALID_INPUT"
                )
                return HybridDecisionResult(
                    prediction="UNKNOWN", confidence=0.0, decision_path="INVALID_INPUT_SHAPE",
                    agreement_count=0, evidence=dummy_ev, rejection_reason="INVALID_INPUT_SHAPE_OR_NON_FINITE",
                    latency_ms=(time.perf_counter() - t_start) * 1000.0,
                )
        except Exception as exc:
            dummy_ev = HybridEvidence(
                lr_prediction="UNKNOWN", lr_confidence=0.0, lr_prob_j=0.0, lr_prob_z=0.0,
                dtw_prediction="UNKNOWN", dtw_best_distance=float("inf"), dtw_margin=0.0, dtw_status="ERROR",
                geom_prediction="UNKNOWN", geom_score_j=0.0, geom_score_z=0.0, geom_margin=0.0, geom_status="ERROR"
            )
            return HybridDecisionResult(
                prediction="UNKNOWN", confidence=0.0, decision_path="INPUT_CONVERSION_ERROR",
                agreement_count=0, evidence=dummy_ev, rejection_reason=f"INPUT_ERROR: {exc}",
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 2. Evaluate constituent models
        # A. Baseline Logistic Regression
        flat_input = arr.reshape(1, -1)
        lr_probs = self.lr_pipeline.predict_proba(flat_input)[0]
        prob_j = float(lr_probs[0])
        prob_z = float(lr_probs[1])
        lr_pred = "J" if prob_j > prob_z else "Z"
        lr_conf = float(max(prob_j, prob_z))

        # B. Enhanced DTW Engine
        dtw_res = self.enhanced_dtw.classify(
            arr, selected_original_frame_indices=selected_original_frame_indices
        )

        # C. Geometric Classifier
        geom_res = self.geom_classifier.classify(arr)

        evidence = HybridEvidence(
            lr_prediction=lr_pred,
            lr_confidence=lr_conf,
            lr_prob_j=prob_j,
            lr_prob_z=prob_z,
            dtw_prediction=dtw_res.prediction,
            dtw_best_distance=dtw_res.best_distance,
            dtw_margin=dtw_res.margin,
            dtw_status=dtw_res.status,
            geom_prediction=geom_res.prediction,
            geom_score_j=geom_res.score_j,
            geom_score_z=geom_res.score_z,
            geom_margin=geom_res.margin,
            geom_status=geom_res.status,
        )

        # 3. Decision Fusion Logic
        d_pred = dtw_res.prediction
        g_pred = geom_res.prediction
        final_pred = "UNKNOWN"
        final_conf = 0.0
        decision_path = "DEFAULT_REJECTION"
        rejection_reason = ""

        # Case 1: Unanimous consensus across all 3 methods
        if lr_pred == d_pred == g_pred and d_pred in ("J", "Z"):
            final_pred = d_pred
            final_conf = float(np.clip(0.6 * lr_conf + 0.2 * (1.0 - min(dtw_res.best_distance / 3.0, 1.0)) + 0.2 * max(geom_res.score_j, geom_res.score_z) + 0.1, 0.0, 1.0))
            decision_path = "UNANIMOUS_CONSENSUS_3_MODELS"

        # Case 2: Out-Of-Distribution / Invalid Motion Gating
        # When BOTH DTW and Geometric reject an attempt (even if binary LR outputs J or Z)
        elif d_pred == "UNKNOWN" and g_pred == "UNKNOWN":
            final_pred = "UNKNOWN"
            final_conf = 0.0
            decision_path = "OOD_REJECTED_BY_DTW_AND_GEOM"
            rejection_reason = f"DTW: {dtw_res.rejection_reason} | Geom: {geom_res.rejection_reason}"

        # Case 3: Consensus between LR and Enhanced DTW (LR == DTW)
        elif lr_pred == d_pred and d_pred in ("J", "Z"):
            # LR and DTW agree on target!
            # Check if Geometric has a strong counter-veto:
            # Tolerant to J hook variation: if target is J and downward drop is present, accept even if hook differs.
            if d_pred == "J":
                j_feat = geom_res.features_j
                has_drop = (j_feat is not None and j_feat.downward_drop >= 0.12)
                if has_drop or dtw_res.best_distance <= 2.20:
                    final_pred = "J"
                    final_conf = float(np.clip(0.65 * lr_conf + 0.35 * (1.0 - min(dtw_res.best_distance / 3.0, 1.0)), 0.0, 0.95))
                    decision_path = "CONSENSUS_LR_DTW_J_CONFIRMED"
                else:
                    final_pred = "UNKNOWN"
                    final_conf = 0.0
                    decision_path = "REJECTED_J_NO_DOWNWARD_PROGRESSION"
                    rejection_reason = "Neither downward drop nor close DTW template match"

            else:  # d_pred == "Z"
                z_feat = geom_res.features_z
                # Z requires some horizontal progression and not moving purely upwards
                has_z_motion = (z_feat is not None and z_feat.aspect_ratio >= 0.30)
                if has_z_motion or dtw_res.best_distance <= 2.20:
                    final_pred = "Z"
                    final_conf = float(np.clip(0.65 * lr_conf + 0.35 * (1.0 - min(dtw_res.best_distance / 3.0, 1.0)), 0.0, 0.95))
                    decision_path = "CONSENSUS_LR_DTW_Z_CONFIRMED"
                else:
                    final_pred = "UNKNOWN"
                    final_conf = 0.0
                    decision_path = "REJECTED_Z_NO_HORIZONTAL_STROKE"
                    rejection_reason = "Z lacked minimal horizontal stroke progression"

        # Case 4: Trajectory consensus between DTW and Geometric (DTW == Geom != LR)
        elif d_pred == g_pred and d_pred in ("J", "Z"):
            # Trajectory-aware models agree on label, overruling linear LR boundary
            if dtw_res.margin >= 0.20 and max(geom_res.score_j, geom_res.score_z) >= 0.45:
                final_pred = d_pred
                final_conf = float(np.clip(0.5 * (1.0 - min(dtw_res.best_distance / 3.0, 1.0)) + 0.5 * max(geom_res.score_j, geom_res.score_z), 0.0, 0.90))
                decision_path = "TRAJECTORY_CONSENSUS_DTW_GEOM_OVERRULE_LR"
            else:
                final_pred = "UNKNOWN"
                final_conf = 0.0
                decision_path = "AMBIGUOUS_DTW_GEOM_MARGIN"
                rejection_reason = f"DTW margin {dtw_res.margin:.3f} or Geom score insufficient to overrule LR"

        # Case 5: DTW is UNKNOWN, but LR and Geometric agree on candidate (LR == Geom)
        elif d_pred == "UNKNOWN" and lr_pred == g_pred and g_pred in ("J", "Z"):
            # Check if DTW's closest class aligns with LR and Geom, and distance is borderline
            closest_dtw_class = "J" if dtw_res.distance_j < dtw_res.distance_z else "Z"
            if closest_dtw_class == lr_pred and dtw_res.best_distance <= 3.20:
                final_pred = lr_pred
                final_conf = float(np.clip(0.5 * lr_conf + 0.5 * max(geom_res.score_j, geom_res.score_z), 0.0, 0.85))
                decision_path = "RECOVERED_LR_GEOM_WITH_BORDERLINE_DTW"
            else:
                final_pred = "UNKNOWN"
                final_conf = 0.0
                decision_path = "REJECTED_DTW_UNKNOWN_AND_CONFLICTING"
                rejection_reason = f"DTW rejected and closest DTW ({closest_dtw_class}) did not support {lr_pred}"

        # Case 6: Irreconcilable disagreement between all three
        else:
            final_pred = "UNKNOWN"
            final_conf = 0.0
            decision_path = "IRRECONCILABLE_DISAGREEMENT"
            rejection_reason = f"LR={lr_pred}, DTW={d_pred}, Geom={g_pred} mutually conflicting"

        # Agreement count calculation
        if final_pred in ("J", "Z"):
            agree_count = sum(1 for m_p in (lr_pred, d_pred, g_pred) if m_p == final_pred)
        else:
            agree_count = 0

        lat_ms = (time.perf_counter() - t_start) * 1000.0

        return HybridDecisionResult(
            prediction=final_pred,
            confidence=final_conf,
            decision_path=decision_path,
            agreement_count=agree_count,
            evidence=evidence,
            rejection_reason=rejection_reason,
            latency_ms=lat_ms,
        )
