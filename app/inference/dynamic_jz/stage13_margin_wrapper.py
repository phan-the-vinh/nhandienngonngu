"""Stage 13: Controlled Integration of Probability Margin Reject Option.

Wraps the production Dynamic J/Z Logistic Regression model (Pipeline) with a
configurable, fail-safe Reject Option based on probability margin:
    margin = |P(J) - P(Z)|
When margin < margin_threshold (default 0.10) and enable_reject_option is True,
the prediction is rejected as 'UNKNOWN'.

Guarantees:
- When enable_reject_option is False, behavior is 100% identical to raw Production LR.
- Fail-safe: Any runtime exception gracefully falls back to base model prediction.
- Raw probabilities P(J) and P(Z) remain unaltered and normalized.
- Backward-compatible with all existing Pipeline attributes and scikit-learn APIs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ProbabilityMarginModelWrapper:
    """Non-invasive wrapper adding a Probability Margin Reject Option to Production LR."""

    def __init__(
        self,
        base_model: Any,
        enable_reject_option: bool = True,
        margin_threshold: float = 0.10,
    ) -> None:
        if base_model is None:
            raise ValueError("base_model cannot be None")
        self.base_model = base_model
        self.enable_reject_option = bool(enable_reject_option)
        self.margin_threshold = float(margin_threshold)

    @property
    def classes_(self) -> np.ndarray:
        return getattr(self.base_model, "classes_", np.asarray(["J", "Z"]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward raw probabilities directly from the base pipeline."""
        return self.base_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels with reject option applied if enabled."""
        if not self.enable_reject_option:
            if hasattr(self.base_model, "predict"):
                return self.base_model.predict(X)
            probs = np.asarray(self.base_model.predict_proba(X), dtype=np.float64)
            return self.classes_[np.argmax(probs, axis=-1)]

        try:
            arr = np.asarray(X, dtype=np.float64)
            probs = np.asarray(self.base_model.predict_proba(arr), dtype=np.float64)
            if probs.ndim == 1:
                probs = probs.reshape(1, -1)
            if hasattr(self.base_model, "predict"):
                raw_preds = self.base_model.predict(arr)
            else:
                raw_preds = self.classes_[np.argmax(probs, axis=-1)]
            preds = np.array(raw_preds, dtype=object)

            if probs.shape[1] >= 2:
                margins = np.abs(probs[:, 0] - probs[:, 1])
                rejected_mask = margins < self.margin_threshold
                preds[rejected_mask] = "UNKNOWN"

            return preds
        except Exception as exc:
            logger.warning(
                "ProbabilityMarginModelWrapper fallback triggered due to error: %s",
                exc,
                exc_info=True,
            )
            if hasattr(self.base_model, "predict"):
                return self.base_model.predict(X)
            probs = np.asarray(self.base_model.predict_proba(X), dtype=np.float64)
            return self.classes_[np.argmax(probs, axis=-1)]

    def predict_sample_detail(
        self, sample_features: np.ndarray
    ) -> Dict[str, Any]:
        """Detailed inference inspection for a single 1104-dim or (16, 69) feature array."""
        flat = np.asarray(sample_features, dtype=np.float64).reshape(1, -1)
        probs = np.asarray(self.base_model.predict_proba(flat), dtype=np.float64)[0]
        prob_j = float(probs[0])
        prob_z = float(probs[1])
        margin = float(abs(prob_j - prob_z))
        base_pred = str(self.classes_[int(np.argmax(probs))])

        if self.enable_reject_option and margin < self.margin_threshold:
            pred = "UNKNOWN"
            is_rejected = True
            reason = f"MARGIN_BELOW_THRESHOLD: {margin:.4f} < {self.margin_threshold:.4f}"
        else:
            pred = base_pred
            is_rejected = False
            reason = ""

        return {
            "prediction": pred,
            "base_prediction": base_pred,
            "probability_J": prob_j,
            "probability_Z": prob_z,
            "margin": margin,
            "is_rejected": is_rejected,
            "rejection_reason": reason,
            "enable_reject_option": self.enable_reject_option,
            "margin_threshold": self.margin_threshold,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "enable_reject_option": self.enable_reject_option,
            "margin_threshold": self.margin_threshold,
        }

    def set_config(
        self,
        enable_reject_option: Optional[bool] = None,
        margin_threshold: Optional[float] = None,
    ) -> None:
        if enable_reject_option is not None:
            self.enable_reject_option = bool(enable_reject_option)
        if margin_threshold is not None:
            self.margin_threshold = float(margin_threshold)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)
