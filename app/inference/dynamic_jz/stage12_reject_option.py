"""Stage 12: Reject Option Architecture for Real-World Dynamic J/Z Recognition.

Provides external decision-rejection wrappers around Production Logistic Regression (LR)
without modifying model weights or retraining checkpoints.

Configurations evaluated:
- Config A: Baseline LR (Argmax, no rejection)
- Config B: Confidence Threshold (Reject if max(P) < theta_conf)
- Config C: Probability Margin (Reject if |P(J) - P(Z)| < theta_margin)
- Config D: Confidence + Margin (Reject if max(P) < theta_conf or margin < theta_margin)
- Config E: Input Quality Rejection (Reject if motion < 0.15 palm scale or duplicate ratio > 0.40)
- Config F: Handshape Consistency Guard (Reject if predicted finger is curled/inconsistent)
- Config G: Best Combined Reject Option (Consistency Guard + Conf/Margin + Quality)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np


@dataclass(frozen=True)
class RejectOptionResult:
    """Output contract for Stage 12 Reject Option."""
    prediction: str                       # 'J', 'Z', or 'UNKNOWN'
    confidence: float                     # Confidence in [0.0, 1.0]
    prob_j: float                         # P(J)
    prob_z: float                         # P(Z)
    margin: float                         # |P(J) - P(Z)|
    is_rejected: bool                     # True if returned UNKNOWN
    rejection_reason: str                 # Reason code if rejected, else ""
    latency_ms: float = 0.0


class ProductionLRRejectOptionClassifier:
    """Wrapper around Production Logistic Regression with configurable Reject Option rules."""

    def __init__(
        self,
        model_pipeline_path: str | Path,
        mode: str = "baseline",
        conf_threshold: float = 0.60,
        margin_threshold: float = 0.05,
        min_motion_amplitude: float = 0.15,
        max_duplicate_ratio: float = 0.45,
        min_finger_extension: float = 0.70,
    ) -> None:
        self.mode = mode.lower()
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.min_motion_amplitude = min_motion_amplitude
        self.max_duplicate_ratio = max_duplicate_ratio
        self.min_finger_extension = min_finger_extension

        pipeline_file = Path(model_pipeline_path)
        if not pipeline_file.exists():
            raise FileNotFoundError(f"Model pipeline not found at {pipeline_file}")
        self.pipeline = joblib.load(pipeline_file)

    def extract_kinematics_and_handshape(
        self, sequence: np.ndarray
    ) -> Dict[str, float]:
        """Extract motion amplitudes and finger extension ratios from (16, 69) features."""
        # 0..62: 21 landmarks x 3
        # 63..64: wrist (x, y)
        # 65..66: index tip (x, y)
        # 67..68: pinky tip (x, y)
        lms = sequence[:, :63].reshape(16, 21, 3)
        palm_scales = np.linalg.norm(lms[:, 0, :] - lms[:, 9, :], axis=1)
        scale = max(float(np.median(palm_scales)), 1e-6)

        idx_ext = float(np.median(np.linalg.norm(lms[:, 8, :] - lms[:, 0, :], axis=1)) / scale)
        pky_ext = float(np.median(np.linalg.norm(lms[:, 20, :] - lms[:, 0, :], axis=1)) / scale)

        idx_path = float(np.sum(np.linalg.norm(np.diff(sequence[:, 65:67], axis=0), axis=1)))
        pky_path = float(np.sum(np.linalg.norm(np.diff(sequence[:, 67:69], axis=0), axis=1)))
        wrist_disp = float(np.linalg.norm(sequence[-1, 63:65] - sequence[0, 63:65]))

        return {
            "index_ext": idx_ext,
            "pinky_ext": pky_ext,
            "index_path": idx_path,
            "pinky_path": pky_path,
            "wrist_disp": wrist_disp,
        }

    def classify(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        duplicate_ratio: float = 0.0,
    ) -> RejectOptionResult:
        """Run classification with the active reject option configuration."""
        t0 = time.perf_counter()
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape != (16, 69):
            lat = (time.perf_counter() - t0) * 1000.0
            return RejectOptionResult(
                prediction="UNKNOWN",
                confidence=0.0,
                prob_j=0.0,
                prob_z=0.0,
                margin=0.0,
                is_rejected=True,
                rejection_reason=f"INVALID_SHAPE: Expected (16, 69), got {arr.shape}",
                latency_ms=lat,
            )

        flat = arr.reshape(1, -1)
        probs = self.pipeline.predict_proba(flat)[0]
        prob_j = float(probs[0])
        prob_z = float(probs[1])
        base_pred = "J" if prob_j >= prob_z else "Z"
        conf = max(prob_j, prob_z)
        margin = abs(prob_j - prob_z)

        kin = self.extract_kinematics_and_handshape(arr)

        prediction = base_pred
        is_rejected = False
        reason = ""

        # Apply Reject Rules according to mode
        if self.mode == "baseline":
            # Config A: Pure LR baseline
            pass

        elif self.mode == "confidence":
            # Config B: Reject if confidence < threshold
            if conf < self.conf_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_CONFIDENCE: {conf:.4f} < {self.conf_threshold:.2f}"

        elif self.mode == "margin":
            # Config C: Reject if probability margin < threshold
            if margin < self.margin_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_MARGIN: {margin:.4f} < {self.margin_threshold:.2f}"

        elif self.mode == "confidence_margin":
            # Config D: Combined confidence + margin
            if conf < self.conf_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_CONFIDENCE: {conf:.4f} < {self.conf_threshold:.2f}"
            elif margin < self.margin_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_MARGIN: {margin:.4f} < {self.margin_threshold:.2f}"

        elif self.mode == "quality":
            # Config E: Quality / Motion rejection
            max_motion = max(kin["index_path"], kin["pinky_path"])
            if max_motion < self.min_motion_amplitude:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"INSUFFICIENT_MOTION: {max_motion:.3f} < {self.min_motion_amplitude:.2f}"
            elif duplicate_ratio > self.max_duplicate_ratio:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"HIGH_DUPLICATE_RATIO: {duplicate_ratio:.3f} > {self.max_duplicate_ratio:.2f}"

        elif self.mode == "consistency":
            # Config F: Handshape Consistency Guard
            if base_pred == "J":
                if kin["pinky_ext"] < self.min_finger_extension or kin["pinky_ext"] <= kin["index_ext"]:
                    prediction = "UNKNOWN"
                    is_rejected = True
                    reason = (
                        f"INCONSISTENT_HANDSHAPE_FOR_J: PinkyExt={kin['pinky_ext']:.2f} "
                        f"(IdxExt={kin['index_ext']:.2f})"
                    )
            elif base_pred == "Z":
                if kin["index_ext"] < self.min_finger_extension or kin["index_ext"] <= kin["pinky_ext"]:
                    prediction = "UNKNOWN"
                    is_rejected = True
                    reason = (
                        f"INCONSISTENT_HANDSHAPE_FOR_Z: IdxExt={kin['index_ext']:.2f} "
                        f"(PkyExt={kin['pinky_ext']:.2f})"
                    )

        elif self.mode == "combined":
            # Config G: Best Combined Reject Option
            max_motion = max(kin["index_path"], kin["pinky_path"])
            if max_motion < self.min_motion_amplitude:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"INSUFFICIENT_MOTION: {max_motion:.3f} < {self.min_motion_amplitude:.2f}"
            elif duplicate_ratio > self.max_duplicate_ratio:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"HIGH_DUPLICATE_RATIO: {duplicate_ratio:.3f} > {self.max_duplicate_ratio:.2f}"
            elif conf < self.conf_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_CONFIDENCE: {conf:.4f} < {self.conf_threshold:.2f}"
            elif margin < self.margin_threshold:
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"LOW_MARGIN: {margin:.4f} < {self.margin_threshold:.2f}"
            elif base_pred == "J" and (kin["pinky_ext"] < self.min_finger_extension or kin["pinky_ext"] <= kin["index_ext"]):
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"INCONSISTENT_HANDSHAPE_FOR_J: PinkyExt={kin['pinky_ext']:.2f} (IdxExt={kin['index_ext']:.2f})"
            elif base_pred == "Z" and (kin["index_ext"] < self.min_finger_extension or kin["index_ext"] <= kin["pinky_ext"]):
                prediction = "UNKNOWN"
                is_rejected = True
                reason = f"INCONSISTENT_HANDSHAPE_FOR_Z: IdxExt={kin['index_ext']:.2f} (PkyExt={kin['pinky_ext']:.2f})"

        else:
            raise ValueError(f"Unknown reject option mode: {self.mode}")

        lat = (time.perf_counter() - t0) * 1000.0

        return RejectOptionResult(
            prediction=prediction,
            confidence=conf,
            prob_j=prob_j,
            prob_z=prob_z,
            margin=margin,
            is_rejected=is_rejected,
            rejection_reason=reason,
            latency_ms=lat,
        )
