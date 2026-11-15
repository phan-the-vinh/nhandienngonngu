"""Stage 3.0 — DTW Shadow Adapter for Dynamic J/Z Gestures.

Provides non-invasive shadow execution of the Multivariate DTW Engine alongside
LogisticRegression in the unified recognition pipeline:
- Strictly observational: never overrides LogisticRegression, never commits text, never changes HoldConfirmation.
- Receives the exact same (16, 69) sequence as LogisticRegression.
- Enforces audited CONFIG_C (9D), VALID_ONLY, K=2 (4 templates), R=3, Theta_reject=2.00, Delta_margin=0.15.
- Controlled by feature flag 'dynamic_jz_dtw_shadow_enabled' (default: False).
- Completely fail-open: any DTW or telemetry failure is isolated and cannot impact production recognition.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from app.inference.dynamic_jz.dtw_engine import DTWError
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher, DTWMatchResult
from app.inference.dynamic_jz.feature_builder import build_dtw_features
from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    FEATURE_VERSION,
    FingerTracker,
    FingerTrackingError,
    FingerTrackingResult,
    SEQUENCE_LENGTH,
)
from app.inference.dynamic_jz.template_store import DTWTemplate, TemplateStore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_PATH = ROOT / "models/dynamic_jz/dtw_templates/templates_config_c_valid_only_k2.json"
EXPECTED_TEMPLATE_SHA256 = "7ac0d6c3bb22a9179a6e63a1fea88c0787a6cc18699e0f89915b395f13377ed1"
DEFAULT_TELEMETRY_DIR = ROOT / "data/experiments/dynamic_jz_v1/shadow"
FEATURE_FLAG_ENV_VAR = "DYNAMIC_JZ_DTW_SHADOW_ENABLED"


@dataclass(frozen=True)
class JzShadowResult:
    """Output contract of the DTW Shadow evaluation."""
    prediction: str                      # 'J', 'Z', or 'UNKNOWN'
    best_distance: float                 # Minimum distance to winning template
    second_best_distance: float          # Minimum distance to competing class
    margin: float                        # |best_distance - second_best_distance|
    tracking_quality: str                # 'VALID', 'SUSPICIOUS', or 'INVALID'
    config_id: str                       # 'CONFIG_C'
    template_id: str                     # Winning template ID or ''
    latency_ms: float                    # Execution time in milliseconds
    status: str                          # 'OK', 'REJECTED', 'INVALID_INPUT', 'ERROR', 'UNAVAILABLE', 'DISABLED'
    rejection_reason: str = ""           # Reason if prediction is UNKNOWN or status is REJECTED
    error_message: str = ""              # Exception message if status is ERROR


@dataclass
class JzShadowTelemetryRecord:
    """Per-attempt comparative telemetry record between LogisticRegression and DTW Shadow."""
    timestamp: str                       # ISO 8601 UTC
    attempt_id: Any                      # Attempt ID or trial index
    target_label: str                    # Target label if supplied by harness, else 'NONE'
    logistic_prediction: str             # Prediction from LogisticRegression ('J', 'Z', 'NONE')
    logistic_confidence: float | None    # Confidence from LogisticRegression
    dtw_prediction: str                  # Prediction from DTW ('J', 'Z', 'UNKNOWN', 'DISABLED')
    dtw_best_distance: float             # DTW winning distance
    dtw_second_distance: float           # DTW losing distance
    dtw_margin: float                    # DTW separation margin
    dtw_status: str                      # 'OK', 'REJECTED', 'INVALID_INPUT', 'ERROR', etc.
    tracking_quality: str                # 'VALID', 'SUSPICIOUS', 'INVALID'
    feature_config: str                  # 'CONFIG_C'
    template_id: str                     # DTW winning template ID
    dtw_latency_ms: float                # DTW execution latency in ms
    agreement: bool                      # Whether logistic_prediction == dtw_prediction
    final_committed_label: str           # Label committed to composer or 'NONE'
    final_commit_status: str             # 'COMMITTED', 'CANCELLED', 'REJECTED', 'PENDING'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JzShadowAdapter:
    """Non-invasive shadow adapter executing Multivariate DTW alongside LogisticRegression."""

    def __init__(
        self,
        enabled: bool | None = None,
        template_path: Path | str | None = None,
        telemetry_dir: Path | str | None = None,
        verify_sha256: bool = True,
        radius: int = 3,
        threshold: float = 2.00,
        min_margin: float = 0.15,
        timeout_ms: float = 25.0,
    ) -> None:
        # 1. Resolve feature flag
        if enabled is None:
            flag_val = os.environ.get(FEATURE_FLAG_ENV_VAR, "").strip().lower()
            self.enabled = flag_val in ("1", "true", "yes", "on")
        else:
            self.enabled = bool(enabled)

        self.template_path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
        self.telemetry_dir = Path(telemetry_dir) if telemetry_dir else DEFAULT_TELEMETRY_DIR
        self.verify_sha256 = bool(verify_sha256)
        self.radius = int(radius)
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.timeout_ms = float(timeout_ms)

        self.available: bool = False
        self.init_error: str = ""
        self.store: TemplateStore | None = None
        self.matcher: DTWMatcher | None = None
        self.tracker: FingerTracker | None = None
        self.latest_result: JzShadowResult | None = None

        # 2. Template verification and initialization (only if enabled)
        if self.enabled:
            self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Safely load and verify audited DTW templates without raising exceptions."""
        try:
            if not self.template_path.exists():
                self.available = False
                self.init_error = f"TEMPLATE_FILE_NOT_FOUND: {self.template_path}"
                return

            # Checksum verification
            if self.verify_sha256 and self.template_path == DEFAULT_TEMPLATE_PATH:
                file_hash = hashlib.sha256(self.template_path.read_bytes()).hexdigest()
                if file_hash != EXPECTED_TEMPLATE_SHA256:
                    self.available = False
                    self.init_error = (
                        f"TEMPLATE_CHECKSUM_MISMATCH: expected {EXPECTED_TEMPLATE_SHA256}, got {file_hash}"
                    )
                    return

            self.store = TemplateStore.load_json(self.template_path)

            # Metadata validation
            j_tmpls = self.store.get_by_label("J")
            z_tmpls = self.store.get_by_label("Z")
            if not j_tmpls or not z_tmpls:
                self.available = False
                self.init_error = f"MISSING_CLASSES_IN_TEMPLATE_STORE: J={len(j_tmpls)}, Z={len(z_tmpls)}"
                return

            for t in self.store.templates:
                if t.feature_config != "CONFIG_C":
                    self.available = False
                    self.init_error = f"INVALID_TEMPLATE_CONFIG: expected CONFIG_C, got {t.feature_config}"
                    return
                if t.source_split != "train":
                    self.available = False
                    self.init_error = f"INVALID_SOURCE_SPLIT: expected train, got {t.source_split}"
                    return

            self.tracker = FingerTracker()
            self.matcher = DTWMatcher(
                self.store,
                config_name="CONFIG_C",
                radius=self.radius,
                threshold_j=self.threshold,
                threshold_z=self.threshold,
                min_margin=self.min_margin,
            )
            self.available = True
            self.init_error = ""

        except Exception as exc:
            self.available = False
            self.init_error = f"INIT_EXCEPTION: {exc}"

    def evaluate_attempt(
        self,
        sequence: Sequence[Sequence[float]] | np.ndarray,
        selected_original_frame_indices: list[int] | None = None,
    ) -> JzShadowResult:
        """Run shadow DTW classification on the exact sequence used by LogisticRegression.

        Parameters
        ----------
        sequence : array_like of shape (16, 69) or flattened (1, 1104)
            Pre-extracted baseline-b feature sequence.
        selected_original_frame_indices : list[int], optional
            Original capture frame indices sampled for this sequence.

        Returns
        -------
        JzShadowResult
            Independent shadow evaluation result.
        """
        # 1. Guard against disabled or unavailable adapter
        if not self.enabled:
            res = JzShadowResult(
                prediction="UNKNOWN",
                best_distance=float("inf"),
                second_best_distance=float("inf"),
                margin=0.0,
                tracking_quality="INVALID",
                config_id="CONFIG_C",
                template_id="",
                latency_ms=0.0,
                status="DISABLED",
                rejection_reason="FEATURE_FLAG_DISABLED",
            )
            self.latest_result = res
            return res

        if not self.available:
            res = JzShadowResult(
                prediction="UNKNOWN",
                best_distance=float("inf"),
                second_best_distance=float("inf"),
                margin=0.0,
                tracking_quality="INVALID",
                config_id="CONFIG_C",
                template_id="",
                latency_ms=0.0,
                status="UNAVAILABLE",
                rejection_reason=self.init_error,
            )
            self.latest_result = res
            return res

        # 2. Defensive copy & shape normalization (never mutate input array)
        t0 = time.perf_counter()
        try:
            arr = np.asarray(sequence, dtype=np.float64).copy()
            if arr.ndim == 2 and arr.shape == (1, SEQUENCE_LENGTH * FEATURE_DIM):
                arr = arr.reshape((SEQUENCE_LENGTH, FEATURE_DIM), order="C")
            elif arr.ndim == 1 and arr.shape == (SEQUENCE_LENGTH * FEATURE_DIM,):
                arr = arr.reshape((SEQUENCE_LENGTH, FEATURE_DIM), order="C")

            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                t1 = time.perf_counter()
                res = JzShadowResult(
                    prediction="UNKNOWN",
                    best_distance=float("inf"),
                    second_best_distance=float("inf"),
                    margin=0.0,
                    tracking_quality="INVALID",
                    config_id="CONFIG_C",
                    template_id="",
                    latency_ms=(t1 - t0) * 1000.0,
                    status="INVALID_INPUT",
                    rejection_reason=f"INVALID_SHAPE: expected ({SEQUENCE_LENGTH}, {FEATURE_DIM}), got {arr.shape}",
                )
                self.latest_result = res
                return res

            # 3. Dual-Hypothesis Feature Extraction
            res_j = self.tracker.extract(
                arr, letter_target="J", selected_original_frame_indices=selected_original_frame_indices
            )
            res_z = self.tracker.extract(
                arr, letter_target="Z", selected_original_frame_indices=selected_original_frame_indices
            )

            # 4. DTW Matching
            match_res: DTWMatchResult = self.matcher.match_dual(res_j, res_z)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0

            # 5. Build Result
            pred = match_res.predicted_label
            if pred == "J":
                best_d = match_res.distance_j
                second_d = match_res.distance_z
            elif pred == "Z":
                best_d = match_res.distance_z
                second_d = match_res.distance_j
            else:
                best_d = min(match_res.distance_j, match_res.distance_z)
                second_d = max(match_res.distance_j, match_res.distance_z)

            status = "OK" if pred in ("J", "Z") else "REJECTED"

            res = JzShadowResult(
                prediction=pred,
                best_distance=float(best_d),
                second_best_distance=float(second_d),
                margin=float(match_res.distance_margin),
                tracking_quality=match_res.tracking_quality,
                config_id="CONFIG_C",
                template_id=match_res.best_template_id,
                latency_ms=float(latency),
                status=status,
                rejection_reason=match_res.rejection_reason,
                error_message="",
            )
            self.latest_result = res
            return res

        except Exception as exc:
            t1 = time.perf_counter()
            res = JzShadowResult(
                prediction="UNKNOWN",
                best_distance=float("inf"),
                second_best_distance=float("inf"),
                margin=0.0,
                tracking_quality="INVALID",
                config_id="CONFIG_C",
                template_id="",
                latency_ms=(t1 - t0) * 1000.0,
                status="ERROR",
                rejection_reason=f"EXCEPTION: {exc}",
                error_message=str(exc),
            )
            self.latest_result = res
            return res

    def log_telemetry(self, record: JzShadowTelemetryRecord | dict[str, Any]) -> None:
        """Safely write telemetry to disk; filesystem errors are isolated and never crash inference."""
        if not self.enabled:
            return

        try:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            data = record.to_dict() if isinstance(record, JzShadowTelemetryRecord) else dict(record)

            # 1. Append to JSONL
            jsonl_file = self.telemetry_dir / "shadow_telemetry.jsonl"
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")

            # 2. Append to CSV
            csv_file = self.telemetry_dir / "shadow_telemetry.csv"
            exists = csv_file.exists()
            fieldnames = list(data.keys())
            with open(csv_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(data)

        except Exception:
            # Observational telemetry must fail open without impacting camera or UI
            pass


class ShadowModelProxy:
    """Transparent proxy wrapping the dynamic LogisticRegression model.

    Preserves 100% of inner model methods and outputs while running DTW Shadow
    analysis on the exact same (16, 69) input vector.
    """

    def __init__(self, inner_model: Any, shadow_adapter: JzShadowAdapter | None = None) -> None:
        self._inner = inner_model
        self.shadow_adapter = shadow_adapter
        self.latest_shadow_result: JzShadowResult | None = None
        self.latest_features: np.ndarray | None = None
        self.call_count_this_attempt = 0

    @property
    def classes_(self) -> Any:
        return self._inner.classes_

    @property
    def named_steps(self) -> Any:
        return getattr(self._inner, "named_steps", None)

    @property
    def n_features_in_(self) -> int:
        return getattr(self._inner, "n_features_in_", SEQUENCE_LENGTH * FEATURE_DIM)

    def predict_proba(self, X: Any) -> np.ndarray:
        # 1. Defensive copy of input array for shadow analysis
        X_copy = np.asarray(X, dtype=np.float64).copy()

        # 2. Execute inner LogisticRegression model (strictly preserving return value)
        probs = self._inner.predict_proba(X)

        # 3. Non-invasive shadow execution
        if self.shadow_adapter and self.shadow_adapter.enabled:
            try:
                self.call_count_this_attempt += 1
                self.latest_features = X_copy
                self.latest_shadow_result = self.shadow_adapter.evaluate_attempt(X_copy)
            except Exception as exc:
                err_res = JzShadowResult(
                    prediction="UNKNOWN",
                    best_distance=float("inf"),
                    second_best_distance=float("inf"),
                    margin=0.0,
                    tracking_quality="INVALID",
                    config_id="CONFIG_C",
                    template_id="",
                    latency_ms=0.0,
                    status="ERROR",
                    error_message=f"PROXY_SHADOW_ERROR: {exc}",
                )
                self.latest_shadow_result = err_res
                if self.shadow_adapter:
                    self.shadow_adapter.latest_result = err_res

        return probs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
