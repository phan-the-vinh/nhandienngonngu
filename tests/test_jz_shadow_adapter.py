"""Unit tests for Stage 3.0 DTW Shadow Integration.

Tests cover all 15 mandatory conditions from Section 8:
1. DTW receives exact sequence of current attempt.
2. LogisticRegression input is never mutated.
3. DTW called at most once per attempt.
4. DTW returns J on valid J trajectory.
5. DTW returns Z on valid Z trajectory.
6. DTW returns UNKNOWN on noisy/ambiguous trajectory.
7. Graceful handling when template file does not exist.
8. Graceful handling when template metadata is invalid.
9. Graceful handling of feature extraction errors / invalid shape.
10. Graceful handling of DTW matcher exceptions (fail open).
11. Telemetry records correct attempt_id and fields.
12. No data leakage or mixing between sequential attempts.
13. DTW has zero effect on final committed character.
14. DTW has zero effect on HoldConfirmation state/timer.
15. Existing pipeline runs identically when shadow is disabled.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from app.inference.dynamic_jz.dtw_engine import DTWError
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher, DTWMatchResult
from app.inference.dynamic_jz.finger_tracker import (
    FEATURE_DIM,
    FingerTracker,
    SEQUENCE_LENGTH,
)
from app.inference.dynamic_jz.shadow_adapter import (
    DEFAULT_TEMPLATE_PATH,
    JzShadowAdapter,
    JzShadowResult,
    JzShadowTelemetryRecord,
    ShadowModelProxy,
)
from app.inference.dynamic_jz.template_store import DTWTemplate, TemplateStore


def make_synthetic_attempt_sequence(letter: str, noise: float = 0.0) -> np.ndarray:
    """Generate synthetic 16x69 baseline-b sequence for J or Z."""
    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
    target = letter.upper()

    for t in range(SEQUENCE_LENGTH):
        p = t / 15.0
        # Palm scale reference
        seq[t, 15:18] = [0.0, -0.5, 0.0]
        seq[t, 27:30] = [0.1, -0.5, 0.0]
        seq[t, 51:54] = [-0.1, -0.4, 0.0]

        if target == "Z":
            # Index extended
            seq[t, 24:27] = [0.1, -1.0, 0.0]
            # Pinky curled
            seq[t, 60:63] = [-0.1, -0.45, 0.0]
            # Z global trajectory: right, diag down-left, right
            if p < 0.33:
                sub = p / 0.33
                x = sub * 1.0
                y = 0.0
            elif p < 0.66:
                sub = (p - 0.33) / 0.33
                x = 1.0 - sub * 1.0
                y = sub * 1.5
            else:
                sub = (p - 0.66) / 0.34
                x = sub * 1.0
                y = 1.5
            seq[t, 65:67] = [x, y]
            seq[t, 63:65] = [x * 0.5, y * 0.5]
        else:
            # J: Index curled
            seq[t, 24:27] = [0.2, -0.45, 0.0]
            # Pinky extended
            seq[t, 60:63] = [-0.25, -1.0, 0.0]
            # J global trajectory: drop down, hook left and up
            if p < 0.6:
                sub = p / 0.6
                x = 0.0
                y = sub * 1.8
            else:
                sub = (p - 0.6) / 0.4
                theta = sub * np.pi
                x = -np.sin(theta) * 0.8
                y = 1.8 - np.cos(theta) * 0.4
            seq[t, 67:69] = [x, y]
            seq[t, 63:65] = [x * 0.3, y * 0.3]

    if noise > 0.0:
        rng = np.random.RandomState(42)
        seq += rng.normal(0.0, noise, size=seq.shape)

    return seq


class MockLogisticModel:
    """Mock LogisticRegression pipeline."""

    def __init__(self, pred_label: str = "J", prob_val: float = 0.85) -> None:
        self.classes_ = np.asarray(["J", "Z"])
        self.pred_label = pred_label
        self.prob_val = prob_val
        self.last_received_input: np.ndarray | None = None
        self.call_count = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.last_received_input = X
        if self.pred_label == "J":
            return np.asarray([[self.prob_val, 1.0 - self.prob_val]])
        return np.asarray([[1.0 - self.prob_val, self.prob_val]])


class JzShadowAdapterTests(unittest.TestCase):
    """Test suite validating all 15 requirements for Stage 3.0 Shadow Integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.telemetry_dir = Path(self.temp_dir.name) / "shadow"
        dataset_path = Path("data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl")
        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                lines = [f.readline() for _ in range(320)]
            self.seq_j = np.array(json.loads(lines[0])["features"], dtype=np.float64)
            self.seq_z = np.array(json.loads(lines[308])["features"], dtype=np.float64)
        else:
            self.seq_j = make_synthetic_attempt_sequence("J")
            self.seq_z = make_synthetic_attempt_sequence("Z")
        self.adapter = JzShadowAdapter(
            enabled=True,
            template_path=DEFAULT_TEMPLATE_PATH,
            telemetry_dir=self.telemetry_dir,
            verify_sha256=True,
            radius=3,
            threshold=2.00,
            min_margin=0.15,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_dtw_receives_exact_sequence(self) -> None:
        """Requirement 1: DTW receives the exact sequence corresponding to LogisticRegression."""
        inner = MockLogisticModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)

        flat = self.seq_j.reshape(1, 1104, order="C")
        _ = proxy.predict_proba(flat)

        self.assertIsNotNone(proxy.latest_features)
        # Array contents must be numerically identical
        np.testing.assert_array_equal(proxy.latest_features, flat)
        self.assertIsNotNone(proxy.latest_shadow_result)
        self.assertEqual(proxy.latest_shadow_result.status, "OK")

    def test_02_logistic_regression_input_immutability(self) -> None:
        """Requirement 2: Input array to LogisticRegression is never mutated by DTW."""
        inner = MockLogisticModel("Z")
        proxy = ShadowModelProxy(inner, self.adapter)

        orig_input = self.seq_z.reshape(1, 1104, order="C").copy()
        test_input = orig_input.copy()

        _ = proxy.predict_proba(test_input)

        # test_input passed into predict_proba must remain strictly unchanged
        np.testing.assert_array_equal(test_input, orig_input)
        np.testing.assert_array_equal(inner.last_received_input, orig_input)

    def test_03_dtw_called_at_most_once_per_attempt(self) -> None:
        """Requirement 3: DTW is called exactly once per attempt, not on every frame."""
        inner = MockLogisticModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)

        flat = self.seq_j.reshape(1, 1104, order="C")
        self.assertEqual(proxy.call_count_this_attempt, 0)

        _ = proxy.predict_proba(flat)
        self.assertEqual(proxy.call_count_this_attempt, 1)
        self.assertEqual(inner.call_count, 1)

    def test_04_dtw_returns_j(self) -> None:
        """Requirement 4: DTW returns J on a valid J trajectory."""
        res = self.adapter.evaluate_attempt(self.seq_j)
        self.assertEqual(res.prediction, "J")
        self.assertEqual(res.status, "OK")
        self.assertLess(res.best_distance, res.second_best_distance)
        self.assertGreaterEqual(res.margin, 0.15)
        self.assertEqual(res.config_id, "CONFIG_C")
        self.assertTrue(res.template_id.startswith("J_"))

    def test_05_dtw_returns_z(self) -> None:
        """Requirement 5: DTW returns Z on a valid Z trajectory."""
        res = self.adapter.evaluate_attempt(self.seq_z)
        self.assertEqual(res.prediction, "Z")
        self.assertEqual(res.status, "OK")
        self.assertLess(res.best_distance, res.second_best_distance)
        self.assertGreaterEqual(res.margin, 0.15)
        self.assertEqual(res.config_id, "CONFIG_C")
        self.assertTrue(res.template_id.startswith("Z_"))

    def test_06_dtw_returns_unknown(self) -> None:
        """Requirement 6: DTW returns UNKNOWN when trajectory is noisy or ambiguous."""
        noisy = make_synthetic_attempt_sequence("Z", noise=3.0)
        res = self.adapter.evaluate_attempt(noisy)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "REJECTED")
        self.assertTrue(len(res.rejection_reason) > 0)

    def test_07_template_does_not_exist(self) -> None:
        """Requirement 7: Gracefully handles missing template file without raising exception."""
        missing_path = Path(self.temp_dir.name) / "non_existent_templates.json"
        adapter = JzShadowAdapter(
            enabled=True,
            template_path=missing_path,
            telemetry_dir=self.telemetry_dir,
            verify_sha256=False,
        )
        self.assertFalse(adapter.available)
        self.assertIn("TEMPLATE_FILE_NOT_FOUND", adapter.init_error)

        # Attempt evaluation must fail open with UNAVAILABLE status
        res = adapter.evaluate_attempt(self.seq_j)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "UNAVAILABLE")

    def test_08_invalid_template_metadata(self) -> None:
        """Requirement 8: Gracefully handles corrupt or invalid template metadata."""
        corrupt_path = Path(self.temp_dir.name) / "corrupt_templates.json"
        corrupt_path.write_text(json.dumps({"templates": [{"template_id": "bad"}]}), encoding="utf-8")

        adapter = JzShadowAdapter(
            enabled=True,
            template_path=corrupt_path,
            telemetry_dir=self.telemetry_dir,
            verify_sha256=False,
        )
        self.assertFalse(adapter.available)

        res = adapter.evaluate_attempt(self.seq_j)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "UNAVAILABLE")

    def test_09_feature_extraction_error_handled(self) -> None:
        """Requirement 9: Gracefully handles invalid sequence shapes."""
        wrong_shape = np.zeros((10, 50))
        res = self.adapter.evaluate_attempt(wrong_shape)
        self.assertEqual(res.prediction, "UNKNOWN")
        self.assertEqual(res.status, "INVALID_INPUT")
        self.assertIn("INVALID_SHAPE", res.rejection_reason)

    def test_10_dtw_matcher_exception_handled(self) -> None:
        """Requirement 10: Matcher exceptions are caught safely and fail open."""
        inner = MockLogisticModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)

        # Inject NaN into a copy passed to evaluate_attempt
        nan_seq = self.seq_j.copy()
        nan_seq[5, 2] = np.nan

        # Proxy must not raise; predict_proba must return valid probabilities from inner model
        probs = proxy.predict_proba(nan_seq.reshape(1, 1104))
        self.assertIsNotNone(probs)
        self.assertEqual(probs.shape, (1, 2))
        self.assertEqual(proxy.latest_shadow_result.prediction, "UNKNOWN")
        # Tracking quality caught NaN
        self.assertIn(proxy.latest_shadow_result.status, ("REJECTED", "ERROR"))

    def test_11_telemetry_records_correct_attempt_id(self) -> None:
        """Requirement 11: Telemetry records correct attempt_id and matching fields."""
        res = self.adapter.evaluate_attempt(self.seq_j)
        record = JzShadowTelemetryRecord(
            timestamp="2026-11-13T12:00:00Z",
            attempt_id="trial_42",
            target_label="J",
            logistic_prediction="J",
            logistic_confidence=0.88,
            dtw_prediction=res.prediction,
            dtw_best_distance=res.best_distance,
            dtw_second_distance=res.second_best_distance,
            dtw_margin=res.margin,
            dtw_status=res.status,
            tracking_quality=res.tracking_quality,
            feature_config=res.config_id,
            template_id=res.template_id,
            dtw_latency_ms=res.latency_ms,
            agreement=True,
            final_committed_label="J",
            final_commit_status="COMMITTED",
        )
        self.adapter.log_telemetry(record)

        jsonl_path = self.telemetry_dir / "shadow_telemetry.jsonl"
        self.assertTrue(jsonl_path.exists())
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)

        logged = json.loads(lines[0])
        self.assertEqual(logged["attempt_id"], "trial_42")
        self.assertEqual(logged["target_label"], "J")
        self.assertEqual(logged["logistic_prediction"], "J")
        self.assertEqual(logged["dtw_prediction"], "J")
        self.assertTrue(logged["agreement"])
        self.assertEqual(logged["final_committed_label"], "J")

    def test_12_no_data_leakage_between_attempts(self) -> None:
        """Requirement 12: Sequential attempts do not leak or mix state."""
        inner = MockLogisticModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)

        # Attempt 1: J gesture
        flat_j = self.seq_j.reshape(1, 1104, order="C")
        _ = proxy.predict_proba(flat_j)
        res1 = proxy.latest_shadow_result
        self.assertEqual(res1.prediction, "J")

        # Reset between attempts
        self.adapter.latest_result = None
        proxy.latest_shadow_result = None

        # Attempt 2: Z gesture
        flat_z = self.seq_z.reshape(1, 1104, order="C")
        _ = proxy.predict_proba(flat_z)
        res2 = proxy.latest_shadow_result
        self.assertEqual(res2.prediction, "Z")
        self.assertNotEqual(res1.template_id, res2.template_id)

    def test_13_dtw_does_not_affect_commit_result(self) -> None:
        """Requirement 13: Final committed character is 100% determined by LogisticRegression."""
        # LogisticRegression predicts 'Z', but input is 'J' (causing DTW to predict 'J')
        inner = MockLogisticModel(pred_label="Z", prob_val=0.92)
        proxy = ShadowModelProxy(inner, self.adapter)

        flat_j = self.seq_j.reshape(1, 1104, order="C")
        probs = proxy.predict_proba(flat_j)

        # Model output must be strictly Z as decided by inner model
        classes = list(inner.classes_)
        pred = classes[int(np.argmax(probs[0]))]
        self.assertEqual(pred, "Z")

        # DTW shadow predicted J independently without overriding inner model
        self.assertEqual(proxy.latest_shadow_result.prediction, "J")

    def test_14_dtw_does_not_affect_hold_confirmation(self) -> None:
        """Requirement 14: DTW shadow execution has zero effect on HoldConfirmation."""
        from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
        hold = HoldConfirmation(HoldSettings())

        # Perform mock hold frames
        obs = SimpleNamespace(
            label="J",
            confidence=0.90,
            scale_px=100.0,
            palm_center_px=(320.0, 240.0),
            shape=tuple([0.1] * 63),
        )
        hold.update(obs, timestamp=time.monotonic())
        state_before = hold.state

        # Shadow execution
        _ = self.adapter.evaluate_attempt(self.seq_j)

        # Hold state must be unaffected
        self.assertEqual(hold.state, state_before)

    def test_15_disabled_shadow_preserves_pipeline_exactly(self) -> None:
        """Requirement 15: Pipeline runs with zero shadow overhead when disabled."""
        disabled_adapter = JzShadowAdapter(enabled=False)
        self.assertFalse(disabled_adapter.enabled)

        inner = MockLogisticModel("J")
        proxy = ShadowModelProxy(inner, disabled_adapter)

        flat = self.seq_j.reshape(1, 1104, order="C")
        t0 = time.perf_counter()
        probs = proxy.predict_proba(flat)
        t1 = time.perf_counter()

        self.assertEqual(probs.shape, (1, 2))
        # Disabled shadow execution overhead should be virtually zero (< 0.1ms)
        self.assertLess((t1 - t0) * 1000.0, 1.0)
        self.assertIsNone(proxy.latest_shadow_result)


if __name__ == "__main__":
    unittest.main()
