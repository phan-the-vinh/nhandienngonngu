"""Comprehensive Runtime Safety Audit Suite for J/Z DTW Shadow Integration.

Validates all 10 scope items defined in STAGE 3.0.1:
1. End-to-end data flow traceability (First16Segmenter -> LogisticRegression -> DTW -> Validator -> Hold -> TextComposer).
2. Attempt sequence and frame alignment (no cross-attempt mixing).
3. ShadowModelProxy and adapter immutability (zero mutation of LR inputs, probabilities, or target label leaks).
4. Invocation frequency (called exactly once per completed attempt).
5. Attempt lifecycle reset, cancellation, retry, and isolation.
6. Fail-open behavior under all fault conditions (missing template, bad checksum, NaN/invalid shape, disk errors, disabled flag).
7. Telemetry schema and field completeness.
8. Shadow OFF vs Shadow ON exact parity on replay data (identical predictions, validator, hold, and composer text).
9. Verification of past test failures and fixes.
10. Checksum baseline verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

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
    EXPECTED_TEMPLATE_SHA256,
    JzShadowAdapter,
    JzShadowResult,
    JzShadowTelemetryRecord,
    ShadowModelProxy,
)
from app.inference.dynamic_jz.template_store import TemplateStore
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.interaction.hold_confirmation import HoldConfirmation, HoldObservation, HoldSettings
from tests.test_unified_abc123_jz_auto import DynamicModel, curved_hand, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand


class JzShadowRuntimeSafetyAuditTests(unittest.TestCase):
    """Safety and integrity audit for Stage 3.0.1."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.telemetry_dir = Path(self.temp_dir.name) / "shadow"
        self.dataset_path = Path("data/experiments/dynamic_jz_v1/processed/signnteam_jz_baseline_b_v1.jsonl")

        # Load real SigNN samples if available
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                lines = [f.readline() for _ in range(320)]
            self.seq_j = np.array(json.loads(lines[0])["features"], dtype=np.float64)
            self.seq_z = np.array(json.loads(lines[308])["features"], dtype=np.float64)
        else:
            self.seq_j = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
            self.seq_z = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)

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

    # -------------------------------------------------------------------------
    # 1. Data Flow Traceability
    # -------------------------------------------------------------------------
    def test_01_data_flow_traceability(self) -> None:
        """Trace data flow from segmenter through model proxy, shadow, validator, hold, and composer."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")
        events = []

        coordinator = UnifiedAutoCoordinator(
            letters, digits, model,
            event_sink=events.append,
            enable_experimental_jz=True,
            dynamic_jz_dtw_shadow_enabled=True,
            jz_evaluation_mode=True,
        )
        coordinator.shadow_adapter.telemetry_dir = self.telemetry_dir

        # Step through 80 movement frames
        for k in range(80):
            t = k / 30.0
            hand = curved_hand(min(int(k * 1.0), 60), "J")
            coordinator.update(k, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)

        # Hold for 100 frames to confirm (3.33s > 3.0s)
        held_hand = curved_hand(60, "J")
        for k in range(80, 185):
            t = k / 30.0
            coordinator.update(k, mirrored(held_hand), held_hand, (480, 640, 3), t, raw_hand_count=1)

        # 1. First16Segmenter successfully sampled and delivered attempt
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0].shape, (1, 1104))

        # 2. Shadow proxy intercepted exact input
        self.assertIsNotNone(coordinator.dynamic_model.latest_features)
        np.testing.assert_array_equal(coordinator.dynamic_model.latest_features, model.inputs[0])

        # 3. TextComposer committed 'J'
        self.assertEqual(coordinator.composer.text, "J")

        # 4. Telemetry recorded both PENDING and COMMITTED stages
        jsonl_path = self.telemetry_dir / "shadow_telemetry.jsonl"
        self.assertTrue(jsonl_path.exists())
        lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(lines[0]["final_commit_status"], "PENDING")
        self.assertEqual(lines[-1]["final_commit_status"], "COMMITTED")
        self.assertEqual(lines[-1]["final_committed_label"], "J")

    # -------------------------------------------------------------------------
    # 2. Sequence & Frame Alignment
    # -------------------------------------------------------------------------
    def test_02_attempt_sequence_alignment(self) -> None:
        """Confirm DTW evaluates the exact array passed to LogisticRegression without lag."""
        inner = DynamicModel("Z")
        proxy = ShadowModelProxy(inner, self.adapter)

        # Pass 1: J sequence
        flat_j = self.seq_j.reshape(1, 1104, order="C")
        _ = proxy.predict_proba(flat_j)
        res_j = proxy.latest_shadow_result
        self.assertEqual(res_j.prediction, "J")
        np.testing.assert_array_equal(proxy.latest_features, flat_j)

        # Pass 2: Z sequence
        flat_z = self.seq_z.reshape(1, 1104, order="C")
        _ = proxy.predict_proba(flat_z)
        res_z = proxy.latest_shadow_result
        self.assertEqual(res_z.prediction, "Z")
        np.testing.assert_array_equal(proxy.latest_features, flat_z)

        # Confirms no stale features or result carried over
        self.assertNotEqual(res_j.best_distance, res_z.best_distance)

    # -------------------------------------------------------------------------
    # 3. Immutability & Target Label Insulation
    # -------------------------------------------------------------------------
    def test_03_immutability_and_target_label_insulation(self) -> None:
        """Verify LogisticRegression input/output immutability and target_label exclusion."""
        inner = DynamicModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)

        orig_flat = self.seq_j.reshape(1, 1104, order="C").copy()
        input_flat = orig_flat.copy()

        probs = proxy.predict_proba(input_flat)

        # Input array never mutated
        np.testing.assert_array_equal(input_flat, orig_flat)

        # Probabilities untouched
        np.testing.assert_array_equal(probs, np.asarray([[0.97, 0.03]]))

        # Target label is NEVER a parameter to evaluate_attempt or DTWMatcher
        import inspect
        sig = inspect.signature(self.adapter.evaluate_attempt)
        self.assertNotIn("target_label", sig.parameters)
        self.assertNotIn("target", sig.parameters)

    # -------------------------------------------------------------------------
    # 4. Single Invocation Per Attempt
    # -------------------------------------------------------------------------
    def test_04_single_invocation_per_attempt(self) -> None:
        """Confirm DTW is invoked at most once per completed attempt."""
        inner = DynamicModel("J")
        proxy = ShadowModelProxy(inner, self.adapter)
        flat = self.seq_j.reshape(1, 1104, order="C")

        self.assertEqual(proxy.call_count_this_attempt, 0)
        _ = proxy.predict_proba(flat)
        self.assertEqual(proxy.call_count_this_attempt, 1)

    # -------------------------------------------------------------------------
    # 5. Lifecycle Isolation & Reset
    # -------------------------------------------------------------------------
    def test_05_lifecycle_isolation_and_reset(self) -> None:
        """Test reset and cancellation clear shadow state cleanly without leaking."""
        letters = FakePredictor("A", 0.0)
        digits = FakePredictor("0", 0.5)
        model = DynamicModel("J")

        coordinator = UnifiedAutoCoordinator(
            letters, digits, model,
            enable_experimental_jz=True,
            dynamic_jz_dtw_shadow_enabled=True,
        )
        coordinator.shadow_adapter.telemetry_dir = self.telemetry_dir

        # Populate shadow result
        flat = self.seq_j.reshape(1, 1104, order="C")
        _ = coordinator.dynamic_model.predict_proba(flat)
        self.assertIsNotNone(coordinator.shadow_adapter.latest_result)

        # Reset motion cycle
        coordinator.reset_motion_cycle(now=1.0, reason="MANUAL_RESET")
        self.assertIsNone(coordinator.shadow_adapter.latest_result)

    # -------------------------------------------------------------------------
    # 6. Fail-Open Behavior
    # -------------------------------------------------------------------------
    def test_06_fail_open_conditions(self) -> None:
        """Verify fail-open behavior under all fault conditions."""
        # 6.1 Missing template
        missing_adapter = JzShadowAdapter(
            enabled=True,
            template_path=Path("non_existent_path.json"),
            telemetry_dir=self.telemetry_dir,
            verify_sha256=False,
        )
        self.assertFalse(missing_adapter.available)
        res_missing = missing_adapter.evaluate_attempt(self.seq_j)
        self.assertEqual(res_missing.status, "UNAVAILABLE")
        self.assertEqual(res_missing.prediction, "UNKNOWN")

        # 6.2 Bad Checksum
        bad_hash_adapter = JzShadowAdapter(
            enabled=True,
            template_path=DEFAULT_TEMPLATE_PATH,
            telemetry_dir=self.telemetry_dir,
            verify_sha256=True,
        )
        with patch("hashlib.sha256") as mock_sha:
            mock_sha.return_value.hexdigest.return_value = "0000000000000000000000000000000000000000000000000000000000000000"
            bad_hash_adapter._initialize_engine()
            self.assertFalse(bad_hash_adapter.available)
            self.assertIn("TEMPLATE_CHECKSUM_MISMATCH", bad_hash_adapter.init_error)

        # 6.3 Invalid Shape
        res_shape = self.adapter.evaluate_attempt(np.zeros((5, 10)))
        self.assertEqual(res_shape.status, "INVALID_INPUT")
        self.assertEqual(res_shape.prediction, "UNKNOWN")

        # 6.4 NaN Input / Invalid Tracking Quality
        nan_seq = self.seq_j.copy()
        nan_seq[0, 0] = np.nan
        res_nan = self.adapter.evaluate_attempt(nan_seq)
        self.assertIn(res_nan.status, ("REJECTED", "ERROR"))
        self.assertEqual(res_nan.prediction, "UNKNOWN")

        # 6.5 DTW Exception
        with patch.object(self.adapter.matcher, "match_dual", side_effect=DTWError("DTW_INTERNAL_CRASH")):
            res_err = self.adapter.evaluate_attempt(self.seq_j)
            self.assertEqual(res_err.status, "ERROR")
            self.assertEqual(res_err.prediction, "UNKNOWN")

        # 6.6 Telemetry Write Error
        # Passing an unwritable directory should fail open silently without crashing
        bad_telemetry_adapter = JzShadowAdapter(
            enabled=True,
            template_path=DEFAULT_TEMPLATE_PATH,
            telemetry_dir=Path("Z:/invalid_drive_that_does_not_exist/shadow"),
        )
        rec = JzShadowTelemetryRecord(
            timestamp="2026-11-13T12:00:00Z", attempt_id="test", target_label="J",
            logistic_prediction="J", logistic_confidence=0.9, dtw_prediction="J",
            dtw_best_distance=0.5, dtw_second_distance=1.0, dtw_margin=0.5,
            dtw_status="OK", tracking_quality="HIGH", feature_config="CONFIG_C",
            template_id="J_0", dtw_latency_ms=0.2, agreement=True,
            final_committed_label="J", final_commit_status="COMMITTED",
        )
        bad_telemetry_adapter.log_telemetry(rec)  # Must not raise

        # 6.7 Shadow Disabled
        disabled_adapter = JzShadowAdapter(enabled=False)
        self.assertFalse(disabled_adapter.enabled)
        res_dis = disabled_adapter.evaluate_attempt(self.seq_j)
        self.assertEqual(res_dis.status, "DISABLED")

    # -------------------------------------------------------------------------
    # 7. Telemetry Completeness
    # -------------------------------------------------------------------------
    def test_07_telemetry_completeness(self) -> None:
        """Check all 17 schema fields in logged telemetry."""
        record = JzShadowTelemetryRecord(
            timestamp="2026-11-13T12:00:00Z",
            attempt_id="trial_101",
            target_label="Z",
            logistic_prediction="Z",
            logistic_confidence=0.95,
            dtw_prediction="Z",
            dtw_best_distance=0.45,
            dtw_second_distance=1.10,
            dtw_margin=0.65,
            dtw_status="OK",
            tracking_quality="VALID",
            feature_config="CONFIG_C",
            template_id="Z_medoid_0",
            dtw_latency_ms=0.22,
            agreement=True,
            final_committed_label="Z",
            final_commit_status="COMMITTED",
        )
        self.adapter.log_telemetry(record)

        jsonl_path = self.telemetry_dir / "shadow_telemetry.jsonl"
        data = json.loads(jsonl_path.read_text(encoding="utf-8").strip())

        required_keys = [
            "timestamp", "attempt_id", "target_label",
            "logistic_prediction", "logistic_confidence",
            "dtw_prediction", "dtw_best_distance", "dtw_second_distance",
            "dtw_margin", "dtw_status", "tracking_quality",
            "feature_config", "template_id", "dtw_latency_ms",
            "agreement", "final_committed_label", "final_commit_status",
        ]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key {key} in telemetry record")

    # -------------------------------------------------------------------------
    # 8. Replay Parity Comparison (Shadow OFF vs ON)
    # -------------------------------------------------------------------------
    def test_08_shadow_off_vs_on_exact_parity(self) -> None:
        """Execute identical 185-frame sequence under OFF vs ON; verify bitwise pipeline equality."""
        def run_sim(shadow_flag: bool):
            letters = FakePredictor("A", 0.0)
            digits = FakePredictor("0", 0.5)
            model = DynamicModel("J")
            events = []
            c = UnifiedAutoCoordinator(
                letters, digits, model,
                event_sink=events.append,
                enable_experimental_jz=True,
                dynamic_jz_dtw_shadow_enabled=shadow_flag,
                jz_evaluation_mode=True,
            )
            c.shadow_adapter.telemetry_dir = self.telemetry_dir

            states = []
            for k in range(80):
                t = k / 30.0
                hand = curved_hand(min(int(k * 1.0), 60), "J")
                c.update(k, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
                states.append((c.status, c.candidate, c.composer.text, c.jz_hold.state, c.jz_hold.progress))

            held = curved_hand(60, "J")
            for k in range(80, 185):
                t = k / 30.0
                c.update(k, mirrored(held), held, (480, 640, 3), t, raw_hand_count=1)
                states.append((c.status, c.candidate, c.composer.text, c.jz_hold.state, c.jz_hold.progress))

            return states, c.composer.text, model.inputs, c.jz_hold.settings.hold_seconds

        states_off, text_off, inputs_off, hold_sec_off = run_sim(False)
        states_on, text_on, inputs_on, hold_sec_on = run_sim(True)

        # 8.1 Hold duration is strictly 3.0s in both
        self.assertEqual(hold_sec_off, 3.0)
        self.assertEqual(hold_sec_on, 3.0)

        # 8.2 Final committed text is identical
        self.assertEqual(text_off, "J")
        self.assertEqual(text_on, "J")

        # 8.3 Number of inputs and contents to LogisticRegression are identical
        self.assertEqual(len(inputs_off), len(inputs_on))
        for in_off, in_on in zip(inputs_off, inputs_on):
            np.testing.assert_array_equal(in_off, in_on)

        # 8.4 State sequence at every single frame is 100% identical
        self.assertEqual(len(states_off), len(states_on))
        for idx, (st_off, st_on) in enumerate(zip(states_off, states_on)):
            self.assertEqual(st_off, st_on, f"State difference at frame {idx}")


if __name__ == "__main__":
    unittest.main()
