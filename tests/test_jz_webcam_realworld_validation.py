"""Offline integration tests for tools/jz_webcam_realworld_validation.py."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_dynamic_jz_motion_gate import curved_hand
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from tools.jz_webcam_realworld_validation import (
    CSV_COLUMNS,
    ValidationModelProxy,
    ValidationTrialTracker,
    compute_session_summary,
)


class StubDynamicModel:
    classes_ = np.asarray(["J", "Z"])

    def __init__(self, label="Z", probs=None):
        self.label = label
        self.probs = probs

    def predict_proba(self, x):
        if self.probs is not None:
            return np.asarray([self.probs], dtype=np.float64)
        return np.asarray([[0.96, 0.04] if self.label == "J" else [0.03, 0.97]], dtype=np.float64)


def mirrored(hand):
    return [SimpleNamespace(x=1.0 - p.x, y=p.y, z=p.z) for p in hand]


class RealworldValidationHarnessTests(unittest.TestCase):
    def _build_env(self, tmpdir: Path, label="Z"):
        tracker = ValidationTrialTracker(tmpdir, phase="baseline", idle_timeout_seconds=12.0)
        proxy = ValidationModelProxy(StubDynamicModel(label=label), tracker=tracker)
        coord = UnifiedAutoCoordinator(
            FakePredictor("A", 0.0),
            FakePredictor("0", 0.5),
            proxy,
            enable_experimental_jz=True,
            jz_evaluation_mode=True,
            event_sink=tracker.on_event,
        )
        return tracker, coord

    def test_pre_model_missing_hand_rejection_and_duplicate_ratio(self):
        with TemporaryDirectory() as tmp:
            tracker, coord = self._build_env(Path(tmp), label="Z")
            tracker.arm("Z", coord, 0.0)

            still = curved_hand(0, "Z")
            # Feed frames with deliberate consecutive duplicate landmarks to verify ratio
            for i in range(6):
                hand = curved_hand(min(i * 3, 20), "Z")
                coord.update(i, mirrored(hand), hand, (480, 640, 3), i / 30.0, raw_hand_count=1)
                row = tracker.observe_frame(hand, coord, i / 30.0, raw_hand_count=1)
                self.assertIsNone(row)

            # Duplicate the last hand explicitly once, then trigger MISSING_HAND_LIMIT
            idx = 6
            coord.update(idx, mirrored(hand), hand, (480, 640, 3), idx / 30.0, raw_hand_count=1)
            tracker.observe_frame(hand, coord, idx / 30.0, raw_hand_count=1)

            final_row = None
            for miss_i in range(1, 20):
                t = (idx + miss_i) / 30.0
                coord.update(idx + miss_i, None, None, (480, 640, 3), t, raw_hand_count=0)
                final_row = tracker.observe_frame(None, coord, t, raw_hand_count=0)
                if final_row is not None:
                    break

            self.assertIsNotNone(final_row)
            self.assertEqual(final_row["ground_truth_label"], "Z")
            self.assertEqual(final_row["motion_segmenter_result"], "REJECTED")
            self.assertIn("MISSING_HAND_LIMIT", final_row["motion_segmenter_reason"])
            self.assertEqual(final_row["model_prediction"], "NONE")
            self.assertAlmostEqual(final_row["duplicate_landmark_ratio"], 1.0 / 6.0, places=3)
            self.assertTrue(final_row["final_recognition_result"].startswith("FAILED:MOTION_SEGMENTER:"))

    def test_trajectory_validator_rejection_records_all_13_fields(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            tracker, coord = self._build_env(tmpdir, label="Z")
            tracker.arm("Z", coord, 0.0)

            # Curved J motion passes MotionSegmenter + First16Segmenter, model predicts Z,
            # and Trajectory Validator must reject the mismatched J trajectory under Z validation.
            curved_j_frames = [curved_hand(min(k, 60), "J") for k in range(80)]
            row = None
            for k, hand in enumerate(curved_j_frames):
                t = k / 30.0
                _p, committed = coord.update(k, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
                row = tracker.observe_frame(hand, coord, t, committed=committed, raw_hand_count=1)
                if row is not None:
                    break

            self.assertIsNotNone(row)
            self.assertEqual(row["ground_truth_label"], "Z")
            self.assertEqual(row["motion_segmenter_result"], "ACCEPTED")
            self.assertEqual(row["first16_segmenter_result"], "COMPLETE")
            self.assertEqual(row["feature_shape"], "(16, 69)")
            self.assertTrue(row["feature_valid"])
            self.assertEqual(row["model_prediction"], "Z")
            self.assertAlmostEqual(row["probability_Z"], 0.97, places=2)
            self.assertEqual(row["trajectory_validator_result"], "REJECTED")
            self.assertNotEqual(row["trajectory_validator_reason"], "NONE")
            self.assertEqual(row["hold_confirmation_result"], "NOT_REACHED")
            self.assertTrue(row["final_recognition_result"].startswith("FAILED:TRAJECTORY_VALIDATOR:"))

            with (tmpdir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = list(csv.DictReader(handle))
            self.assertEqual(len(reader), 1)
            self.assertEqual(tuple(reader[0].keys()), CSV_COLUMNS)

    def test_end_to_end_z_commit_and_cancelled_hold_reporting(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            tracker, coord = self._build_env(tmpdir, label="Z")

            # Trial 1: Complete curved Z gesture + quiet endpoint + 3.0s hold -> COMMITTED:Z
            tracker.arm("Z", coord, 0.0)
            row1 = None
            idx = 0
            for step in range(95):
                hand = curved_hand(min(int(step * 1.8), 60), "Z")
                t = idx / 30.0
                _p, committed = coord.update(idx, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
                row1 = tracker.observe_frame(hand, coord, t, committed=committed, raw_hand_count=1)
                idx += 1
                if row1 is not None:
                    break

            # Hold stationary endpoint for >= 3.2s if trial hasn't resolved yet
            end_hand = curved_hand(60, "Z")
            for extra in range(130):
                if row1 is not None:
                    break
                t = idx / 30.0
                _p, committed = coord.update(idx, mirrored(end_hand), end_hand, (480, 640, 3), t, raw_hand_count=1)
                row1 = tracker.observe_frame(end_hand, coord, t, committed=committed, raw_hand_count=1)
                idx += 1

            self.assertIsNotNone(row1)
            self.assertEqual(row1["model_prediction"], "Z")
            self.assertEqual(row1["trajectory_validator_result"], "ACCEPTED")
            self.assertEqual(row1["hold_confirmation_result"], "CONFIRMED")
            self.assertEqual(row1["final_recognition_result"], "COMMITTED:Z")
            self.assertTrue(row1["committed_correct"])

            # Trial 2: Armed J gesture -> passes validator -> hand vanishes -> CANCELLED_PENDING_ENDPOINT
            coord.dynamic_model._inner.label = "J"
            t0 = idx / 30.0
            tracker.arm("J", coord, t0)
            row2 = None
            for step in range(40):
                hand = curved_hand(min(step, 60), "J")
                t = idx / 30.0
                _p, committed = coord.update(idx, mirrored(hand), hand, (480, 640, 3), t, raw_hand_count=1)
                row2 = tracker.observe_frame(hand, coord, t, committed=committed, raw_hand_count=1)
                idx += 1
                if row2 is not None:
                    break
            for drop_i in range(5):
                if row2 is not None:
                    break
                t = idx / 30.0
                _p, committed = coord.update(idx, None, None, (480, 640, 3), t, raw_hand_count=0)
                row2 = tracker.observe_frame(None, coord, t, committed=committed, raw_hand_count=0)
                idx += 1

            self.assertIsNotNone(row2)
            self.assertEqual(row2["trajectory_validator_result"], "ACCEPTED")
            self.assertEqual(row2["hold_confirmation_result"], "CANCELLED_PENDING_ENDPOINT")
            self.assertEqual(row2["hold_confirmation_reason"], "CANDIDATE_TRACKING_LOST_OR_AMBIGUOUS")
            self.assertFalse(row2["committed_correct"])
            self.assertTrue(row2["model_prediction_correct"])

            # Summary report must cleanly distinguish model accuracy from end-to-end rate
            summary = compute_session_summary(tracker.completed, phase="baseline")
            self.assertEqual(summary["overall"]["total_attempts"], 2)
            self.assertEqual(summary["overall"]["model_prediction_accuracy"], 1.0)
            self.assertEqual(summary["overall"]["end_to_end_webcam_recognition_rate"], 0.5)
            self.assertEqual(summary["Z"]["successful_confirmations"], 1)
            self.assertEqual(summary["J"]["successful_confirmations"], 0)
            self.assertTrue((tmpdir / "session_summary.md").is_file())
            self.assertTrue((tmpdir / "session_summary.json").is_file())

    def test_hybrid_shadow_telemetry_and_safety(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            from app.inference.dynamic_jz.hybrid_jz_classifier import HybridJzClassifier
            hybrid_cls = HybridJzClassifier()
            tracker, coord = self._build_env(tmpdir, label="Z")
            # Upgrade proxy to enable hybrid shadow
            coord.dynamic_model.enable_hybrid_shadow = True
            coord.dynamic_model.hybrid_classifier = hybrid_cls

            self.assertEqual(len(CSV_COLUMNS), 52)
            self.assertIn("hybrid_prediction", CSV_COLUMNS)
            self.assertIn("methods_agreement", CSV_COLUMNS)

            # Test prediction call with 16x69 array
            tracker.arm("Z", coord, 0.0)
            active = tracker.active
            active.trial_attempt_id = 1

            sample_16x69 = np.zeros((16, 69), dtype=np.float64)
            probs = coord.dynamic_model.predict_proba(sample_16x69)

            # 1. Output from predict_proba must remain strictly identical to inner model
            expected_probs = coord.dynamic_model._inner.predict_proba(sample_16x69.reshape(1, -1))
            np.testing.assert_array_almost_equal(probs, expected_probs)

            # 2. Shadow telemetry fields must be populated on active trial
            self.assertEqual(active.lr_prediction, "Z")
            self.assertIn(active.hybrid_prediction, ("J", "Z", "UNKNOWN"))
            self.assertIn(active.hybrid_decision_path, ("UNANIMOUS_CONSENSUS_3_MODELS", "OOD_REJECTED_BY_DTW_AND_GEOM", "CONSENSUS_LR_DTW_Z_CONFIRMED", "CONSENSUS_LR_DTW_J_CONFIRMED", "TRAJECTORY_CONSENSUS_DTW_GEOM_OVERRULE_LR", "RECOVERED_LR_GEOM_WITH_BORDERLINE_DTW", "IRRECONCILABLE_DISAGREEMENT", "INVALID_INPUT_SHAPE"))
            self.assertGreater(active.lr_latency_ms, 0.0)
            self.assertGreater(active.dtw_latency_ms, 0.0)
            self.assertGreater(active.geom_latency_ms, 0.0)
            self.assertGreater(active.hybrid_latency_ms, 0.0)

            tracker._finalize_active(1.5)
            # Verify telemetry file was written
            telem_path = tmpdir / "hybrid_shadow_telemetry.jsonl"
            self.assertTrue(telem_path.is_file())
            record = json.loads(telem_path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["attempt_id"], 1)
            self.assertEqual(record["target_label"], "Z")
            self.assertIn("baseline_lr", record)
            self.assertIn("enhanced_dtw", record)
            self.assertIn("geometric", record)
            self.assertIn("hybrid", record)


if __name__ == "__main__":
    unittest.main()
