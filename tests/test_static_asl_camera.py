import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))

from app.inference import letter_inference as camera_app  # noqa: E402
from landmark_utils import normalize_landmarks  # noqa: E402


class StaticAslCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = camera_app.load_official_artifacts("v1")

    def test_official_artifact_contract(self):
        self.assertEqual(self.artifacts.labels, tuple("ABCDEFGHIKLMNOPQRSTUVWXY"))
        self.assertNotIn("J", self.artifacts.labels)
        self.assertNotIn("Z", self.artifacts.labels)
        self.assertEqual(self.artifacts.model.n_features_in_, 63)
        self.assertEqual(self.artifacts.scaler.n_features_in_, 63)
        self.assertEqual(int(self.artifacts.scaler.n_samples_seen_), 62591)
        self.assertEqual(self.artifacts.model_name, "v1_baseline")

    def test_uvw_medium_artifact_contract_and_inference(self):
        artifacts = camera_app.load_official_artifacts("uvw_medium")
        self.assertEqual(artifacts.model_name, "uvw_medium")
        self.assertEqual(artifacts.best_epoch, 81)
        self.assertEqual(artifacts.labels, tuple("ABCDEFGHIKLMNOPQRSTUVWXY"))
        self.assertEqual(artifacts.model.n_features_in_, 63)
        self.assertEqual(artifacts.scaler.n_features_in_, 63)
        self.assertEqual(int(artifacts.scaler.n_samples_seen_), 62648)

        validation = pd.read_csv(
            ROOT / "ai/dataset/train_validation_test_split_v1/validation.csv"
        )
        saved = pd.read_csv(
            ROOT / "ai/dataset/mlp_uvw_validation_comparison_v1/"
            "uvw_medium_validation_predictions.csv"
        ).set_index("file")
        pipeline = camera_app.StaticAslInference(artifacts)
        for row in validation.iloc[:5].itertuples(index=False):
            pipeline.reset()
            features = np.asarray([getattr(row, name) for name in camera_app.FEATURES])
            result = pipeline.predict_features(features)
            expected = saved.loc[row.file]
            self.assertEqual(result.instant_label, expected.predicted_label)
            self.assertAlmostEqual(result.instant_confidence, expected.predicted_confidence, places=12)

    def test_cli_defaults_to_uvw_medium_and_keeps_v1_fallback(self):
        self.assertEqual(camera_app.parse_args([]).model, "uvw_medium")
        self.assertEqual(camera_app.load_official_artifacts().model_name, "uvw_medium")
        self.assertEqual(camera_app.parse_args(["--model", "v1"]).model, "v1")
        self.assertEqual(
            camera_app.parse_args(["--model", "uvw_medium"]).model,
            "uvw_medium",
        )

    def test_retired_and_invalid_models_are_rejected(self):
        for model_name in ("v2_debug_5pct", "unknown"):
            with self.subTest(model_name=model_name):
                with self.assertRaises(SystemExit):
                    camera_app.parse_args(["--model", model_name])
                with self.assertRaises(camera_app.IntegrationError):
                    camera_app.load_official_artifacts(model_name)
        self.assertTrue(callable(camera_app._manifest_output))

    def test_canonical_normalization_is_reused_exactly(self):
        points = [SimpleNamespace(x=i / 50, y=(i * i) / 1000, z=-i / 500) for i in range(21)]
        actual = camera_app.features_from_landmarks(points)
        expected = np.asarray(normalize_landmarks(points), dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_landmarks_are_rejected(self):
        point = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        with self.assertRaises(ValueError):
            camera_app.features_from_landmarks([point] * 20)
        with self.assertRaises(ValueError):
            camera_app.features_from_landmarks([point] * 21)
        invalid = [SimpleNamespace(x=float(i), y=0.0, z=0.0) for i in range(21)]
        invalid[3] = SimpleNamespace(x=float("nan"), y=0.0, z=0.0)
        with self.assertRaises(ValueError):
            camera_app.features_from_landmarks(invalid)

    def test_smoother_and_no_hand_reset(self):
        smoother = camera_app.TemporalSmoother(24, window_size=7, minimum_votes=4)
        probabilities = np.zeros(24, dtype=np.float64)
        probabilities[2] = 1.0
        for _ in range(3):
            self.assertEqual(smoother.update(2, probabilities), (None, None))
        label, confidence = smoother.update(2, probabilities)
        self.assertEqual(label, 2)
        self.assertEqual(confidence, 1.0)
        smoother.reset()
        self.assertEqual(smoother.update(2, probabilities), (None, None))

    def test_known_validation_samples_match_saved_results(self):
        validation = pd.read_csv(
            ROOT / "ai/dataset/train_validation_test_split_v1/validation.csv"
        )
        saved = pd.read_csv(
            ROOT / "ai/dataset/mlp_validation_comparison_v1/baseline_validation_predictions.csv"
        ).set_index("file")
        selected = pd.concat([
            validation.loc[validation.label == label].head(1)
            for label in ("A", "K", "M", "N", "T", "Y")
        ], ignore_index=True)
        pipeline = camera_app.StaticAslInference(self.artifacts, window_size=7, minimum_votes=4)
        for row in selected.itertuples(index=False):
            pipeline.reset()
            features = np.asarray([getattr(row, name) for name in camera_app.FEATURES])
            result = pipeline.predict_features(features)
            expected = saved.loc[row.file]
            self.assertEqual(result.instant_label, expected.predicted_label)
            self.assertAlmostEqual(result.instant_confidence, expected.predicted_confidence, places=12)
            self.assertIsNone(result.stable_label)

    def test_saved_raw_landmarks_match_frozen_known_feature_row(self):
        target = r"dulieu\chucai\a\A10.jpg"
        validation = pd.read_csv(
            ROOT / "ai/dataset/train_validation_test_split_v1/validation.csv",
            nrows=1,
        )
        self.assertEqual(validation.iloc[0].file, target)
        raw_row = None
        with (ROOT / "ai/raw_data/raw_landmarks_chucai_all.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if row["file"] == target:
                    raw_row = row
                    break
        self.assertIsNotNone(raw_row)
        landmarks = [
            SimpleNamespace(
                x=float(raw_row[f"x{i}"]),
                y=float(raw_row[f"y{i}"]),
                z=float(raw_row[f"z{i}"]),
            )
            for i in range(21)
        ]
        actual = camera_app.features_from_landmarks(landmarks)
        expected = validation.loc[0, list(camera_app.FEATURES)].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_test_data_not_loaded_by_camera_module(self):
        source = Path(camera_app.__file__).read_text(encoding="utf-8")
        self.assertNotIn("test.csv", source)
        self.assertNotIn("fit(", source)
        self.assertNotIn("partial_fit", source)

    @patch.object(camera_app.cv2, "VideoCapture")
    def test_camera_open_failure_returns_without_hanging(self, video_capture):
        closed_camera = Mock()
        closed_camera.isOpened.return_value = False
        video_capture.return_value = closed_camera
        pipeline = camera_app.StaticAslInference(self.artifacts)
        self.assertEqual(camera_app.run_camera(pipeline, camera_index=99), 3)
        closed_camera.release.assert_called_once_with()

    def test_explicit_debug_capture_writes_complete_evidence(self):
        probabilities = np.zeros(24, dtype=np.float64)
        probabilities[camera_app.STATIC_LABELS.index("P")] = 0.08
        probabilities[camera_app.STATIC_LABELS.index("Q")] = 0.92
        prediction = camera_app.Prediction(
            instant_label="Q",
            instant_confidence=0.92,
            stable_label="Q",
            stable_confidence=0.90,
            features=tuple(np.linspace(-1.0, 1.0, 63)),
            probabilities=tuple(probabilities),
        )
        landmarks = [
            SimpleNamespace(x=i / 30, y=i / 40, z=-i / 100) for i in range(21)
        ]
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            collector = camera_app.DebugCollector(
                Path(temporary), manual_label="P", view="horizontal", mirror=True
            )
            details_path = collector.capture(
                image, image, landmarks, prediction, handedness="Left"
            )
            details = camera_app._load_json(details_path)
            self.assertEqual(details["manual_label"], "P")
            self.assertEqual(details["manual_view"], "horizontal")
            self.assertEqual(details["instant_prediction"], "Q")
            self.assertEqual(details["probability_P"], 0.08)
            self.assertEqual(details["probability_Q"], 0.92)
            self.assertEqual(len(details["features"]), 63)
            self.assertEqual(len(details["raw_landmarks"]), 21)
            self.assertTrue((details_path.parent / details["raw_image"]).is_file())
            self.assertTrue((details_path.parent / details["overlay_image"]).is_file())
            with collector.index_path.open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 2)

    def test_uv_debug_labels_use_separate_condition_directories(self):
        probabilities = np.zeros(24, dtype=np.float64)
        probabilities[camera_app.STATIC_LABELS.index("U")] = 0.75
        probabilities[camera_app.STATIC_LABELS.index("V")] = 0.25
        prediction = camera_app.Prediction(
            instant_label="U",
            instant_confidence=0.75,
            stable_label="U",
            stable_confidence=0.70,
            features=tuple(np.linspace(-1.0, 1.0, 63)),
            probabilities=tuple(probabilities),
        )
        landmarks = [
            SimpleNamespace(x=i / 30, y=i / 40, z=-i / 100) for i in range(21)
        ]
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            collector_u = camera_app.DebugCollector(
                output_dir, manual_label="U", view="front", mirror=True
            )
            collector_v = camera_app.DebugCollector(
                output_dir, manual_label="V", view="horizontal", mirror=True
            )
            self.assertEqual(collector_u.output_dir, output_dir / "U_front")
            self.assertEqual(collector_v.output_dir, output_dir / "V_horizontal")
            self.assertNotEqual(collector_u.output_dir, collector_v.output_dir)
            details_path = collector_u.capture(
                image, image, landmarks, prediction, handedness="Left"
            )
            details = camera_app._load_json(details_path)
            self.assertEqual(details["manual_label"], "U")
            self.assertEqual(details["probability_manual_label"], 0.75)
            self.assertEqual(details["probabilities"]["V"], 0.25)
            self.assertTrue(collector_u.index_path.is_file())

    def test_cli_accepts_uv_debug_labels(self):
        args_u = camera_app.parse_args(
            ["--debug-dir", "ai/camera_debug_uv", "--debug-label", "U", "--debug-view", "front"]
        )
        args_v = camera_app.parse_args(
            ["--debug-dir", "ai/camera_debug_uv", "--debug-label", "V", "--debug-view", "horizontal"]
        )
        self.assertEqual(args_u.debug_label, "U")
        self.assertEqual(args_v.debug_label, "V")

    def test_debug_prediction_filter_keeps_only_v_as_u_failures(self):
        probabilities_u = np.zeros(24, dtype=np.float64)
        probabilities_u[camera_app.STATIC_LABELS.index("U")] = 1.0
        prediction_u = camera_app.Prediction(
            instant_label="U", instant_confidence=1.0,
            stable_label="U", stable_confidence=1.0,
            features=tuple(np.linspace(-1.0, 1.0, 63)),
            probabilities=tuple(probabilities_u),
        )
        probabilities_v = np.zeros(24, dtype=np.float64)
        probabilities_v[camera_app.STATIC_LABELS.index("V")] = 1.0
        prediction_v = camera_app.Prediction(
            instant_label="V", instant_confidence=1.0,
            stable_label="V", stable_confidence=1.0,
            features=tuple(np.linspace(-1.0, 1.0, 63)),
            probabilities=tuple(probabilities_v),
        )
        landmarks = [
            SimpleNamespace(x=i / 30, y=i / 40, z=-i / 100) for i in range(21)
        ]
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            collector = camera_app.DebugCollector(
                Path(temporary), manual_label="V", view="front", mirror=True,
                model_name="v1_baseline", required_instant_prediction="U",
            )
            with self.assertRaises(ValueError):
                collector.capture(image, image, landmarks, prediction_v, handedness="Left")
            saved = collector.capture(
                image, image, landmarks, prediction_u, handedness="Left"
            )
            details = camera_app._load_json(saved)
            self.assertEqual(details["manual_label"], "V")
            self.assertEqual(details["instant_prediction"], "U")
            self.assertEqual(details["required_instant_prediction"], "U")
            self.assertEqual(details["model"], "v1_baseline")

        args = camera_app.parse_args([
            "--model", "v1",
            "--debug-dir", "ai/camera_debug_uv_v1_v_as_u_new",
            "--debug-label", "V", "--debug-view", "front",
            "--debug-prediction-filter", "U",
        ])
        self.assertEqual(args.model, "v1")
        self.assertEqual(args.debug_prediction_filter, "U")


if __name__ == "__main__":
    unittest.main()
