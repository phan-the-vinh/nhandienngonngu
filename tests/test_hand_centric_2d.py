"""Offline compatibility checks for the stable digit alignment transformer."""

import io
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = ROOT / "ai" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from hand_centric_2d import DegenerateHandAxisError, HandCentric2DAligner
from orientation_2d_experimental import HandCentric2DAligner as ExperimentalAligner


def example_features():
    hand = np.zeros((21, 3), dtype=float)
    hand[:, 0] = np.linspace(-0.2, 0.2, 21)
    hand[:, 1] = np.linspace(0.0, -0.4, 21)
    hand[:, 2] = np.linspace(0.0, 0.1, 21)
    hand[0] = (0.0, 0.0, 0.0)
    hand[5] = (0.25, -0.35, 0.01)
    hand[9] = (0.0, -0.45, 0.02)
    hand[17] = (-0.25, -0.35, -0.01)
    rotated = hand.copy()
    rotated[:, :2] = hand[:, :2] @ np.array([[0.0, -1.0], [1.0, 0.0]]).T
    return np.stack((hand, rotated)).reshape(2, 63)


class StableHandCentric2DTests(unittest.TestCase):
    def test_fit_transform_and_transform_match_experimental_formula(self):
        features = example_features()
        stable = HandCentric2DAligner(axis_epsilon=1e-6)
        experimental = ExperimentalAligner(axis_epsilon=1e-6)
        expected = experimental.fit_transform(features)
        for actual in (stable.fit_transform(features), stable.transform(features)):
            self.assertEqual(actual.shape, (2, 63))
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                actual.reshape(2, 21, 3)[:, :, 2],
                features.reshape(2, 21, 3)[:, :, 2],
            )

    def test_degenerate_axis_is_rejected(self):
        with self.assertRaises(DegenerateHandAxisError):
            HandCentric2DAligner().transform(np.zeros((1, 63)))

    def test_stable_pipeline_joblib_round_trip_without_training(self):
        pipeline = Pipeline([
            ("orientation_2d", HandCentric2DAligner()),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64, 32))),
        ])
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        buffer.seek(0)
        loaded = joblib.load(buffer)
        self.assertEqual(list(loaded.named_steps), ["orientation_2d", "scaler", "mlp"])
        self.assertEqual(loaded.named_steps["orientation_2d"].__class__.__module__, "hand_centric_2d")
        self.assertEqual(loaded.named_steps["orientation_2d"].transform(example_features()).shape, (2, 63))

    def test_current_production_model_still_loads(self):
        path = ROOT / "models" / "digits" / "production" / "mlp_large_v2a.joblib"
        model = joblib.load(path)
        self.assertEqual(list(model.named_steps), ["scaler", "mlp"])
        self.assertEqual(model.n_features_in_, 63)
        self.assertEqual(set(map(str, model.classes_)), set(map(str, range(10))))


if __name__ == "__main__":
    unittest.main()
