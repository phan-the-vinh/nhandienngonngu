"""Offline smoke tests for the experimental digit orientation transformer."""

import sys
import unittest
from pathlib import Path

import numpy as np


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "ai" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from orientation_2d_experimental import HandCentric2DAligner


class HandCentric2DAlignerTests(unittest.TestCase):
    def test_fit_transform_and_transform_return_finite_63_features(self):
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
        features = np.stack((hand, rotated)).reshape(2, 63)

        aligner = HandCentric2DAligner()
        fitted_output = aligner.fit_transform(features)
        inference_output = aligner.transform(features)

        for output in (fitted_output, inference_output):
            self.assertEqual(output.shape, (2, 63))
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_allclose(
                output.reshape(2, 21, 3)[:, :, 2],
                features.reshape(2, 21, 3)[:, :, 2],
            )


if __name__ == "__main__":
    unittest.main()
