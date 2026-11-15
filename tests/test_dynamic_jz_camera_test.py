"""Offline tests for the experimental J/Z camera; never open a real camera."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.inference import dynamic_jz_camera_test as camera_test
from ai.feature_extraction.dynamic_sequence import sample_valid_frame_indices


class FakeModel:
    classes_ = np.asarray(["J", "Z"])

    def predict_proba(self, features):
        assert features.shape == (1, 1104)
        assert np.isfinite(features).all()
        return np.asarray([[0.3, 0.7]])


def synthetic_entries(indices):
    entries = []
    for frame_index in indices:
        hand = [SimpleNamespace(x=0.35 + (i % 5) * 0.02 + frame_index * 0.001,
                                y=0.35 + (i // 5) * 0.02, z=i * 0.001)
                for i in range(21)]
        entries.append((frame_index, hand, (640, 480)))
    return entries


class DynamicJzCameraTest(unittest.TestCase):
    def test_valid_frame_sampling_matches_official_sequence(self):
        valid_indices = list(range(0, 40, 2))
        result = camera_test.predict_attempt(synthetic_entries(valid_indices), 40, FakeModel())
        self.assertEqual(result["status"], "PREDICTED")
        self.assertEqual(result["selected_original_frame_indices"],
                         sample_valid_frame_indices(valid_indices))
        self.assertEqual(result["prediction"], "Z")
        self.assertEqual(result["probability_J"], 0.3)

    def test_insufficient_valid_frames_never_predicts(self):
        result = camera_test.predict_attempt(synthetic_entries(list(range(15))), 20, FakeModel())
        self.assertEqual(result["status"], "INVALID_ATTEMPT")
        self.assertEqual(result["failure_reason"], "INSUFFICIENT_VALID_LANDMARK_FRAMES")

    def test_check_only_never_opens_camera_or_writes_log(self):
        with patch.object(camera_test.cv2, "VideoCapture", side_effect=AssertionError("camera opened")):
            with patch.object(camera_test, "_append_trial", side_effect=AssertionError("log written")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(camera_test.main(["--check-only", "--save-results"]), 0)
        self.assertFalse(json.loads(output.getvalue())["camera_opened"])


if __name__ == "__main__":
    unittest.main()
