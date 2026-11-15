"""Synthetic-only tests for saved-sequence quality audit helpers."""
import unittest

from tools.audit_dynamic_sequence_quality import metrics, percentile, validate


def sample_record():
    frames = [[0.0] * 69 for _ in range(16)]
    for t, frame in enumerate(frames):
        frame[63] = float(t)       # wrist x
        frame[65] = float(2 * t)   # index-tip x
        frame[68] = float(3 * t)   # little-tip y
    return {
        "source_path": "data/raw/dynamic/external/signnteam_jz/SigNN Video Data/J/1.avi",
        "label": "J", "z_variant": "NOT_APPLICABLE",
        "extraction_status": "SUCCESS", "sequence_length": 16, "feature_dim": 69,
        "selected_original_frame_indices": list(range(16)),
        "decoded_frame_count": 16, "valid_landmark_frames": 16,
        "hand_scale_px": 80.0, "features": frames,
    }


class SequenceQualityHelperTests(unittest.TestCase):
    def test_saved_feature_schema_and_motion_metrics(self):
        row = sample_record()
        key, indices, features, errors = validate(row)
        self.assertEqual(key, "J/1.avi")
        self.assertEqual(errors, [])
        values = metrics(features, indices, "J")
        self.assertAlmostEqual(values["landmark_0_path"], 15.0)
        self.assertAlmostEqual(values["landmark_8_path"], 30.0)
        self.assertAlmostEqual(values["landmark_20_path"], 45.0)
        self.assertAlmostEqual(values["label_tip_path"], 45.0)
        self.assertEqual(values["max_selected_frame_gap"], 1)
        self.assertAlmostEqual(values["shape_rms"], 0.0)

    def test_invalid_indices_and_nonfinite_feature_are_reported(self):
        row = sample_record()
        row["selected_original_frame_indices"][4] = 3
        row["features"][5][68] = float("nan")
        _, indices, features, errors = validate(row)
        self.assertEqual(indices, [])
        self.assertEqual(features, [])
        self.assertIn("INVALID_FRAME_INDICES", errors)
        self.assertIn("NONFINITE_OR_NONNUMERIC_FEATURE", errors)

    def test_percentile_is_deterministic(self):
        self.assertEqual(percentile([4.0, 1.0, 2.0, 3.0], 0.5), 2.5)


if __name__ == "__main__":
    unittest.main()
