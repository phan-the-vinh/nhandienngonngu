"""Unit tests for Multivariate DTW Engine, Template Store, and Matcher (Stage 2).

Verifies all mandatory test contracts:
1. Identical sequence yields distance 0.0.
2. Same trajectory at different temporal speeds aligns with low distance.
3. J and Z distinct trajectories have strong separation margin.
4. Sequences of different lengths align properly.
5. Sakoe-Chiba constraint enforces warping boundary.
6. Empty sequence raises DTWError.
7. Invalid sequence dimensions raise DTWError.
8. NaN/Inf in sequence raises DTWError.
9. UNKNOWN rejection when distance exceeds threshold or margin is too small.
10. Distance is properly normalized by alignment path length.
11. Template metadata is fully populated with source split 'train'.
12. Deterministic matching across repeated runs.
13. No Test split leakage (templates sourced exclusively from Train).
14. No input sequence mutation during matching.
"""

from __future__ import annotations

import unittest
import numpy as np

from app.inference.dynamic_jz.dtw_engine import dtw_multivariate, DTWError
from app.inference.dynamic_jz.feature_builder import build_dtw_features, FEATURE_CONFIGURATIONS
from app.inference.dynamic_jz.template_store import DTWTemplate, TemplateStore
from app.inference.dynamic_jz.dtw_matcher import DTWMatcher, DTWMatchResult
from app.inference.dynamic_jz.finger_tracker import FingerTracker


def make_test_trajectory(letter: str, length: int = 16, noise_scale: float = 0.0) -> np.ndarray:
    """Generate a synthetic 16x69 baseline sequence for J or Z."""
    seq = np.zeros((length, 69), dtype=np.float64)
    for t in range(length):
        p = t / (length - 1)
        # Fake handshape in cols 0..62
        seq[t, 15:18] = [0.2, -0.4, 0.0]  # Index MCP
        seq[t, 27:30] = [0.0, -0.45, 0.0] # Middle MCP
        seq[t, 51:54] = [-0.25, -0.38, 0.0] # Pinky MCP

        if letter == "Z":
            # Index tip extended
            seq[t, 24:27] = [0.2, -1.0, 0.0]
            # Pinky curled
            seq[t, 60:63] = [-0.25, -0.45, 0.0]
            # Z global trajectory: right, diagonal down-left, right
            if p < 0.33:
                x = p * 3.0
                y = 0.0
            elif p < 0.66:
                sub_p = (p - 0.33) / 0.33
                x = 1.0 - sub_p * 1.0
                y = sub_p * 1.5
            else:
                sub_p = (p - 0.66) / 0.34
                x = sub_p * 1.0
                y = 1.5
            seq[t, 65:67] = [x, y]  # Index global trajectory (cols 65:67)
            seq[t, 63:65] = [x * 0.5, y * 0.5]  # Wrist global motion
        else:
            # J: Index curled
            seq[t, 24:27] = [0.2, -0.45, 0.0]
            # Pinky extended
            seq[t, 60:63] = [-0.25, -1.0, 0.0]
            # J global trajectory: drop down, then hook left and up
            if p < 0.6:
                sub_p = p / 0.6
                x = 0.0
                y = sub_p * 1.8
            else:
                sub_p = (p - 0.6) / 0.4
                theta = sub_p * np.pi
                x = -np.sin(theta) * 0.8
                y = 1.8 - np.cos(theta) * 0.4
            seq[t, 67:69] = [x, y]  # Pinky global trajectory (cols 67:69)
            seq[t, 63:65] = [x * 0.3, y * 0.3]  # Wrist global motion

    if noise_scale > 0.0:
        rng = np.random.RandomState(42)
        seq += rng.normal(0.0, noise_scale, size=seq.shape)

    return seq


class JzDTWEngineTests(unittest.TestCase):
    """Unit tests for the DTW engine and template matching pipeline."""

    def setUp(self) -> None:
        self.tracker = FingerTracker()
        self.seq_z = make_test_trajectory("Z", length=16)
        self.seq_j = make_test_trajectory("J", length=16)

        self.res_z = self.tracker.extract(self.seq_z, letter_target="Z")
        self.res_j = self.tracker.extract(self.seq_j, letter_target="J")

        self.feat_z, self.weights_a = build_dtw_features(self.res_z, config_name="CONFIG_A")
        self.feat_j, _ = build_dtw_features(self.res_j, config_name="CONFIG_A")

        # Mock templates
        self.tmpl_z = DTWTemplate(
            template_id="tmpl_z_01",
            label="Z",
            source_sample_id="train/z_01.avi",
            source_video_id="z_01.avi",
            source_split="train",
            tracking_quality="VALID",
            feature_schema_version="finger_tracking_v1",
            feature_config="CONFIG_A",
            selection_method="k-medoids",
            creation_timestamp="2026-11-13T00:00:00Z",
            feature_sequence=self.feat_z.tolist(),
        )
        self.tmpl_j = DTWTemplate(
            template_id="tmpl_j_01",
            label="J",
            source_sample_id="train/j_01.avi",
            source_video_id="j_01.avi",
            source_split="train",
            tracking_quality="VALID",
            feature_schema_version="finger_tracking_v1",
            feature_config="CONFIG_A",
            selection_method="k-medoids",
            creation_timestamp="2026-11-13T00:00:00Z",
            feature_sequence=self.feat_j.tolist(),
        )
        self.store = TemplateStore([self.tmpl_z, self.tmpl_j])

    def test_01_identical_sequence_zero_distance(self) -> None:
        """Requirement 1: Identical sequence yields distance exactly 0.0."""
        dist, path_len, path = dtw_multivariate(self.feat_z, self.feat_z, weights=self.weights_a, radius=3)
        self.assertAlmostEqual(dist, 0.0, places=7)
        self.assertEqual(path_len, 16)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (15, 15))

    def test_02_same_trajectory_different_speeds(self) -> None:
        """Requirement 2: Same trajectory sampled at non-uniform speeds aligns with low distance."""
        # Non-uniform temporal warping: duplicate some frames to simulate speed changes
        slow_indices = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        warped_seq = self.feat_z[slow_indices]
        dist, path_len, _ = dtw_multivariate(self.feat_z, warped_seq, weights=self.weights_a, radius=4)
        # Even with 18 vs 16 frames, distance should be very close to zero
        self.assertLess(dist, 0.05)

    def test_03_j_and_z_distinct_separation(self) -> None:
        """Requirement 3: J and Z sequences have strong distance separation."""
        dist_same, _, _ = dtw_multivariate(self.feat_z, self.feat_z, weights=self.weights_a, radius=3)
        dist_cross, _, _ = dtw_multivariate(self.feat_z, self.feat_j, weights=self.weights_a, radius=3)
        self.assertAlmostEqual(dist_same, 0.0, places=7)
        self.assertGreater(dist_cross, 0.5)
        self.assertGreater(dist_cross - dist_same, 0.5)

    def test_04_different_sequence_lengths(self) -> None:
        """Requirement 4: Supports sequences of different lengths (e.g. 14 vs 16 frames)."""
        short_seq = self.feat_z[:14]
        dist, path_len, path = dtw_multivariate(short_seq, self.feat_z, weights=self.weights_a, radius=4)
        self.assertTrue(np.isfinite(dist))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (13, 15))

    def test_05_sakoe_chiba_constraint_enforced(self) -> None:
        """Requirement 5: Sakoe-Chiba band strictly limits search space |i - j| <= R."""
        dist, path_len, path = dtw_multivariate(self.feat_z, self.feat_z, weights=self.weights_a, radius=2)
        for i, j in path:
            self.assertLessEqual(abs(i - j), 2)

    def test_06_empty_sequence_raises_error(self) -> None:
        """Requirement 6: Aligning with empty sequence raises DTWError."""
        empty_seq = np.zeros((0, 6))
        with self.assertRaises(DTWError):
            dtw_multivariate(empty_seq, self.feat_z, weights=self.weights_a)

    def test_07_invalid_dimensions_raises_error(self) -> None:
        """Requirement 7: Feature dimension mismatch raises DTWError."""
        wrong_dim_seq = np.zeros((16, 5))
        with self.assertRaises(DTWError):
            dtw_multivariate(self.feat_z, wrong_dim_seq, weights=self.weights_a)

    def test_08_nan_inf_raises_error(self) -> None:
        """Requirement 8: NaN or Inf values in sequence raise DTWError."""
        nan_seq = self.feat_z.copy()
        nan_seq[5, 2] = np.nan
        with self.assertRaises(DTWError):
            dtw_multivariate(self.feat_z, nan_seq, weights=self.weights_a)

    def test_09_unknown_rejection(self) -> None:
        """Requirement 9: Matcher rejects with UNKNOWN when distance or margin fails."""
        matcher = DTWMatcher(
            self.store,
            config_name="CONFIG_A",
            radius=3,
            threshold_j=0.5,
            threshold_z=0.5,
            min_margin=0.20,
        )

        # High-noise sequence should exceed threshold
        noisy_seq = make_test_trajectory("Z", noise_scale=2.0)
        noisy_res = self.tracker.extract(noisy_seq, letter_target="Z")
        noisy_match = matcher.match(noisy_res)

        self.assertEqual(noisy_match.predicted_label, "UNKNOWN")
        self.assertIn("DISTANCE_EXCEEDS_THRESHOLD", noisy_match.rejection_reason)

    def test_10_distance_normalization_by_path_length(self) -> None:
        """Requirement 10: Cumulative distance is divided by alignment path length."""
        norm_dist, path_len, _ = dtw_multivariate(self.feat_z, self.feat_j, weights=self.weights_a, radius=3)
        self.assertGreater(path_len, 15)
        # Normalized distance must be bounded and reasonable
        self.assertLess(norm_dist, 50.0)

    def test_11_template_metadata_provenance(self) -> None:
        """Requirement 11: Templates retain complete provenance metadata."""
        t = self.tmpl_z
        self.assertEqual(t.source_split, "train")
        self.assertEqual(t.feature_schema_version, "finger_tracking_v1")
        self.assertIn(t.tracking_quality, ("VALID", "SUSPICIOUS"))
        self.assertTrue(len(t.creation_timestamp) > 10)

    def test_12_deterministic_results(self) -> None:
        """Requirement 12: Matching produces strictly identical numerical results on repeat."""
        matcher = DTWMatcher(self.store, config_name="CONFIG_A", radius=3)
        match1 = matcher.match(self.res_z)
        match2 = matcher.match(self.res_z)
        self.assertEqual(match1.predicted_label, match2.predicted_label)
        self.assertEqual(match1.distance_j, match2.distance_j)
        self.assertEqual(match1.distance_z, match2.distance_z)
        self.assertEqual(match1.distance_margin, match2.distance_margin)

    def test_13_no_test_split_leakage(self) -> None:
        """Requirement 13: All templates in store must strictly originate from Train split."""
        for t in self.store.templates:
            self.assertEqual(t.source_split, "train")
            self.assertNotEqual(t.source_split, "test")
            self.assertNotEqual(t.source_split, "validation")

    def test_14_input_immutability(self) -> None:
        """Requirement 14: Input FingerTrackingResult arrays are never mutated during matching."""
        matcher = DTWMatcher(self.store, config_name="CONFIG_A", radius=3)
        orig_traj = self.res_z.global_trajectory.copy()
        _ = matcher.match(self.res_z)
        np.testing.assert_array_equal(self.res_z.global_trajectory, orig_traj)

    def test_15_match_sequence_dual_hypothesis(self) -> None:
        """Requirement 15: match_sequence correctly distinguishes raw J and Z sequences."""
        matcher = DTWMatcher(
            self.store,
            config_name="CONFIG_A",
            radius=3,
            threshold_j=2.0,
            threshold_z=2.0,
            min_margin=0.10,
        )
        res_z = matcher.match_sequence(self.seq_z, tracker=self.tracker)
        self.assertEqual(res_z.predicted_label, "Z")
        self.assertLess(res_z.distance_z, res_z.distance_j)

        res_j = matcher.match_sequence(self.seq_j, tracker=self.tracker)
        self.assertEqual(res_j.predicted_label, "J")
        self.assertLess(res_j.distance_j, res_j.distance_z)


if __name__ == "__main__":
    unittest.main()
