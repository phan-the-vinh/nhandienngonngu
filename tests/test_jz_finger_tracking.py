"""Unit tests for standalone Finger Tracking Engine (Stage 1).

Verifies all 13 mandatory test contracts:
1. J uses landmark 20 (pinky tip).
2. Z uses landmark 8 (index tip).
3. Global and wrist-relative trajectories differ when hand translates.
4. Correct normalization matching original preprocessing.
5. No double normalization applied.
6. Missing landmarks detection.
7. Duplicate frames detection.
8. Frame indices out-of-order rejection.
9. Sequence with insufficient frames (< 16) rejected.
10. No NaN or Inf produced under valid inputs.
11. Input sequence immutability (no in-place modification).
12. Handshape computed per-frame.
13. No synthetic timestamps used.
"""

from __future__ import annotations

import unittest
import numpy as np

from app.inference.dynamic_jz.finger_tracker import (
    FingerTracker,
    FingerTrackingResult,
    FingerTrackingError,
    extract_finger_tracking,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
)


def create_synthetic_hand_sequence(
    translation_x: float = 0.5,
    translation_y: float = 0.5,
    active_finger_length: float = 0.8,
) -> np.ndarray:
    """Create a physically plausible 16x69 synthetic hand sequence.

    - Landmarks 0..20 generated with wrist at (0,0,0) and MCPs around radius 0.4.
    - Active finger tips extended.
    - Global trajectory simulates smooth translation.
    """
    seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)

    # Base hand template (relative to wrist)
    # Landmark 0: wrist at (0, 0, 0)
    # Landmark 5 (Index MCP): (0.2, -0.4, 0.0)
    # Landmark 8 (Index TIP): (0.2, -0.8, 0.0)
    # Landmark 9 (Middle MCP): (0.0, -0.45, 0.0)
    # Landmark 12 (Middle TIP): (0.0, -0.5, 0.0) [curled]
    # Landmark 13 (Ring MCP): (-0.15, -0.42, 0.0)
    # Landmark 16 (Ring TIP): (-0.15, -0.5, 0.0) [curled]
    # Landmark 17 (Pinky MCP): (-0.25, -0.38, 0.0)
    # Landmark 20 (Pinky TIP): (-0.25, -0.8, 0.0)
    # Landmark 4 (Thumb TIP): (0.3, -0.2, 0.0)

    for t in range(SEQUENCE_LENGTH):
        progress = t / (SEQUENCE_LENGTH - 1)

        # Columns 0..62: relative landmarks normalized by D_max
        # 5: Index MCP
        seq[t, 15:18] = [0.2, -0.4, 0.0]
        # 8: Index TIP
        seq[t, 24:27] = [0.2, -0.4 - active_finger_length, 0.0]
        # 9: Middle MCP
        seq[t, 27:30] = [0.0, -0.45, 0.0]
        # 12: Middle TIP
        seq[t, 36:39] = [0.0, -0.5, 0.0]
        # 13: Ring MCP
        seq[t, 39:42] = [-0.15, -0.42, 0.0]
        # 16: Ring TIP
        seq[t, 48:51] = [-0.15, -0.5, 0.0]
        # 17: Pinky MCP
        seq[t, 51:54] = [-0.25, -0.38, 0.0]
        # 20: Pinky TIP
        seq[t, 60:63] = [-0.25, -0.38 - active_finger_length, 0.0]
        # 4: Thumb TIP
        seq[t, 12:15] = [0.3, -0.2, 0.0]

        # Columns 63..68: Global trajectory displacements from frame 0 / S_palm
        # Wrist displacement (moving right and down)
        seq[t, 63:65] = [progress * translation_x, progress * translation_y]
        # Index tip displacement (moving in a Z-stroke: right, diagonal down-left, right)
        seq[t, 65:67] = [progress * (translation_x + 0.3), progress * (translation_y - 0.2)]
        # Pinky tip displacement (moving in a J-stroke: down and hook left)
        seq[t, 67:69] = [progress * (translation_x - 0.4), progress * (translation_y + 0.6)]

    return seq


class JzFingerTrackingTests(unittest.TestCase):
    """Test suite verifying the finger tracking engine implementation."""

    def setUp(self) -> None:
        self.tracker = FingerTracker()
        self.seq = create_synthetic_hand_sequence()
        self.frame_indices = [i * 2 for i in range(16)]

    def test_01_j_uses_landmark_20(self) -> None:
        """Requirement 1: J must extract pinky tip (Landmark 20) as active trajectory."""
        res = self.tracker.extract(self.seq, letter_target="J")
        self.assertEqual(res.letter_target, "J")

        # Global trajectory must match columns 67..68 (pinky displacement)
        expected_global = self.seq[:, 67:69]
        np.testing.assert_allclose(res.global_trajectory, expected_global)

        # Wrist-relative trajectory must match columns 60..62 (landmark 20)
        expected_wrist_rel = self.seq[:, 60:63]
        np.testing.assert_allclose(res.wrist_relative_trajectory, expected_wrist_rel)

        # Active extension must track pinky extension
        self.assertEqual(res.handshape_features.endpoint_handshape["active_finger"], "pinky")
        self.assertGreater(res.handshape_features.endpoint_handshape["active_extension"], 1.0)

    def test_02_z_uses_landmark_8(self) -> None:
        """Requirement 2: Z must extract index tip (Landmark 8) as active trajectory."""
        res = self.tracker.extract(self.seq, letter_target="Z")
        self.assertEqual(res.letter_target, "Z")

        # Global trajectory must match columns 65..66 (index displacement)
        expected_global = self.seq[:, 65:67]
        np.testing.assert_allclose(res.global_trajectory, expected_global)

        # Wrist-relative trajectory must match columns 24..26 (landmark 8)
        expected_wrist_rel = self.seq[:, 24:27]
        np.testing.assert_allclose(res.wrist_relative_trajectory, expected_wrist_rel)

        # Active extension must track index extension
        self.assertGreater(res.handshape_features.endpoint_handshape["active_extension"], 1.0)

    def test_03_global_and_wrist_relative_differ_when_translating(self) -> None:
        """Requirement 3: Global trajectory reflects hand translation; wrist-relative does not."""
        # When hand translates across space, wrist-relative coordinates remain constant
        # if the finger doesn't articulate, while global displacement changes.
        res = self.tracker.extract(self.seq, letter_target="Z")

        # Global trajectory changes from frame 0 to frame 15
        global_disp_magnitude = np.linalg.norm(res.global_trajectory[-1] - res.global_trajectory[0])
        self.assertGreater(global_disp_magnitude, 0.2)

        # Wrist-relative coordinates in our static-pose sequence remain constant
        wrist_rel_change = np.linalg.norm(res.wrist_relative_trajectory[-1] - res.wrist_relative_trajectory[0])
        self.assertAlmostEqual(wrist_rel_change, 0.0, places=5)

        # Global and wrist-relative must have completely distinct shapes and semantics
        self.assertEqual(res.global_trajectory.shape, (16, 2))
        self.assertEqual(res.wrist_relative_trajectory.shape, (16, 3))

    def test_04_normalization_matches_original_preprocessing(self) -> None:
        """Requirement 4: Features conform exactly to project normalization contracts."""
        res = self.tracker.extract(self.seq, letter_target="J")

        # At frame 0, global trajectory and wrist motion must be exactly [0, 0]
        np.testing.assert_allclose(res.global_trajectory[0], [0.0, 0.0])
        np.testing.assert_allclose(res.wrist_motion[0], [0.0, 0.0])

        # Handshape palm scale is computed from median distance to MCPs 5, 9, 17
        mcp_pts = self.seq[0, [15, 16, 17, 27, 28, 29, 51, 52, 53]].reshape((3, 3))
        expected_palm_scale = float(np.median(np.linalg.norm(mcp_pts, axis=1)))
        actual_palm_scale = res.handshape_features.endpoint_handshape["palm_scale"]
        self.assertAlmostEqual(actual_palm_scale, expected_palm_scale, places=5)

    def test_05_no_double_normalization(self) -> None:
        """Requirement 5: Pre-normalized features in input are never re-divided by palm scale."""
        res = self.tracker.extract(self.seq, letter_target="Z")

        # Global trajectory values must match input sequence columns 65:67 without scaling
        self.assertEqual(res.global_trajectory[10, 0], self.seq[10, 65])
        self.assertEqual(res.global_trajectory[10, 1], self.seq[10, 66])

        # Wrist motion must match input sequence columns 63:65 without scaling
        self.assertEqual(res.wrist_motion[10, 0], self.seq[10, 63])
        self.assertEqual(res.wrist_motion[10, 1], self.seq[10, 64])

    def test_06_missing_landmarks_detection(self) -> None:
        """Requirement 6: Detect missing landmarks when hand coordinates are zeroed."""
        corrupted = self.seq.copy()
        # Zero out landmarks in frame 5
        corrupted[5, :63] = 0.0

        res = self.tracker.extract(corrupted, letter_target="Z")
        self.assertEqual(res.tracking_quality.missing_landmark_count, 1)
        self.assertEqual(res.validity_status, "INVALID")
        self.assertIn("MISSING_LANDMARKS", res.invalidation_reason)

    def test_07_duplicate_frames_detection(self) -> None:
        """Requirement 7: Detect duplicate / frozen tracking frames."""
        corrupted = self.seq.copy()
        # Duplicate frame 2 into frames 3, 4, 5, 6, 7
        for t in range(3, 8):
            corrupted[t] = corrupted[2]

        res = self.tracker.extract(corrupted, letter_target="J")
        self.assertEqual(res.tracking_quality.duplicate_frame_count, 5)
        self.assertEqual(res.validity_status, "SUSPICIOUS")
        self.assertIn("EXCESSIVE_DUPLICATES", res.invalidation_reason)

    def test_08_frame_indices_out_of_order_rejected(self) -> None:
        """Requirement 8: Reject non-monotonic or disordered original frame indices."""
        disordered_indices = [0, 2, 4, 3, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

        res = self.tracker.extract(self.seq, letter_target="Z", selected_original_frame_indices=disordered_indices)
        self.assertEqual(res.tracking_quality.frame_indices_monotonic, "FAIL")
        self.assertEqual(res.validity_status, "INVALID")
        self.assertIn("NON_MONOTONIC_FRAME_INDICES", res.invalidation_reason)

    def test_09_sequence_with_insufficient_frames_rejected(self) -> None:
        """Requirement 9: Sequence shorter than 16 frames must raise FingerTrackingError."""
        short_seq = self.seq[:12, :]
        with self.assertRaises(FingerTrackingError) as ctx:
            self.tracker.extract(short_seq, letter_target="J")
        self.assertIn("Invalid sequence shape", str(ctx.exception))

    def test_10_no_nan_or_inf_produced(self) -> None:
        """Requirement 10: Valid sequence produces zero NaNs or Infs across all feature groups."""
        res = self.tracker.extract(self.seq, letter_target="Z", selected_original_frame_indices=self.frame_indices)

        self.assertFalse(res.tracking_quality.has_nan_inf)
        self.assertTrue(np.all(np.isfinite(res.global_trajectory)))
        self.assertTrue(np.all(np.isfinite(res.wrist_motion)))
        self.assertTrue(np.all(np.isfinite(res.wrist_relative_trajectory)))
        self.assertTrue(np.all(np.isfinite(res.handshape_features.active_extension)))
        self.assertTrue(np.all(np.isfinite(res.handshape_features.curled_extension_mean)))
        self.assertTrue(np.all(np.isfinite(res.velocity_profile.delta_displacement)))
        self.assertTrue(np.all(np.isfinite(res.velocity_profile.speed_per_frame)))

    def test_11_input_sequence_immutability(self) -> None:
        """Requirement 11: The input array must never be mutated in-place."""
        original = self.seq.copy()
        _ = self.tracker.extract(self.seq, letter_target="J")

        # Verify exact equality before and after extraction
        np.testing.assert_array_equal(self.seq, original)

    def test_12_handshape_computed_per_frame(self) -> None:
        """Requirement 12: Handshape features vary when finger articulates over time."""
        articulated_seq = self.seq.copy()
        # Progressively extend pinky over time in articulated sequence
        for t in range(SEQUENCE_LENGTH):
            ext_factor = 0.5 + 0.8 * (t / (SEQUENCE_LENGTH - 1))
            articulated_seq[t, 60:63] = [-0.25, -0.38 - ext_factor, 0.0]

        res = self.tracker.extract(articulated_seq, letter_target="J")
        active_ext = res.handshape_features.active_extension

        # Handshape at frame 0 must be strictly less than handshape at frame 15
        self.assertLess(active_ext[0], active_ext[-1])
        # Values must be computed for all 16 distinct frames
        self.assertEqual(len(active_ext), 16)
        self.assertNotEqual(active_ext[0], active_ext[8])

    def test_13_no_synthetic_timestamps(self) -> None:
        """Requirement 13: Engine must not invent synthetic millisecond timestamps."""
        res = self.tracker.extract(self.seq, letter_target="Z")

        # Verify no timestamp attribute exists in the output contract
        self.assertFalse(hasattr(res, "timestamps"))
        self.assertFalse(hasattr(res, "time_ms"))
        self.assertFalse(hasattr(res, "fps"))

        # Displacement deltas must be frame-relative, not per-second
        self.assertEqual(res.velocity_profile.delta_displacement.shape, (16, 2))


if __name__ == "__main__":
    unittest.main()
