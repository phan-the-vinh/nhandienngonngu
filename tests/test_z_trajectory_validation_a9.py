import os
import json
import unittest
from pathlib import Path
import numpy as np

from app.inference.integrated_jz_validation import validate_z_trajectory, ZTrajectorySettings

ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = str(ROOT / "data" / "experiments" / "dynamic_jz_v1" / "diagnostics" / "20261108T143709941895Z_44ef47a7")


def _load_replay_features(attempt_id: int) -> np.ndarray:
    folder = os.path.join(DIAG_PATH, f"generation_000_attempt_{attempt_id:05d}")
    npz_path = os.path.join(folder, "replay_data.npz")
    if not os.path.exists(npz_path):
        raise unittest.SkipTest(f"Replay data not found at {npz_path}")
    npz = np.load(npz_path)
    return npz["features"]


class TestRealWebcamReplayZ(unittest.TestCase):
    """Test real webcam telemetry from A7 session."""

    def test_clean_real_z_passes(self):
        """Attempts 23, 34, 46 must all be validated as genuine Z."""
        clean_attempts = [23, 34, 46]
        for att in clean_attempts:
            feat = _load_replay_features(att)
            res = validate_z_trajectory(feat)
            assert res["valid"] is True, f"Attempt {att} failed: {res['reason']}"
            assert res["start_index"] in range(0, 5)
            assert res["corner1_index"] > res["start_index"]
            assert res["corner2_index"] > res["corner1_index"]
            assert res["endpoint_index"] > res["corner2_index"]

    def test_incomplete_real_z_rejected(self):
        """Incomplete Z gestures must be blocked (0/6 PASS)."""
        incomplete_attempts = [24, 27, 32, 33, 49, 50]
        for att in incomplete_attempts:
            feat = _load_replay_features(att)
            res = validate_z_trajectory(feat)
            assert res["valid"] is False, f"Incomplete attempt {att} was falsely validated!"

    def test_static_l_attempts_rejected(self):
        """Static L holds with jitter must NEVER be validated as Z (0/12 PASS)."""
        l_attempts = [4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 20]
        for att in l_attempts:
            feat = _load_replay_features(att)
            res = validate_z_trajectory(feat)
            assert res["valid"] is False, f"Static L attempt {att} was falsely validated as Z!"


class TestSyntheticZTrajectoryGeometry(unittest.TestCase):
    """Unit tests for geometric constraints in validate_z_trajectory."""

    def _create_synthetic_features(self, path_2d: np.ndarray) -> np.ndarray:
        features = np.zeros((16, 69), dtype=np.float32)
        features[:, 65:67] = path_2d
        return features

    def test_dynamic_t0_with_quiescent_leadin(self):
        """A Z that starts at frame 2 with minor lead-in drift must pass."""
        # Frames 0..2: stationary with minor drift <= 0.05
        # Frame 2..6: top stroke (dx=0.30, dy=-0.10)
        # Frame 6..11: diagonal stroke (dx=-0.25, dy=0.30)
        # Frame 11..15: bottom stroke (dx=0.30, dy=0.05)
        path = np.zeros((16, 2), dtype=np.float32)
        path[0] = [0.0, 0.0]
        path[1] = [0.02, 0.01]
        path[2] = [0.03, 0.02]  # t0 = 2

        # top stroke: frame 2 to 6
        for i in range(3, 7):
            frac = (i - 2) / 4.0
            path[i] = path[2] + [0.30 * frac, -0.10 * frac]

        # diagonal: frame 6 to 11
        for i in range(7, 12):
            frac = (i - 6) / 5.0
            path[i] = path[6] + [-0.25 * frac, 0.30 * frac]

        # bottom stroke: frame 11 to 15
        for i in range(12, 16):
            frac = (i - 11) / 4.0
            path[i] = path[11] + [0.30 * frac, 0.05 * frac]

        feat = self._create_synthetic_features(path)
        res = validate_z_trajectory(feat)
        assert res["valid"] is True
        assert res["start_index"] in (0, 1, 2)

    def test_dynamic_t0_rejected_if_lead_in_has_large_horizontal_drift(self):
        """If lead-in has horizontal drift > 0.15, starting at t0=3 must be blocked."""
        path = np.zeros((16, 2), dtype=np.float32)
        # Large spurious horizontal motion before stroke
        path[0] = [0.0, 0.0]
        path[1] = [0.40, 0.0]  # drift = 0.40 > 0.15, also breaks stroke 0 monotonicity for t0 < 3
        path[2] = [0.0, 0.0]
        path[3] = [0.0, 0.0]  # attempt start here

        # top stroke from 3 to 7
        for i in range(4, 8):
            frac = (i - 3) / 4.0
            path[i] = path[3] + [0.30 * frac, 0.0]

        # diagonal from 7 to 11
        for i in range(8, 12):
            frac = (i - 7) / 4.0
            path[i] = path[7] + [-0.25 * frac, 0.35 * frac]

        # bottom stroke from 11 to 15
        for i in range(12, 16):
            frac = (i - 11) / 4.0
            path[i] = path[11] + [0.30 * frac, 0.0]

        feat = self._create_synthetic_features(path)
        settings = ZTrajectorySettings(max_lead_in_horizontal_drift=0.15)
        res = validate_z_trajectory(feat, settings=settings)
        # Should be rejected because frame 1 violates lead-in horizontal drift
        assert res["valid"] is False

    def test_upward_slope_boundary(self):
        """Upward slope <= 2.85 is accepted; > 2.85 is rejected."""
        def make_path(top_dx, top_dy):
            p = np.zeros((16, 2), dtype=np.float32)
            p[0] = [0.0, 0.0]
            p[4] = [top_dx, top_dy]
            p[10] = [top_dx - 0.30, top_dy + 0.50]
            p[15] = [top_dx, top_dy + 0.50]
            for i in range(1, 4):
                p[i] = p[0] + (p[4] - p[0]) * (i / 4.0)
            for i in range(5, 10):
                p[i] = p[4] + (p[10] - p[4]) * ((i - 4) / 6.0)
            for i in range(11, 15):
                p[i] = p[10] + (p[15] - p[10]) * ((i - 10) / 5.0)
            return self._create_synthetic_features(p)

        # slope = 0.50 / 0.20 = 2.50 <= 2.85 (pass)
        res_pass = validate_z_trajectory(make_path(0.20, -0.50))
        assert res_pass["valid"] is True

        # slope = 0.70 / 0.20 = 3.50 > 2.85 (fail)
        res_fail = validate_z_trajectory(make_path(0.20, -0.70))
        assert res_fail["valid"] is False

    def test_chord_ratio_rejects_erratic_jitter(self):
        """Erratic noise path with chord ratio > 1.85 must be rejected."""
        # Clean Z baseline
        path = np.zeros((16, 2), dtype=np.float32)
        path[0] = [0.0, 0.0]
        path[5] = [0.30, 0.0]
        path[10] = [0.05, 0.30]
        path[15] = [0.35, 0.30]
        # Linear fill
        for i in range(1, 5):
            path[i] = path[0] + (path[5] - path[0]) * (i / 5.0)
        for i in range(6, 10):
            path[i] = path[5] + (path[10] - path[5]) * ((i - 5) / 5.0)
        for i in range(11, 15):
            path[i] = path[10] + (path[15] - path[10]) * ((i - 10) / 5.0)

        # Add heavy zigzag jitter to make path length exceed 1.85x chord sum
        for i in range(1, 15):
            if i % 2 == 1:
                path[i, 1] += 0.25
            else:
                path[i, 1] -= 0.25

        feat = self._create_synthetic_features(path)
        res = validate_z_trajectory(feat)
        assert res["valid"] is False


if __name__ == "__main__":
    unittest.main()
