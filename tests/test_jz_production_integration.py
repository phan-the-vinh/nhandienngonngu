"""Production integration tests for J/Z across Groups A (Static), B (Dynamic), C (Regression), D (Integration)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai.feature_extraction.dynamic_sequence import SEQUENCE_LENGTH, FEATURE_DIM
from app.camera import realtime_inference
from app.inference import integrated_jz_validation as jz_val
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from app.inference.unified_abc123_jz_auto_experiment import prepare_inference_frame
from app.interaction.hold_confirmation import HoldConfirmation, HoldSettings
from src.model import APPLICATION_CONFIG, load_configuration
from src.recognition.alphabet import AlphabetRecognizer
from tests.test_dynamic_jz_motion_gate import curved_hand, hand_at
from tests.test_static_l_protection import static_l_hand
from tests.test_unified_abc123_jz_auto import DynamicModel, mirrored
from tests.test_unified_camera_experiment import FakePredictor, synthetic_hand
from training.dynamic.evaluate_dynamic import validate_experiment

ROOT = Path(__file__).resolve().parent.parent


def _signer_mirrored_curved_z(step_idx: int):
    base = curved_hand(step_idx, "Z")
    return [SimpleNamespace(x=1.0 - p.x, y=p.y, z=p.z) for p in base]


def _make_static_features(path_xy: np.ndarray) -> np.ndarray:
    feat = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float64)
    feat[:, 65:67] = path_xy
    feat[:, 67:69] = path_xy
    return feat


def _hold_static(coord: UnifiedAutoCoordinator, idx: int, label: str, shape_val: float,
                 hand_fn, duration_s: float = 3.2, fps: float = 30.0) -> tuple[int, list[str]]:
    if coord.mode == "ABC":
        coord.static.predictors["ABC"].label = label
        coord.static.predictors["ABC"].shape = shape_val
    else:
        coord.static.predictors["123"].label = label
        coord.static.predictors["123"].shape = shape_val
    commits = []
    frames = int(round(duration_s * fps)) + 1
    for _ in range(frames):
        raw = hand_fn()
        _pred, committed = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / fps, raw_hand_count=1)
        if committed:
            commits.append(committed)
        idx += 1
    return idx, commits


def _run_dynamic_bout(coord: UnifiedAutoCoordinator, idx: int, label: str, *,
                      signer_mirrored_z: bool = False,
                      motion_frames: int = 80,
                      hold_frames: int = 120,
                      fps: float = 30.0) -> tuple[int, dict]:
    step = 1.8 if label == "Z" else 1.0
    for k in range(motion_frames):
        t_param = min(int(k * step), 60)
        if label == "Z" and signer_mirrored_z:
            hand = _signer_mirrored_curved_z(t_param)
        else:
            hand = curved_hand(t_param, label)
        coord.update(idx, mirrored(hand), hand, (480, 640, 3), idx / fps, raw_hand_count=1)
        idx += 1

    terminal = _signer_mirrored_curved_z(60) if (label == "Z" and signer_mirrored_z) else curved_hand(60, label)
    early_commit = False
    hold_started = False
    commits = []
    for _ in range(hold_frames):
        now = idx / fps
        _pred, committed = coord.update(idx, mirrored(terminal), terminal, (480, 640, 3), now, raw_hand_count=1)
        idx += 1
        if coord.jz_hold.state in ("STABILIZING", "HOLDING") and not committed:
            hold_started = True
        if committed:
            commits.append(committed)
            if coord.jz_hold._candidate_started_at is not None and (now - coord.jz_hold._candidate_started_at) < 2.95:
                early_commit = True
    return idx, {
        "commits": commits,
        "hold_started": hold_started,
        "early_commit": early_commit,
        "text": coord.composer.text,
    }


def _release_hand(coord: UnifiedAutoCoordinator, idx: int, frames: int = 25, fps: float = 30.0) -> int:
    for _ in range(frames):
        coord.update(idx, None, None, (480, 640, 3), idx / fps, raw_hand_count=0)
        idx += 1
    return idx


class GroupAStaticTests(unittest.TestCase):
    """Group A: Static ABC, 123, and I/L behavior under production unified coordinator."""

    def test_abc_static_letters_commit_with_experimental_jz_enabled(self):
        coord = AlphabetRecognizer(
            FakePredictor("A", 0.0), FakePredictor("0", 0.5), DynamicModel("Z"),
            enable_experimental_jz=True,
        )
        idx = 0
        for ch, shp in (("A", 0.0), ("B", 0.3), ("C", 0.6)):
            idx, commits = _hold_static(coord, idx, ch, shp, lambda: hand_at(0.0))
            self.assertEqual(commits, [ch])
            idx = _release_hand(coord, idx)
        self.assertEqual(coord.composer.text, "ABC")

    def test_123_digits_commit_and_never_touch_dynamic_model(self):
        dyn = DynamicModel("Z")
        coord = AlphabetRecognizer(FakePredictor("A", 0.0), FakePredictor("1", 0.1), dyn, enable_experimental_jz=True)
        coord.switch_mode("123")
        idx = 0
        for ch, shp in (("1", 0.1), ("2", 0.4), ("3", 0.7)):
            idx, commits = _hold_static(coord, idx, ch, shp, lambda: synthetic_hand())
            self.assertEqual(commits, [ch])
            idx = _release_hand(coord, idx)
        self.assertEqual(coord.composer.text, "123")
        self.assertEqual(dyn.inputs, [])

    def test_static_i_and_l_with_micro_jitter_commit_without_false_jz(self):
        for label_char, shp, hand_fn in (("I", 0.15, lambda: hand_at(0.0)), ("L", 0.45, static_l_hand)):
            dyn = DynamicModel("Z" if label_char == "L" else "J")
            coord = AlphabetRecognizer(FakePredictor(label_char, shp), FakePredictor("0", 0.5), dyn, enable_experimental_jz=True)
            commits = []
            for idx in range(135):  # 4.5s > 2s historical bug & > 3.0s hold
                raw = hand_fn()
                dx = 0.0005 * np.sin(idx * 0.4)
                for p in raw:
                    p.x += dx
                _, c = coord.update(idx, mirrored(raw), raw, (480, 640, 3), idx / 30.0, raw_hand_count=1)
                if c:
                    commits.append(c)
            self.assertEqual(commits, [label_char])
            self.assertEqual(coord.composer.text, label_char)


class GroupBDynamicTests(unittest.TestCase):
    """Group B: Dynamic J and Z (both viewer-LR and signer/mirrored-LR orientations)."""

    def test_j_trajectory_and_end_to_end_commit_preserved(self):
        coord = AlphabetRecognizer(
            FakePredictor("I", 0.0), FakePredictor("0", 0.5), DynamicModel("J"),
            enable_experimental_jz=True,
        )
        _idx, out = _run_dynamic_bout(coord, 0, "J")
        self.assertTrue(out["hold_started"])
        self.assertFalse(out["early_commit"])
        self.assertEqual(out["commits"], ["J"])
        self.assertEqual(out["text"], "J")

    def test_z_both_horizontal_orientations_end_to_end_and_signn_heldout_test(self):
        for mirrored_x, expected_orientation in ((False, "VIEWER_LR"), (True, "SIGNER_OR_MIRRORED_LR")):
            coord = AlphabetRecognizer(
                FakePredictor("I", 0.0), FakePredictor("0", 0.5), DynamicModel("Z"),
                enable_experimental_jz=True,
            )
            _idx, out = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=mirrored_x)
            self.assertTrue(out["hold_started"], expected_orientation)
            self.assertFalse(out["early_commit"], expected_orientation)
            self.assertEqual(out["commits"], ["Z"], expected_orientation)
            self.assertEqual(out["text"], "Z", expected_orientation)

        # Verify on held-out SigNN test set without altering thresholds or ML inputs
        _, records, X, _y, info = validate_experiment(ROOT / "data/experiments/dynamic_jz_binary_baseline_v1")
        z_test = [i for i in info["indices"]["test"] if records[i]["label"] == "Z"]
        j_test = [i for i in info["indices"]["test"] if records[i]["label"] == "J"]
        z_passed = [i for i in z_test if jz_val.validate_z_trajectory(X[i].reshape(16, 69))["valid"]]
        j_passed = [i for i in j_test if jz_val.validate_j_trajectory(X[i].reshape(16, 69))["valid"]]
        self.assertEqual(len(z_test), 37)
        self.assertEqual(len(z_passed), 36)
        self.assertEqual(len(j_test), 27)
        self.assertEqual(len(j_passed), 24)


class GroupCRegressionTests(unittest.TestCase):
    """Group C: Non-regression on I/L, duplicate commits, 3s hold, rearm, SPACE, TextComposer & Qwen hook."""

    def test_no_duplicate_commit_and_strict_3s_hold_then_release_rearms(self):
        coord = AlphabetRecognizer(
            FakePredictor("A", 0.0), FakePredictor("0", 0.5), DynamicModel("Z"),
            enable_experimental_jz=True,
        )
        # Check sub-3s boundary: hold finishes only after 3.0s, no repeat while held past 6.5s
        idx, out = _run_dynamic_bout(coord, 0, "Z", signer_mirrored_z=True, hold_frames=200)
        self.assertFalse(out["early_commit"])
        self.assertEqual(out["commits"], ["Z"])
        self.assertEqual(coord.composer.text, "Z")

        # Release hand -> rearm -> static A should commit cleanly as next sign
        idx = _release_hand(coord, idx, frames=25)
        idx, a_commits = _hold_static(coord, idx, "A", 0.8, lambda: hand_at(0.0))
        self.assertEqual(a_commits, ["A"])
        self.assertEqual(coord.composer.text, "ZA")

    def test_confusing_shapes_never_pass_z_validator(self):
        t = np.linspace(0.0, 1.0, 16)
        upward_z = np.stack([-np.concatenate([t[:5], 1 - t[5:10], t[10:]]), -t], axis=1)
        swipe = np.stack([t, np.zeros(16)], axis=1)
        l_shape = np.stack([np.concatenate([np.zeros(8), t[:8]]), np.concatenate([t[:8], np.ones(8)])], axis=1)
        for path in (upward_z, swipe, l_shape):
            self.assertFalse(jz_val.validate_z_trajectory(_make_static_features(path))["valid"])

    def test_space_text_composer_and_suggestion_manager_preserved(self):
        class StubSuggestionManager:
            def __init__(self):
                self.texts = []
                self.modes = []
                self.closed = False

            def update_text(self, text: str):
                self.texts.append(text)

            def set_mode(self, mode: str):
                self.modes.append(mode)

            def close(self):
                self.closed = True

        class StubSpacePredictor:
            def __init__(self, active: bool = False):
                self.active = active

            def predict_hand(self, _landmarks, mode: str = "ABC"):
                if mode == "ABC" and self.active:
                    return 0.99, True
                return 0.05, False

            def reset(self):
                return None

        sugg = StubSuggestionManager()
        space = StubSpacePredictor(active=False)
        coord = AlphabetRecognizer(
            FakePredictor("A", 0.0), FakePredictor("0", 0.5), DynamicModel("Z"),
            enable_experimental_jz=True,
            space_predictor=space,
            enable_space=True,
            suggestion_manager=sugg,
        )
        idx, _ = _hold_static(coord, 0, "A", 0.0, lambda: hand_at(0.0))
        idx = _release_hand(coord, idx)

        space.active = True
        idx, sp_commits = _hold_static(coord, idx, "A", 0.3, lambda: hand_at(0.0))
        self.assertEqual(sp_commits, ["SPACE"])
        space.active = False
        idx = _release_hand(coord, idx)

        idx, z_out = _run_dynamic_bout(coord, idx, "Z", signer_mirrored_z=True)
        self.assertEqual(z_out["commits"], ["Z"])
        self.assertEqual(coord.composer.text, "A Z")
        self.assertEqual(sugg.texts[-1], "A Z")
        coord.close()
        self.assertTrue(sugg.closed)


class GroupDIntegrationTests(unittest.TestCase):
    """Group D: Production entrypoint, config contracts, tensor shapes, and camera-to-commit dataflow."""

    def test_config_and_feature_pipeline_contracts_unchanged(self):
        cfg = load_configuration(APPLICATION_CONFIG)
        self.assertEqual(cfg["dynamic_experiment_dir"], "data/experiments/dynamic_jz_binary_baseline_v1")
        self.assertEqual(SEQUENCE_LENGTH, 16)
        self.assertEqual(FEATURE_DIM, 69)
        self.assertEqual(HoldSettings().hold_seconds, 3.0)
        self.assertTrue(issubclass(AlphabetRecognizer, UnifiedAutoCoordinator))

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[:, :320] = 255
        mirrored_abc = prepare_inference_frame(dummy_frame, "ABC")
        self.assertEqual(int(mirrored_abc[0, 0, 0]), 0)
        self.assertEqual(int(mirrored_abc[0, -1, 0]), 255)

    def test_dispatcher_and_run_py_check_only_integration(self):
        # Verify dispatcher forwards --enable-experimental-jz by default for unified mode
        captured = {}
        with patch("app.camera.realtime_inference.runpy.run_path") as mock_run:
            def _capture(path, run_name=None):
                captured["path"] = path
                captured["argv"] = list(sys.argv)
            mock_run.side_effect = _capture
            rc = realtime_inference.main(["--check-only"])
        self.assertEqual(rc, 0)
        self.assertEqual(Path(captured["path"]).resolve(), (ROOT / "src" / "main.py").resolve())
        self.assertIn("--enable-experimental-jz", captured["argv"])
        self.assertIn("--check-only", captured["argv"])

        # Subprocess run.py --check-only end-to-end check
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", "run.py", "--check-only"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["jz_commits_enabled"])
        self.assertEqual(payload["dynamic_feature_shape"], [16, 69])
        self.assertEqual(payload["hold_seconds"], 3.0)
        self.assertFalse(payload["camera_opened"])


if __name__ == "__main__":
    unittest.main()
