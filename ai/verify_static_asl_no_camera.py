"""Run automated ASL verification without permission to open a real webcam.

This runs the unit suite and both model modes via ``--check-only``. A guard
replaces ``cv2.VideoCapture`` for the entire run and fails if any unmocked
camera access is attempted.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai" / "preprocessing"))


def main() -> int:
    attempts: list[tuple] = []

    def forbid_video_capture(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("Automated verification must not open cv2.VideoCapture")

    with patch.object(cv2, "VideoCapture", side_effect=forbid_video_capture):
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT)
        )
        unit_output = io.StringIO()
        with contextlib.redirect_stderr(unit_output):
            result = unittest.TextTestRunner(stream=unit_output, verbosity=0).run(suite)
        if not result.wasSuccessful() or attempts:
            print(json.dumps({
                "status": "FAIL",
                "unit_tests": result.testsRun,
                "camera_open_attempts": len(attempts),
                "test_output_tail": unit_output.getvalue()[-2000:],
            }))
            return 1

        from app.inference import letter_inference as camera

        model_checks = []
        for argv, expected_model, expected_epoch in (
            (["--check-only"], "uvw_medium", 81),
            (["--model", "v1", "--check-only"], "v1_baseline", 143),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = camera.main(argv)
            status = json.loads(output.getvalue())
            if (
                code != 0
                or status.get("status") != "CAMERA_INFERENCE_CHECK_PASS"
                or status.get("model") != expected_model
                or status.get("best_epoch") != expected_epoch
                or status.get("features") != 63
                or len(status.get("classes", [])) != 24
            ):
                print(json.dumps({"status": "FAIL", "model_check": expected_model}))
                return 1
            model_checks.append(expected_model)

    if attempts:
        print(json.dumps({"status": "FAIL", "camera_open_attempts": len(attempts)}))
        return 1
    print(json.dumps({
        "status": "PASS",
        "unit_tests": result.testsRun,
        "model_checks": model_checks,
        "camera_open_attempts": 0,
        "verification_mode": "--check-only",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
