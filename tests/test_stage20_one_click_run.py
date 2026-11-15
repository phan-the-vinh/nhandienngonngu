"""Stage 20 Verification: One-Click Application Startup via run.py and VS Code Configuration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestStage20OneClickStartup(unittest.TestCase):
    def test_vscode_settings_interpreter_configured_to_nn(self):
        settings_path = ROOT / ".vscode" / "settings.json"
        self.assertTrue(settings_path.is_file(), f"Missing {settings_path}")
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertIn("python.defaultInterpreterPath", data)
        interpreter = Path(data["python.defaultInterpreterPath"])
        self.assertTrue(interpreter.is_file(), f"Interpreter executable not found: {interpreter}")
        self.assertIn("nn", str(interpreter).lower())
        self.assertTrue(data.get("python.terminal.activateEnvironment", False))

    def test_vscode_launch_json_configured(self):
        launch_path = ROOT / ".vscode" / "launch.json"
        self.assertTrue(launch_path.is_file(), f"Missing {launch_path}")
        data = json.loads(launch_path.read_text(encoding="utf-8"))
        configurations = data.get("configurations", [])
        self.assertGreater(len(configurations), 0)
        default_config = configurations[0]
        self.assertIn("run.py", default_config.get("program", ""))

    def test_run_py_default_check_only_enables_space_jz_and_suggestions(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "--check-only"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Process failed:\n{proc.stderr}")
        last_line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(last_line)
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["space_enabled"], "SPACE gesture recognition must be enabled by default")
        self.assertIsNotNone(payload["space_model"])
        self.assertTrue(payload["jz_commits_enabled"], "J/Z confirmation commits must be enabled by default")
        self.assertTrue(payload["word_suggestion_enabled"], "Word suggestion must be enabled by default")
        self.assertTrue(payload["word_suggestion_initialized"], "Word suggestion manager must be initialized")
        self.assertFalse(payload["camera_opened"])

    def test_run_py_explicit_disable_space(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "--check-only", "--disable-space"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Process failed:\n{proc.stderr}")
        last_line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(last_line)
        self.assertEqual(payload["status"], "PASS")
        self.assertFalse(payload["space_enabled"], "SPACE must be disabled when --disable-space is supplied")

    def test_run_py_explicit_disable_jz(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "--check-only", "--disable-experimental-jz"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Process failed:\n{proc.stderr}")
        last_line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(last_line)
        self.assertEqual(payload["status"], "PASS")
        self.assertFalse(payload["jz_commits_enabled"], "J/Z commits must be disabled with --disable-experimental-jz")

    def test_run_py_explicit_disable_word_suggestion(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "run.py"), "--check-only", "--disable-word-suggestion"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Process failed:\n{proc.stderr}")
        last_line = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(last_line)
        self.assertEqual(payload["status"], "PASS")
        self.assertFalse(payload["word_suggestion_enabled"])


if __name__ == "__main__":
    unittest.main()
