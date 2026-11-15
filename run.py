"""Primary entry point for the NhanDienCuChi realtime recognizers.

Usage (from the project root):
    python run.py                                # unified ABC/123/JZ experiment
    python run.py --check-only                   # unified checks, no camera
    python run.py --mode letters --check-only
    python run.py --mode letters --camera-index 0
    python run.py --mode digits
    python run.py --mode unified --check-only
    python run.py --mode unified --enable-experimental-jz
    python run.py --disable-experimental-jz       # unified, J/Z diagnostic-only
    python run.py --mode letters --model v1       # fallback letter model

Recognition arguments are forwarded to the selected program; the dispatcher
applies the unified-only J/Z launch switches. Programs can still run directly, e.g.
``python app/inference/digit_inference.py``.
VS Code Run Python File (no arguments) uses unified mode with:
- Static ASL (A-Y except J/Z) + Digits (0-9) recognition
- Dynamic J/Z confirmation commits ON
- SPACE gesture recognition ON (configs/space.json)
- Offline Word & Sentence suggestions ON
Use --disable-experimental-jz to turn J/Z confirmation off.
Use --disable-space to turn SPACE recognition off.
Use --disable-word-suggestion to turn suggestions off.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.camera.realtime_inference import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
