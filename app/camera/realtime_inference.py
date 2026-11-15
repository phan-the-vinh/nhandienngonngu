"""Runtime camera dispatcher.

Launches a standalone recognizer or the unified application without importing it as a
module, so a program's top-level setup/loop behavior is preserved exactly:

* ``letters`` -> ``app/inference/letter_inference.py`` (24 static ASL classes)
* ``digits``  -> ``app/inference/digit_inference.py``  (0-9)
* ``unified`` -> ``src/main.py`` (ABC/123; J/Z experimental commits ON at launch)

Programs remain independently runnable. This launch policy explicitly enables
experimental J/Z for unified unless --disable-experimental-jz is supplied.
Direct src/comparison-camera defaults are unchanged. No recognition logic lives here.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]

PROGRAMS = {
    "letters": ROOT / "app" / "inference" / "letter_inference.py",
    "digits": ROOT / "app" / "inference" / "digit_inference.py",
    "unified": ROOT / "src" / "main.py",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NhanDienCuChi realtime camera dispatcher",
        add_help=True,
    )
    parser.add_argument(
        "--mode",
        choices=tuple(PROGRAMS),
        default="unified",
        help="Which recognizer to run (default: unified; J/Z commits EXPERIMENTAL ON).",
    )
    jz = parser.add_mutually_exclusive_group()
    jz.add_argument("--enable-experimental-jz", dest="experimental_jz", action="store_true",
                   default=None, help="Enable experimental J/Z confirmation in unified (launch default).")
    jz.add_argument("--disable-experimental-jz", dest="experimental_jz", action="store_false",
                   help="Disable J/Z confirmation in unified; keep diagnostic-only behavior.")
    mode, passthrough = parser.parse_known_args(argv)
    if mode.mode != "unified" and mode.experimental_jz is not None:
        parser.error("J/Z confirmation switches apply only to --mode unified")
    if mode.mode == "unified" and mode.experimental_jz is not False:
        passthrough = [*passthrough, "--enable-experimental-jz"]

    program = PROGRAMS[mode.mode]
    if not program.is_file():
        print(f"ERROR: Program not found: {program}", file=sys.stderr)
        return 2

    # Preflight check for unverified models on fresh clones
    missing_models = []
    if mode.mode in ("letters", "unified"):
        letter_file = ROOT / "models" / "letters" / "production" / "uvw_medium_model.joblib"
        if not letter_file.is_file():
            missing_models.append(f"Static ASL letter model ({letter_file})")
    if mode.mode in ("digits", "unified"):
        digit_file = ROOT / "models" / "digits" / "production" / "digits_final.joblib"
        if not digit_file.is_file():
            missing_models.append(f"Static digit model ({digit_file})")
    if missing_models:
        print(
            "ERROR: Missing required pre-trained model file(s):\n  "
            + "\n  ".join(missing_models)
            + "\n\nThis public repository is distributed source-only without unverified pre-trained weights.\n"
            "Please refer to docs/MODEL_DISTRIBUTION.md for instructions.",
            file=sys.stderr,
        )
        return 2

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.argv = [str(program), *passthrough]
    try:
        runpy.run_path(str(program), run_name="__main__")
    except SystemExit as exc:  # forwarded from the target program
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
