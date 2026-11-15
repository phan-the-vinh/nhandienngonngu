"""Explicit desktop application entry. Import never starts inference."""
from pathlib import Path
import sys
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main(argv=None):
    from src.camera.realtime import main as run_unified
    return run_unified(argv)

if __name__=="__main__":
    raise SystemExit(main())
