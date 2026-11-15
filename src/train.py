"""Training facade; --execute is user-only, existing rights/confirmation gates remain."""
from pathlib import Path
import sys
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src._workflow import main as dispatch
def main(argv=None):
    return dispatch("train",argv)
if __name__=="__main__":
    raise SystemExit(main())
