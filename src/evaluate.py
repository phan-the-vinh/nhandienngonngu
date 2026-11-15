"""Evaluation facade; unsupported production pipelines fail explicitly."""
from pathlib import Path
import sys
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src._workflow import main as dispatch
def main(argv=None):
    return dispatch("evaluate",argv)
if __name__=="__main__":
    raise SystemExit(main())
