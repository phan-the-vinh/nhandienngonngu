"""Public application API; importing src never loads models or opens a camera."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
__all__ = ["ROOT"]
