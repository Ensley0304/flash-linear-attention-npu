"""Compatibility import; mathematical source is ../reference.py."""
from pathlib import Path
import runpy
globals().update({k: v for k, v in runpy.run_path(str(Path(__file__).resolve().parents[1] / "reference.py")).items() if not k.startswith("__")})
