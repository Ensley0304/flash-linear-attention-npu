"""Forward to the unique repository CPU reference; no formulas here."""
from pathlib import Path
import runpy
globals().update({k: v for k, v in runpy.run_path(str(Path(__file__).resolve().parents[3] / "tests/atk/chunk_kda_bwd_finalize/reference.py")).items() if not k.startswith("__")})
