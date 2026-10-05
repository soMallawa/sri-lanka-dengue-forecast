from __future__ import annotations

# ruff: noqa: I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dengue_forecast.historical_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
