"""Run every analysis script in order on the downloaded data (run download_data.py first)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = ["run_stock_risk.py", "run_correlation.py", "run_factors.py", "run_portfolio.py", "run_tails.py"]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        print(f"\n=== {name} ===")
        sys.argv = [name, *sys.argv[1:]]
        runpy.run_path(str(here / name), run_name="__main__")


if __name__ == "__main__":
    main()
