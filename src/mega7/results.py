"""Writes CSV tables and charts to `results/`, and refuses to write anything that looks like a price or
return series.

Yahoo Finance data is for personal use, so the committed results may hold only summary statistics
(metrics, correlation matrices, betas, rolling statistics sampled at month end or coarser) and charts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# A table is treated as a time series at daily or weekly frequency if it has more than this many
# distinct dates and their median spacing is shorter than MIN_MEDIAN_GAP_DAYS.
MAX_DENSE_DATES = 120
MIN_MEDIAN_GAP_DAYS = 25


def looks_like_high_frequency_series(frame: pd.DataFrame) -> bool:
    """True if the index, or any date column, holds many dates spaced less than about a month apart."""
    candidates = []
    if isinstance(frame.index, pd.DatetimeIndex):
        candidates.append(pd.Series(frame.index))
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            candidates.append(frame[col])
        elif frame[col].dtype == object and len(frame) and hasattr(frame[col].iloc[0], "isoformat"):
            candidates.append(pd.to_datetime(frame[col], errors="coerce"))
    for dates in candidates:
        unique = pd.Series(pd.to_datetime(dates.dropna().unique())).sort_values()
        if len(unique) > MAX_DENSE_DATES and unique.diff().dt.days.median() < MIN_MEDIAN_GAP_DAYS:
            return True
    return False


class ResultsWriter:
    """Saves every table and chart the scripts produce, in one folder.

    Why a class: it owns the output folder and the rule about what may be published, so no script
    can write a daily or weekly series by mistake. Public interface:

    - `csv(name, frame)` writes `results/<name>.csv` (floats to 6 significant digits) and returns the path.
    - `figure(name, fig)` writes `results/<name>.png` and closes the figure.
    - `text(name, lines)` writes a short notes file.
    """

    def __init__(self, folder: str | Path = "results"):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def csv(self, name: str, frame: pd.DataFrame, index: bool = False) -> Path:
        if looks_like_high_frequency_series(frame):
            raise ValueError(
                f"{name}: this looks like a daily or weekly series. Only summary statistics may go in "
                "results/ (Yahoo data are for personal use); sample it at month end or summarise it."
            )
        path = self.folder / f"{name}.csv"
        frame.to_csv(path, index=index, float_format="%.6g")
        return path

    def figure(self, name: str, fig) -> Path:
        import matplotlib.pyplot as plt

        path = self.folder / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def text(self, name: str, lines: list[str]) -> Path:
        path = self.folder / f"{name}.txt"
        path.write_text("\n".join(lines) + "\n")
        return path
