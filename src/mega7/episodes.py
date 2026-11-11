"""Market episodes: windows and single days worth looking at closely.

Window boundaries are not typed in by hand. Each sell-off is identified by one date inside it, and
`spy_drawdown_window` finds the SPY drawdown that contains that date, returning its peak and trough
from the data. Calendar windows (the 2023-24 rally) and single days are given as dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .correlation import average_pairwise, cross_sectional_dispersion, pca
from .metrics import drawdown_episodes


@dataclass(frozen=True)
class Episode:
    """A named window of trading days. `start` is the first day whose return counts (the day after
    the SPY peak, for a drawdown); `end` is the last."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    how_dated: str


def spy_drawdown_window(name: str, spy: pd.Series, inside: str) -> Episode:
    """The SPY peak-to-trough window of the drawdown episode that contains the date `inside`."""
    day = pd.Timestamp(inside)
    for e in drawdown_episodes(spy):
        recovery = e.recovery if e.recovery is not None else spy.index[-1]
        if e.peak is not None and e.peak < day <= recovery:
            first = spy.index[spy.index.get_loc(e.peak) + 1]
            return Episode(name, first, e.trough, f"SPY peak {e.peak.date()} to trough {e.trough.date()}, -{e.depth:.1%}")
    raise ValueError(f"No SPY drawdown contains {inside}")


def calendar_window(name: str, returns: pd.DataFrame, start: str, end: str) -> Episode:
    days = returns.loc[start:end].index
    return Episode(name, days[0], days[-1], f"calendar window {days[0].date()} to {days[-1].date()}")


def episode_summary(returns: pd.DataFrame, stocks: list[str], episode: Episode) -> dict:
    """Compound return of each ticker over the window, and the co-movement of the stocks inside it.

    Average pairwise correlation and PC1 share are only reported for windows of 15 days or more.
    """
    window = returns.loc[episode.start : episode.end]
    row: dict = {"episode": episode.name, "start": episode.start.date(), "end": episode.end.date(),
                 "days": len(window), "dated_by": episode.how_dated}
    for t in returns.columns:
        row[f"ret_{t}"] = float(np.prod(1 + window[t]) - 1)
    stock_returns = pd.Series({t: row[f"ret_{t}"] for t in stocks})
    row["mega7_equal_weight_buy_and_hold"] = float(stock_returns.mean())
    row["stocks_worse_than_spy"] = int((stock_returns < row["ret_SPY"]).sum())
    row["spread_best_minus_worst"] = float(stock_returns.max() - stock_returns.min())
    row["avg_daily_dispersion"] = float(cross_sectional_dispersion(window[stocks]).mean())
    if len(window) >= 15:
        corr = window[stocks].corr()
        row["avg_pairwise_corr"] = average_pairwise(corr)
        row["pc1_share"] = float(pca(corr)[0].iloc[0])
    else:
        row["avg_pairwise_corr"] = np.nan
        row["pc1_share"] = np.nan
    return row
