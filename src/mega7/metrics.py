"""Risk and return metrics for one return series.

These are plain functions: each takes a series of simple returns (and sometimes a benchmark or the
risk-free rate) and returns a number or a small table. They keep no state, so they do not need to be
classes. Conventions:

- Returns are simple (not log) returns per period; `periods` is the number of periods in a year
  (252 for daily, 12 for monthly).
- Annualised return is geometric (compound growth per year); volatility is the sample standard
  deviation times the square root of `periods`.
- Losses (VaR, ES, drawdowns) are reported as positive numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------------------------------------------------------- return and risk
def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate: (product of 1 + r) ** (periods / n) - 1."""
    r = returns.dropna()
    return float(np.prod(1.0 + r.to_numpy()) ** (periods / len(r)) - 1.0)


def annualised_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Sample standard deviation (n - 1 in the denominator) times the square root of `periods`."""
    return float(returns.dropna().std(ddof=1) * math.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Mean excess return over its standard deviation, both annualised (arithmetic)."""
    excess = (returns - rf).dropna()
    return float(excess.mean() / excess.std(ddof=1) * math.sqrt(periods))


def downside_deviation(excess: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised root-mean-square of the negative excess returns, over *all* periods.

    Periods with a positive excess return count as zero (Sortino and Price, 1994), so the
    denominator is the full number of periods, not just the losing ones.
    """
    x = excess.dropna().to_numpy()
    return float(math.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * math.sqrt(periods))


def sortino_ratio(returns: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised mean excess return over the annualised downside deviation (target: the risk-free rate)."""
    excess = (returns - rf).dropna()
    return float(excess.mean() * periods / downside_deviation(excess, periods))


def skewness(returns: pd.Series) -> float:
    """Sample skewness, adjusted for bias (the same estimator as Excel's SKEW and pandas)."""
    return float(returns.dropna().skew())


def excess_kurtosis(returns: pd.Series) -> float:
    """Sample excess kurtosis (0 for a normal distribution), adjusted for bias as in pandas."""
    return float(returns.dropna().kurt())


# ---------------------------------------------------------------------------------- tail losses
def historical_var_es(returns: pd.Series, level: float = 0.99) -> tuple[float, float]:
    """Historical Value-at-Risk and Expected Shortfall at `level`, as positive losses.

    With n returns sorted from worst to best, k = ceil(n * (1 - level)) is the number of returns in
    the tail. VaR is the loss on the k-th worst day and ES is the average loss over the k worst days.
    This order-statistic definition needs no interpolation, so it can be checked by hand.
    """
    r = np.sort(returns.dropna().to_numpy())
    if len(r) == 0:
        raise ValueError("no returns")
    k = max(1, math.ceil(len(r) * (1 - level) - 1e-9))
    return float(-r[k - 1]), float(-r[:k].mean())


# ------------------------------------------------------------------------------------ drawdowns
def wealth_index(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Growth of `start_value`, with the starting value placed one period before the first return."""
    r = returns.dropna()
    return start_value * (1.0 + r).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fall from the running peak of wealth, as a positive fraction (0 at a new high).

    The running peak starts at 1, the wealth before the first return, so a series that falls from
    its first day has a drawdown on that day.
    """
    wealth = wealth_index(returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth.to_numpy()]))[1:]
    return pd.Series(1.0 - wealth.to_numpy() / peak, index=wealth.index, name="drawdown")


@dataclass(frozen=True)
class DrawdownEpisode:
    """One fall from a peak to a trough, and the date wealth regained the peak (None if not yet).

    `peak` is the last date at the old high (None if the peak is the starting value, before the first
    return). Lengths are counted in periods (trading days for daily data).
    """

    depth: float
    peak: pd.Timestamp | None
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    periods_to_trough: int
    periods_to_recover: int | None

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "peak": self.peak.date() if self.peak is not None else None,
            "trough": self.trough.date(),
            "recovery": self.recovery.date() if self.recovery is not None else None,
            "days_peak_to_trough": self.periods_to_trough,
            "days_trough_to_recovery": self.periods_to_recover,
        }


def drawdown_episodes(returns: pd.Series) -> list[DrawdownEpisode]:
    """Every drawdown episode, deepest first.

    An episode starts when wealth falls below its running peak and ends on the first date it is
    back at or above that peak. The last episode may still be open (no recovery).
    """
    dd = drawdown_series(returns)
    values, dates = dd.to_numpy(), dd.index
    episodes: list[DrawdownEpisode] = []
    i, n = 0, len(values)
    while i < n:
        if values[i] <= 0:
            i += 1
            continue
        start = i  # first day below the peak
        while i < n and values[i] > 0:
            i += 1
        segment = values[start:i]
        trough_i = start + int(np.argmax(segment))
        peak = dates[start - 1] if start > 0 else None
        recovery = dates[i] if i < n else None
        episodes.append(
            DrawdownEpisode(
                depth=float(values[trough_i]),
                peak=peak,
                trough=dates[trough_i],
                recovery=recovery,
                periods_to_trough=trough_i - start + 1,
                periods_to_recover=(i - trough_i) if recovery is not None else None,
            )
        )
    return sorted(episodes, key=lambda e: e.depth, reverse=True)


def max_drawdown(returns: pd.Series) -> DrawdownEpisode:
    """The deepest drawdown episode."""
    episodes = drawdown_episodes(returns)
    if not episodes:
        first = returns.dropna().index[0]
        return DrawdownEpisode(0.0, first, first, first, 0, 0)
    return episodes[0]


# ---------------------------------------------------------------------------------- vs benchmark
def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Covariance with the benchmark over the benchmark's variance, on common dates."""
    both = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    cov = np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def capture_ratios(returns: pd.Series, benchmark: pd.Series, periods: int = 12) -> tuple[float, float]:
    """Upside and downside capture ratios (Morningstar's definition), usually on monthly returns.

    Upside capture: the stock's compound return over the periods when the benchmark rose, annualised,
    divided by the benchmark's annualised compound return over the same periods. Downside capture
    does the same for the periods when the benchmark fell. Above 1 on the upside means the stock gained
    more than the benchmark in up markets; below 1 on the downside means it lost less in down markets.
    """
    both = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    r, m = both.iloc[:, 0], both.iloc[:, 1]

    def ratio(mask: pd.Series) -> float:
        if mask.sum() == 0:
            return float("nan")
        return annualised_return(r[mask], periods) / annualised_return(m[mask], periods)

    return ratio(m > 0), ratio(m < 0)


# ------------------------------------------------------------------------------- concentration
def share_of_log_return_from_best_days(returns: pd.Series, fraction: float = 0.01) -> tuple[float, float]:
    """How much of the total return came from the best `fraction` of days.

    Returns (share, annualised return without those days). The share is measured in log returns,
    which add up over time: the sum of log(1 + r) over the best days divided by the sum over all
    days. A share above 1 means the stock would have lost money without its best days. The share
    is only meaningful when the total log return is positive, as it is for every series here.
    """
    r = returns.dropna()
    k = max(1, math.ceil(len(r) * fraction - 1e-9))
    logs = np.log1p(r.to_numpy())
    best = np.sort(logs)[-k:]
    total = logs.sum()
    rest = np.sort(logs)[:-k]
    without = math.exp(rest.sum() * TRADING_DAYS / len(rest)) - 1.0
    return float(best.sum() / total), float(without)


def risk_profile(
    returns: pd.Series,
    rf: pd.Series,
    benchmark_daily: pd.Series | None = None,
    monthly: pd.Series | None = None,
    benchmark_monthly: pd.Series | None = None,
) -> dict:
    """All the per-stock metrics of the risk table, as one row (a dict)."""
    r = returns.dropna()
    var99, es99 = historical_var_es(r, 0.99)
    mdd = max_drawdown(r)
    best_share, without_best = share_of_log_return_from_best_days(r, 0.01)
    row = {
        "first_return": r.index[0].date(),
        "last_return": r.index[-1].date(),
        "days": len(r),
        "ann_return": annualised_return(r),
        "ann_volatility": annualised_volatility(r),
        "sharpe": sharpe_ratio(r, rf.reindex(r.index)),
        "sortino": sortino_ratio(r, rf.reindex(r.index)),
        "max_drawdown": mdd.depth,
        "mdd_peak": mdd.peak.date() if mdd.peak is not None else None,
        "mdd_trough": mdd.trough.date(),
        "mdd_recovery": mdd.recovery.date() if mdd.recovery is not None else None,
        "skewness": skewness(r),
        "excess_kurtosis": excess_kurtosis(r),
        "var99_1d": var99,
        "es99_1d": es99,
        "best_day": float(r.max()),
        "best_day_date": r.idxmax().date(),
        "worst_day": float(r.min()),
        "worst_day_date": r.idxmin().date(),
        "best_1pct_days_share_of_log_return": best_share,
        "ann_return_without_best_1pct_days": without_best,
    }
    if benchmark_daily is not None:
        row["beta_spy_daily"] = beta(r, benchmark_daily)
    if monthly is not None and benchmark_monthly is not None:
        up, down = capture_ratios(monthly, benchmark_monthly)
        row["upside_capture_monthly"] = up
        row["downside_capture_monthly"] = down
    return row
