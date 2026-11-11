"""Event study: split a stock's daily return into market, sector, alpha and company-specific parts.

The model is a two-factor market model,

    r_t = alpha + beta_m m_t + beta_s s*_t + e_t,      s*_t = s_t - (a + b m_t),

where r is the stock's simple return, m the market's (SPY) and s the sector's (SMH, or an equal-weight
basket of peers). The sector return is orthogonalised to the market: a and b come from a regression of
s on m over the estimation window, so s* is the part of the sector's return that the market does not
explain. Inside the estimation window s* has mean zero and is uncorrelated with m, so alpha and beta_m
are exactly the one-factor (market model) estimates. Two things follow:

- the one-factor abnormal return splits exactly into the sector part beta_s s*_t and the two-factor
  abnormal return e_t;
- the orthogonalisation only decides how the part that the market and the sector share is labelled
  (it goes to the market). The fitted values, and so the abnormal returns, are the same as in a
  regression on m and s directly.

On any day the return is the sum of four parts:

    market     beta_m m_t
    sector     beta_s s*_t   (absent from the one-factor model)
    alpha      alpha
    abnormal   e_t, the company-specific part

The z-score divides the abnormal return by the residual standard deviation of the estimation window
(sigma, with n - k degrees of freedom); over a window of L days the cumulative abnormal return (the
sum of the daily ones) is divided by sigma sqrt(L). Estimation error in the coefficients is ignored.

One rule sets every estimation window: `estimation_days` trading days (250) ending `gap` trading days
(21) before the first day of the event window, so the window ends on the trading day 21 days before
it. Totals over several days add up the daily parts. Their sum is the sum of the daily returns, not
the compound return; the `compounding` column is the difference, so the parts plus `compounding` give
the compound return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

ESTIMATION_DAYS = 250
GAP = 21


def equal_weight_basket(returns: pd.DataFrame, min_members: int = 1) -> pd.Series:
    """The average return of the members that have a return each day: an equal-weight basket
    rebalanced daily.

    Days on which fewer than `min_members` members have a return are NaN, so that a basket whose
    members list at different dates is not represented by one or two of them early on.
    """
    if min_members < 1:
        raise ValueError("min_members must be at least 1")
    count = returns.notna().sum(axis=1)
    return returns.mean(axis=1).where(count >= min_members).rename("basket")


def _estimate(r: np.ndarray, m: np.ndarray, s: np.ndarray | None) -> tuple[float, ...]:
    """OLS of r on a constant, m and (if given) s orthogonalised to m.

    Returns alpha, beta_m, beta_s, a, b, residual standard deviation and R-squared; beta_s, a and b
    are NaN without a sector.
    """
    n = len(r)
    k = 2 if s is None else 3
    if n <= k + 1:
        raise ValueError("not enough observations for the market model")
    base = np.column_stack([np.ones(n), m])
    a = b = beta_s = math.nan
    x = base
    if s is not None:
        (a, b), *_ = np.linalg.lstsq(base, s, rcond=None)
        x = np.column_stack([base, s - a - b * m])
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coef
    if s is not None:
        beta_s = coef[2]
    residual_sd = math.sqrt(resid @ resid / (n - k))
    r_squared = 1.0 - resid @ resid / np.sum((r - r.mean()) ** 2)
    return float(coef[0]), float(coef[1]), float(beta_s), float(a), float(b), residual_sd, float(r_squared)


@dataclass(frozen=True)
class MarketModelFit:
    """One estimate of the market model, and the rule that turns returns into parts.

    Why a class: the coefficients of one estimation window are used together, and always in the same
    way, to split returns; a frozen dataclass keeps them together with the window they come from.
    `parts(...)` splits returns, `to_series()` gives one row of a results table. `alpha` is per day.
    The sector fields are NaN in the one-factor model.
    """

    alpha: float
    beta_market: float
    beta_sector: float
    sector_intercept: float
    sector_beta_to_market: float
    residual_sd: float
    r_squared: float
    observations: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def two_factor(self) -> bool:
        return not math.isnan(self.beta_sector)

    def parts(self, stock: pd.Series, market: pd.Series, sector: pd.Series | None = None) -> pd.DataFrame:
        """Each day's return split into market, sector (two-factor only), alpha and abnormal, and z.

        Columns: return, market_return, [sector_return, sector_orthogonal,] market, [sector,] alpha,
        abnormal, z. The parts add up to the return exactly.
        """
        if self.two_factor and sector is None:
            raise ValueError("a two-factor fit needs the sector's returns")
        out = pd.DataFrame({"return": stock, "market_return": market})
        if self.two_factor:
            out["sector_return"] = sector
            out["sector_orthogonal"] = sector - (self.sector_intercept + self.sector_beta_to_market * market)
        out["market"] = self.beta_market * out["market_return"]
        explained = out["market"] + self.alpha
        if self.two_factor:
            out["sector"] = self.beta_sector * out["sector_orthogonal"]
            explained = explained + out["sector"]
        out["alpha"] = self.alpha
        out["abnormal"] = out["return"] - explained
        out["z"] = out["abnormal"] / self.residual_sd
        return out

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "estimation_start": self.start.date(),
                "estimation_end": self.end.date(),
                "observations": self.observations,
                "alpha_daily": self.alpha,
                "beta_market": self.beta_market,
                "beta_sector": self.beta_sector,
                "sector_intercept": self.sector_intercept,
                "sector_beta_to_market": self.sector_beta_to_market,
                "residual_sd_daily": self.residual_sd,
                "r_squared": self.r_squared,
            }
        )


def fit_market_model(stock: pd.Series, market: pd.Series, sector: pd.Series | None = None) -> MarketModelFit:
    """Estimate the market model on the dates where every input has a return."""
    columns = {"r": stock, "m": market} | ({"s": sector} if sector is not None else {})
    data = pd.concat(columns, axis=1, join="inner").dropna()
    s = data["s"].to_numpy() if sector is not None else None
    alpha, beta_m, beta_s, a, b, sd, r2 = _estimate(data["r"].to_numpy(), data["m"].to_numpy(), s)
    return MarketModelFit(alpha, beta_m, beta_s, a, b, sd, r2, len(data), data.index[0], data.index[-1])


def rank_in_history(history: pd.Series, value: float) -> dict:
    """Where `value` sits in `history`: its rank counted from the lowest (1 = lower than every other
    observation), the number of observations, and the share of observations at or below it."""
    h = history.dropna()
    return {
        "rank_from_lowest": int((h < value).sum()) + 1,
        "observations": len(h),
        "share_at_or_below": float((h <= value).mean()),
    }


class EventStudy:
    """The returns of one stock, the market and (optionally) a sector, and the estimation-window rule.

    Why a class: it owns the aligned returns and the rule that places the estimation window, so every
    event, every window and the rolling history are estimated on the same calendar with the same rule.
    Public interface:

    - `estimation_window(first_day)` the trading days used for an event whose window starts on `first_day`.
    - `fit(first_day)` the `MarketModelFit` on that window.
    - `days(first_day, dates)` the parts of each listed day.
    - `window(first_day, start, end)` the parts added up over the trading days from `start` to `end`.
    - `rolling(start)` every day's parts from the fit for an event starting that day: the estimation
      window ends `gap` trading days earlier, so no day is explained with its own or later data.

    The calendar is the set of days on which every input has a return. Days and windows must come
    after the estimation window; anything else raises.
    """

    def __init__(
        self,
        stock: pd.Series,
        market: pd.Series,
        sector: pd.Series | None = None,
        estimation_days: int = ESTIMATION_DAYS,
        gap: int = GAP,
    ):
        if estimation_days < 10:
            raise ValueError("estimation_days must be at least 10")
        if gap < 1:
            raise ValueError("gap must be at least 1, so that no event day is in its own estimation window")
        columns = {"stock": stock, "market": market} | ({"sector": sector} if sector is not None else {})
        self.data = pd.concat(columns, axis=1, join="inner").dropna().sort_index()
        self.has_sector = sector is not None
        self.estimation_days = estimation_days
        self.gap = gap

    def _position(self, day) -> int:
        day = pd.Timestamp(day)
        if day not in self.data.index:
            raise KeyError(f"{day.date()} is not a day on which every series has a return")
        return int(self.data.index.get_loc(day))

    def _sector(self, rows: pd.DataFrame) -> pd.Series | None:
        return rows["sector"] if self.has_sector else None

    def estimation_window(self, first_day) -> pd.DatetimeIndex:
        end = self._position(first_day) - self.gap
        start = end - self.estimation_days + 1
        if start < 0:
            raise ValueError(
                f"not enough history before {pd.Timestamp(first_day).date()}: "
                f"{self.estimation_days + self.gap} trading days are needed"
            )
        return self.data.index[start : end + 1]

    def fit(self, first_day) -> MarketModelFit:
        rows = self.data.loc[self.estimation_window(first_day)]
        return fit_market_model(rows["stock"], rows["market"], self._sector(rows))

    def days(self, first_day, dates) -> pd.DataFrame:
        """The parts of each day in `dates`, from the fit for the event that starts on `first_day`."""
        fit = self.fit(first_day)
        index = pd.DatetimeIndex([self.data.index[self._position(d)] for d in dates])
        if (index <= fit.end).any():
            raise ValueError("an event day falls inside the estimation window")
        rows = self.data.loc[index]
        return fit.parts(rows["stock"], rows["market"], self._sector(rows))

    def window(self, first_day, start, end) -> pd.Series:
        """Totals over the trading days from `start` to `end` (inclusive), from the fit for the event
        that starts on `first_day`.

        Fields: start, end, days, compound_return, sum_of_returns, the parts (each the sum of its daily
        values; abnormal is the cumulative abnormal return), compounding (compound_return minus
        sum_of_returns), z (abnormal / (sigma sqrt(days))), and the compound returns of the market and
        the sector over the same days.
        """
        fit = self.fit(first_day)
        rows = self.data.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if rows.empty:
            raise ValueError(f"no trading days between {start} and {end}")
        if rows.index[0] <= fit.end:
            raise ValueError("the window overlaps the estimation window")
        parts = fit.parts(rows["stock"], rows["market"], self._sector(rows))
        days = len(parts)
        compound = float(np.prod(1.0 + parts["return"]) - 1.0)
        total: dict = {
            "start": rows.index[0].date(),
            "end": rows.index[-1].date(),
            "days": days,
            "compound_return": compound,
            "sum_of_returns": float(parts["return"].sum()),
        }
        for name in ("market", "sector", "alpha", "abnormal"):
            if name in parts:
                total[name] = float(parts[name].sum())
        total["compounding"] = compound - total["sum_of_returns"]
        total["z"] = total["abnormal"] / (fit.residual_sd * math.sqrt(days))
        total["market_compound_return"] = float(np.prod(1.0 + rows["market"]) - 1.0)
        if self.has_sector:
            total["sector_compound_return"] = float(np.prod(1.0 + rows["sector"]) - 1.0)
        return pd.Series(total)

    def rolling(self, start=None) -> pd.DataFrame:
        """Each day's parts, z and coefficients, from the fit for an event starting on that day.

        Rows begin on the first day with a full estimation window (or on `start`, if later). Columns:
        market, [sector,] alpha, abnormal, z, beta_market, [beta_sector,] residual_sd.
        """
        index = self.data.index
        r = self.data["stock"].to_numpy()
        m = self.data["market"].to_numpy()
        s = self.data["sector"].to_numpy() if self.has_sector else None
        first = self.estimation_days + self.gap - 1
        if start is not None:
            first = max(first, int(index.searchsorted(pd.Timestamp(start))))
        rows = []
        for i in range(first, len(index)):
            lo, hi = i - self.gap - self.estimation_days + 1, i - self.gap + 1
            window_s = s[lo:hi] if s is not None else None
            alpha, beta_m, beta_s, a, b, sd, _ = _estimate(r[lo:hi], m[lo:hi], window_s)
            row = {"market": beta_m * m[i]}
            explained = row["market"] + alpha
            if s is not None:
                row["sector"] = beta_s * (s[i] - a - b * m[i])
                explained += row["sector"]
            row |= {"alpha": alpha, "abnormal": r[i] - explained}
            row |= {"z": row["abnormal"] / sd, "beta_market": beta_m}
            if s is not None:
                row["beta_sector"] = beta_s
            row["residual_sd"] = sd
            rows.append(row)
        return pd.DataFrame(rows, index=index[first:])
