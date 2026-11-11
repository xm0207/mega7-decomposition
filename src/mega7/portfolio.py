"""The seven stocks as a portfolio: an equal-weight basket, and how its risk splits across the stocks.

Risk contributions use the Euler split of portfolio volatility, sigma_p = sqrt(w' S w):

- marginal contribution of stock i: d sigma_p / d w_i = (S w)_i / sigma_p
- component contribution: w_i (S w)_i / sigma_p; the components add up to sigma_p
- percentage contribution: component / sigma_p; these add up to 1

The diversification ratio (Choueifaty and Coignard, 2008) is the weighted average of the stocks'
volatilities over the portfolio's volatility: 1 with no diversification (perfect correlation), and
higher the more the stocks offset each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EqualWeightPortfolio:
    """An equal-weight portfolio of N stocks, rebalanced daily, monthly or never (buy and hold).

    Why a class: a drifting portfolio has state that changes every day (the weights), and its daily
    return depends on that state. The class runs the simulation once, on construction, and then
    exposes the results. Public interface:

    - `returns`: the portfolio's daily simple return (a Series).
    - `weights`: each stock's weight at the start of each day (a DataFrame; rows sum to 1).
    - `turnover()`: average one-way turnover per year from rebalancing back to 1/N.

    With monthly rebalancing the weights are reset to 1/N at the close of each month's last trading
    day, so every month starts equal-weighted and drifts with prices until the month ends. With
    `rebalance='never'` the weights start at 1/N and drift for the whole sample. Costs are ignored.
    """

    REBALANCE = ("daily", "monthly", "never")

    def __init__(self, returns: pd.DataFrame, rebalance: str = "monthly"):
        if rebalance not in self.REBALANCE:
            raise ValueError(f"rebalance must be one of {self.REBALANCE}")
        data = returns.dropna()
        if data.empty:
            raise ValueError("no dates on which every stock has a return")
        self.rebalance = rebalance
        self.stocks = list(data.columns)
        n = len(self.stocks)
        r = data.to_numpy()
        months = data.index.to_period("M")
        start_weights = np.empty_like(r)
        port = np.empty(len(r))
        traded = np.zeros(len(r))
        w = np.full(n, 1.0 / n)
        for t in range(len(r)):
            new_month = t > 0 and months[t] != months[t - 1]
            if rebalance == "daily" or (rebalance == "monthly" and new_month):
                target = np.full(n, 1.0 / n)
                traded[t] = 0.5 * np.abs(target - w).sum()
                w = target
            start_weights[t] = w
            port[t] = w @ r[t]
            w = w * (1.0 + r[t]) / (1.0 + port[t])
        self.returns = pd.Series(port, index=data.index, name=f"Mega 7 equal weight ({rebalance})")
        self.weights = pd.DataFrame(start_weights, index=data.index, columns=self.stocks)
        self._traded = pd.Series(traded, index=data.index)

    def turnover(self, periods: int = 252) -> float:
        """Average one-way turnover per year: half the sum of absolute weight changes at rebalances."""
        return float(self._traded.mean() * periods)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Each stock's weight, standalone volatility, marginal, component and percentage risk contribution.

    `cov` must be in the same units as the volatility you want (annualise it first for annual vol).
    The `component` column adds up to the portfolio volatility and `percent` to 1.
    """
    names = list(weights.index)
    w = weights.to_numpy(dtype=float)
    s = cov.loc[names, names].to_numpy(dtype=float)
    sigma = float(np.sqrt(w @ s @ w))
    marginal = s @ w / sigma
    component = w * marginal
    return pd.DataFrame(
        {
            "weight": w,
            "volatility": np.sqrt(np.diag(s)),
            "marginal": marginal,
            "component": component,
            "percent": component / sigma,
        },
        index=names,
    )


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    names = list(weights.index)
    w = weights.to_numpy(dtype=float)
    return float(np.sqrt(w @ cov.loc[names, names].to_numpy(dtype=float) @ w))


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Weighted average of standalone volatilities over the portfolio volatility (>= 1 for long-only)."""
    names = list(weights.index)
    vols = np.sqrt(np.diag(cov.loc[names, names].to_numpy(dtype=float)))
    return float(weights.to_numpy(dtype=float) @ vols / portfolio_volatility(weights, cov))


def rolling_risk_split(returns: pd.DataFrame, window: int = 252, periods: int = 252) -> pd.DataFrame:
    """Percentage risk contributions of an equal-weight portfolio and its diversification ratio,
    from the covariance of each trailing `window` of daily returns (dated at the window's end).
    """
    data = returns.dropna()
    n = data.shape[1]
    w = pd.Series(1.0 / n, index=data.columns)
    rows = {}
    x = data.to_numpy()
    for end in range(window, len(data) + 1):
        cov = pd.DataFrame(np.cov(x[end - window : end], rowvar=False) * periods, index=data.columns, columns=data.columns)
        rc = risk_contributions(w, cov)
        row = rc["percent"].to_dict()
        row["portfolio_vol"] = float(rc["component"].sum())
        row["diversification_ratio"] = diversification_ratio(w, cov)
        rows[data.index[end - 1]] = row
    return pd.DataFrame.from_dict(rows, orient="index")
