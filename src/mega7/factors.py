"""Factor regressions: how much of each stock's return the Fama-French five factors and momentum explain.

The model is a time-series regression of excess returns on factor returns,

    r_t - rf_t = alpha + sum_k beta_k f_k,t + e_t,

estimated by ordinary least squares. The t-statistics use Newey-West (1987) standard errors, which
allow the residuals to be heteroskedastic and autocorrelated. The R-squared is the share of the
stock's variance that the factors explain (its *systematic* share); 1 - R-squared is idiosyncratic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def newey_west_lags(observations: int) -> int:
    """The usual rule of thumb for the Newey-West lag length: floor(4 (T / 100) ^ (2/9))."""
    return int(math.floor(4 * (observations / 100) ** (2 / 9)))


def ols_newey_west(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, Newey-West standard errors and residuals. `x` must include the constant column.

    The covariance is (X'X)^-1 S (X'X)^-1, where S adds to the White (heteroskedasticity-robust) term
    the autocovariances of the scores up to `lags`, with Bartlett weights 1 - lag / (lags + 1).
    """
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    resid = y - x @ coef
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    return coef, np.sqrt(np.diag(cov)), resid


@dataclass(frozen=True)
class FactorFit:
    """The result of one factor regression.

    Why a class: a regression produces several related numbers (coefficients, t-statistics, fit,
    sample size) that are always reported together; a frozen dataclass keeps them together and
    `to_series()` turns them into one row of a results table. `alpha` is annualised (times the
    number of periods in a year); betas are per unit of factor return.
    """

    alpha: float
    alpha_t: float
    betas: pd.Series
    t_stats: pd.Series
    r_squared: float
    residual_vol: float
    observations: int
    lags: int
    start: pd.Timestamp
    end: pd.Timestamp

    def to_series(self) -> pd.Series:
        row: dict = {"start": self.start.date(), "end": self.end.date(), "observations": self.observations}
        row |= {"alpha_ann": self.alpha, "alpha_t": self.alpha_t}
        for name in self.betas.index:
            row[name] = self.betas[name]
        for name in self.betas.index:
            row[f"{name}_t"] = self.t_stats[name]
        row |= {"r_squared": self.r_squared, "residual_vol_ann": self.residual_vol, "nw_lags": self.lags}
        return pd.Series(row)


def factor_regression(excess: pd.Series, factors: pd.DataFrame, periods: int = 12, lags: int | None = None) -> FactorFit:
    """Regress excess returns on factor returns over their common dates.

    `periods` annualises alpha and residual volatility (12 for monthly data, 252 for daily). `lags`
    defaults to the Newey-West rule of thumb for the sample size.
    """
    data = pd.concat([excess.rename("_y"), factors], axis=1, join="inner").dropna()
    if len(data) <= factors.shape[1] + 2:
        raise ValueError("not enough observations for the regression")
    y = data["_y"].to_numpy()
    x = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    lags = newey_west_lags(len(data)) if lags is None else lags
    coef, se, resid = ols_newey_west(y, x, lags)
    t = coef / se
    names = list(factors.columns)
    r2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)
    dof = len(y) - x.shape[1]
    return FactorFit(
        alpha=float(coef[0] * periods),
        alpha_t=float(t[0]),
        betas=pd.Series(coef[1:], index=names),
        t_stats=pd.Series(t[1:], index=names),
        r_squared=float(r2),
        residual_vol=float(math.sqrt(resid @ resid / dof) * math.sqrt(periods)),
        observations=len(y),
        lags=lags,
        start=data.index[0],
        end=data.index[-1],
    )


def rolling_factor_regression(excess: pd.Series, factors: pd.DataFrame, window: int) -> pd.DataFrame:
    """Betas and R-squared from a regression over each trailing `window` of observations.

    Each row is dated at the window's last observation. Columns: one per factor, then r_squared.
    """
    data = pd.concat([excess.rename("_y"), factors], axis=1, join="inner").dropna()
    y_all = data["_y"].to_numpy()
    x_all = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    rows = []
    for end in range(window, len(data) + 1):
        y = y_all[end - window : end]
        x = x_all[end - window : end]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        r2 = 1.0 - resid @ resid / np.sum((y - y.mean()) ** 2)
        rows.append([*coef[1:], r2])
    return pd.DataFrame(rows, index=data.index[window - 1 :], columns=[*factors.columns, "r_squared"])


def variance_decomposition(excess: pd.Series, factors: pd.DataFrame, betas: pd.Series) -> pd.Series:
    """Split a stock's variance into each factor's contribution and an idiosyncratic remainder.

    Systematic variance is b' S b, where S is the factor covariance matrix. Factor k contributes
    b_k (S b)_k (an Euler split, which can be negative when factors offset each other); the
    contributions add up to the systematic variance. Everything is reported as a share of the stock's
    total variance, so the factor shares add up to the regression's R-squared and the idiosyncratic
    share is 1 - R-squared.
    """
    data = pd.concat([excess.rename("_y"), factors], axis=1, join="inner").dropna()
    s = data[factors.columns].cov().to_numpy()
    b = betas[factors.columns].to_numpy()
    total = data["_y"].var(ddof=1)
    contrib = b * (s @ b) / total
    out = pd.Series(contrib, index=[f"share_{c}" for c in factors.columns])
    out["systematic_share"] = contrib.sum()
    out["idiosyncratic_share"] = 1.0 - contrib.sum()
    return out


def residuals(excess: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Each column's OLS residuals from a full-sample regression on `factors` (with a constant)."""
    data = excess.join(factors, how="inner").dropna()
    x = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    out = {}
    for col in excess.columns:
        y = data[col].to_numpy()
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[col] = y - x @ coef
    return pd.DataFrame(out, index=data.index)
