"""Tail co-movement: how often several of the stocks fall (or rise) sharply on the same day.

The observed count of 'joint crash' days is compared with two benchmarks that use each stock's own
crash frequency:

- independence: each stock crashes on its own, with its observed daily probability; the number of
  stocks crashing on a day then follows a Poisson-binomial distribution, computed exactly;
- a Gaussian copula: the stocks are linked by a multivariate normal with their observed (rank)
  correlations. This keeps the ordinary correlation but has no extra tail dependence, so a count
  well above it is evidence that crashes cluster more than correlation alone implies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def distribution_of_counts(counts: pd.Series, n_stocks: int) -> pd.Series:
    """Number of days on which exactly k stocks moved beyond the threshold, for k = 0..n_stocks."""
    return counts.value_counts().reindex(range(n_stocks + 1), fill_value=0).sort_index().rename("days")


def poisson_binomial(probabilities: list[float] | np.ndarray) -> np.ndarray:
    """P(exactly k successes), k = 0..n, for independent trials with different success probabilities."""
    dist = np.array([1.0])
    for p in probabilities:
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def normal_scores_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the normal scores (van der Waerden): each column's ranks mapped through the normal
    quantile function. This is the correlation matrix a Gaussian copula fitted by ranks would use."""
    data = returns.dropna()
    n = len(data)
    scores = np.column_stack([norm.ppf(rankdata(data[c]) / (n + 1)) for c in data.columns])
    return pd.DataFrame(np.corrcoef(scores, rowvar=False), index=data.columns, columns=data.columns)


def gaussian_copula_counts(
    probabilities: pd.Series, corr: pd.DataFrame, draws: int = 1_000_000, seed: int = 11, lower: bool = True
) -> np.ndarray:
    """Expected share of days with exactly k joint moves under a Gaussian copula, k = 0..n.

    Stock i 'moves' when its latent normal is below norm.ppf(p_i) (for falls; above norm.ppf(1 - p_i)
    for rises), so each stock keeps its observed marginal probability p_i.
    """
    names = list(probabilities.index)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr.loc[names, names].to_numpy())
    z = rng.standard_normal((draws, len(names))) @ chol.T
    p = probabilities.to_numpy()
    hits = z < norm.ppf(p) if lower else z > norm.ppf(1.0 - p)
    k = hits.sum(axis=1)
    return np.bincount(k, minlength=len(names) + 1) / draws


def co_move_matrix(returns: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    """P(column stock moves beyond threshold | row stock does), and each stock's own number of such days.

    A negative threshold looks at falls (return below it), a positive one at rises.
    """
    hits = returns < threshold if threshold < 0 else returns > threshold
    hits = hits.astype(float)
    joint = hits.T @ hits
    own = np.diag(joint)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = joint.to_numpy() / own[:, None]
    cond = pd.DataFrame(cond, index=returns.columns, columns=returns.columns)
    return cond, pd.Series(own, index=returns.columns, name="days")


def joint_move_summary(returns: pd.DataFrame, threshold: float, at_least: list[int], draws: int = 1_000_000) -> pd.DataFrame:
    """Observed days with at least k joint moves, against independence and a Gaussian copula.

    One row per k in `at_least`: observed days, expected days under independence, expected days under
    the Gaussian copula, and the ratio observed / copula.
    """
    data = returns.dropna()
    lower = threshold < 0
    hits = data < threshold if lower else data > threshold
    counts = hits.sum(axis=1)
    probabilities = hits.mean()
    days = len(data)
    indep = poisson_binomial(probabilities.to_numpy())
    copula = gaussian_copula_counts(probabilities, normal_scores_correlation(data), draws=draws, lower=lower)
    rows = []
    for k in at_least:
        observed = int((counts >= k).sum())
        exp_indep = float(indep[k:].sum() * days)
        exp_cop = float(copula[k:].sum() * days)
        rows.append(
            {
                "threshold": threshold,
                "at_least": k,
                "observed_days": observed,
                "expected_if_independent": exp_indep,
                "expected_gaussian_copula": exp_cop,
                "observed_over_copula": observed / exp_cop if exp_cop > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)
