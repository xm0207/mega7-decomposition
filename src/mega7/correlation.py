"""Correlation structure of a small set of stocks: levels, changes over time, conditional and tail
correlations, clustering, principal components and dispersion.

Plain functions on a frame of returns (dates x stocks). A few definitions used throughout:

- Average pairwise correlation: the mean of the correlations above the diagonal (21 pairs for 7 stocks).
- PC1 share: the largest eigenvalue of the correlation matrix over the number of stocks. It is the
  share of the total (standardised) variance that one common component explains: 1/N if the stocks
  were uncorrelated, 1 if they moved as one.
- Dispersion: the cross-sectional standard deviation of the stocks' returns on a given day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


# ------------------------------------------------------------------------------------- levels
def average_pairwise(corr: pd.DataFrame | np.ndarray) -> float:
    """Mean of the off-diagonal correlations (each pair counted once)."""
    c = np.asarray(corr, dtype=float)
    upper = np.triu_indices_from(c, k=1)
    return float(np.mean(c[upper]))


def pairwise_table(corr: pd.DataFrame) -> pd.DataFrame:
    """The correlation matrix as one row per pair: first, second, correlation."""
    names = list(corr.columns)
    rows = [
        {"first": names[i], "second": names[j], "correlation": float(corr.iloc[i, j])}
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- over time
def _rolling_corr_matrices(returns: pd.DataFrame, window: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Correlation matrix of each trailing `window` of rows, dated at the window's last row."""
    x = returns.dropna().to_numpy()
    n = len(x)
    if n < window:
        raise ValueError("fewer rows than the window")
    out = np.empty((n - window + 1, x.shape[1], x.shape[1]))
    for end in range(window, n + 1):
        out[end - window] = np.corrcoef(x[end - window : end], rowvar=False)
    return returns.dropna().index[window - 1 :], out


def rolling_average_correlation(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Average pairwise correlation over each trailing window."""
    index, mats = _rolling_corr_matrices(returns, window)
    upper = np.triu_indices(mats.shape[1], k=1)
    return pd.Series(mats[:, upper[0], upper[1]].mean(axis=1), index=index, name="avg_pairwise_corr")


def rolling_pc1_share(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """Share of standardised variance explained by the first principal component, over each trailing window."""
    index, mats = _rolling_corr_matrices(returns, window)
    largest = np.linalg.eigvalsh(mats)[:, -1]
    return pd.Series(largest / mats.shape[1], index=index, name="pc1_share")


def rolling_correlation_to(returns: pd.DataFrame, benchmark: pd.Series, window: int = 252) -> pd.DataFrame:
    """Each column's correlation with `benchmark` over each trailing window."""
    data = returns.join(benchmark.rename("_benchmark"), how="inner").dropna()
    return data.drop(columns="_benchmark").rolling(window).corr(data["_benchmark"]).dropna(how="all")


# ------------------------------------------------------------------------------ conditional
def conditional_correlation(returns: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Correlation matrix over the dates where `mask` is True."""
    mask = mask.reindex(returns.index).fillna(False).astype(bool)
    return returns[mask].corr()


def one_factor_implied_correlation(returns: pd.DataFrame, market: pd.Series, mask: pd.Series) -> pd.DataFrame:
    """The correlation matrix we would expect on the `mask` dates if only the market's variance changed.

    Each stock is split into a market part and a residual using full-sample betas:
    r_i = a_i + b_i m + e_i. The model keeps the betas and the full-sample residual covariance fixed and
    replaces the market's variance by its variance on the `mask` dates:

        Cov = b b' Var(m | mask) + Cov(e)

    Comparing this with the actual conditional correlation separates two effects. Correlations rise
    on volatile days even when nothing about the stocks changes, because the common market component
    is a larger part of each stock's variance on those days (Forbes and Rigobon, 2002). Only the part of
    the actual correlation above this benchmark is co-movement beyond that mechanical effect.
    """
    data = returns.join(market.rename("_m"), how="inner").dropna()
    m = data.pop("_m").to_numpy()
    x = data.to_numpy()
    m_c = m - m.mean()
    betas = (x - x.mean(axis=0)).T @ m_c / (m_c @ m_c)
    resid = x - x.mean(axis=0) - np.outer(m_c, betas)
    resid_cov = np.cov(resid, rowvar=False, ddof=1)
    sel = mask.reindex(data.index).fillna(False).astype(bool).to_numpy()
    var_m = np.var(m[sel], ddof=1)
    cov = np.outer(betas, betas) * var_m + resid_cov
    sd = np.sqrt(np.diag(cov))
    return pd.DataFrame(cov / np.outer(sd, sd), index=data.columns, columns=data.columns)


def exceedance_correlation(x: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, int]:
    """Correlation of two standardised series on the days when both are beyond `threshold`.

    For a negative threshold: days when both x < threshold and y < threshold (joint falls). For a
    positive threshold: days when both are above it (joint rises). Returns (correlation, days used);
    the correlation is NaN with fewer than 10 days. Longin and Solnik (2001); Ang and Chen (2002).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    sel = (x < threshold) & (y < threshold) if threshold <= 0 else (x > threshold) & (y > threshold)
    n = int(sel.sum())
    if n < 10:
        return float("nan"), n
    return float(np.corrcoef(x[sel], y[sel])[0, 1]), n


def average_exceedance_correlation(returns: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Exceedance correlation at each threshold, averaged over all pairs (returns standardised first).

    Columns: threshold, avg_correlation, avg_days (average number of joint exceedance days per pair).
    """
    z = (returns - returns.mean()) / returns.std(ddof=1)
    cols = list(z.columns)
    rows = []
    for t in thresholds:
        values, days = [], []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c, n = exceedance_correlation(z[cols[i]].to_numpy(), z[cols[j]].to_numpy(), t)
                values.append(c)
                days.append(n)
        rows.append({"threshold": t, "avg_correlation": float(np.nanmean(values)), "avg_days": float(np.mean(days))})
    return pd.DataFrame(rows)


def normal_exceedance_correlation(rho: float, thresholds: list[float], draws: int = 2_000_000, seed: int = 7) -> pd.Series:
    """Exceedance correlations of a bivariate normal with correlation `rho`, by simulation.

    This is the benchmark: even for normal returns the exceedance correlation falls as the threshold
    moves into the tails, symmetrically on both sides. Real returns are judged against this curve.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(draws)
    b = rho * a + np.sqrt(1 - rho**2) * rng.standard_normal(draws)
    return pd.Series({t: exceedance_correlation(a, b, t)[0] for t in thresholds}, name="normal")


def normal_exceedance_benchmark(corr: pd.DataFrame, thresholds: list[float], draws: int = 1_000_000) -> pd.Series:
    """The normal-distribution exceedance correlation for each pair's own correlation, averaged over pairs."""
    c = corr.to_numpy()
    upper = np.triu_indices_from(c, k=1)
    curves = [normal_exceedance_correlation(float(rho), thresholds, draws=draws, seed=7 + k)
              for k, rho in enumerate(c[upper])]
    return pd.concat(curves, axis=1).mean(axis=1).rename("normal_benchmark")


# ------------------------------------------------------------------------- structure: clusters
def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Mantegna's (1999) distance, sqrt(2 (1 - rho)): 0 for perfectly correlated, 2 for opposite."""
    d = np.sqrt(np.clip(2.0 * (1.0 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def hierarchical_clustering(corr: pd.DataFrame, method: str = "average") -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Agglomerative clustering of the stocks on correlation distance.

    Returns the scipy linkage matrix, the leaf order (for a dendrogram or a reordered heatmap) and a
    table of merges: which clusters joined, at what distance and at what implied correlation.
    """
    names = list(corr.columns)
    dist = correlation_distance(corr)
    z = linkage(squareform(dist.to_numpy(), checks=False), method=method)
    members: dict[int, list[str]] = {i: [n] for i, n in enumerate(names)}
    rows = []
    for k, (a, b, d, size) in enumerate(z):
        left, right = members[int(a)], members[int(b)]
        members[len(names) + k] = left + right
        rows.append(
            {
                "step": k + 1,
                "left": " + ".join(left),
                "right": " + ".join(right),
                "distance": float(d),
                "implied_correlation": float(1 - d**2 / 2),
                "size": int(size),
            }
        )
    order = [names[i] for i in leaves_list(z)]
    return z, order, pd.DataFrame(rows)


# ---------------------------------------------------------------------------- structure: PCA
def pca(corr: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Principal components of a correlation matrix.

    Returns the share of standardised variance explained by each component (largest first) and the
    loadings (eigenvectors, one column per component). Each eigenvector's sign is set so that its
    loadings sum to a positive number, so PC1 reads as 'all stocks up'.
    """
    values, vectors = np.linalg.eigh(corr.to_numpy())
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    signs = np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    vectors = vectors * signs
    labels = [f"PC{i + 1}" for i in range(len(values))]
    share = pd.Series(values / values.sum(), index=labels, name="variance_share")
    loadings = pd.DataFrame(vectors, index=corr.index, columns=labels)
    return share, loadings


# ------------------------------------------------------------------------------- dispersion
def cross_sectional_dispersion(returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional standard deviation of the stocks' returns on each date (n - 1 denominator)."""
    return returns.std(axis=1, ddof=1).rename("dispersion")
