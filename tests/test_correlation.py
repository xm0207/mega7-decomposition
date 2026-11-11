"""Correlation, clustering, PCA and dispersion arithmetic."""

import numpy as np
import pandas as pd
import pytest

from mega7.correlation import (
    average_exceedance_correlation,
    average_pairwise,
    conditional_correlation,
    correlation_distance,
    cross_sectional_dispersion,
    exceedance_correlation,
    hierarchical_clustering,
    normal_exceedance_correlation,
    one_factor_implied_correlation,
    pairwise_table,
    pca,
    rolling_average_correlation,
    rolling_correlation_to,
    rolling_pc1_share,
)


def equicorrelation(n: int, rho: float) -> pd.DataFrame:
    names = [f"S{i}" for i in range(n)]
    c = np.full((n, n), rho)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=names, columns=names)


def factor_returns(days=800, n=5, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, days)
    idx = pd.bdate_range("2020-01-01", periods=days)
    data = {f"S{i}": (0.6 + 0.2 * i) * m + rng.normal(0, 0.01, days) for i in range(n)}
    return pd.DataFrame(data, index=idx), pd.Series(m, index=idx, name="M")


def test_average_pairwise_counts_each_pair_once():
    c = pd.DataFrame([[1, 0.2, 0.4], [0.2, 1, 0.6], [0.4, 0.6, 1]], columns=list("ABC"), index=list("ABC"))
    assert average_pairwise(c) == pytest.approx(0.4)
    table = pairwise_table(c)
    assert len(table) == 3 and table["correlation"].tolist() == [0.2, 0.4, 0.6]


def test_pca_of_an_equicorrelation_matrix():
    # the largest eigenvalue is 1 + (N - 1) rho, with equal loadings 1 / sqrt(N)
    share, loadings = pca(equicorrelation(7, 0.4))
    assert share.iloc[0] == pytest.approx((1 + 6 * 0.4) / 7)
    assert share.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(loadings["PC1"], 1 / np.sqrt(7))
    assert share.iloc[1] == pytest.approx((1 - 0.4) / 7)


def test_rolling_statistics_match_the_last_window():
    r, _ = factor_returns()
    window = 120
    avg = rolling_average_correlation(r, window)
    pc1 = rolling_pc1_share(r, window)
    last = r.iloc[-window:].corr()
    assert avg.iloc[-1] == pytest.approx(average_pairwise(last))
    assert pc1.iloc[-1] == pytest.approx(pca(last)[0].iloc[0])
    assert avg.index[0] == r.index[window - 1]
    with pytest.raises(ValueError):
        rolling_average_correlation(r.iloc[:10], window)


def test_rolling_correlation_to_a_benchmark():
    r, m = factor_returns()
    out = rolling_correlation_to(r, m, 100)
    assert out.iloc[-1]["S0"] == pytest.approx(r["S0"].iloc[-100:].corr(m.iloc[-100:]))


def test_one_factor_benchmark_reproduces_the_sample_correlation_on_all_days():
    # with the full-sample market variance, b b' var(m) + Cov(e) is exactly the sample covariance
    r, m = factor_returns()
    everything = pd.Series(True, index=r.index)
    implied = one_factor_implied_correlation(r, m, everything)
    np.testing.assert_allclose(implied.to_numpy(), r.corr().to_numpy(), atol=1e-10)


def test_conditioning_on_a_volatile_market_raises_the_benchmark_correlation():
    r, m = factor_returns(days=3000)
    big = m.abs() > m.abs().quantile(0.8)
    calm = one_factor_implied_correlation(r, m, ~big)
    stressed = one_factor_implied_correlation(r, m, big)
    assert average_pairwise(stressed) > average_pairwise(calm)
    # for one-factor data with independent noise the actual conditional correlation is close to it
    assert average_pairwise(conditional_correlation(r, big)) == pytest.approx(average_pairwise(stressed), abs=0.05)


def test_exceedance_correlation_edge_cases():
    x = np.linspace(-3, 3, 1000)
    c, n = exceedance_correlation(x, x, -1.0)
    assert c == pytest.approx(1.0) and n == int((x < -1).sum())
    c, n = exceedance_correlation(x, x, 2.99)
    assert np.isnan(c) and n < 10
    # for independent normals the exceedance correlation is about zero at any threshold
    assert abs(normal_exceedance_correlation(0.0, [0.0], draws=400_000)[0.0]) < 0.01


def test_normal_exceedance_falls_symmetrically_into_the_tails():
    curve = normal_exceedance_correlation(0.5, [-1.0, 0.0, 1.0], draws=400_000)
    assert curve[0.0] > curve[-1.0]
    assert curve[-1.0] == pytest.approx(curve[1.0], abs=0.02)


def test_average_exceedance_reports_days():
    r, _ = factor_returns()
    out = average_exceedance_correlation(r, [-1.0, 1.0])
    assert list(out.columns) == ["threshold", "avg_correlation", "avg_days"]
    assert (out["avg_days"] > 10).all()


def test_clustering_merges_the_closest_pair_first():
    names = ["A", "B", "C", "D"]
    c = pd.DataFrame(
        [[1.0, 0.9, 0.1, 0.2], [0.9, 1.0, 0.15, 0.1], [0.1, 0.15, 1.0, 0.7], [0.2, 0.1, 0.7, 1.0]],
        index=names, columns=names,
    )
    d = correlation_distance(c)
    assert d.loc["A", "B"] == pytest.approx(np.sqrt(2 * 0.1))
    assert d.loc["A", "A"] == 0.0
    _, order, merges = hierarchical_clustering(c)
    assert {merges.iloc[0]["left"], merges.iloc[0]["right"]} == {"A", "B"}
    assert merges.iloc[0]["implied_correlation"] == pytest.approx(0.9)
    assert {merges.iloc[1]["left"], merges.iloc[1]["right"]} == {"C", "D"}
    assert sorted(order) == names


def test_dispersion_is_the_cross_sectional_standard_deviation():
    r = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.02], "C": [0.02, 0.02]})
    np.testing.assert_allclose(cross_sectional_dispersion(r), [0.01, 0.0])
