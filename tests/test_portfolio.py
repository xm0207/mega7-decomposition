"""Equal-weight portfolio mechanics and the Euler split of portfolio volatility."""

import numpy as np
import pandas as pd
import pytest

from mega7 import EqualWeightPortfolio
from mega7.portfolio import diversification_ratio, portfolio_volatility, risk_contributions, rolling_risk_split


def returns(days=300, n=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=days)
    return pd.DataFrame(rng.normal(0.0005, 0.02, (days, n)), index=idx, columns=[f"S{i}" for i in range(n)])


def random_cov(n=4, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    names = [f"S{i}" for i in range(n)]
    return pd.DataFrame(a @ a.T / n + np.eye(n) * 0.01, index=names, columns=names)


def test_risk_contributions_add_up_to_portfolio_volatility():
    cov = random_cov()
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    rc = risk_contributions(w, cov)
    sigma = portfolio_volatility(w, cov)
    assert rc["component"].sum() == pytest.approx(sigma)
    assert rc["percent"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(rc["volatility"], np.sqrt(np.diag(cov)))


def test_marginal_contribution_is_the_derivative_of_volatility():
    cov = random_cov()
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    rc = risk_contributions(w, cov)
    h = 1e-6
    for name in w.index:
        bumped = w.copy()
        bumped[name] += h
        numeric = (portfolio_volatility(bumped, cov) - portfolio_volatility(w, cov)) / h
        assert rc.loc[name, "marginal"] == pytest.approx(numeric, rel=1e-4)


def test_diversification_ratio_extremes():
    names = ["A", "B", "C", "D"]
    w = pd.Series(0.25, index=names)
    perfectly_correlated = pd.DataFrame(0.04, index=names, columns=names)
    assert diversification_ratio(w, perfectly_correlated) == pytest.approx(1.0)
    independent = pd.DataFrame(np.eye(4) * 0.04, index=names, columns=names)
    assert diversification_ratio(w, independent) == pytest.approx(2.0)  # sqrt(4)


def test_daily_rebalancing_returns_the_cross_sectional_mean():
    r = returns()
    p = EqualWeightPortfolio(r, "daily")
    np.testing.assert_allclose(p.returns, r.mean(axis=1))
    np.testing.assert_allclose(p.weights.to_numpy(), 1 / 3)


def test_buy_and_hold_equals_the_average_of_individual_wealth():
    r = returns()
    p = EqualWeightPortfolio(r, "never")
    wealth = (1 + p.returns).cumprod()
    expected = (1 + r).cumprod().mean(axis=1)
    np.testing.assert_allclose(wealth, expected)
    assert p.turnover() == 0.0
    np.testing.assert_allclose(p.weights.sum(axis=1), 1.0)


def test_monthly_rebalancing_resets_weights_on_the_first_day_of_each_month():
    r = returns()
    p = EqualWeightPortfolio(r, "monthly")
    firsts = p.weights.groupby(p.weights.index.to_period("M")).head(1)
    np.testing.assert_allclose(firsts.to_numpy(), 1 / 3)
    mid = p.weights.iloc[5]
    assert not np.allclose(mid, 1 / 3)  # weights drift inside the month
    assert p.turnover() > 0


def test_rebalance_option_is_checked():
    with pytest.raises(ValueError):
        EqualWeightPortfolio(returns(), "weekly")


def test_rolling_risk_split_rows_sum_to_one():
    r = returns(days=200)
    out = rolling_risk_split(r, window=60)
    np.testing.assert_allclose(out[r.columns].sum(axis=1), 1.0)
    assert (out["diversification_ratio"] >= 1.0).all()
    assert len(out) == 200 - 60 + 1
