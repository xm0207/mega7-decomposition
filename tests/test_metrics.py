"""Metric functions on toy series whose answers can be worked out by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from mega7.metrics import (
    annualised_return,
    annualised_volatility,
    beta,
    capture_ratios,
    downside_deviation,
    drawdown_episodes,
    drawdown_series,
    historical_var_es,
    max_drawdown,
    risk_profile,
    share_of_log_return_from_best_days,
    sharpe_ratio,
    skewness,
    sortino_ratio,
)


def series(values, start="2026-01-05"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_annualised_return_is_geometric():
    # two periods a year: 1.1 * 0.9 = 0.99 after one year
    assert annualised_return(series([0.10, -0.10]), periods=2) == pytest.approx(-0.01)
    # four quarters of 2% compound to 8.243%
    assert annualised_return(series([0.02] * 4), periods=4) == pytest.approx(1.02**4 - 1)


def test_annualised_volatility_uses_the_sample_standard_deviation():
    r = series([0.01, -0.01, 0.03, -0.03])
    sd = math.sqrt((0.01**2 * 2 + 0.03**2 * 2) / 3)
    assert annualised_volatility(r, periods=252) == pytest.approx(sd * math.sqrt(252))


def test_sharpe_ratio_by_hand():
    r = series([0.02, 0.00, 0.04])
    excess = np.array([0.01, -0.01, 0.03])  # rf = 1% a period
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(12)
    assert sharpe_ratio(r, 0.01, periods=12) == pytest.approx(expected)


def test_sortino_ratio_by_hand():
    r = series([0.02, -0.01, 0.03, -0.02])
    # downside deviation over all four periods: sqrt((0 + 0.01^2 + 0 + 0.02^2) / 4) = 0.0111803
    assert downside_deviation(r, periods=1) == pytest.approx(math.sqrt(0.0005 / 4))
    assert sortino_ratio(r, 0.0, periods=1) == pytest.approx(0.005 / math.sqrt(0.0005 / 4))
    # annualising multiplies the mean by 12 and the deviation by sqrt(12)
    assert sortino_ratio(r, 0.0, periods=12) == pytest.approx(0.005 * 12 / (math.sqrt(0.0005 / 4) * math.sqrt(12)))


def test_drawdown_series_and_episodes_by_hand():
    r = series([0.10, -0.20, 0.05, 0.20, -0.10])
    # wealth 1.1, 0.88, 0.924, 1.1088, 0.99792; peaks 1.1, 1.1, 1.1, 1.1088, 1.1088
    np.testing.assert_allclose(drawdown_series(r), [0.0, 0.2, 0.16, 0.0, 0.1], atol=1e-12)
    first, second = drawdown_episodes(r)
    assert first.depth == pytest.approx(0.2)
    assert (first.peak, first.trough, first.recovery) == (r.index[0], r.index[1], r.index[3])
    assert (first.periods_to_trough, first.periods_to_recover) == (1, 2)
    assert second.depth == pytest.approx(0.1)
    assert second.recovery is None and second.periods_to_recover is None
    assert max_drawdown(r) == first


def test_drawdown_from_the_first_day_has_no_peak_date():
    r = series([-0.10, 0.20])
    e = max_drawdown(r)
    assert e.depth == pytest.approx(0.10)
    assert e.peak is None and e.trough == r.index[0] and e.recovery == r.index[1]


def test_no_drawdown_for_a_rising_series():
    e = max_drawdown(series([0.01, 0.02, 0.03]))
    assert e.depth == 0.0


def test_historical_var_es_uses_order_statistics():
    # 200 returns: the two worst are -10% and -8%, so at 99% k = 2
    r = series(np.concatenate([[-0.10, -0.08], np.linspace(-0.05, 0.05, 198)]))
    var, es = historical_var_es(r, 0.99)
    assert var == pytest.approx(0.08)
    assert es == pytest.approx(0.09)
    # 100 returns at 99%: k = 1, so VaR = ES = the worst loss
    r = series(np.linspace(-0.07, 0.03, 100))
    assert historical_var_es(r, 0.99) == pytest.approx((0.07, 0.07))
    # 20 returns at 90%: k = 2
    r = series(np.arange(1, 21) / 100 - 0.105)  # -9.5%, -8.5%, ...
    assert historical_var_es(r, 0.90) == pytest.approx((0.085, 0.09))


def test_capture_ratios_by_hand():
    bench = series([0.10, -0.05, 0.02, -0.10])
    stock = series([0.15, -0.02, 0.04, -0.12])
    up, down = capture_ratios(stock, bench, periods=12)
    up_expected = ((1.15 * 1.04) ** (12 / 2) - 1) / ((1.10 * 1.02) ** (12 / 2) - 1)
    down_expected = ((0.98 * 0.88) ** (12 / 2) - 1) / ((0.95 * 0.90) ** (12 / 2) - 1)
    assert up == pytest.approx(up_expected)
    assert down == pytest.approx(down_expected)
    # the benchmark captures itself exactly
    assert capture_ratios(bench, bench) == pytest.approx((1.0, 1.0))


def test_beta_recovers_a_known_slope():
    rng = np.random.default_rng(3)
    m = series(rng.normal(0, 0.01, 500))
    assert beta(2.0 * m + 0.001, m) == pytest.approx(2.0)


def test_share_of_log_return_from_best_days():
    # 99 flat days and one +10% day: all of the return comes from the best 1% of days
    r = series([0.0] * 99 + [0.10])
    share, without = share_of_log_return_from_best_days(r, 0.01)
    assert share == pytest.approx(1.0)
    assert without == pytest.approx(0.0)
    # two good days of +10% out of 200 days with a small loss elsewhere: share > 1
    r = series([0.10, 0.10] + [-0.0005] * 198)
    share, without = share_of_log_return_from_best_days(r, 0.01)
    total = 2 * math.log(1.1) + 198 * math.log(0.9995)
    assert total > 0
    assert share == pytest.approx(2 * math.log(1.1) / total)
    assert share > 1 and without < 0


def test_skewness_of_a_symmetric_series_is_zero():
    assert skewness(series([-0.02, -0.01, 0.0, 0.01, 0.02])) == pytest.approx(0.0)


def test_risk_profile_has_the_expected_fields():
    rng = np.random.default_rng(5)
    m = series(rng.normal(0.0004, 0.01, 600))
    r = 1.3 * m + series(rng.normal(0, 0.01, 600))
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    monthly_m = (1 + m).groupby(m.index.to_period("M")).prod() - 1
    row = risk_profile(r, pd.Series(0.0001, index=r.index), m, monthly, monthly_m)
    for key in ("ann_return", "sharpe", "sortino", "max_drawdown", "var99_1d", "es99_1d", "beta_spy_daily",
                "upside_capture_monthly", "downside_capture_monthly", "best_1pct_days_share_of_log_return"):
        assert key in row
    assert row["es99_1d"] >= row["var99_1d"] > 0
    assert 1.0 < row["beta_spy_daily"] < 1.6
