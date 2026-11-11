"""The event study on synthetic returns whose market, sector, alpha and company-specific parts are known.

Inside the estimation window the sector's own shock is made exactly orthogonal to the market, and the
stock's own shock exactly orthogonal to both, so OLS recovers the true coefficients to rounding error
and every part of an event day's return can be checked exactly.
"""

import math

import numpy as np
import pandas as pd
import pytest

from mega7.event_study import (
    EventStudy,
    equal_weight_basket,
    fit_market_model,
    rank_in_history,
)

TRUE = {"alpha": 0.0008, "beta_market": 1.6, "beta_sector": 0.9, "a": 0.0003, "b": 1.2}
EST, GAP, FIRST = 250, 21, 300


def project_out(values: np.ndarray, regressors: list[np.ndarray]) -> np.ndarray:
    """`values` minus its least-squares projection on a constant and `regressors`."""
    x = np.column_stack([np.ones(len(values)), *regressors])
    coef, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ coef


def synthetic_event(n: int = 320, seed: int = 0, orthogonal: bool = True):
    """Market m, sector s = a + b m + u and stock r = alpha + beta_m m + beta_s u + e.

    Days FIRST and FIRST + 1 are event days with chosen shocks: on the first the market and the sector
    fall and the stock's own shock is small; on the second the stock alone falls 9%.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    u = rng.normal(0.0, 0.008, n)
    e = rng.normal(0.0, 0.006, n)
    window = slice(FIRST - GAP - EST + 1, FIRST - GAP + 1)
    if orthogonal:
        u[window] = project_out(u[window], [m[window]])
        e[window] = project_out(e[window], [m[window], u[window]])
    m[FIRST], u[FIRST], e[FIRST] = -0.04, -0.05, 0.01
    m[FIRST + 1], u[FIRST + 1], e[FIRST + 1] = 0.02, 0.03, -0.09
    s = TRUE["a"] + TRUE["b"] * m + u
    r = TRUE["alpha"] + TRUE["beta_market"] * m + TRUE["beta_sector"] * u + e
    series = {name: pd.Series(v, index=index, name=name) for name, v in (("stock", r), ("market", m), ("sector", s))}
    shocks = pd.DataFrame({"m": m, "u": u, "e": e}, index=index)
    return series["stock"], series["market"], series["sector"], shocks


def test_two_factor_model_recovers_the_true_parts():
    r, m, s, shocks = synthetic_event()
    study = EventStudy(r, m, s, estimation_days=EST, gap=GAP)
    first = r.index[FIRST]
    fit = study.fit(first)
    assert fit.alpha == pytest.approx(TRUE["alpha"], abs=1e-12)
    assert fit.beta_market == pytest.approx(TRUE["beta_market"], abs=1e-10)
    assert fit.beta_sector == pytest.approx(TRUE["beta_sector"], abs=1e-10)
    assert fit.sector_intercept == pytest.approx(TRUE["a"], abs=1e-12)
    assert fit.sector_beta_to_market == pytest.approx(TRUE["b"], abs=1e-10)
    window = shocks.loc[fit.start : fit.end, "e"].to_numpy()
    assert fit.residual_sd == pytest.approx(math.sqrt(window @ window / (EST - 3)))

    parts = study.days(first, r.index[FIRST : FIRST + 2])
    truth = shocks.iloc[FIRST : FIRST + 2]
    np.testing.assert_allclose(parts["market"], TRUE["beta_market"] * truth["m"], atol=1e-12)
    np.testing.assert_allclose(parts["sector"], TRUE["beta_sector"] * truth["u"], atol=1e-12)
    np.testing.assert_allclose(parts["alpha"], TRUE["alpha"], atol=1e-12)
    np.testing.assert_allclose(parts["abnormal"], truth["e"], atol=1e-12)
    np.testing.assert_allclose(parts["z"], truth["e"] / fit.residual_sd, atol=1e-9)
    total = parts[["market", "sector", "alpha", "abnormal"]].sum(axis=1)
    np.testing.assert_allclose(total, parts["return"], atol=1e-15)
    # the second day: the market and the sector rose, the stock's own shock was -9%
    assert parts["abnormal"].iloc[1] == pytest.approx(-0.09)
    assert parts["market"].iloc[1] > 0 and parts["sector"].iloc[1] > 0


def test_estimation_window_is_250_days_ending_21_days_before_the_event():
    r, m, s, _ = synthetic_event()
    study = EventStudy(r, m, s)  # the defaults are the rule used in the project
    window = study.estimation_window(r.index[FIRST])
    assert len(window) == 250
    assert window[-1] == r.index[FIRST - 21]
    assert window[0] == r.index[FIRST - 21 - 249]
    assert len(r.loc[window[-1] : r.index[FIRST]]) - 2 == 20  # 20 trading days strictly in between
    with pytest.raises(ValueError, match="not enough history"):
        study.estimation_window(r.index[250])
    with pytest.raises(KeyError):
        study.estimation_window("2024-06-01")  # a Saturday
    with pytest.raises(ValueError):
        study.days(r.index[FIRST], [r.index[FIRST - 21]])  # inside the estimation window


def test_one_factor_abnormal_return_splits_into_sector_and_company_parts():
    r, m, s, _ = synthetic_event(orthogonal=False, seed=3)
    first = r.index[FIRST]
    two = EventStudy(r, m, s, EST, GAP)
    one = EventStudy(r, m, None, EST, GAP)
    fit2, fit1 = two.fit(first), one.fit(first)
    # orthogonalising the sector leaves the market model's alpha and beta unchanged
    assert fit2.beta_market == pytest.approx(fit1.beta_market, abs=1e-12)
    assert fit2.alpha == pytest.approx(fit1.alpha, abs=1e-14)
    assert math.isnan(fit1.beta_sector) and not fit1.two_factor
    days = r.index[FIRST : FIRST + 2]
    p2, p1 = two.days(first, days), one.days(first, days)
    assert "sector" not in p1
    np.testing.assert_allclose(p1["abnormal"], p2["sector"] + p2["abnormal"], atol=1e-14)
    # inside the estimation window the orthogonalised sector has mean zero and no correlation with SPY
    window = two.estimation_window(first)
    inside = fit2.parts(r[window], m[window], s[window])
    assert inside["sector_orthogonal"].mean() == pytest.approx(0.0, abs=1e-15)
    assert np.corrcoef(inside["sector_orthogonal"], m[window])[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_orthogonalising_does_not_change_the_abnormal_return():
    r, m, s, _ = synthetic_event(orthogonal=False, seed=5)
    study = EventStudy(r, m, s, EST, GAP)
    first = r.index[FIRST]
    window = study.estimation_window(first)
    x = np.column_stack([np.ones(EST), m[window], s[window]])
    coef, *_ = np.linalg.lstsq(x, r[window].to_numpy(), rcond=None)
    day = r.index[FIRST + 1]
    direct = r[day] - coef @ [1.0, m[day], s[day]]
    assert study.days(first, [day])["abnormal"].iloc[0] == pytest.approx(direct, abs=1e-14)


def test_window_totals_add_up_to_the_compound_return():
    r, m, s, _ = synthetic_event()
    study = EventStudy(r, m, s, EST, GAP)
    first = r.index[FIRST]
    days = r.index[FIRST : FIRST + 5]
    total = study.window(first, days[0], days[-1])
    daily = study.days(first, days)
    assert total["days"] == 5
    for name in ("market", "sector", "alpha", "abnormal"):
        assert total[name] == pytest.approx(daily[name].sum())
    assert total["compound_return"] == pytest.approx(np.prod(1 + r[days]) - 1)
    parts = total[["market", "sector", "alpha", "abnormal", "compounding"]].sum()
    assert parts == pytest.approx(total["compound_return"], abs=1e-15)
    fit = study.fit(first)
    assert total["z"] == pytest.approx(daily["abnormal"].sum() / (fit.residual_sd * math.sqrt(5)))
    assert total["market_compound_return"] == pytest.approx(np.prod(1 + m[days]) - 1)
    # a window may start before the event's first day, but not inside the estimation window
    earlier = study.window(first, r.index[FIRST - 20], days[-1])
    assert earlier["days"] == 25
    with pytest.raises(ValueError, match="overlaps"):
        study.window(first, r.index[FIRST - 21], days[-1])


def test_rolling_uses_no_later_data_and_matches_the_event_fit():
    r, m, s, _ = synthetic_event(orthogonal=False, seed=7)
    r, m, s = r.iloc[:200], m.iloc[:200], s.iloc[:200]
    study = EventStudy(r, m, s, estimation_days=60, gap=5)
    roll = study.rolling()
    assert roll.index[0] == r.index[60 + 5 - 1]
    day = r.index[150]
    one_day = study.days(day, [day]).iloc[0]
    for name in ("market", "sector", "alpha", "abnormal", "z"):
        assert roll.loc[day, name] == pytest.approx(one_day[name], abs=1e-14)
    # changing the stock's return on that day and after it changes only that day's abnormal return
    shocked = r.copy()
    shocked.iloc[150:] -= 0.05
    again = EventStudy(shocked, m, s, estimation_days=60, gap=5).rolling()
    assert again.loc[day, "beta_market"] == pytest.approx(roll.loc[day, "beta_market"], abs=1e-14)
    assert again.loc[day, "abnormal"] == pytest.approx(roll.loc[day, "abnormal"] - 0.05, abs=1e-14)
    pd.testing.assert_frame_equal(again.loc[: r.index[149]], roll.loc[: r.index[149]])
    later = study.rolling(start=r.index[120])
    assert later.index[0] == r.index[120]
    pd.testing.assert_frame_equal(later, roll.loc[r.index[120] :])


def test_one_factor_fit_recovers_alpha_and_beta():
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2025-01-01", periods=300)
    m = rng.normal(0, 0.01, 300)
    e = project_out(rng.normal(0, 0.01, 300), [m])
    fit = fit_market_model(pd.Series(0.001 + 1.3 * m + e, index=index), pd.Series(m, index=index))
    assert fit.alpha == pytest.approx(0.001)
    assert fit.beta_market == pytest.approx(1.3)
    assert fit.observations == 300 and fit.start == index[0] and fit.end == index[-1]
    assert fit.to_series()["observations"] == 300
    with pytest.raises(ValueError, match="needs the sector"):
        EventStudy(pd.Series(m, index=index), pd.Series(m, index=index), pd.Series(m, index=index), 50, 5).fit(
            index[100]
        ).parts(pd.Series([0.0]), pd.Series([0.0]))


def test_equal_weight_basket_averages_the_members_present():
    returns = pd.DataFrame({"A": [0.01, 0.02, np.nan], "B": [0.03, np.nan, np.nan], "C": [0.05, 0.04, 0.06]})
    basket = equal_weight_basket(returns, min_members=2)
    assert basket.iloc[0] == pytest.approx(0.03)
    assert basket.iloc[1] == pytest.approx(0.03)
    assert np.isnan(basket.iloc[2])  # only one member had a return
    with pytest.raises(ValueError):
        equal_weight_basket(returns, min_members=0)


def test_rank_in_history_counts_from_the_lowest():
    history = pd.Series([-0.10, -0.05, 0.0, 0.02, np.nan])
    assert rank_in_history(history, -0.10) == {"rank_from_lowest": 1, "observations": 4, "share_at_or_below": 0.25}
    assert rank_in_history(history, -0.07)["rank_from_lowest"] == 2
    assert rank_in_history(history, 0.05)["rank_from_lowest"] == 5
