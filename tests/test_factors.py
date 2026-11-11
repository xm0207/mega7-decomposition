"""Factor regressions recover known betas, and the variance split adds up."""

import numpy as np
import pandas as pd
import pytest

from mega7.factors import (
    factor_regression,
    newey_west_lags,
    ols_newey_west,
    residuals,
    rolling_factor_regression,
    variance_decomposition,
)

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
TRUE_BETAS = np.array([1.3, -0.4, -0.6, 0.3, -0.8, 0.1])


def synthetic(n=6000, alpha=0.001, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="D")
    f = pd.DataFrame(rng.normal(0, 0.02, (n, 6)), index=idx, columns=FACTOR_NAMES)
    y = alpha + f.to_numpy() @ TRUE_BETAS + rng.normal(0, noise, n)
    return pd.Series(y, index=idx), f


def test_regression_recovers_known_betas_and_alpha():
    y, f = synthetic()
    fit = factor_regression(y, f, periods=12)
    np.testing.assert_allclose(fit.betas.to_numpy(), TRUE_BETAS, atol=0.02)
    assert fit.alpha == pytest.approx(0.001 * 12, abs=0.003)
    assert fit.alpha_t > 3  # an alpha of 0.1% a period with noise of 1% over 6,000 periods is clearly significant
    assert fit.r_squared > 0.9
    assert fit.observations == 6000
    row = fit.to_series()
    assert row["Mkt-RF"] == fit.betas["Mkt-RF"] and "Mkt-RF_t" in row


def test_newey_west_with_zero_lags_is_white():
    y, f = synthetic(n=500)
    x = np.column_stack([np.ones(len(f)), f.to_numpy()])
    coef, se, resid = ols_newey_west(y.to_numpy(), x, lags=0)
    xtx_inv = np.linalg.inv(x.T @ x)
    white = xtx_inv @ (x.T * resid**2) @ x @ xtx_inv
    np.testing.assert_allclose(se, np.sqrt(np.diag(white)))
    np.testing.assert_allclose(coef, np.linalg.lstsq(x, y.to_numpy(), rcond=None)[0])


def test_newey_west_lag_rule():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(172) == 4
    assert newey_west_lags(3600) == 8


def test_variance_shares_add_up_to_r_squared():
    y, f = synthetic(n=2000, noise=0.03)
    fit = factor_regression(y, f)
    split = variance_decomposition(y, f, fit.betas)
    shares = split[[f"share_{c}" for c in FACTOR_NAMES]].sum()
    assert shares == pytest.approx(fit.r_squared)
    assert split["systematic_share"] + split["idiosyncratic_share"] == pytest.approx(1.0)


def test_rolling_regression_last_window_matches_the_full_fit_on_that_window():
    y, f = synthetic(n=400)
    roll = rolling_factor_regression(y, f, window=150)
    fit = factor_regression(y.iloc[-150:], f.iloc[-150:])
    np.testing.assert_allclose(roll.iloc[-1][FACTOR_NAMES].to_numpy(dtype=float), fit.betas.to_numpy())
    assert roll.iloc[-1]["r_squared"] == pytest.approx(fit.r_squared)
    assert len(roll) == 400 - 150 + 1


def test_residuals_are_orthogonal_to_the_factors():
    y, f = synthetic(n=500)
    res = residuals(pd.DataFrame({"a": y, "b": 0.5 * y}), f)
    np.testing.assert_allclose(f.to_numpy().T @ res["a"].to_numpy(), 0.0, atol=1e-10)
    assert res["a"].mean() == pytest.approx(0.0, abs=1e-12)


def test_too_few_observations_raise():
    y, f = synthetic(n=7)
    with pytest.raises(ValueError):
        factor_regression(y, f)
