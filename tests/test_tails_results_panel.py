"""Tail co-movement benchmarks, the results writer's publishing rule, the panel and episodes."""

import numpy as np
import pandas as pd
import pytest

from mega7 import STOCKS, ResultsWriter, ReturnPanel
from mega7.episodes import calendar_window, episode_summary, spy_drawdown_window
from mega7.panel import drop_unfinished_last_period
from mega7.results import looks_like_high_frequency_series
from mega7.tails import (
    co_move_matrix,
    distribution_of_counts,
    gaussian_copula_counts,
    joint_move_summary,
    poisson_binomial,
)

from conftest import synthetic_prices, synthetic_store


# ----------------------------------------------------------------------------------------- tails
def test_poisson_binomial_by_hand():
    np.testing.assert_allclose(poisson_binomial([0.5, 0.5]), [0.25, 0.5, 0.25])
    dist = poisson_binomial([0.1, 0.2, 0.3])
    assert dist.sum() == pytest.approx(1.0)
    assert dist[3] == pytest.approx(0.1 * 0.2 * 0.3)
    assert dist[0] == pytest.approx(0.9 * 0.8 * 0.7)


def test_distribution_of_counts_fills_missing_values():
    dist = distribution_of_counts(pd.Series([0, 0, 2, 3]), 3)
    assert dist.tolist() == [2, 0, 1, 1]


def test_gaussian_copula_limits():
    p = pd.Series([0.05, 0.05, 0.05], index=list("ABC"))
    independent = gaussian_copula_counts(p, pd.DataFrame(np.eye(3), index=p.index, columns=p.index), draws=400_000)
    np.testing.assert_allclose(independent, poisson_binomial(p.to_numpy()), atol=0.002)
    c = np.full((3, 3), 0.999999)
    np.fill_diagonal(c, 1.0)
    almost_one = pd.DataFrame(c, index=p.index, columns=p.index)
    together = gaussian_copula_counts(p, almost_one, draws=200_000)
    assert together[3] == pytest.approx(0.05, abs=0.003)  # all crash together or none do
    assert together[1] + together[2] < 0.005


def test_co_move_matrix_and_summary_count_correctly():
    r = pd.DataFrame({"A": [-0.06, -0.06, 0.0, -0.10], "B": [-0.06, 0.0, 0.0, -0.07], "C": [0.0, 0.0, 0.0, -0.08]})
    cond, own = co_move_matrix(r, -0.05)
    assert own.tolist() == [3, 2, 1]
    assert cond.loc["A", "B"] == pytest.approx(2 / 3)
    assert cond.loc["B", "A"] == pytest.approx(1.0)
    summary = joint_move_summary(r, -0.05, at_least=[1, 2, 3], draws=10_000).set_index("at_least")
    assert summary["observed_days"].tolist() == [3, 2, 1]


# --------------------------------------------------------------------------------------- results
def test_results_writer_refuses_daily_series(tmp_path):
    writer = ResultsWriter(tmp_path)
    daily = pd.DataFrame({"r": np.zeros(300)}, index=pd.bdate_range("2025-01-01", periods=300))
    with pytest.raises(ValueError, match="daily or weekly"):
        writer.csv("bad", daily, index=True)
    weekly_column = pd.DataFrame({"date": pd.date_range("2020-01-03", periods=200, freq="W-FRI").date, "r": 0.0})
    with pytest.raises(ValueError):
        writer.csv("bad_weekly", weekly_column)
    assert not (tmp_path / "bad.csv").exists()


def test_results_writer_accepts_summaries(tmp_path):
    writer = ResultsWriter(tmp_path)
    month_end = pd.DataFrame({"date": pd.date_range("2012-06-30", periods=170, freq="ME").date, "corr": 0.5})
    assert not looks_like_high_frequency_series(month_end)
    path = writer.csv("ok_monthly", month_end)
    assert path.exists()
    few_days = pd.DataFrame({"date": pd.bdate_range("2020-03-09", periods=10).date, "r": 0.0})
    writer.csv("ok_few_days", few_days)
    writer.text("notes", ["a", "b"])
    assert (tmp_path / "notes.txt").read_text() == "a\nb\n"


# ----------------------------------------------------------------------------------------- panel
def test_drop_unfinished_last_period():
    idx = pd.to_datetime(["2026-10-30", "2026-11-06", "2026-11-10"])  # Fri, Fri, Tue
    weekly = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
    assert len(drop_unfinished_last_period(weekly, "weekly", idx[-1])) == 2
    assert len(drop_unfinished_last_period(weekly.iloc[:2], "weekly", idx[1])) == 2
    monthly = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-10-30", "2026-11-10"]))
    assert len(drop_unfinished_last_period(monthly, "monthly", pd.Timestamp("2026-11-10"))) == 1
    finished = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2026-10-30"]))  # last business day of October
    assert len(drop_unfinished_last_period(finished, "monthly", pd.Timestamp("2026-10-30"))) == 1


def test_panel_common_sample_starts_when_the_last_stock_has_a_return(panel, prices):
    first_meta_close = prices["META"].dropna().index[0]
    daily = panel.returns("daily")
    assert daily.index[0] > first_meta_close  # the first META return is the day after its first close
    assert panel.start == daily.index[0]
    assert not daily.isna().any().any()
    assert len(panel.history("AAPL")) > len(daily)  # own history is longer than the common sample


def test_panel_risk_free_rate_is_carried_forward_and_converted(panel):
    daily = panel.returns("daily")
    rf = panel.rf("daily", daily.index)
    assert rf.iloc[-1] == pytest.approx(0.0001)  # 0.01 percent a day
    assert panel.rf_carried_forward("daily", daily.index) == 10
    excess = panel.excess("daily", ["AAPL"])
    np.testing.assert_allclose(excess["AAPL"], daily["AAPL"] - rf)
    assert list(panel.factors("monthly").columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    with pytest.raises(ValueError):
        panel.rf("weekly", daily.index)


def test_panel_monthly_returns_are_labelled_at_month_end(panel):
    monthly = panel.returns("monthly")
    assert (monthly.index == monthly.index + pd.offsets.MonthEnd(0)).all()


# -------------------------------------------------------------------------------------- episodes
def test_spy_drawdown_window_reads_peak_and_trough_from_the_data():
    idx = pd.bdate_range("2026-01-05", periods=8)
    spy = pd.Series([0.01, 0.01, -0.05, -0.05, 0.02, 0.10, 0.01, 0.0], index=idx)
    e = spy_drawdown_window("test", spy, inside=str(idx[3].date()))
    assert e.start == idx[2] and e.end == idx[3]  # first falling day to the trough
    with pytest.raises(ValueError):
        spy_drawdown_window("none", spy, inside=str(idx[0].date()))


def test_episode_summary_compounds_returns():
    prices = synthetic_prices()
    store = synthetic_store(prices)
    daily = ReturnPanel.from_store(store).returns("daily")
    e = calendar_window("window", daily, str(daily.index[10].date()), str(daily.index[40].date()))
    row = episode_summary(daily, STOCKS, e)
    window = daily.loc[e.start : e.end]
    assert row["days"] == 31
    assert row["ret_SPY"] == pytest.approx(np.prod(1 + window["SPY"]) - 1)
    assert 0 <= row["stocks_worse_than_spy"] <= 7
    assert 1 / 7 <= row["pc1_share"] <= 1
