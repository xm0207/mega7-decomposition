"""The SQL in PriceStore against the same calculations done by hand in pandas."""

import numpy as np
import pandas as pd
import pytest

from mega7 import PriceStore

from conftest import long_prices


def toy_store() -> PriceStore:
    # Thursday 26 Feb to Tuesday 10 Mar 2026: crosses an ISO week and a month end
    dates = pd.bdate_range("2026-02-26", "2026-03-10")
    prices = pd.DataFrame({"A": np.arange(100.0, 100.0 + len(dates)), "B": 50.0 * 1.01 ** np.arange(len(dates))}, index=dates)
    store = PriceStore()
    store.upsert("price", long_prices(prices))
    return store


def test_daily_returns_match_pct_change():
    store = toy_store()
    prices = store.wide("price")
    expected = prices.pct_change().iloc[1:]
    pd.testing.assert_frame_equal(store.returns("daily"), expected, check_freq=False, check_names=False)


def test_weekly_returns_use_the_last_close_of_each_iso_week():
    store = toy_store()
    weekly = store.returns("weekly")
    prices = store.wide("price")["A"]
    # weeks end Fri 27 Feb, Fri 6 Mar, and Tue 10 Mar (unfinished); the first week has no previous close
    assert list(weekly.index.strftime("%Y-%m-%d")) == ["2026-03-06", "2026-03-10"]
    assert weekly.loc["2026-03-06", "A"] == pytest.approx(prices["2026-03-06"] / prices["2026-02-27"] - 1)


def test_monthly_returns_use_the_last_close_of_each_month():
    store = toy_store()
    monthly = store.returns("monthly")
    prices = store.wide("price")["B"]
    assert list(monthly.index.strftime("%Y-%m-%d")) == ["2026-03-10"]
    assert monthly["B"].iloc[0] == pytest.approx(prices["2026-03-10"] / prices["2026-02-27"] - 1)


def test_calendar_year_returns_compound_daily_returns():
    dates = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
    store = PriceStore()
    store.upsert("price", long_prices(pd.DataFrame({"X": [100.0, 110.0, 99.0, 108.9]}, index=dates)))
    years = store.calendar_year_returns().set_index("year")
    assert years.loc[2025, "total_return"] == pytest.approx(0.10)
    assert years.loc[2026, "total_return"] == pytest.approx(108.9 / 110.0 - 1)
    assert years.loc[2026, "days"] == 2


def test_extreme_days_ranks_best_and_worst():
    dates = pd.bdate_range("2026-01-05", periods=5)
    store = PriceStore()
    store.upsert("price", long_prices(pd.DataFrame({"X": [100.0, 120.0, 90.0, 99.0, 99.0]}, index=dates)))
    ext = store.extreme_days(n=1)
    best = ext[ext["side"] == "best"].iloc[0]
    worst = ext[ext["side"] == "worst"].iloc[0]
    assert best["ret"] == pytest.approx(0.20) and pd.Timestamp(best["date"]) == dates[1]
    assert worst["ret"] == pytest.approx(-0.25) and pd.Timestamp(worst["date"]) == dates[2]


def test_joint_move_counts_only_counts_days_when_every_ticker_trades():
    dates = pd.bdate_range("2026-01-05", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 90.0, 95.0, 85.0], "B": [100.0, 94.0, 100.0, 89.0], "C": [np.nan, 100.0, 90.0, 80.0]}, index=dates
    )
    store = PriceStore()
    store.upsert("price", long_prices(prices))
    counts = store.joint_move_counts(["A", "B", "C"], -0.05).set_index("date")["n_moves"]
    # day 2: C has no return yet, so the day is skipped; day 3: only C fell >5%; day 4: all three
    assert list(pd.DatetimeIndex(counts.index)) == list(dates[2:])
    assert list(counts) == [1, 3]
    ups = store.joint_move_counts(["A", "B"], 0.05).set_index("date")["n_moves"]
    assert list(ups) == [0, 2, 0]
    with pytest.raises(ValueError):
        store.joint_move_counts(["A"], 0.0)


def test_upsert_replaces_the_whole_history_of_a_series():
    store = toy_store()
    new = pd.DataFrame({"series": "A", "date": pd.to_datetime(["2026-03-02"]), "value": [1.0]})
    store.upsert("price", new)
    prices = store.wide("price")
    assert prices["A"].count() == 1
    assert prices["B"].count() == 9  # other series untouched
    cov = store.coverage().set_index("series")
    assert cov.loc["A", "observations"] == 1


def test_market_caps_round_trip():
    store = PriceStore()
    store.save_market_caps(pd.DataFrame({"series": ["X", "Y"], "as_of": ["2026-11-11"] * 2, "market_cap": [1.0, 3.0]}))
    caps = store.market_caps()
    assert list(caps["series"]) == ["Y", "X"]


def test_unknown_frequency_and_missing_series_raise():
    store = toy_store()
    with pytest.raises(ValueError):
        store.returns("hourly")
    with pytest.raises(KeyError):
        store.wide("price", ["NOPE"])
