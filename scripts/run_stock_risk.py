"""Per-stock risk and return, over the common sample and over each stock's own history.

Outputs (results/): stock_metrics_common.csv, stock_metrics_full_history.csv, drawdown_episodes.csv,
extreme_days.csv, calendar_year_returns.csv, wealth_common_sample.png, drawdowns_common_sample.png.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mega7 import NAMES, STOCKS, TICKERS, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.metrics import drawdown_episodes, drawdown_series, risk_profile


def metrics_table(panel: ReturnPanel, common: bool) -> pd.DataFrame:
    rows = []
    if common:
        daily = panel.returns("daily")
        monthly = panel.returns("monthly")
    for t in TICKERS:
        if common:
            r, m = daily[t], monthly[t]
            spy_d, spy_m = daily["SPY"], monthly["SPY"]
        else:
            r, m = panel.history(t, "daily"), panel.history(t, "monthly")
            spy_d, spy_m = panel.history("SPY", "daily"), panel.history("SPY", "monthly")
        rf = panel.rf("daily", r.index)
        row = {"ticker": t, "name": NAMES[t]}
        row |= risk_profile(r, rf, spy_d if t != "SPY" else None, m if t != "SPY" else None, spy_m)
        if t != "SPY":
            overlap = r.index.intersection(spy_d.index)
            row["vs_spy_from"] = overlap[0].date()
            mo = m.dropna().index.intersection(spy_m.dropna().index)
            row["capture_months"] = len(mo)
        row["rf_days_carried_forward"] = panel.rf_carried_forward("daily", r.index)
        rows.append(row)
    return pd.DataFrame(rows)


def episodes_table(panel: ReturnPanel, top: int = 5) -> pd.DataFrame:
    rows = []
    for sample in ("common", "full_history"):
        for t in TICKERS:
            r = panel.returns("daily")[t] if sample == "common" else panel.history(t)
            for rank, e in enumerate(drawdown_episodes(r)[:top], start=1):
                rows.append({"sample": sample, "ticker": t, "rank": rank} | e.to_dict())
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    panel = ReturnPanel.from_store(store)
    writer = ResultsWriter(args.results)
    print(f"Common sample: {panel.start.date()} to {panel.end.date()}")

    common = metrics_table(panel, common=True)
    writer.csv("stock_metrics_common", common)
    full = metrics_table(panel, common=False)
    writer.csv("stock_metrics_full_history", full)
    cols = ["ticker", "first_return", "ann_return", "ann_volatility", "sharpe", "sortino", "max_drawdown",
            "mdd_peak", "mdd_trough", "mdd_recovery", "var99_1d", "es99_1d", "beta_spy_daily"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(common[cols].round(3).to_string(index=False))
        print(full[cols].round(3).to_string(index=False))

    writer.csv("drawdown_episodes", episodes_table(panel))

    extremes = store.extreme_days(n=5)
    extremes = extremes[extremes["series"].isin(TICKERS)].copy()  # the store also holds the event study's semiconductors
    spy = panel.history("SPY")
    extremes["date"] = pd.to_datetime(extremes["date"])
    extremes["spy_same_day"] = extremes["date"].map(spy)
    extremes["date"] = extremes["date"].dt.date
    writer.csv("extreme_days", extremes.rename(columns={"series": "ticker", "ret": "return"}))

    years = store.calendar_year_returns(start=str(panel.start.date()))
    years = years[years["series"].isin(TICKERS)]
    table = years.pivot(index="year", columns="series", values="total_return")[TICKERS]
    table["days"] = years.groupby("year")["days"].max()
    writer.csv("calendar_year_returns", table.reset_index())

    daily = panel.returns("daily")
    fig = charts.wealth_chart(daily[STOCKS + ["SPY"]], f"Growth of $1, {panel.start:%d %b %Y} to {panel.end:%d %b %Y}")
    writer.figure("wealth_common_sample", fig)
    dd = pd.DataFrame({t: drawdown_series(daily[t]) for t in STOCKS})
    fig = charts.small_multiples(-dd * 100, "Drawdown from previous peak, common sample", "Drawdown, %",
                                 reference=-drawdown_series(daily["SPY"]) * 100, reference_label="SPY")
    writer.figure("drawdowns_common_sample", fig)
    store.close()


if __name__ == "__main__":
    main()
