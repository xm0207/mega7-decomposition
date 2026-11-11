"""Download prices (Yahoo Finance) and factors (Ken French) into data/mega7.duckdb, and check them.

Writes to results/ only coverage and data-check summaries, never prices or returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mega7 import STOCKS, TICKERS, KenFrenchClient, PriceStore, ResultsWriter, YahooClient
from mega7.sources import first_table

FRENCH_FILES = {
    "F-F_Research_Data_5_Factors_2x3_daily": "daily",
    "F-F_Momentum_Factor_daily": "daily",
    "F-F_Research_Data_5_Factors_2x3": "monthly",
    "F-F_Momentum_Factor": "monthly",
}


def long_format(wide: pd.DataFrame) -> pd.DataFrame:
    frame = wide.copy()
    frame.index.name = "date"
    long = frame.reset_index().melt(id_vars="date", var_name="series", value_name="value")
    return long.dropna(subset=["value"])


def data_checks(store: PriceStore) -> pd.DataFrame:
    """Per ticker: calendar gaps against SPY in the common sample, zero-return days, very large moves."""
    daily = store.returns("daily")
    common_start = daily[STOCKS].dropna().index[0]
    spy_days = daily["SPY"].dropna().loc[common_start:].index
    rows = []
    for t in TICKERS:
        r = daily[t].dropna()
        common = r.loc[common_start:]
        closes = store.wide("price", [t])[t].dropna()
        gaps = closes.index.to_series().diff().dt.days
        rows.append(
            {
                "ticker": t,
                "first_close": closes.index[0].date(),
                "first_return": r.index[0].date(),
                "last_close": closes.index[-1].date(),
                "closes": len(closes),
                "common_sample_days_missing_vs_spy": int(len(spy_days.difference(common.index))),
                "common_sample_zero_return_days": int((common == 0).sum()),
                "full_history_days_abs_return_over_20pct": int((r.abs() > 0.20).sum()),
                "longest_gap_between_closes_calendar_days": int(gaps.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    writer = ResultsWriter(args.results)

    yahoo = YahooClient()
    for ticker in TICKERS:
        frame = yahoo.fetch(ticker)
        rows = store.upsert("price", frame)
        print(f"{ticker:<6} {rows:>6} closes  {frame['date'].min().date()} to {frame['date'].max().date()}")

    as_of = pd.Timestamp.now(tz="America/New_York").date()
    caps = pd.DataFrame(
        [{"series": t, "as_of": as_of, "market_cap": yahoo.market_cap(t)} for t in STOCKS]
    ).dropna(subset=["market_cap"])
    store.save_market_caps(caps)
    print(f"current market caps for {len(caps)} stocks, as of {as_of}")

    french = KenFrenchClient(raw_dir=Path(args.raw) / "french")
    for frequency in ("daily", "monthly"):
        tables = [first_table(french.fetch(name), frequency) for name, f in FRENCH_FILES.items() if f == frequency]
        wide = pd.concat(tables, axis=1, join="outer", sort=True)
        store.upsert(f"french_{frequency}", long_format(wide))
        print(f"Ken French {frequency}: {wide.index[0].date()} to {wide.index[-1].date()}, columns {list(wide.columns)}")

    coverage = store.coverage()
    coverage = coverage[coverage["dataset"] == "price"].drop(columns="dataset")
    french_cov = store.coverage()
    french_cov = (
        french_cov[french_cov["dataset"] != "price"]
        .groupby("dataset")
        .agg(first_date=("first_date", "min"), last_date=("last_date", "max"), observations=("observations", "max"))
        .reset_index()
        .rename(columns={"dataset": "series"})
    )
    writer.csv("data_coverage", pd.concat([coverage, french_cov], ignore_index=True))

    checks = data_checks(store)
    writer.csv("data_checks", checks)
    print(checks.to_string(index=False))

    # META traded as FB until 9 June 2022; Yahoo keeps the whole history under META.
    meta = store.wide("price", ["META"])["META"]
    around = meta.loc["2022-06-01":"2022-06-15"]
    print("META closes around the FB -> META ticker change (9 June 2022):")
    print(around.round(2).to_string())

    capped = store.market_caps()
    capped["share_of_seven"] = capped["market_cap"] / capped["market_cap"].sum()
    capped["market_cap_usd_bn"] = capped["market_cap"] / 1e9
    writer.csv("current_market_caps", capped[["series", "as_of", "market_cap_usd_bn", "share_of_seven"]])
    store.close()


if __name__ == "__main__":
    main()
