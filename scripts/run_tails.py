"""Tail co-movement: days on which several of the seven fall (or rise) sharply together.

The day counts come from SQL (`PriceStore.joint_move_counts`); the benchmarks (independence and a
Gaussian copula with the observed rank correlations) from `mega7.tails`.

Outputs (results/): joint_moves_distribution.csv, joint_moves_vs_benchmarks.csv, joint_crash_days.csv,
co_crash_matrix.csv, joint_crashes_by_year.csv, joint_crashes.png.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mega7 import STOCKS, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.tails import co_move_matrix, distribution_of_counts, joint_move_summary

THRESHOLDS = [-0.03, -0.05, -0.07, 0.05]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    panel = ReturnPanel.from_store(store)
    writer = ResultsWriter(args.results)
    daily = panel.returns("daily")
    stocks = daily[STOCKS]
    start = str(panel.start.date())

    rows, bench = [], []
    counts_5 = None
    for threshold in THRESHOLDS:
        counts = store.joint_move_counts(STOCKS, threshold, start=start).set_index("date")["n_moves"]
        counts.index = pd.DatetimeIndex(counts.index)
        # the SQL count and the pandas count must agree
        check = (stocks < threshold).sum(axis=1) if threshold < 0 else (stocks > threshold).sum(axis=1)
        assert counts.reindex(check.index).eq(check).all(), "SQL and pandas joint-move counts differ"
        dist = distribution_of_counts(counts, len(STOCKS))
        rows.append({"threshold": threshold, "days": int(dist.sum())} | {f"exactly_{k}": int(v) for k, v in dist.items()})
        bench.append(joint_move_summary(stocks, threshold, at_least=[1, 2, 3, 4, 5]))
        if threshold == -0.05:
            counts_5 = counts
    writer.csv("joint_moves_distribution", pd.DataFrame(rows))
    bench = pd.concat(bench, ignore_index=True)
    writer.csv("joint_moves_vs_benchmarks", bench)
    print(pd.DataFrame(rows).to_string(index=False))
    print(bench.round(2).to_string(index=False))

    # Days on which at least three of the seven fell more than 5%
    big = counts_5[counts_5 >= 3].index
    days = []
    for d in big:
        r = daily.loc[d]
        days.append({"date": d.date(), "stocks_down_over_5pct": int((r[STOCKS] < -0.05).sum()),
                     "which": " ".join(t for t in STOCKS if r[t] < -0.05),
                     "mega7_equal_weight": float(r[STOCKS].mean()), "SPY": float(r["SPY"]), "QQQ": float(r["QQQ"]),
                     "RSP": float(r["RSP"])})
    days = pd.DataFrame(days)
    writer.csv("joint_crash_days", days)
    print(days.round(3).to_string(index=False))

    by_year = pd.DataFrame({
        "days_any_down_5pct": (counts_5 >= 1).groupby(counts_5.index.year).sum(),
        "days_3plus_down_5pct": (counts_5 >= 3).groupby(counts_5.index.year).sum(),
        "stock_days_down_5pct": counts_5.groupby(counts_5.index.year).sum(),
        "trading_days": counts_5.groupby(counts_5.index.year).size(),
    })
    by_year.index.name = "year"
    writer.csv("joint_crashes_by_year", by_year.reset_index())
    print(by_year.to_string())

    cond, own = co_move_matrix(stocks, -0.05)
    out = cond.copy()
    out.insert(0, "given", out.index)
    out["own_days_down_5pct"] = own
    writer.csv("co_crash_matrix", out)
    print(out.round(2).to_string(index=False))

    # Chart: observed days with at least k falls of 5%, against the two benchmarks
    b = bench[bench["threshold"] == -0.05]
    charts.style()
    fig, ax = charts.plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(b))
    width = 0.26
    for i, (col, label, c) in enumerate([
        ("observed_days", "Observed", charts.INK),
        ("expected_gaussian_copula", "Gaussian copula, same rank correlations", charts.COLOURS["AAPL"]),
        ("expected_if_independent", "Independent stocks", charts.MUTED),
    ]):
        bars = ax.bar(x + (i - 1) * (width + 0.02), b[col], width=width, color=c, label=label)
        for rect, v in zip(bars, b[col]):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}" if v >= 1 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, color=charts.INK_2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x, [f"at least {k}" for k in b["at_least"]])
    ax.set_ylabel("Days (symmetric log scale)")
    ax.set_title(f"Days on which several of the seven fell more than 5%, {panel.start:%b %Y} to {panel.end:%b %Y}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    writer.figure("joint_crashes", fig)
    store.close()


if __name__ == "__main__":
    main()
