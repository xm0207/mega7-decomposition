"""Correlation structure of the seven: by frequency, over time, in calm and stressed markets, in the
tails, as clusters and principal components, plus dispersion and named episodes.

Outputs (results/): correlation_{daily,weekly,monthly}.csv, correlation_by_frequency.csv,
rolling_correlation_month_end.csv, conditional_correlation.csv, exceedance_correlation.csv,
clustering_merges.csv, pca.csv, dispersion_by_year.csv, episodes.csv, episode_days.csv and charts.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from mega7 import STOCKS, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.correlation import (
    average_exceedance_correlation,
    average_pairwise,
    conditional_correlation,
    cross_sectional_dispersion,
    hierarchical_clustering,
    normal_exceedance_benchmark,
    one_factor_implied_correlation,
    pairwise_table,
    pca,
    rolling_average_correlation,
    rolling_correlation_to,
    rolling_pc1_share,
)
from mega7.episodes import calendar_window, episode_summary, spy_drawdown_window

WINDOW = 252
THRESHOLDS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
# Each sell-off is located by one date inside it; its window is SPY's peak-to-trough, read from the data.
SELL_OFFS = {
    "Q4 2018 sell-off": "2018-12-24",
    "COVID crash": "2020-03-16",
    "2022 bear market": "2022-06-16",
    "Jul-Aug 2024 sell-off": "2024-08-05",
    "2025 tariff sell-off": "2025-04-08",
    "Early-2026 drawdown": "2026-03-02",
}
NAMED_DAYS = {
    "2020-03-16": "SPY's worst day in the sample",
    "2022-02-03": "META's worst day in the sample",
    "2023-05-25": "NVDA after its May 2023 results",
    "2024-07-11": "Rotation day: RSP up, Mega 7 down",
    "2024-08-05": "August 2024 sell-off low",
    "2025-01-27": "27 January 2025",
    "2025-04-04": "Second day of the April 2025 tariff sell-off",
    "2025-04-09": "SPY's best day in the sample",
}


def month_end(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Keep the last trading day of each month: a coarse sample of a rolling statistic."""
    return frame.groupby(frame.index.to_period("M")).tail(1)


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
    spy = daily["SPY"]

    # 1. Levels by frequency --------------------------------------------------------------------
    matrices, by_pair, summary = {}, [], []
    for freq in ("daily", "weekly", "monthly"):
        r = panel.returns(freq)[STOCKS + ["SPY"]]
        corr = r.corr()
        matrices[freq] = corr
        out = corr.copy()
        out.insert(0, "ticker", out.index)
        writer.csv(f"correlation_{freq}", out)
        table = pairwise_table(corr.loc[STOCKS, STOCKS]).rename(columns={"correlation": freq})
        by_pair.append(table.set_index(["first", "second"]))
        ranked = table.sort_values(freq)
        summary.append({
            "frequency": freq, "first_return": r.index[0].date(), "last_return": r.index[-1].date(),
            "observations": len(r), "avg_pairwise_corr": average_pairwise(corr.loc[STOCKS, STOCKS]),
            "avg_corr_to_spy": float(corr.loc[STOCKS, "SPY"].mean()),
            "highest_pair": f"{ranked.iloc[-1]['first']}-{ranked.iloc[-1]['second']}", "highest": ranked.iloc[-1][freq],
            "lowest_pair": f"{ranked.iloc[0]['first']}-{ranked.iloc[0]['second']}", "lowest": ranked.iloc[0][freq],
        })
        print(f"{freq:<8} obs={len(r):>5}  avg pairwise={average_pairwise(corr.loc[STOCKS, STOCKS]):.3f}  "
              f"avg corr to SPY={corr.loc[STOCKS, 'SPY'].mean():.3f}")
    writer.csv("correlation_summary", pd.DataFrame(summary))
    pairs = pd.concat(by_pair, axis=1).reset_index()
    pairs["monthly_minus_daily"] = pairs["monthly"] - pairs["daily"]
    avg = {"first": "Average of 21 pairs", "second": ""} | {c: pairs[c].mean() for c in pairs.columns[2:]}
    to_spy = [{"first": t, "second": "SPY", "daily": matrices["daily"].loc[t, "SPY"], "weekly": matrices["weekly"].loc[t, "SPY"],
               "monthly": matrices["monthly"].loc[t, "SPY"]} for t in STOCKS]
    for row in to_spy:
        row["monthly_minus_daily"] = row["monthly"] - row["daily"]
    pairs = pd.concat([pairs, pd.DataFrame([avg]), pd.DataFrame(to_spy)], ignore_index=True)
    writer.csv("correlation_by_frequency", pairs)

    fig, axes = charts.plt.subplots(1, 2, figsize=(10.5, 4.6))
    charts.heatmap(matrices["daily"].loc[STOCKS, STOCKS], "Daily returns", ax=axes[0])
    charts.heatmap(matrices["monthly"].loc[STOCKS, STOCKS], "Monthly returns", ax=axes[1])
    fig.suptitle(f"Correlations, {panel.start:%b %Y} to {panel.end:%b %Y}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    writer.figure("correlation_heatmaps", fig)

    # 2. Over time ------------------------------------------------------------------------------
    avg_corr = rolling_average_correlation(stocks, WINDOW)
    pc1 = rolling_pc1_share(stocks, WINDOW)
    to_spy_roll = rolling_correlation_to(stocks, spy, WINDOW)
    disp = cross_sectional_dispersion(stocks).rolling(WINDOW).mean().rename("avg_daily_dispersion_252d")
    rolling = pd.concat([avg_corr, pc1, disp, to_spy_roll.add_prefix("corr_to_spy_")], axis=1, sort=True).dropna()
    monthly_view = month_end(rolling)
    out = monthly_view.copy()
    out.insert(0, "date", out.index.date)
    writer.csv("rolling_correlation_month_end", out)
    print("Rolling 252-day avg pairwise correlation: min "
          f"{avg_corr.min():.3f} ({avg_corr.idxmin().date()}), max {avg_corr.max():.3f} ({avg_corr.idxmax().date()}), "
          f"last {avg_corr.iloc[-1]:.3f}")
    print(f"Rolling PC1 share: min {pc1.min():.3f} ({pc1.idxmin().date()}), max {pc1.max():.3f} ({pc1.idxmax().date()}), "
          f"last {pc1.iloc[-1]:.3f}")

    # A trailing window jumps when an extreme day enters it and again 252 days later when it leaves.
    # List the largest one-day changes with the day that entered and the day that left the window.
    change = avg_corr.diff().dropna()
    rows = []
    for day, delta in change.reindex(change.abs().nlargest(8).index).items():
        i = stocks.index.get_loc(day)
        entered, left = stocks.index[i], stocks.index[i - WINDOW]
        rows.append({"date": day.date(), "change_in_avg_corr": delta, "day_entering": entered.date(),
                     "spy_on_day_entering": float(spy[entered]), "day_leaving": left.date(),
                     "spy_on_day_leaving": float(spy[left])})
    window = stocks.loc[:"2026-03-31"].iloc[-WINDOW:]
    without = window.drop(window.loc["2025-04-03":"2025-04-10"].index)
    rows.append({"date": window.index[-1].date(), "change_in_avg_corr": np.nan,
                 "note": f"window {window.index[0].date()} to {window.index[-1].date()}: average pairwise "
                         f"{average_pairwise(window.corr()):.3f}; without 3-10 April 2025 "
                         f"({len(window) - len(without)} days): {average_pairwise(without.corr()):.3f}"})
    writer.csv("rolling_correlation_jumps", pd.DataFrame(rows))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 3. Episodes -------------------------------------------------------------------------------
    windows = [spy_drawdown_window(name, spy, inside) for name, inside in SELL_OFFS.items()]
    windows.insert(3, calendar_window("2023-24 AI rally", daily, "2023-01-01", "2024-12-31"))
    episodes = pd.DataFrame([episode_summary(daily, STOCKS, w) for w in windows])
    writer.csv("episodes", episodes)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(episodes.drop(columns=["dated_by"]).round(3).to_string(index=False))

    rows = []
    for day, label in NAMED_DAYS.items():
        d = pd.Timestamp(day)
        r = daily.loc[d]
        row = {"date": d.date(), "label": label} | {f"ret_{t}": float(r[t]) for t in daily.columns}
        row["mega7_equal_weight"] = float(r[STOCKS].mean())
        row["dispersion"] = float(r[STOCKS].std(ddof=1))
        row["stocks_down_over_5pct"] = int((r[STOCKS] < -0.05).sum())
        row["stocks_up"] = int((r[STOCKS] > 0).sum())
        rows.append(row)
    writer.csv("episode_days", pd.DataFrame(rows))

    shade = [(w.start, w.end, w.name.replace(" sell-off", "").replace(" drawdown", "")) for w in windows]
    charts.style()
    fig, axes = charts.plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, series, label in [
        (axes[0], avg_corr, "Average pairwise correlation"),
        (axes[1], pc1, "PC1 share of variance"),
        (axes[2], disp * 100, "Avg daily dispersion, %"),
    ]:
        charts.shade_windows(ax, shade, label=ax is axes[0])
        ax.plot(series.index, series, color=charts.INK, lw=1.3)
        ax.set_ylabel(label)
    axes[0].set_title(f"How much the seven move together: trailing {WINDOW}-day windows", pad=26)
    axes[1].axhline(1 / len(STOCKS), color=charts.MUTED, lw=0.8)
    axes[1].text(daily.index[WINDOW], 1 / len(STOCKS) + 0.01, "1/7: no common component", fontsize=7, color=charts.MUTED)
    fig.tight_layout()
    writer.figure("rolling_correlation_pc1", fig)

    fig = charts.small_multiples(to_spy_roll, f"Correlation with SPY, trailing {WINDOW} days", "Correlation",
                                 ylim=(0, 1), windows=shade)
    writer.figure("rolling_correlation_to_spy", fig)

    # 4. Calm vs stressed -------------------------------------------------------------------------
    worst10 = spy <= spy.quantile(0.10)
    best10 = spy >= spy.quantile(0.90)
    conditions = {
        "All days": pd.Series(True, index=spy.index),
        "SPY up days": spy > 0,
        "SPY down days": spy < 0,
        "SPY worst 10% of days": worst10,
        "SPY middle 80% of days": ~worst10 & ~best10,
        "SPY best 10% of days": best10,
    }
    rows = []
    for name, mask in conditions.items():
        actual = conditional_correlation(stocks, mask)
        implied = one_factor_implied_correlation(stocks, spy, mask)
        rows.append({
            "condition": name,
            "days": int(mask.sum()),
            "spy_daily_vol": float(spy[mask].std(ddof=1)),
            "avg_pairwise_corr": average_pairwise(actual),
            "avg_pairwise_corr_if_only_spy_vol_changed": average_pairwise(implied),
            "excess_over_benchmark": average_pairwise(actual) - average_pairwise(implied),
        })
    cond = pd.DataFrame(rows)
    writer.csv("conditional_correlation", cond)
    print(cond.round(3).to_string(index=False))

    exceed = average_exceedance_correlation(stocks, THRESHOLDS)
    z = (stocks - stocks.mean()) / stocks.std(ddof=1)
    exceed["normal_benchmark"] = normal_exceedance_benchmark(z.corr(), THRESHOLDS).to_numpy()
    exceed["minus_normal"] = exceed["avg_correlation"] - exceed["normal_benchmark"]
    writer.csv("exceedance_correlation", exceed)
    print(exceed.round(3).to_string(index=False))

    charts.style()
    fig, ax = charts.plt.subplots(figsize=(7, 4))
    ax.plot(exceed["threshold"], exceed["avg_correlation"], color=charts.INK, marker="o", ms=5, label="Mega 7, average of 21 pairs")
    ax.plot(exceed["threshold"], exceed["normal_benchmark"], color=charts.MUTED, marker="o", ms=5,
            label="Bivariate normal with the same correlations")
    ax.axvline(0, color=charts.AXIS, lw=0.8)
    ax.set_xlabel("Threshold, standard deviations (left: both fall below it; right: both rise above it)")
    ax.set_ylabel("Correlation on those days")
    ax.set_title("Exceedance correlations: joint falls versus joint rises")
    ax.legend(loc="upper center")
    writer.figure("exceedance_correlation", fig)

    # 5. Clusters and principal components ------------------------------------------------------
    corr_d = matrices["daily"].loc[STOCKS, STOCKS]
    resid = stocks.sub(stocks.mean()).copy()
    spy_c = spy - spy.mean()
    for t in STOCKS:
        b = (resid[t] * spy_c).sum() / (spy_c @ spy_c)
        resid[t] = resid[t] - b * spy_c
    corr_resid = resid.corr()
    merges = []
    charts.style()
    fig, axes = charts.plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, (label, c) in zip(axes, [("Raw daily returns", corr_d), ("Residuals after removing SPY", corr_resid)]):
        z_link, order, table = hierarchical_clustering(c, "average")
        table.insert(0, "input", label)
        merges.append(table)
        dendrogram(z_link, labels=list(c.columns), ax=ax, color_threshold=0, above_threshold_color=charts.INK_2)
        ax.set_title(label)
        ax.set_ylabel("Distance sqrt(2(1 - correlation))")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Hierarchical clustering (average linkage)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    writer.figure("clustering_dendrograms", fig)
    merges = pd.concat(merges, ignore_index=True)
    writer.csv("clustering_merges", merges)
    print(merges.round(3).to_string(index=False))
    resid_out = corr_resid.copy()
    resid_out.insert(0, "ticker", resid_out.index)
    writer.csv("correlation_residual_vs_spy", resid_out)
    print(f"Average pairwise residual correlation after removing SPY: {average_pairwise(corr_resid):.3f}")

    rows = []
    for freq in ("daily", "weekly", "monthly"):
        share, loadings = pca(matrices[freq].loc[STOCKS, STOCKS])
        rho = average_pairwise(matrices[freq].loc[STOCKS, STOCKS])
        row = {"frequency": freq} | {f"share_{k}": v for k, v in share.items()}
        row["pc1_if_equicorrelated"] = (1 + (len(STOCKS) - 1) * rho) / len(STOCKS)
        row |= {f"pc1_loading_{t}": loadings.loc[t, "PC1"] for t in STOCKS}
        row |= {f"pc2_loading_{t}": loadings.loc[t, "PC2"] for t in STOCKS}
        rows.append(row)
    pca_table = pd.DataFrame(rows)
    writer.csv("pca", pca_table)
    print(pca_table.round(3).T.to_string())

    # 6. Dispersion by year -----------------------------------------------------------------------
    years = store.calendar_year_returns(start=str(panel.start.date()))
    annual = years[years["series"].isin(STOCKS)].pivot(index="year", columns="series", values="total_return")
    by_year = pd.DataFrame({
        "avg_daily_dispersion": cross_sectional_dispersion(stocks).groupby(stocks.index.year).mean(),
        "avg_pairwise_corr": stocks.groupby(stocks.index.year).apply(lambda x: average_pairwise(x.corr())),
        "pc1_share": stocks.groupby(stocks.index.year).apply(lambda x: float(pca(x.corr())[0].iloc[0])),
        "best_stock": annual.idxmax(axis=1),
        "best_return": annual.max(axis=1),
        "worst_stock": annual.idxmin(axis=1),
        "worst_return": annual.min(axis=1),
        "days": stocks.groupby(stocks.index.year).size(),
    })
    by_year["spread_best_minus_worst"] = by_year["best_return"] - by_year["worst_return"]
    by_year.index.name = "year"
    writer.csv("dispersion_by_year", by_year.reset_index())
    print(by_year.round(3).to_string())
    store.close()


if __name__ == "__main__":
    main()
