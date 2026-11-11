"""The seven as a portfolio, and their role in the index.

- An equal-weight Mega 7 portfolio (rebalanced monthly; daily-rebalanced and buy-and-hold variants)
  against SPY, QQQ and RSP.
- Each stock's marginal and component contribution to the portfolio's volatility, and the
  diversification ratio, in the full sample, over the last year and over time.
- Cap-weighted minus equal-weighted S&P 500 (SPY minus RSP) as a proxy for index concentration, and
  how much of it lines up with the Mega 7.

Historical index weights are not used (they are not in the data). Current market caps from Yahoo are
used once, labelled as current, to weight the seven among themselves.

Outputs (results/): portfolio_vs_index.csv, portfolio_calendar_years.csv, risk_contributions.csv,
diversification.csv, buy_and_hold_weights.csv, rolling_risk_month_end.csv, concentration_month_end.csv,
concentration_regression.csv and charts.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mega7 import STOCKS, EqualWeightPortfolio, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.correlation import rolling_correlation_to
from mega7.factors import factor_regression
from mega7.metrics import (
    annualised_return,
    annualised_volatility,
    beta,
    capture_ratios,
    drawdown_series,
    historical_var_es,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from mega7.portfolio import diversification_ratio, portfolio_volatility, risk_contributions, rolling_risk_split

WINDOW = 252
EW = "Mega 7 EW"


def monthly_from_daily(daily: pd.Series) -> pd.Series:
    out = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


def month_end(frame):
    return frame.groupby(frame.index.to_period("M")).tail(1)


def performance_row(name: str, r: pd.Series, spy: pd.Series, rf: pd.Series, monthly: pd.Series, spy_m: pd.Series) -> dict:
    mdd = max_drawdown(r)
    var99, es99 = historical_var_es(r, 0.99)
    active = r - spy
    row = {
        "portfolio": name,
        "ann_return": annualised_return(r),
        "ann_volatility": annualised_volatility(r),
        "sharpe": sharpe_ratio(r, rf),
        "sortino": sortino_ratio(r, rf),
        "max_drawdown": mdd.depth,
        "mdd_peak": mdd.peak.date() if mdd.peak is not None else None,
        "mdd_trough": mdd.trough.date(),
        "mdd_recovery": mdd.recovery.date() if mdd.recovery is not None else None,
        "var99_1d": var99,
        "es99_1d": es99,
    }
    if name != "SPY":
        up, down = capture_ratios(monthly, spy_m)
        row |= {
            "beta_spy_daily": beta(r, spy),
            "corr_spy_daily": float(r.corr(spy)),
            "tracking_error_vs_spy": float(active.std(ddof=1) * np.sqrt(252)),
            "information_ratio_vs_spy": float(active.mean() * 252 / (active.std(ddof=1) * np.sqrt(252))),
            "upside_capture_monthly": up,
            "downside_capture_monthly": down,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    panel = ReturnPanel.from_store(store)
    writer = ResultsWriter(args.results)
    daily = panel.returns("daily")
    monthly = panel.returns("monthly")
    rf = panel.rf("daily", daily.index)

    # 1. Equal-weight portfolio against the index funds --------------------------------------------
    portfolios = {
        f"{EW} (monthly rebalance)": EqualWeightPortfolio(daily[STOCKS], "monthly"),
        f"{EW} (daily rebalance)": EqualWeightPortfolio(daily[STOCKS], "daily"),
        f"{EW} (buy and hold from {panel.start.date()})": EqualWeightPortfolio(daily[STOCKS], "never"),
    }
    series = {name: p.returns for name, p in portfolios.items()} | {t: daily[t] for t in ("SPY", "QQQ", "RSP")}
    spy_m = monthly["SPY"]
    rows = []
    for name, r in series.items():
        m = monthly_from_daily(r).reindex(monthly.index) if name.startswith(EW) else monthly[name]
        row = performance_row(name, r, daily["SPY"], rf, m, spy_m)
        if name in portfolios:
            row["turnover_per_year"] = portfolios[name].turnover()
        rows.append(row)
    perf = pd.DataFrame(rows)
    writer.csv("portfolio_vs_index", perf)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(perf.round(3).to_string(index=False))

    bh = portfolios[f"{EW} (buy and hold from {panel.start.date()})"]
    end_weights = bh.weights.iloc[-1] * (1 + daily[STOCKS].iloc[-1])
    end_weights = end_weights / end_weights.sum()
    print("Buy-and-hold weights at the end:", end_weights.round(3).to_dict())
    writer.csv("buy_and_hold_weights", pd.DataFrame({
        "ticker": STOCKS, "weight_at_start": 1 / len(STOCKS), "start": panel.start.date(),
        "weight_at_end": end_weights[STOCKS].to_numpy(), "end": daily.index[-1].date(),
    }))

    ew = portfolios[f"{EW} (monthly rebalance)"].returns.rename(EW)
    years = store.calendar_year_returns(start=str(panel.start.date()))
    annual = years.pivot(index="year", columns="series", values="total_return")[["SPY", "QQQ", "RSP"]]
    annual.insert(0, EW, (1 + ew).groupby(ew.index.year).prod() - 1)
    annual["SPY_minus_RSP"] = annual["SPY"] - annual["RSP"]
    annual[f"{EW}_minus_SPY"] = annual[EW] - annual["SPY"]
    annual["days"] = ew.groupby(ew.index.year).size()
    annual.index.name = "year"
    writer.csv("portfolio_calendar_years", annual.reset_index())
    print(annual.round(3).to_string())

    frame = pd.concat([ew, daily[["SPY", "QQQ", "RSP"]]], axis=1)
    fig = charts.wealth_chart(frame, f"Equal-weight Mega 7 (monthly rebalance) and index funds, growth of $1")
    writer.figure("portfolio_wealth", fig)
    fig = charts.drawdown_chart(pd.DataFrame({c: drawdown_series(frame[c]) for c in frame.columns}),
                                "Drawdown from previous peak")
    writer.figure("portfolio_drawdowns", fig)

    # 2. Risk contributions ---------------------------------------------------------------------
    w_eq = pd.Series(1 / len(STOCKS), index=STOCKS)
    caps = store.market_caps().set_index("series")
    w_cap = (caps["market_cap"] / caps["market_cap"].sum()).reindex(STOCKS)
    as_of = pd.Timestamp(caps["as_of"].iloc[0]).date()
    last_year = daily[STOCKS].iloc[-WINDOW:]
    covs = {
        f"full sample {daily.index[0].date()} to {daily.index[-1].date()}": daily[STOCKS].cov() * 252,
        f"last {WINDOW} days {last_year.index[0].date()} to {last_year.index[-1].date()}": last_year.cov() * 252,
    }
    rows, summary = [], []
    for cov_label, cov in covs.items():
        for w_label, w in (("equal weight", w_eq), (f"current market caps ({as_of})", w_cap)):
            if w_label.startswith("current") and cov_label.startswith("full"):
                continue  # today's weights with a 14-year covariance mixes two different times
            rc = risk_contributions(w, cov)
            rc.insert(0, "covariance", cov_label)
            rc.insert(0, "weights", w_label)
            rows.append(rc.rename_axis("ticker").reset_index())
            summary.append({
                "weights": w_label, "covariance": cov_label,
                "portfolio_vol": portfolio_volatility(w, cov),
                "weighted_avg_stock_vol": float(w @ np.sqrt(np.diag(cov))),
                "diversification_ratio": diversification_ratio(w, cov),
                "largest_contributor": rc["percent"].idxmax(),
                "largest_share": float(rc["percent"].max()),
            })
    contributions = pd.concat(rows, ignore_index=True)
    writer.csv("risk_contributions", contributions)
    writer.csv("diversification", pd.DataFrame(summary))
    print(contributions.round(3).to_string(index=False))
    print(pd.DataFrame(summary).round(3).to_string(index=False))

    roll = rolling_risk_split(daily[STOCKS], WINDOW)
    roll_m = month_end(roll)
    out = roll_m.copy()
    out.insert(0, "date", out.index.date)
    writer.csv("rolling_risk_month_end", out)
    print("Rolling diversification ratio: min", round(roll["diversification_ratio"].min(), 3),
          roll["diversification_ratio"].idxmin().date(), "max", round(roll["diversification_ratio"].max(), 3),
          roll["diversification_ratio"].idxmax().date(), "last", round(roll["diversification_ratio"].iloc[-1], 3))
    print("Largest rolling percent contribution by stock (max over time):",
          roll[STOCKS].max().round(3).to_dict())

    fig, ax = charts.plt.subplots(figsize=(9, 4.2))
    charts.style()
    ax.stackplot(roll_m.index, *[roll_m[t] * 100 for t in STOCKS], labels=STOCKS,
                 colors=[charts.colour(t) for t in STOCKS], edgecolor=charts.SURFACE, linewidth=0.6)
    ax.axhline(100 / len(STOCKS), color=charts.SURFACE, lw=0.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of portfolio volatility, %")
    ax.set_title(f"Who carries the risk of an equal-weight Mega 7? Trailing {WINDOW}-day covariance")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.margins(x=0)
    writer.figure("rolling_risk_contributions", fig)

    # 3. Concentration: SPY minus RSP --------------------------------------------------------------
    gap_252 = (1 + daily["SPY"]).rolling(WINDOW).apply(np.prod, raw=True) - (1 + daily["RSP"]).rolling(WINDOW).apply(np.prod, raw=True)
    ew_rel_252 = (1 + ew).rolling(WINDOW).apply(np.prod, raw=True) - (1 + daily["RSP"]).rolling(WINDOW).apply(np.prod, raw=True)
    r2_spy = rolling_correlation_to(daily[["SPY", "RSP"]], ew, WINDOW) ** 2
    conc = pd.concat([gap_252.rename("spy_minus_rsp_trailing_252d"), ew_rel_252.rename("mega7_ew_minus_rsp_trailing_252d"),
                      r2_spy["SPY"].rename("r2_spy_on_mega7_ew_252d"), r2_spy["RSP"].rename("r2_rsp_on_mega7_ew_252d")],
                     axis=1, sort=True).dropna()
    conc_m = month_end(conc)
    out = conc_m.copy()
    out.insert(0, "date", out.index.date)
    writer.csv("concentration_month_end", out)
    print(conc_m.iloc[[0, -1]].round(3).to_string())
    print("R2 of SPY on Mega 7 EW, by calendar year (252d at year end):",
          conc["r2_spy_on_mega7_ew_252d"].groupby(conc.index.year).last().round(3).to_dict())

    ew_m = monthly_from_daily(ew).reindex(monthly.index)
    y = (monthly["SPY"] - monthly["RSP"]).rename("SPY-RSP")
    rows = []
    for label, x in (("Mega 7 EW minus RSP", ew_m - monthly["RSP"]), ("QQQ minus RSP", monthly["QQQ"] - monthly["RSP"])):
        fit = factor_regression(y, x.rename(label).to_frame(), periods=12)
        rows.append({"y": "SPY minus RSP (monthly)", "x": label, "start": fit.start.date(), "end": fit.end.date(),
                     "months": fit.observations, "alpha_ann": fit.alpha, "alpha_t": fit.alpha_t,
                     "beta": float(fit.betas.iloc[0]), "beta_t": float(fit.t_stats.iloc[0]), "r_squared": fit.r_squared,
                     "nw_lags": fit.lags})
    reg = pd.DataFrame(rows)
    writer.csv("concentration_regression", reg)
    print(reg.round(3).to_string(index=False))

    charts.style()
    fig, axes = charts.plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(conc.index, conc["spy_minus_rsp_trailing_252d"] * 100, color=charts.INK, lw=1.3)
    axes[0].axhline(0, color=charts.AXIS, lw=0.8)
    axes[0].set_ylabel("SPY minus RSP, % points")
    axes[0].set_title(f"Cap-weighted minus equal-weighted S&P 500, trailing {WINDOW}-day return")
    axes[1].plot(conc.index, conc["r2_spy_on_mega7_ew_252d"], color=charts.INK, lw=1.3, label="SPY")
    axes[1].plot(conc.index, conc["r2_rsp_on_mega7_ew_252d"], color=charts.MUTED, lw=1.3, label="RSP")
    axes[1].set_ylabel("R-squared")
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Share of daily index variance explained by the equal-weight Mega 7, trailing {WINDOW} days")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    writer.figure("concentration_proxy", fig)

    fig = charts.bar_chart(
        contributions[(contributions["weights"] == "equal weight") & contributions["covariance"].str.startswith("last")]
        .set_index("ticker")["percent"],
        f"Share of equal-weight Mega 7 volatility, last {WINDOW} days", "Share of portfolio volatility",
        reference=1 / len(STOCKS), reference_label="1/7")
    writer.figure("risk_contributions_last_year", fig)
    store.close()


if __name__ == "__main__":
    main()
