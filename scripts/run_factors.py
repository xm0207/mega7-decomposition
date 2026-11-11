"""Factor decomposition: Fama-French five factors plus momentum.

- Full-sample monthly regressions (CAPM and the six-factor model) with Newey-West t-statistics, over
  the common sample and over each stock's own history.
- Systematic versus idiosyncratic variance, with each factor's share, monthly and daily.
- Rolling one-year (252-day) daily regressions: betas and R-squared over time, sampled at month end.
- Correlation of the residuals: what the seven still share after the factors are removed.

Outputs (results/): factor_regressions.csv, factor_variance_decomposition.csv,
rolling_factor_month_end.csv, residual_correlation_ff6.csv, factor_betas.png, rolling_r_squared.png,
rolling_market_beta.png.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mega7 import FACTORS, STOCKS, EqualWeightPortfolio, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.correlation import average_pairwise, pca
from mega7.factors import factor_regression, residuals, rolling_factor_regression, variance_decomposition

WINDOW = 252
EW = "Mega 7 EW"


def monthly_from_daily(daily: pd.Series) -> pd.Series:
    """Compound daily returns within each calendar month (labelled at month end)."""
    out = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    panel = ReturnPanel.from_store(store)
    writer = ResultsWriter(args.results)

    f_m = panel.factors("monthly")
    f_d = panel.factors("daily")
    daily = panel.returns("daily")
    monthly = panel.returns("monthly")
    ew_daily = EqualWeightPortfolio(daily[STOCKS], "monthly").returns
    ew_monthly = monthly_from_daily(ew_daily)
    ew_monthly = ew_monthly.loc[monthly.index[0] : monthly.index[-1]]
    rf_m = panel.rf("monthly", monthly.index)
    excess_m = monthly.sub(rf_m, axis=0)
    excess_m[EW] = ew_monthly.reindex(monthly.index) - rf_m
    excess_d = daily.sub(panel.rf("daily", daily.index), axis=0)
    excess_d[EW] = ew_daily - panel.rf("daily", daily.index)
    names = STOCKS + [EW, "SPY", "QQQ", "RSP"]

    # 1. Full-sample monthly regressions ----------------------------------------------------------
    rows = []
    for ticker in names:
        samples = {"common": excess_m[ticker]}
        if ticker in STOCKS:
            own = panel.history(ticker, "monthly")
            own = own.loc[own.index[0] :]
            samples["full_history"] = own - panel.rf("monthly", own.index)
        for sample, y in samples.items():
            for model, cols in (("CAPM", ["Mkt-RF"]), ("FF5+Mom", FACTORS)):
                fit = factor_regression(y, f_m[cols], periods=12)
                rows.append(pd.concat([pd.Series({"ticker": ticker, "sample": sample, "model": model}), fit.to_series()]))
    regressions = pd.DataFrame(rows)
    writer.csv("factor_regressions", regressions)
    show = regressions[(regressions["model"] == "FF5+Mom")]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(show[["ticker", "sample", "start", "end", "observations", "alpha_ann", "alpha_t", *FACTORS, "r_squared"]]
              .round(3).to_string(index=False))
        capm = regressions[regressions["model"] == "CAPM"]
        print(capm[["ticker", "sample", "alpha_ann", "alpha_t", "Mkt-RF", "r_squared"]].round(3).to_string(index=False))
        print(show[["ticker", "sample", *[f"{f}_t" for f in FACTORS]]].round(2).to_string(index=False))

    # 2. Variance decomposition -------------------------------------------------------------------
    rows = []
    for freq, excess, factors, periods in (("monthly", excess_m, f_m, 12), ("daily", excess_d, f_d, 252)):
        for ticker in names:
            fit = factor_regression(excess[ticker], factors[FACTORS], periods=periods)
            split = variance_decomposition(excess[ticker], factors[FACTORS], fit.betas)
            rows.append({"frequency": freq, "ticker": ticker, "start": fit.start.date(), "end": fit.end.date(),
                         "observations": fit.observations, "r_squared": fit.r_squared} | split.to_dict())
    decomposition = pd.DataFrame(rows)
    writer.csv("factor_variance_decomposition", decomposition)
    print(decomposition.round(3).to_string(index=False))

    # 3. Rolling daily regressions ----------------------------------------------------------------
    rolling = []
    for ticker in STOCKS + [EW]:
        roll = rolling_factor_regression(excess_d[ticker], f_d[FACTORS], WINDOW)
        roll = roll.groupby(roll.index.to_period("M")).tail(1)
        roll.insert(0, "ticker", ticker)
        roll.insert(0, "date", roll.index.date)
        rolling.append(roll)
    rolling = pd.concat(rolling, ignore_index=True)
    writer.csv("rolling_factor_month_end", rolling)
    r2 = rolling.pivot(index="date", columns="ticker", values="r_squared")
    r2.index = pd.to_datetime(r2.index)
    mkt = rolling.pivot(index="date", columns="ticker", values="Mkt-RF")
    mkt.index = pd.to_datetime(mkt.index)
    print("Rolling R-squared (month end): first, min, max, last")
    print(pd.DataFrame({"first": r2.iloc[0], "min": r2.min(), "max": r2.max(), "last": r2.iloc[-1],
                        "last_date": r2.index[-1].date()}).round(3).to_string())

    fig = charts.small_multiples(r2[STOCKS], f"Share of daily variance explained by FF5 + momentum, trailing {WINDOW} days",
                                 "R-squared", reference=r2[EW], reference_label="Mega 7 equal weight", ylim=(0, 1))
    writer.figure("rolling_r_squared", fig)
    fig = charts.small_multiples(mkt[STOCKS], f"Market beta (Mkt-RF) in FF5 + momentum, trailing {WINDOW} days",
                                 "Beta", reference=mkt[EW], reference_label="Mega 7 equal weight")
    for ax in fig.axes:
        ax.axhline(1.0, color=charts.AXIS, lw=0.8)
    writer.figure("rolling_market_beta", fig)

    # 4. Residual correlation --------------------------------------------------------------------
    res = residuals(excess_d[STOCKS], f_d[FACTORS])
    corr = res.corr()
    out = corr.copy()
    out.insert(0, "ticker", out.index)
    writer.csv("residual_correlation_ff6", out)
    share, _ = pca(corr)
    print(f"Daily FF5+Mom residuals, {res.index[0].date()} to {res.index[-1].date()}: average pairwise correlation "
          f"{average_pairwise(corr):.3f}; PC1 share {share.iloc[0]:.3f}")
    raw = daily.loc[res.index, STOCKS].corr()
    print(f"Raw daily over the same dates: average pairwise {average_pairwise(raw):.3f}")
    writer.text("factor_notes", [
        f"Monthly regressions: common sample {regressions.loc[regressions['sample'] == 'common', 'start'].iloc[0]} "
        f"to {regressions['end'].max()}; Ken French files end {panel.factor_end.date()}.",
        f"Daily FF5+Mom residual correlation ({res.index[0].date()} to {res.index[-1].date()}): average pairwise "
        f"{average_pairwise(corr):.4f}, PC1 share {share.iloc[0]:.4f}; raw returns over the same days: "
        f"average pairwise {average_pairwise(raw):.4f}.",
        "Newey-West lags follow floor(4 (T/100)^(2/9)); the lag used is in the nw_lags column.",
    ])

    # 5. Betas chart -------------------------------------------------------------------------------
    common = show[show["sample"] == "common"].set_index("ticker")[FACTORS]
    fig, _ = charts.heatmap(common, f"Monthly factor betas, FF5 + momentum, {monthly.index[0]:%b %Y} to "
                            f"{panel.factor_end:%b %Y}", vmin=-1.8, vmax=1.8, diverging=True)
    fig.set_size_inches(6.4, 4.6)
    writer.figure("factor_betas", fig)
    store.close()


if __name__ == "__main__":
    main()
