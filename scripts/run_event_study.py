"""Event study: Nvidia's sell-off of November 2026, compared with the DeepSeek day (27 January 2025).

Each day's NVDA return is split into market (SPY), semiconductor-sector, alpha and company-specific
(abnormal) parts with the two-factor market model in `mega7.event_study`. The sector factor is SMH in
the primary specification. Nvidia is SMH's largest holding, so when Nvidia falls on its own news SMH
falls with it, and part of Nvidia's own move is labelled "sector". The robustness check replaces SMH
with an equal-weight basket of eight large semiconductor peers that excludes Nvidia.

Estimation windows: 250 trading days ending 21 trading days before the first day of the event window
(4 November 2026; 27 January 2025). The other six of the seven get a one-factor (SPY) model on the same
windows. NVDA's abnormal returns on 9 November 2026 and 27 January 2025 are ranked against a rolling
version of the same model, re-estimated for every day with the same rule (no look-ahead), over all the
history that SMH allows.

Outputs (results/): event_study_estimation.csv, event_study_days.csv, event_study_windows.csv,
event_study_sector_members.csv, event_study_megacaps.csv, event_study_ranks.csv, event_study_parts.png.
Only event days and window totals are written; the rolling history behind the ranks stays in memory.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from mega7 import SECTOR_ETF, SEMI_PEERS, STOCKS, EventStudy, PriceStore, ResultsWriter, ReturnPanel
from mega7 import charts
from mega7.event_study import equal_weight_basket, rank_in_history

STOCK, MARKET = "NVDA", "SPY"
PEERS = "Peers ex-NVDA (equal weight)"
MIN_PEERS = 5  # the basket needs at least five of the eight peers on a day (AVGO lists only in 2009)
NOVEMBER, DEEPSEEK = "November 2026", "DeepSeek, January 2025"
EVENTS = {
    NOVEMBER: {
        "first_day": "2026-11-04",
        "days": {
            "2026-11-04": "Chip and AI stocks sell off",
            "2026-11-05": "Chip and AI stocks sell off, second day",
            "2026-11-06": "Last trading day before the board's statement",
            "2026-11-09": "First trading day after the board's statement of 8 November",
            "2026-11-10": "Second trading day after the statement",
        },
    },
    DEEPSEEK: {
        "first_day": "2025-01-27",
        "days": {"2025-01-27": "DeepSeek day", "2025-01-28": "The day after"},
    },
}
# (event, window, first day, last day); the start of the peak window is read from the data below
WINDOWS = [
    (NOVEMBER, "9-10 Nov 2026 (after the board's statement)", "2026-11-09", "2026-11-10"),
    (NOVEMBER, "4-10 Nov 2026", "2026-11-04", "2026-11-10"),
    (NOVEMBER, "peak", None, "2026-11-10"),
    (DEEPSEEK, "27 Jan 2025", "2025-01-27", "2025-01-27"),
    (DEEPSEEK, "27-28 Jan 2025", "2025-01-27", "2025-01-28"),
]
RANKED_DAYS = {"2026-11-09": NOVEMBER, "2025-01-27": DEEPSEEK}
PART_LABELS = {
    "market": "Market (SPY)",
    "sector": "Semiconductor sector beyond SPY",
    "alpha": "Alpha",
    "abnormal": "Company-specific (abnormal)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mega7.duckdb")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    store = PriceStore(args.db)
    panel = ReturnPanel.from_store(store)
    writer = ResultsWriter(args.results)

    nvda, spy = panel.history(STOCK), panel.history(MARKET)
    peers = pd.concat({t: panel.history(t) for t in SEMI_PEERS}, axis=1, sort=True)
    sectors = {SECTOR_ETF: panel.history(SECTOR_ETF), PEERS: equal_weight_basket(peers, MIN_PEERS)}
    studies = {name: EventStudy(nvda, spy, sector) for name, sector in sectors.items()}
    one_factor = EventStudy(nvda, spy)

    # The peak: Nvidia's highest close before the November event window, read from the data
    closes = store.wide("price", [STOCK])[STOCK].dropna()
    first_nov = pd.Timestamp(EVENTS[NOVEMBER]["first_day"])
    peak = closes.loc[closes.index < first_nov].idxmax()
    after_peak = closes.index[closes.index.get_loc(peak) + 1]
    windows = [(e, f"{peak.day} {peak:%b %Y} peak to 10 Nov 2026", str(after_peak.date()), end) if w == "peak"
               else (e, w, start, end) for e, w, start, end in WINDOWS]
    print(f"NVDA peak close {closes[peak]:.2f} on {peak.date()}; close on 10 Nov 2026 "
          f"{closes['2026-11-10']:.2f} ({closes['2026-11-10'] / closes[peak] - 1:+.2%})")

    # 1. Estimation windows -------------------------------------------------------------------------
    rows = []
    for event, spec in EVENTS.items():
        first = spec["first_day"]
        for name, study in [*studies.items(), ("none (one-factor)", one_factor)]:
            fit = study.fit(first)
            row = {"event": event, "first_event_day": first, "model": "two-factor" if fit.two_factor else "one-factor",
                   "sector_factor": name} | fit.to_series().to_dict()
            if name == PEERS:
                window = peers.loc[fit.start : fit.end]
                row["peer_members_min"] = int(window.notna().sum(axis=1).min())
            rows.append(row)
    estimation = pd.DataFrame(rows)
    writer.csv("event_study_estimation", estimation)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(estimation.round(4).to_string(index=False))

    # 2. Event days ---------------------------------------------------------------------------------
    rows = []
    for event, spec in EVENTS.items():
        first, days = spec["first_day"], list(spec["days"])
        single = one_factor.days(first, days)
        for name, study in studies.items():
            parts = study.days(first, days)
            # the one-factor abnormal return is the sector part plus the two-factor abnormal return
            assert np.allclose(single["abnormal"], parts["sector"] + parts["abnormal"], atol=1e-12)
            parts["abnormal_one_factor"] = single["abnormal"]
            parts["z_one_factor"] = single["z"]
            parts.insert(0, "label", [spec["days"][d] for d in days])
            parts.insert(0, "date", parts.index.date)
            parts.insert(0, "sector_factor", name)
            parts.insert(0, "event", event)
            rows.append(parts)
    days_table = pd.concat(rows, ignore_index=True)
    writer.csv("event_study_days", days_table)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(days_table.drop(columns=["label"]).round(4).to_string(index=False))

    # 3. Windows ------------------------------------------------------------------------------------
    rows = []
    for event, label, start, end in windows:
        first = EVENTS[event]["first_day"]
        single = one_factor.window(first, start, end)
        for name, study in studies.items():
            total = study.window(first, start, end)
            row = {"event": event, "window": label, "sector_factor": name} | total.to_dict()
            row |= {"abnormal_one_factor": single["abnormal"], "z_one_factor": single["z"]}
            rows.append(row)
    window_table = pd.DataFrame(rows)
    writer.csv("event_study_windows", window_table)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(window_table.round(4).to_string(index=False))

    # The sector factors' members on the event days: SMH, each peer and the basket
    rows = []
    for event, spec in EVENTS.items():
        for day in spec["days"]:
            d = pd.Timestamp(day)
            row = {"event": event, "date": d.date(), SECTOR_ETF: sectors[SECTOR_ETF][d]}
            row |= {t: peers.loc[d, t] for t in SEMI_PEERS} | {"peer_basket": sectors[PEERS][d]}
            rows.append(row)
    members = pd.DataFrame(rows)
    writer.csv("event_study_sector_members", members)
    print(members.round(4).to_string(index=False))

    # 4. The seven on the two headline days (one-factor model on the same estimation windows) ------
    rows = []
    for day, event in RANKED_DAYS.items():
        first = EVENTS[event]["first_day"]
        for ticker in STOCKS:
            study = EventStudy(panel.history(ticker), spy)
            fit = study.fit(first)
            p = study.days(first, [day]).iloc[0]
            rows.append({"event": event, "date": pd.Timestamp(day).date(), "ticker": ticker, "return": p["return"],
                         "spy_return": p["market_return"], "beta_market": fit.beta_market, "alpha_daily": fit.alpha,
                         "market": p["market"], "abnormal": p["abnormal"], "z": p["z"],
                         "residual_sd_daily": fit.residual_sd, "estimation_start": fit.start.date(),
                         "estimation_end": fit.end.date()})
    megacaps = pd.DataFrame(rows)
    writer.csv("event_study_megacaps", megacaps)
    print(megacaps.round(4).to_string(index=False))

    # 5. Ranks against NVDA's own history of abnormal returns ------------------------------------------
    rolling = {SECTOR_ETF: studies[SECTOR_ETF].rolling()}
    history_start = rolling[SECTOR_ETF].index[0]  # the first day SMH allows: its 271st common return
    rolling[PEERS] = studies[PEERS].rolling(start=history_start)
    rows = []
    for day, event in RANKED_DAYS.items():
        d = pd.Timestamp(day)
        for name, roll in rolling.items():
            event_value = days_table.loc[(days_table["sector_factor"] == name) &
                                         (days_table["date"] == d.date()), "abnormal"].iloc[0]
            by_value = rank_in_history(roll["abnormal"], roll.loc[d, "abnormal"])
            by_z = rank_in_history(roll["z"], roll.loc[d, "z"])
            rows.append({
                "date": d.date(), "event": event, "sector_factor": name,
                "abnormal_rolling": roll.loc[d, "abnormal"], "z_rolling": roll.loc[d, "z"],
                "abnormal_event_study": event_value,
                "rank_abnormal_from_lowest": by_value["rank_from_lowest"], "rank_z_from_lowest": by_z["rank_from_lowest"],
                "history_days": by_value["observations"], "share_at_or_below_abnormal": by_value["share_at_or_below"],
                "share_at_or_below_z": by_z["share_at_or_below"],
                "history_start": roll.index[0].date(), "history_end": roll.index[-1].date(),
                "history_min_abnormal": roll["abnormal"].min(), "history_min_abnormal_date": roll["abnormal"].idxmin().date(),
                "history_min_z": roll["z"].min(), "history_min_z_date": roll["z"].idxmin().date(),
                "history_days_z_below_minus_4": int((roll["z"] < -4).sum()),
                "history_days_z_below_minus_4_if_normal": len(roll) * norm.cdf(-4.0),
            })
    ranks = pd.DataFrame(rows)
    writer.csv("event_study_ranks", ranks)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ranks.round(4).to_string(index=False))

    # 6. Chart --------------------------------------------------------------------------------------
    panels = {}
    for name in studies:
        frame = days_table[days_table["sector_factor"] == name].set_index(pd.to_datetime(days_table.loc[
            days_table["sector_factor"] == name, "date"]))
        panels[f"Sector factor: {name}"] = frame
    groups = [(event, list(spec["days"])) for event, spec in EVENTS.items()]
    fig = charts.stacked_parts(panels, groups, PART_LABELS,
                               "Nvidia's daily return split into market, sector, alpha and company-specific parts")
    writer.figure("event_study_parts", fig)
    store.close()


if __name__ == "__main__":
    main()
