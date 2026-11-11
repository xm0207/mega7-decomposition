"""The aligned return panel that every analysis module reads from.

The seven stocks list on different dates (Apple in 1980, Meta in 2012), the factor files end a month
or two before the prices, and weekly and monthly returns need a rule for the unfinished last period.
`ReturnPanel` settles all of that in one place, so the analysis modules receive clean, aligned frames.
"""

from __future__ import annotations

import pandas as pd

STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
BENCHMARKS = ["SPY", "QQQ", "RSP"]
TICKERS = STOCKS + BENCHMARKS
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (class A)",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "SPY": "S&P 500 (SPY)",
    "QQQ": "Nasdaq-100 (QQQ)",
    "RSP": "Equal-weight S&P 500 (RSP)",
}

TRADING_DAYS = 252


def drop_unfinished_last_period(returns: pd.DataFrame, frequency: str, last_close: pd.Timestamp) -> pd.DataFrame:
    """Drop the final weekly or monthly return if the data end before that period is over.

    A week is treated as finished if the last close is a Friday; a month if the last close is on or
    after the month's last business day. (A holiday on the last business day would make this rule
    drop a finished period; that only matters for the final period and costs one observation.)
    """
    if returns.empty or frequency == "daily":
        return returns
    last = returns.index[-1]
    if last != last_close:
        return returns
    if frequency == "weekly":
        finished = last_close.day_of_week == 4
    elif frequency == "monthly":
        finished = last_close >= last_close + pd.offsets.BMonthEnd(0)
    else:
        raise ValueError(f"unknown frequency {frequency!r}")
    return returns if finished else returns.iloc[:-1]


class ReturnPanel:
    """Daily, weekly and monthly returns of the ten tickers, plus factor returns and the risk-free rate.

    Why a class: the panel is built once from the database and then asked for many slightly
    different views (one frequency, the common sample or each stock's own history, excess returns).
    Keeping the alignment rules in one object means every analysis uses the same sample. Public
    interface:

    - `returns(frequency, tickers, common)` simple returns; `common=True` keeps only dates where all
      seven stocks have a return (from META's first full trading day).
    - `rf(frequency, index)` the risk-free rate per period, aligned to `index`.
    - `excess(frequency, tickers, common)` returns minus the risk-free rate.
    - `factors(frequency)` the six Ken French factors, in decimals.
    - `start`, `end`, `factor_end` the sample boundaries, for labels in the output.

    Factor and risk-free returns are stored in percent (as Ken French publishes them) and converted
    to decimals here.
    """

    def __init__(
        self,
        daily: pd.DataFrame,
        weekly: pd.DataFrame,
        monthly: pd.DataFrame,
        french_daily: pd.DataFrame,
        french_monthly: pd.DataFrame,
    ):
        self._returns = {"daily": daily.sort_index(), "weekly": weekly.sort_index(), "monthly": monthly.sort_index()}
        self._french = {"daily": french_daily.sort_index() / 100.0, "monthly": french_monthly.sort_index() / 100.0}
        stocks = [s for s in STOCKS if s in daily.columns]
        common = daily[stocks].dropna()
        if common.empty:
            raise ValueError("No date on which all the stocks have a return")
        self.start = common.index[0]
        self.end = daily.index[-1]

    @classmethod
    def from_store(cls, store) -> "ReturnPanel":
        """Build the panel from a `PriceStore`: returns via SQL, then drop unfinished last periods."""
        daily = store.returns("daily")
        last_close = daily.index[-1]
        weekly = drop_unfinished_last_period(store.returns("weekly"), "weekly", last_close)
        monthly = drop_unfinished_last_period(store.returns("monthly"), "monthly", last_close)
        monthly.index = monthly.index + pd.offsets.MonthEnd(0)  # month-end labels, to match Ken French
        return cls(daily, weekly, monthly, store.wide("french_daily"), store.wide("french_monthly"))

    # --------------------------------------------------------------------------------- returns
    def returns(self, frequency: str = "daily", tickers: list[str] | None = None, common: bool = True) -> pd.DataFrame:
        """Simple returns. With `common=True`, rows from the common sample start onwards, with no gaps."""
        full = self._returns[frequency]
        tickers = tickers or [t for t in TICKERS if t in full.columns]
        frame = full[tickers]
        if common:
            # the common sample is set by all seven stocks, whichever tickers were asked for
            stocks = [s for s in STOCKS if s in full.columns]
            first = full[stocks].dropna().index[0]
            frame = frame.loc[first:].dropna()
        return frame

    def history(self, ticker: str, frequency: str = "daily") -> pd.Series:
        """One ticker's returns over its own full history (from its first return)."""
        return self._returns[frequency][ticker].dropna()

    # ---------------------------------------------------------------------------------- factors
    @property
    def factor_end(self) -> pd.Timestamp:
        return self._french["daily"].index[-1]

    def factors(self, frequency: str = "monthly") -> pd.DataFrame:
        """The five Fama-French factors and momentum, in decimals."""
        frame = self._french[frequency]
        return frame[[f for f in FACTORS if f in frame.columns]].dropna()

    def rf(self, frequency: str, index: pd.DatetimeIndex) -> pd.Series:
        """The risk-free return per period, on `index`.

        Ken French's files end before the prices do. Periods after the last published rate reuse it
        (carried forward); `rf_carried_forward(index)` says how many periods that affects.
        """
        source = "monthly" if frequency == "monthly" else "daily"
        rf = self._french[source]["RF"].dropna()
        if frequency == "weekly":
            raise ValueError("weekly risk-free rates are not needed in this project")
        aligned = rf.reindex(rf.index.union(index)).ffill().reindex(index)
        return aligned.rename("RF")

    def rf_carried_forward(self, frequency: str, index: pd.DatetimeIndex) -> int:
        source = "monthly" if frequency == "monthly" else "daily"
        return int((index > self._french[source]["RF"].dropna().index[-1]).sum())

    def excess(self, frequency: str = "daily", tickers: list[str] | None = None, common: bool = True) -> pd.DataFrame:
        """Returns minus the risk-free rate of the same period."""
        frame = self.returns(frequency, tickers, common)
        return frame.sub(self.rf(frequency, frame.index), axis=0)
