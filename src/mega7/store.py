"""A DuckDB database that keeps the daily panel and answers the SQL-shaped questions asked of it.

Prices and factors are stored in one long table, `observations(dataset, series, date, value)`:

- dataset `price`: adjusted closes from Yahoo Finance (dollars)
- dataset `french_daily` and `french_monthly`: Ken French factor returns (percent, as published)

A second table, `market_caps(series, as_of, market_cap)`, keeps the *current* market capitalisations
reported by Yahoo on the day of the download. They are a snapshot, not a history.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# Returns are computed from closes in SQL. The first row of each series has no previous close and
# is dropped. Weekly and monthly returns use the last close of each ISO week or calendar month.
_PERIOD_CLOSE = {
    "daily": "SELECT series, date, value FROM observations WHERE dataset = 'price'",
    "weekly": """
        SELECT series, max(date) AS date, arg_max(value, date) AS value
        FROM observations WHERE dataset = 'price'
        GROUP BY series, date_trunc('week', date)
    """,
    "monthly": """
        SELECT series, max(date) AS date, arg_max(value, date) AS value
        FROM observations WHERE dataset = 'price'
        GROUP BY series, last_day(date)
    """,
}


class PriceStore:
    """The project's local database (DuckDB), with every SQL query in one place.

    Why a class: it owns the database connection, so scripts open the store once and ask it
    questions, and no other module writes SQL against the raw tables. Public interface:

    - `upsert(dataset, frame)` replaces a dataset with new rows (series, date, value).
    - `save_market_caps(frame)` / `market_caps()` keep the current market-cap snapshot.
    - `coverage()` reports the first and last date and the number of rows of each series.
    - `wide(dataset)` returns a dates x series table.
    - `returns(frequency)` computes daily, weekly or monthly simple returns from closes.
    - `calendar_year_returns()` compounds daily returns within each calendar year.
    - `extreme_days(n)` lists each series' n best and n worst days.
    - `joint_move_counts(tickers, threshold)` counts, for each day, how many tickers moved beyond a threshold.
    """

    def __init__(self, path: str | Path = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                dataset VARCHAR NOT NULL,
                series  VARCHAR NOT NULL,
                date    DATE    NOT NULL,
                value   DOUBLE  NOT NULL,
                PRIMARY KEY (dataset, series, date)
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS market_caps (
                series     VARCHAR NOT NULL PRIMARY KEY,
                as_of      DATE    NOT NULL,
                market_cap DOUBLE  NOT NULL
            )
            """
        )

    def close(self) -> None:
        self.con.close()

    # ----------------------------------------------------------------------------------- writing
    def upsert(self, dataset: str, frame: pd.DataFrame, replace_series: bool = True) -> int:
        """Write rows (columns series, date, value) into `dataset`; returns the number of rows written.

        With `replace_series=True` the rows already stored for the incoming series are deleted first,
        so re-downloading a ticker replaces its whole history (Yahoo revises adjusted closes after
        every dividend).
        """
        incoming = frame[["series", "date", "value"]].dropna(subset=["value"]).assign(dataset=dataset)
        incoming["date"] = pd.to_datetime(incoming["date"])
        self.con.register("incoming", incoming)
        try:
            if replace_series:
                self.con.execute(
                    "DELETE FROM observations WHERE dataset = ? AND series IN (SELECT DISTINCT series FROM incoming)",
                    [dataset],
                )
            self.con.execute(
                "INSERT OR REPLACE INTO observations SELECT dataset, series, CAST(date AS DATE), value FROM incoming"
            )
        finally:
            self.con.unregister("incoming")
        return len(incoming)

    def save_market_caps(self, frame: pd.DataFrame) -> None:
        """Replace the snapshot of current market caps (columns series, as_of, market_cap)."""
        self.con.register("caps", frame[["series", "as_of", "market_cap"]])
        try:
            self.con.execute("DELETE FROM market_caps")
            self.con.execute("INSERT INTO market_caps SELECT series, CAST(as_of AS DATE), market_cap FROM caps")
        finally:
            self.con.unregister("caps")

    # ----------------------------------------------------------------------------------- reading
    def market_caps(self) -> pd.DataFrame:
        return self.con.execute("SELECT series, as_of, market_cap FROM market_caps ORDER BY market_cap DESC").df()

    def coverage(self) -> pd.DataFrame:
        """First date, last date and number of rows of each series of each dataset."""
        return self.con.execute(
            """
            SELECT dataset, series,
                   min(date) AS first_date,
                   max(date) AS last_date,
                   count(*)  AS observations
            FROM observations
            GROUP BY dataset, series
            ORDER BY dataset, min(date), series
            """
        ).df()

    def wide(self, dataset: str, series: list[str] | None = None) -> pd.DataFrame:
        """One column per series of `dataset`, one row per date."""
        long = self.con.execute(
            "SELECT series, date, value FROM observations WHERE dataset = ? ORDER BY date", [dataset]
        ).df()
        if long.empty:
            raise KeyError(f"No rows stored for dataset {dataset!r}")
        table = long.pivot(index="date", columns="series", values="value")
        table.index = pd.DatetimeIndex(table.index, name="date")
        table.columns.name = None
        if series is not None:
            missing = [s for s in series if s not in table.columns]
            if missing:
                raise KeyError(f"Series not in {dataset!r}: {missing}")
            table = table[series]
        return table

    def returns(self, frequency: str = "daily") -> pd.DataFrame:
        """Simple returns (one column per ticker) from the stored closes, computed in SQL.

        `frequency` is 'daily', 'weekly' (last close of each ISO week, Monday to Sunday) or 'monthly'
        (last close of each calendar month). Each return is dated at the period's last trading day.
        """
        if frequency not in _PERIOD_CLOSE:
            raise ValueError(f"frequency must be one of {sorted(_PERIOD_CLOSE)}")
        long = self.con.execute(
            f"""
            WITH closes AS ({_PERIOD_CLOSE[frequency]}),
            returns AS (
                SELECT series, date,
                       value / lag(value) OVER (PARTITION BY series ORDER BY date) - 1 AS ret
                FROM closes
            )
            SELECT series, date, ret FROM returns WHERE ret IS NOT NULL ORDER BY date
            """
        ).df()
        table = long.pivot(index="date", columns="series", values="ret")
        table.index = pd.DatetimeIndex(table.index, name="date")
        table.columns.name = None
        return table

    def calendar_year_returns(self, start: str | None = None) -> pd.DataFrame:
        """Each series' compound return in each calendar year (exp of the sum of log returns).

        Only daily returns on or after `start` are used, so the first year can be a partial year.
        The `days` column says how many daily returns each figure is built from.
        """
        return self.con.execute(
            """
            WITH daily AS (
                SELECT series, date,
                       value / lag(value) OVER (PARTITION BY series ORDER BY date) - 1 AS ret
                FROM observations WHERE dataset = 'price'
            )
            SELECT series, year(date) AS year,
                   exp(sum(ln(1 + ret))) - 1 AS total_return,
                   count(*) AS days
            FROM daily
            WHERE ret IS NOT NULL AND date >= coalesce(CAST(? AS DATE), DATE '1900-01-01')
            GROUP BY series, year(date)
            ORDER BY series, year
            """,
            [start],
        ).df()

    def extreme_days(self, n: int = 5, start: str | None = None) -> pd.DataFrame:
        """The n best and n worst daily returns of each series, with dates and rank (1 = most extreme)."""
        return self.con.execute(
            """
            WITH daily AS (
                SELECT series, date,
                       value / lag(value) OVER (PARTITION BY series ORDER BY date) - 1 AS ret
                FROM observations WHERE dataset = 'price'
            ),
            ranked AS (
                SELECT series, date, ret,
                       row_number() OVER (PARTITION BY series ORDER BY ret DESC) AS best_rank,
                       row_number() OVER (PARTITION BY series ORDER BY ret ASC)  AS worst_rank
                FROM daily
                WHERE ret IS NOT NULL AND date >= coalesce(CAST(? AS DATE), DATE '1900-01-01')
            )
            SELECT series, 'best' AS side, best_rank AS rank, date, ret FROM ranked WHERE best_rank <= ?
            UNION ALL
            SELECT series, 'worst' AS side, worst_rank AS rank, date, ret FROM ranked WHERE worst_rank <= ?
            ORDER BY series, side, rank
            """,
            [start, n, n],
        ).df()

    def joint_move_counts(self, tickers: list[str], threshold: float, start: str | None = None) -> pd.DataFrame:
        """For each trading day on or after `start` on which every ticker has a return: how many of
        `tickers` returned below `threshold` (for a negative threshold) or above it (for a positive one).

        Returns columns date, n_moves. Only days where all tickers trade are counted.
        """
        if threshold == 0:
            raise ValueError("threshold must be non-zero: negative for falls, positive for rises")
        comparison = "ret < ?" if threshold < 0 else "ret > ?"
        placeholders = ", ".join("?" for _ in tickers)
        return self.con.execute(
            f"""
            WITH daily AS (
                SELECT series, date,
                       value / lag(value) OVER (PARTITION BY series ORDER BY date) - 1 AS ret
                FROM observations WHERE dataset = 'price' AND series IN ({placeholders})
            )
            SELECT date, count_if({comparison}) AS n_moves
            FROM daily
            WHERE ret IS NOT NULL AND date >= coalesce(CAST(? AS DATE), DATE '1900-01-01')
            GROUP BY date
            HAVING count(*) = ?
            ORDER BY date
            """,
            [*tickers, threshold, start, len(tickers)],
        ).df()
