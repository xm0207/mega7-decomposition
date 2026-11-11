"""Download prices from Yahoo Finance and factor returns from the Ken French data library.

Two classes, each with one job:

- `YahooClient` downloads daily closes, adjusted for splits and dividends, with retries, and drops
  today's unfinished bar. It can also report a ticker's *current* market capitalisation.
- `KenFrenchClient` downloads a zipped CSV from the Ken French data library and returns its daily
  or monthly tables, parsed by `parse_french_csv`.

Yahoo Finance data is for personal use only. Keep the downloaded prices on your own machine (the
`data/` folder is not committed) and publish only statistics derived from them.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

KEN_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FRENCH_MISSING = (-99.99, -999.0)


class YahooClient:
    """Fetches the daily adjusted closes of one ticker from Yahoo Finance.

    Why a class: the retry settings (how many attempts, how long to wait) belong together and are
    set once, then every download uses them. Public interface:

    - `fetch(ticker)` returns one row per finished trading day: series, date, value (adjusted close).
    - `market_cap(ticker)` returns today's market capitalisation in dollars, or None if Yahoo has none.

    Yahoo sometimes answers with an empty table when it is rate limiting, so `fetch` tries `attempts`
    times, waiting a little longer each time, before giving up. While a market is open Yahoo reports
    the latest price as today's close, so `fetch` keeps only days before today, in the exchange's
    time zone: every value it returns is a finished close.
    """

    def __init__(self, attempts: int = 4, pause: float = 2.0):
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.attempts = attempts
        self.pause = pause

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Return one row per finished trading day: series (the ticker), date, value (adjusted close)."""
        yf = self._yfinance()
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                history = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True, actions=False)
                frame = self.parse_history(ticker, history)
                if not frame.empty:
                    return frame
            except Exception as error:  # yfinance raises its own errors, e.g. when rate limited
                last_error = error
            if attempt < self.attempts:
                time.sleep(self.pause * attempt)
        raise RuntimeError(
            f"Yahoo Finance returned no prices for {ticker!r} after {self.attempts} attempts. "
            "Check the ticker; if it is right, Yahoo is probably rate limiting, so wait a few minutes and retry."
        ) from last_error

    def market_cap(self, ticker: str) -> float | None:
        """Today's market capitalisation in dollars, as Yahoo reports it (a current value, not history)."""
        yf = self._yfinance()
        for attempt in range(1, self.attempts + 1):
            try:
                value = yf.Ticker(ticker).fast_info["marketCap"]
                if value:
                    return float(value)
            except Exception:
                pass
            if attempt < self.attempts:
                time.sleep(self.pause * attempt)
        return None

    @staticmethod
    def parse_history(ticker: str, history: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
        """Turn yfinance's `Ticker.history()` frame into long rows, dropping missing and unfinished closes.

        `today` is only for tests; by default it is the current date in the exchange's time zone.
        """
        if history is None or history.empty or "Close" not in history:
            return pd.DataFrame(columns=["series", "date", "value"])
        close = history["Close"].dropna()
        index = pd.DatetimeIndex(close.index)
        if today is None:
            today = pd.Timestamp.now(tz=index.tz).normalize()
        elif index.tz is not None and today.tz is None:
            today = today.tz_localize(index.tz)
        finished = index < today
        dates = (index.tz_localize(None) if index.tz is not None else index).normalize()
        frame = pd.DataFrame(
            {"series": ticker, "date": dates[finished], "value": close.to_numpy(dtype=float)[finished]}
        )
        return frame.drop_duplicates("date", keep="last").reset_index(drop=True)

    @staticmethod
    def _yfinance():
        try:
            import yfinance as yf  # imported here so the rest of the package works without it
        except ImportError as error:
            raise ImportError("YahooClient needs yfinance: pip install yfinance") from error
        return yf


@dataclass(frozen=True)
class FrenchTable:
    """One table from a Ken French CSV file: its title, its frequency ('daily' or 'monthly') and its data."""

    title: str
    frequency: str
    data: pd.DataFrame


def parse_french_csv(text: str) -> list[FrenchTable]:
    """Split a Ken French CSV file into its daily and monthly tables.

    The files hold one or more tables one after another. Each starts with some text, then a header
    row that begins with a comma, then data rows keyed by YYYYMMDD
    (daily), YYYYMM (monthly) or YYYY (annual). Daily and monthly tables are returned; annual ones are
    skipped. Values stay in the file's units (percent). Missing values (-99.99 or -999) become NaN.
    Monthly dates are month ends.
    """
    lines = text.replace("\r", "").split("\n")
    tables: list[FrenchTable] = []
    recent_text: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(","):
            columns = [c.strip() for c in line.split(",")[1:]]
            rows = []
            i += 1
            while i < len(lines) and lines[i].strip()[:1].isdigit():
                rows.append([p.strip() for p in lines[i].split(",")])
                i += 1
            key_length = len(rows[0][0]) if rows else 0
            if key_length in (6, 8):
                frame = pd.DataFrame([r[1 : len(columns) + 1] for r in rows], columns=columns).astype(float)
                keys = [r[0] for r in rows]
                if key_length == 8:
                    frame.index = pd.to_datetime(keys, format="%Y%m%d")
                    frequency = "daily"
                else:
                    frame.index = pd.to_datetime(keys, format="%Y%m") + pd.offsets.MonthEnd(0)
                    frequency = "monthly"
                frame.index.name = "date"
                frame = frame.mask(frame.isin(FRENCH_MISSING))
                tables.append(FrenchTable(" ".join(recent_text), frequency, frame))
            recent_text = []
            continue
        if line.strip():
            recent_text.append(line.strip())
        i += 1
    return tables


class KenFrenchClient:
    """Downloads zipped CSV files from the Ken French data library (public, no key needed).

    Why a class: it holds the HTTP session, the timeout and the folder where raw zips are kept, so
    every download is configured the same way. Public interface: `fetch(name)` returns the daily and
    monthly tables of one file, and saves the zip in `raw_dir` if one was given.
    """

    def __init__(self, session: requests.Session | None = None, timeout: float = 60.0, raw_dir: str | Path | None = None):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None

    def fetch(self, name: str) -> list[FrenchTable]:
        response = self.session.get(KEN_FRENCH_URL.format(name=name), timeout=self.timeout)
        response.raise_for_status()
        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_dir / f"{name}_CSV.zip").write_bytes(response.content)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected one CSV in {name}, found {csv_names}")
            text = archive.read(csv_names[0]).decode("latin-1")
        return parse_french_csv(text)


def first_table(tables: list[FrenchTable], frequency: str) -> pd.DataFrame:
    """The first table of the given frequency ('daily' or 'monthly'): the returns, not the annual summary."""
    for table in tables:
        if table.frequency == frequency:
            return table.data
    raise KeyError(f"No {frequency} table; found {[(t.title[:40], t.frequency) for t in tables]}")
