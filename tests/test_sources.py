"""YahooClient against a fake yfinance module, and the Ken French parser and client against fixtures.

No network: yfinance is replaced by a stand-in, and the Ken French zip is built in memory.
"""

import io
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

from mega7 import KenFrenchClient, YahooClient
from mega7.sources import first_table, parse_french_csv


def yahoo_history(closes, start="2024-01-02", tz="America/New_York"):
    """A frame shaped like yfinance's Ticker.history(): tz-aware dates and OHLC columns."""
    index = pd.bdate_range(start, periods=len(closes), tz=tz, name="Date")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1}, index=index)


def fake_yfinance(monkeypatch, *responses, market_cap=None):
    """Install a fake yfinance whose Ticker.history() returns (or raises) each response in turn."""
    calls = []
    queue = list(responses)

    class Ticker:
        def __init__(self, ticker):
            self.ticker = ticker
            self.fast_info = {"marketCap": market_cap}

        def history(self, **kwargs):
            calls.append((self.ticker, kwargs))
            response = queue.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    return calls


def test_fetch_returns_long_rows_of_adjusted_closes(monkeypatch):
    calls = fake_yfinance(monkeypatch, yahoo_history([10.0, float("nan"), 12.0]))
    frame = YahooClient(pause=0).fetch("AAPL")
    assert list(frame.columns) == ["series", "date", "value"]
    assert list(frame["series"].unique()) == ["AAPL"]
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]  # missing close dropped
    assert list(frame["value"]) == [10.0, 12.0]
    assert frame["date"].dt.tz is None
    assert calls[0][1]["auto_adjust"] is True and calls[0][1]["period"] == "max"


def test_fetch_drops_todays_unfinished_bar(monkeypatch):
    # while the market is open, Yahoo labels the latest price as today's close
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    dates = pd.DatetimeIndex([today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today], name="Date")
    fake_yfinance(monkeypatch, pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=dates))
    assert list(YahooClient(pause=0).fetch("SPY")["value"]) == [1.0, 2.0]


def test_parse_history_with_an_explicit_today():
    history = yahoo_history([1.0, 2.0, 3.0, 4.0], start="2026-11-09")  # Mon 9 to Thu 12 Nov
    frame = YahooClient.parse_history("NVDA", history, today=pd.Timestamp("2026-11-11"))
    assert list(frame["date"].dt.day) == [9, 10]


def test_fetch_retries_empty_answers_and_errors(monkeypatch):
    calls = fake_yfinance(monkeypatch, pd.DataFrame(), RuntimeError("Too Many Requests"), yahoo_history([5.0, 6.0]))
    frame = YahooClient(attempts=3, pause=0).fetch("SPY")
    assert len(calls) == 3
    assert list(frame["value"]) == [5.0, 6.0]


def test_fetch_gives_a_clear_error_when_yahoo_stays_empty(monkeypatch):
    calls = fake_yfinance(monkeypatch, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    with pytest.raises(RuntimeError, match="no prices for 'NOPE' after 3 attempts"):
        YahooClient(attempts=3, pause=0).fetch("NOPE")
    assert len(calls) == 3


def test_market_cap_reads_fast_info(monkeypatch):
    fake_yfinance(monkeypatch, market_cap=3.5e12)
    assert YahooClient(pause=0).market_cap("MSFT") == 3.5e12


def test_client_rejects_zero_attempts():
    with pytest.raises(ValueError):
        YahooClient(attempts=0)


FRENCH_TEXT = """This file was created using the 202609 CRSP database.
Some description line.

,Mkt-RF,SMB,RF
20260929,   -0.26,    0.10,    0.01
20260930,    1.50,  -99.99,    0.01

 Annual Factors: January-December
,Mkt-RF,SMB,RF
  2025,   13.30,   -9.31,    4.25
"""

FRENCH_MONTHLY = """Monthly file

,Mom
202608,   1.20
202609,  -0.40

Annual Factors:
January-December
,Mom
  2025,  -2.27
"""


def test_parse_french_csv_daily_table_with_missing_values():
    tables = parse_french_csv(FRENCH_TEXT)
    assert [t.frequency for t in tables] == ["daily"]  # the annual table is skipped
    daily = first_table(tables, "daily")
    assert list(daily.columns) == ["Mkt-RF", "SMB", "RF"]
    assert list(daily.index.strftime("%Y-%m-%d")) == ["2026-09-29", "2026-09-30"]
    assert daily.loc["2026-09-30", "Mkt-RF"] == 1.5
    assert np.isnan(daily.loc["2026-09-30", "SMB"])  # -99.99 means missing


def test_parse_french_csv_monthly_table_uses_month_ends():
    monthly = first_table(parse_french_csv(FRENCH_MONTHLY), "monthly")
    assert list(monthly.index.strftime("%Y-%m-%d")) == ["2026-08-31", "2026-09-30"]
    assert list(monthly["Mom"]) == [1.2, -0.4]
    with pytest.raises(KeyError):
        first_table(parse_french_csv(FRENCH_MONTHLY), "daily")


def test_ken_french_client_unzips_and_keeps_the_raw_file(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Test_daily.csv", FRENCH_TEXT)

    class Response:
        content = buffer.getvalue()

        def raise_for_status(self):
            pass

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout):
            self.urls.append(url)
            return Response()

    session = Session()
    tables = KenFrenchClient(session=session, raw_dir=tmp_path).fetch("F-F_Test_daily")
    assert session.urls == ["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Test_daily_CSV.zip"]
    assert (tmp_path / "F-F_Test_daily_CSV.zip").exists()
    assert first_table(tables, "daily").shape == (2, 3)
