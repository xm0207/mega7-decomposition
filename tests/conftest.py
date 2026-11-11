"""Small synthetic fixtures, so every test runs offline in well under a second."""

import numpy as np
import pandas as pd
import pytest

from mega7 import BENCHMARKS, STOCKS, PriceStore, ReturnPanel


def long_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide prices (dates x tickers) as the long rows a store expects."""
    frame = prices.copy()
    frame.index.name = "date"
    return frame.reset_index().melt(id_vars="date", var_name="series", value_name="value").dropna()


def synthetic_prices(days: int = 600, seed: int = 0, late_listing: int = 20) -> pd.DataFrame:
    """Ten tickers driven by one market factor. META lists `late_listing` days after the others."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=days)
    market = rng.normal(0.0004, 0.01, days)
    rets = {}
    for i, t in enumerate(STOCKS):
        rets[t] = (0.8 + 0.15 * i) * market + rng.normal(0.0003, 0.012, days)
    rets["SPY"] = market
    rets["QQQ"] = 1.1 * market + rng.normal(0, 0.003, days)
    rets["RSP"] = 0.95 * market + rng.normal(0, 0.003, days)
    prices = 100 * (1 + pd.DataFrame(rets, index=index)).cumprod()
    prices.loc[prices.index[:late_listing], "META"] = np.nan
    return prices


def synthetic_store(prices: pd.DataFrame | None = None) -> PriceStore:
    prices = synthetic_prices() if prices is None else prices
    store = PriceStore()
    store.upsert("price", long_prices(prices))
    days = prices.index
    rng = np.random.default_rng(1)
    french_d = pd.DataFrame(rng.normal(0, 0.5, (len(days), 6)), index=days,
                            columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    french_d["RF"] = 0.01
    months = pd.date_range(days[0], days[-1], freq="ME")
    french_m = pd.DataFrame(rng.normal(0, 3, (len(months), 6)), index=months, columns=french_d.columns[:6])
    french_m["RF"] = 0.2
    store.upsert("french_daily", long_prices(french_d.iloc[:-10]))  # factor files end before prices
    store.upsert("french_monthly", long_prices(french_m))
    return store


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic_prices()


@pytest.fixture
def store(prices) -> PriceStore:
    return synthetic_store(prices)


@pytest.fixture
def panel(store) -> ReturnPanel:
    return ReturnPanel.from_store(store)


__all__ = ["BENCHMARKS", "STOCKS", "long_prices", "synthetic_prices", "synthetic_store"]
