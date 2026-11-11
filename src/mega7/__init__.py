"""Mega 7 decomposition: risk, correlation, factor and index analysis of seven large US stocks.

Data flow:

    YahooClient, KenFrenchClient -> PriceStore (DuckDB, SQL) -> ReturnPanel -> analysis functions
                                                                             -> EventStudy (market model)
                                                                             -> ResultsWriter (summary CSV/PNG)
"""

from .event_study import EventStudy, MarketModelFit
from .panel import BENCHMARKS, FACTORS, NAMES, SECTOR_ETF, SEMI_PEERS, SEMIS, STOCKS, TICKERS, ReturnPanel
from .portfolio import EqualWeightPortfolio
from .results import ResultsWriter
from .sources import KenFrenchClient, YahooClient
from .store import PriceStore

__all__ = [
    "BENCHMARKS",
    "FACTORS",
    "NAMES",
    "SECTOR_ETF",
    "SEMIS",
    "SEMI_PEERS",
    "STOCKS",
    "TICKERS",
    "EqualWeightPortfolio",
    "EventStudy",
    "KenFrenchClient",
    "MarketModelFit",
    "PriceStore",
    "ResultsWriter",
    "ReturnPanel",
    "YahooClient",
]
