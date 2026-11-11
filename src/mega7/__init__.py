"""Mega 7 decomposition: risk, correlation, factor and index analysis of seven large US stocks.

Data flow:

    YahooClient, KenFrenchClient -> PriceStore (DuckDB, SQL) -> ReturnPanel -> analysis functions
                                                                             -> ResultsWriter (summary CSV/PNG)
"""

from .panel import BENCHMARKS, FACTORS, NAMES, STOCKS, TICKERS, ReturnPanel
from .portfolio import EqualWeightPortfolio
from .results import ResultsWriter
from .sources import KenFrenchClient, YahooClient
from .store import PriceStore

__all__ = [
    "BENCHMARKS",
    "FACTORS",
    "NAMES",
    "STOCKS",
    "TICKERS",
    "EqualWeightPortfolio",
    "KenFrenchClient",
    "PriceStore",
    "ResultsWriter",
    "ReturnPanel",
    "YahooClient",
]
