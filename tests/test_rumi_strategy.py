"""Tests for RUMI indicator and RumiStrategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.strategy.context import StrategyContext


def make_data(closes: list[float], iid: str = "AAPL", ticker: str = "AAPL") -> pl.DataFrame:
    start = date(2020, 1, 2)
    n = len(closes)
    dates = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({
        "instrument_id": [iid] * n,
        "date": dates,
        "ticker": [ticker] * n,
        "open": [c - 0.5 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
        "adj_factor": [1.0] * n,
    })


class TestRumiIndicator:
    def test_rumi_indicator_basic(self):
        from trendspec.strategy.indicators import rumi as rumi_indicator

        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        data = make_data(closes)
        result = rumi_indicator(data, fast_period=2, slow_period=3, signal_period=2)

        assert "RUMI_DIFF" in result.columns
        assert "RUMI" in result.columns

        # Last row: uptrend → RUMI > 0
        last = result.sort("date").filter(pl.col("instrument_id") == "AAPL")["RUMI"].to_list()[-1]
        assert last is not None
        assert last > 0

    def test_rumi_indicator_warmup(self):
        from trendspec.strategy.indicators import rumi as rumi_indicator

        closes = list(range(10, 20))  # 10 rows, uptrend
        data = make_data([float(c) for c in closes])
        result = rumi_indicator(data, fast_period=2, slow_period=3, signal_period=2)

        rumi_col = result.sort("date").filter(pl.col("instrument_id") == "AAPL")["RUMI"].to_list()
        # First slow_period + signal_period - 2 = 3 + 2 - 2 = 3 rows must be None
        assert rumi_col[0] is None
        assert rumi_col[1] is None
        assert rumi_col[2] is None
        assert rumi_col[3] is not None

    def test_rumi_downtrend_negative(self):
        from trendspec.strategy.indicators import rumi as rumi_indicator

        closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
        data = make_data(closes)
        result = rumi_indicator(data, fast_period=2, slow_period=3, signal_period=2)

        last = result.sort("date").filter(pl.col("instrument_id") == "AAPL")["RUMI"].to_list()[-1]
        assert last is not None
        assert last < 0

    def test_rumi_registered(self):
        from trendspec.strategy.indicators import list_indicators
        assert "RUMI" in list_indicators()
