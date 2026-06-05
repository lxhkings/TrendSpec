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
        # First 3 rows must be None (insufficient data for WMA(3) + SMA(2))
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


class TestRumiStrategy:
    def _make_strategy(self, params=None):
        from trendspec.strategy.examples.rumi import RumiStrategy
        return RumiStrategy(params=params or {"fast_period": 2, "slow_period": 3, "signal_period": 2})

    def _run_next(self, closes, has_pos=False):
        """Build context, run init+next on last bar, return pending signals."""
        strategy = self._make_strategy()
        data = make_data(closes)
        ctx = StrategyContext(Market.US, strategy, data=data)
        strategy.init(ctx)
        last_date = data.sort("date")["date"].to_list()[-1]
        if has_pos:
            ctx.update_positions({"AAPL": 100.0}, 0.0)
        ctx.update_bar(last_date, "AAPL", "AAPL", data)
        strategy.next(ctx)
        return ctx.pending_signals()

    def test_buy_signal_when_rumi_positive_no_position(self):
        # Uptrend → RUMI > 0, no position → BUY
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        signals = self._run_next(closes, has_pos=False)
        assert len(signals) == 1
        assert signals[0].direction == "BUY"
        assert signals[0].instrument_id == "AAPL"

    def test_sell_signal_when_rumi_negative_with_position(self):
        # Downtrend → RUMI < 0, has position → SELL
        closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
        signals = self._run_next(closes, has_pos=True)
        assert len(signals) == 1
        assert signals[0].direction == "SELL"

    def test_no_signal_rumi_negative_no_position(self):
        # Downtrend, no position → no signal (not shorting)
        closes = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0]
        signals = self._run_next(closes, has_pos=False)
        assert len(signals) == 0

    def test_no_signal_rumi_positive_already_held(self):
        # Uptrend, already in position → no repeated BUY
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        signals = self._run_next(closes, has_pos=True)
        assert len(signals) == 0

    def test_no_signal_during_warmup(self):
        # Only 2 rows → RUMI = None → no signal regardless
        closes = [10.0, 11.0]
        signals = self._run_next(closes, has_pos=False)
        assert len(signals) == 0

    def test_strategy_registered(self):
        from trendspec.strategy.base import list_strategies
        assert "rumi" in list_strategies()
