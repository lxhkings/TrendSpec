"""Tests for StrategyContext weekly indicator API."""
from datetime import date

import polars as pl
import pytest


@pytest.fixture
def weekly_df():
    """Hand-crafted weekly DataFrame: AAPL with 5 consecutive weekly bars."""
    return pl.DataFrame({
        "instrument_id": ["AAPL"] * 5,
        "ticker": ["AAPL"] * 5,
        "date": [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19),
                  date(2024, 1, 26), date(2024, 2, 2)],
        "open":   [180.0, 187.0, 190.0, 192.0, 195.0],
        "high":   [188.0, 192.0, 194.0, 196.0, 200.0],
        "low":    [179.0, 185.0, 188.0, 191.0, 193.0],
        "close":  [187.0, 190.0, 193.0, 195.0, 199.0],
        "volume": [250_000_000] * 5,
        "adj_factor": [1.0] * 5,
    })


@pytest.fixture
def ctx_with_weekly(weekly_df):
    """StrategyContext with weekly_data injected."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import BaseStrategy
    from trendspec.strategy.context import StrategyContext

    class _Dummy(BaseStrategy):
        name = "dummy"
        def init(self, ctx): pass
        def next(self, ctx): pass

    strat = _Dummy()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=None,
                          weekly_data=weekly_df)
    return ctx


def test_weekly_indicator_value_returns_completed_week(ctx_with_weekly):
    """as_of_date=周三 (after most recent Friday) should return last Friday's value."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    # 1/22 是周一; 上一已完成周线 bar = 1/19 (周五)
    val = ctx_with_weekly.weekly_indicator_value(
        "EMA", instrument_id="AAPL", as_of_date=date(2024, 1, 22), period=2
    )
    assert val is not None
    # EMA(period=2, smoothing=2) on [187, 190, 193] at 1/19 row:
    #   sf = 2 / (1+2) = 0.6667; iterative EMA result, just sanity-check non-None.


def test_weekly_indicator_value_no_lookahead(ctx_with_weekly):
    """as_of_date 当周尚未结束时不应读未来 bar."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    # 1/8 周一: 已完成的最近周是 1/5, 不能偷看 1/12
    val_mon = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2024, 1, 8), period=2)
    val_fri = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2024, 1, 5), period=2)
    assert val_mon == val_fri   # 都指向 1/5 那条 bar


def test_weekly_indicator_value_before_any_data_returns_none(ctx_with_weekly):
    """as_of_date 早于第一周 → None."""
    ctx_with_weekly.precompute_weekly_indicator("EMA", period=2)
    val = ctx_with_weekly.weekly_indicator_value(
        "EMA", "AAPL", date(2023, 12, 1), period=2)
    assert val is None


def test_weekly_indicator_value_missing_weekly_data():
    """ctx without weekly_data → weekly_indicator_value returns None."""
    from trendspec.data.markets import Market
    from trendspec.strategy.base import BaseStrategy
    from trendspec.strategy.context import StrategyContext

    class _Dummy(BaseStrategy):
        name = "dummy"
        def init(self, ctx): pass
        def next(self, ctx): pass

    ctx = StrategyContext(market=Market.US, strategy=_Dummy(), data=None,
                          weekly_data=None)
    val = ctx.weekly_indicator_value("EMA", "AAPL", date(2024, 1, 22), period=20)
    assert val is None