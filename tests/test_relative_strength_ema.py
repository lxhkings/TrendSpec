# tests/test_relative_strength_ema.py
"""Tests for rs_ema_cross relative-strength EMA cross strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.ingest.writer import write_parquet
from trendspec.strategy.base import get_strategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.examples.relative_strength_ema import RelativeStrengthEMACross


def test_strategy_registered() -> None:
    """rs_ema_cross is discoverable via the registry."""
    cls = get_strategy("rs_ema_cross")
    assert cls is RelativeStrengthEMACross


def test_default_params() -> None:
    """Defaults present even when constructed with no params."""
    strat = RelativeStrengthEMACross()
    assert strat.get_param("benchmark_id") == "QQQ"
    assert strat.get_param("ema_short") == 60
    assert strat.get_param("ema_long") == 120


def _make_stock_and_qqq(temp_root, iid="AAPL_US", n=200):
    """写 QQQ 到临时 data_lake，返回 (stock_df, dates)。QQQ 收盘恒为 100。"""
    start = date(2023, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n)]
    # 股票收盘：前 100 天下行、后 100 天上行 → 比值后段 EMA60>EMA120
    closes = [100.0 - i * 0.2 for i in range(100)] + [80.0 + (i - 100) * 0.5 for i in range(100, n)]
    ticker = iid.split("_")[0]  # AAPL
    stock_df = pl.DataFrame(
        {
            "instrument_id": [iid] * n,
            "ticker": [ticker] * n,
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
            "adj_factor": [1.0] * n,
        }
    )
    # QQQ 需要 OHLCV 列（compute_indicator 检查 REQUIRED_COLUMNS）
    qqq_close = 100.0
    qqq_df = pl.DataFrame(
        {
            "instrument_id": ["QQQ"] * n,
            "ticker": ["QQQ"] * n,
            "date": dates,
            "open": [qqq_close] * n,
            "high": [qqq_close * 1.01] * n,
            "low": [qqq_close * 0.99] * n,
            "close": [qqq_close] * n,
            "volume": [10_000_000] * n,
            "adj_factor": [1.0] * n,
        }
    )
    write_parquet(qqq_df, Market.US, "indices", temp_root)
    return stock_df, dates


def test_init_populates_rs_dicts_and_skips_warmup(temp_root) -> None:
    """init() 填充比值 EMA 快查 dict，且前 ema_long 个 bar 被剔除。"""
    stock_df, dates = _make_stock_and_qqq(temp_root, n=200)
    strat = RelativeStrengthEMACross()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=stock_df, root=temp_root)
    strat.init(ctx)

    # 预热：第 120 个 bar 之前（索引 0..118）不应入字典
    assert ("AAPL_US", dates[50]) not in strat._rs_short
    # 预热满足后（>=120 个 bar）应入字典
    assert ("AAPL_US", dates[150]) in strat._rs_short
    assert ("AAPL_US", dates[150]) in strat._rs_long
    # 上行段末尾应为金叉状态
    assert strat._rs_short[("AAPL_US", dates[199])] > strat._rs_long[("AAPL_US", dates[199])]


def test_init_raises_when_benchmark_missing(temp_root) -> None:
    """无 indices 数据时 init() 抛 RuntimeError 提示先摄入。"""
    start = date(2023, 1, 2)
    stock_df = pl.DataFrame(
        {
            "instrument_id": ["AAPL_US"] * 3,
            "ticker": ["AAPL"] * 3,
            "date": [start + timedelta(days=i) for i in range(3)],
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [1, 1, 1],
            "adj_factor": [1.0, 1.0, 1.0],
        }
    )
    strat = RelativeStrengthEMACross()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=stock_df, root=temp_root)
    with pytest.raises(RuntimeError, match="ingest indices"):
        strat.init(ctx)