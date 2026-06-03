"""Tests for rs_ema_cross v2 relative-strength Top-N rotation strategy."""

from datetime import date, timedelta

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.ingest.writer import write_parquet
from trendspec.strategy.base import get_strategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.examples.relative_strength_ema import RelativeStrengthEMACross


def test_strategy_registered() -> None:
    assert get_strategy("rs_ema_cross") is RelativeStrengthEMACross


def test_default_params() -> None:
    strat = RelativeStrengthEMACross()
    assert strat.get_param("benchmark_id") == "QQQ"
    assert strat.get_param("ema_short") == 60
    assert strat.get_param("ema_long") == 120
    assert strat.get_param("top_n") == 20
    assert strat.get_param("rebalance_weekday") == 0
    assert strat.get_param("min_adv_us") == 1e8
    assert strat.get_param("min_adv_cn") == 0.0


def _make_stock_and_qqq(temp_root, iid="AAPL_US", n=200):
    """写 QQQ 到临时 data_lake，返回 stock_df（含 volume）。QQQ 收盘恒 100。"""
    start = date(2023, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n)]
    closes = [100.0 - i * 0.2 for i in range(100)] + [80.0 + (i - 100) * 0.5 for i in range(100, n)]
    ticker = iid.split("_")[0]
    stock_df = pl.DataFrame(
        {
            "instrument_id": [iid] * n,
            "ticker": [ticker] * n,
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [2_000_000] * n,
            "adj_factor": [1.0] * n,
        }
    )
    qqq_df = pl.DataFrame(
        {
            "instrument_id": ["QQQ"] * n,
            "ticker": ["QQQ"] * n,
            "date": dates,
            "close": [100.0] * n,
        }
    )
    write_parquet(qqq_df, Market.US, "indices", temp_root)
    return stock_df, dates


def test_init_populates_rs_dicts_adv_and_warmup(temp_root) -> None:
    stock_df, dates = _make_stock_and_qqq(temp_root, n=200)
    strat = RelativeStrengthEMACross()
    ctx = StrategyContext(market=Market.US, strategy=strat, data=stock_df, root=temp_root)
    strat.init(ctx)

    # 预热剔除：第 120 bar 前不入字典
    assert ("AAPL_US", dates[50]) not in strat._rs_short
    assert ("AAPL_US", dates[150]) in strat._rs_short
    assert ("AAPL_US", dates[150]) in strat._rs_long
    # ADV20 可经 ctx 取得（成交额 ≈ 100*2_000_000 数量级）
    adv = ctx.indicator_value("ADV", "AAPL_US", dates[150], period=20)
    assert adv is not None and adv > 0
    # _full_data 与调仓状态初始化
    assert strat._full_data is not None
    assert strat._last_rebalance_date is None


def test_init_raises_when_benchmark_missing(temp_root) -> None:
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