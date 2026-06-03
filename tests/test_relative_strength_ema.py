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


REBAL = date(2024, 1, 8)  # Monday: weekday()==0
TUES = date(2024, 1, 9)  # Tuesday


def _build(strat, specs, positions=None, cash=1_000_000.0, n_hist=25, cur_date=REBAL):
    """
    specs: list of (iid, e60, e120, dollar_vol)。
      常量价量构造使 ADV20 == dollar_vol（close=100, volume=dv/100）。
    种入 rs dicts(在 cur_date) + _full_data + ctx ADV 预计算 + pit_universe。
    ctx 定位到首个 iid（触发组合级首调用逻辑）。
    """
    dates = [cur_date - timedelta(days=(n_hist - 1 - i)) for i in range(n_hist)]
    rows = []
    for iid, _e60, _e120, dv in specs:
        close = 100.0
        vol = dv / close
        ticker = iid.split("_")[0]
        for dt in dates:
            rows.append(
                {
                    "instrument_id": iid,
                    "ticker": ticker,
                    "date": dt,
                    "close": close,
                    "volume": vol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "adj_factor": 1.0,
                }
            )
    data = pl.DataFrame(rows)
    ctx = StrategyContext(market=Market.US, strategy=strat, data=data)
    ctx.precompute_indicator("ADV", period=20)

    strat._full_data = data
    strat._last_rebalance_date = None
    strat._rs_short = {(iid, cur_date): e60 for iid, e60, _e120, _dv in specs}
    strat._rs_long = {(iid, cur_date): e120 for iid, _e60, e120, _dv in specs}

    ids = [s[0] for s in specs]
    ctx.pit_universe = lambda _d: ids
    ctx._current_date = cur_date
    ctx.update_positions(positions or {}, cash)

    first = ids[0]
    first_row = data.filter(
        (pl.col("instrument_id") == first) & (pl.col("date") == cur_date)
    ).to_dicts()[0]
    ctx.update_bar(cur_date, first, first, data, current_row=first_row)
    return ctx


def _dirs(ctx, direction):
    return {s.instrument_id for s in ctx.pending_signals() if s.direction == direction}


def test_top_n_selection() -> None:
    """5 个金叉候选、top_n=3 → 只买相对强度最高的 3 只。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [
        ("A_US", 1.30, 1.00, 2e8),  # score .30
        ("B_US", 1.25, 1.00, 2e8),  # .25
        ("C_US", 1.20, 1.00, 2e8),  # .20
        ("D_US", 1.10, 1.00, 2e8),  # .10
        ("E_US", 1.05, 1.00, 2e8),  # .05
    ]
    ctx = _build(strat, specs)
    strat.next(ctx)
    assert _dirs(ctx, "BUY") == {"A_US", "B_US", "C_US"}


def test_non_golden_excluded() -> None:
    """死叉 (e60<=e120) 不进候选。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [
        ("A_US", 1.30, 1.00, 2e8),
        ("B_US", 0.90, 1.00, 2e8),  # 死叉
        ("C_US", 1.20, 1.00, 2e8),
    ]
    ctx = _build(strat, specs)
    strat.next(ctx)
    assert _dirs(ctx, "BUY") == {"A_US", "C_US"}


def test_adv_gate_excludes_illiquid() -> None:
    """强势但 ADV20 < min_adv_us 的被剔除。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [
        ("A_US", 1.40, 1.00, 5e7),  # 最强但 5千万 < 1亿
        ("B_US", 1.20, 1.00, 2e8),
        ("C_US", 1.10, 1.00, 2e8),
    ]
    ctx = _build(strat, specs)
    strat.next(ctx)
    assert _dirs(ctx, "BUY") == {"B_US", "C_US"}


def test_equal_weight_shares() -> None:
    """等权：shares == int((NAV/top_n)/close)。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [("A_US", 1.30, 1.00, 2e8), ("B_US", 1.20, 1.00, 2e8), ("C_US", 1.10, 1.00, 2e8)]
    ctx = _build(strat, specs, cash=1_000_000.0)  # NAV=1e6, per=333_333, close=100
    strat.next(ctx)
    buy = {s.instrument_id: s.shares for s in ctx.pending_signals() if s.direction == "BUY"}
    assert buy["A_US"] == 3333.0


def test_sell_dropout_full_close() -> None:
    """持仓 250 股但本周跌出 top_n → SELL 且 shares==250（不被默认100截断）。"""
    strat = RelativeStrengthEMACross(params={"top_n": 2})
    specs = [
        ("A_US", 1.30, 1.00, 2e8),
        ("B_US", 1.20, 1.00, 2e8),
        ("OLD_US", 1.05, 1.00, 2e8),  # 仍金叉但排名第3，跌出 top2
    ]
    ctx = _build(strat, specs, positions={"OLD_US": 250.0})
    strat.next(ctx)
    sells = [s for s in ctx.pending_signals() if s.direction == "SELL"]
    assert len(sells) == 1 and sells[0].instrument_id == "OLD_US"
    assert sells[0].shares == 250.0


def test_sell_on_death_cross() -> None:
    """持仓死叉 (e60<=e120) → 不进候选 → SELL。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [
        ("A_US", 1.30, 1.00, 2e8),
        ("B_US", 1.20, 1.00, 2e8),
        ("OLD_US", 0.95, 1.00, 2e8),  # 死叉
    ]
    ctx = _build(strat, specs, positions={"OLD_US": 100.0})
    strat.next(ctx)
    assert _dirs(ctx, "SELL") == {"OLD_US"}


def test_weekly_guard_and_no_double_run() -> None:
    """非调仓日不动作；同日二次调用不重复。"""
    strat = RelativeStrengthEMACross(params={"top_n": 3})
    specs = [("A_US", 1.30, 1.00, 2e8), ("B_US", 1.20, 1.00, 2e8)]
    # 周二、非 screening → 无信号
    ctx = _build(strat, specs, cur_date=TUES)
    strat.next(ctx)
    assert ctx.pending_signals() == []
    # 周一首调有信号，二次调用无新增
    ctx2 = _build(strat, specs)
    strat.next(ctx2)
    n_first = len(ctx2.pending_signals())
    strat.next(ctx2)
    assert len(ctx2.pending_signals()) == n_first


def test_screening_ignores_weekday() -> None:
    """screening 模式任意 weekday 都出 Top-N。"""
    strat = RelativeStrengthEMACross(params={"top_n": 2})
    specs = [("A_US", 1.30, 1.00, 2e8), ("B_US", 1.20, 1.00, 2e8), ("C_US", 1.10, 1.00, 2e8)]
    ctx = _build(strat, specs, cur_date=TUES)
    ctx.is_screening = True
    strat.next(ctx)
    assert _dirs(ctx, "BUY") == {"A_US", "B_US"}
