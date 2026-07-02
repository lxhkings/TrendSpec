from datetime import date, timedelta

import polars as pl

from trendspec.data.markets import Market
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.factor_strategy import FactorStrategy


def _make_bars(iid: str, n: int, start_close: float, drift: float) -> pl.DataFrame:
    rows = []
    close = start_close
    ticker = iid.split("_")[0]
    for i in range(n):
        d = date(2024, 1, 1) + timedelta(days=i)
        rows.append({
            "instrument_id": iid, "date": d, "ticker": ticker,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000_000, "adj_factor": 1.0,
        })
        close *= drift
    return pl.DataFrame(rows)


def _two_stock_data() -> pl.DataFrame:
    # FAST 强动量, SLOW 弱动量 —— 同一日截面可比较
    fast = _make_bars("FAST_US", 120, 100.0, 1.004)
    slow = _make_bars("SLOW_US", 120, 100.0, 1.0005)
    return pl.concat([fast, slow])


def _spec_dict():
    return {
        "spec": {
            "market": "us",
            "factors": [{"name": "momentum", "params": {"period": 60},
                         "direction": "high", "weight": 1.0}],
            "top_k": 1,
            "rebalance": 5,
        }
    }


def test_init_builds_score_cache():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    ranked = strat._ranked_by_date[last_date]
    # 强动量股排第一
    assert ranked[0] == "FAST_US"
    assert strat._score_by_date[(last_date, "FAST_US")] > strat._score_by_date[(last_date, "SLOW_US")]


def test_direction_low_inverts_rank():
    df = _two_stock_data()
    d = _spec_dict()
    d["spec"]["factors"][0]["direction"] = "low"
    strat = FactorStrategy(params=d)
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    # direction=low → 弱动量股反而排第一
    assert strat._ranked_by_date[last_date][0] == "SLOW_US"


def _run_next_once(strat, ctx, df, target_date):
    """模拟引擎：对某交易日逐 instrument 调 next()，收集信号。"""
    universe = df["instrument_id"].unique().to_list()
    ctx.set_universe(_StubUniverse(universe))
    day = df.filter(pl.col("date") == target_date)
    rows = {r["instrument_id"]: r for r in day.iter_rows(named=True)}
    ctx.clear_signals()
    for iid in universe:
        row = rows.get(iid)
        if row is None:
            continue
        ctx.update_bar(target_date, iid, row["ticker"], df, current_row=row)
        strat.next(ctx)
    return ctx.pending_signals()


class _StubUniverse:
    def __init__(self, ids):
        self._ids = ids

    def tickers(self, _as_of_date):
        return self._ids


def test_next_emits_buy_for_top_k_on_rebalance():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())  # top_k=1
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    last_date = df["date"].max()
    sigs = _run_next_once(strat, ctx, df, last_date)
    buys = [s for s in sigs if s.direction == "BUY"]
    # top_k=1 → 只买强动量股
    assert len(buys) == 1
    assert buys[0].instrument_id == "FAST_US"


def test_next_respects_rebalance_interval():
    df = _two_stock_data()
    strat = FactorStrategy(params=_spec_dict())  # rebalance=5
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)

    all_dates = sorted(df["date"].unique().to_list())
    # 第一次调仓日（day index 60）出信号
    sigs1 = _run_next_once(strat, ctx, df, all_dates[60])
    assert len(sigs1) >= 1
    # 紧邻下一日（间隔 1 < 5）不再调仓
    sigs2 = _run_next_once(strat, ctx, df, all_dates[61])
    assert sigs2 == []


def test_init_normalizes_lowercase_market_string_for_cross_sectional_factor():
    df = _two_stock_data()
    spec = {
        "spec": {
            "market": "us",
            "factors": [{"name": "rank_within_sector",
                         "params": {"factor_name": "returns", "market": "us"},
                         "direction": "low", "weight": 1.0}],
            "top_k": 1,
            "rebalance": 5,
        }
    }
    strat = FactorStrategy(params=spec)
    ctx = StrategyContext(market=Market.US, strategy=strat, data=df)
    strat.init(ctx)  # pre-fix: AttributeError, 'str' object has no attribute 'path'
    last_date = df["date"].max()
    assert last_date in strat._ranked_by_date
