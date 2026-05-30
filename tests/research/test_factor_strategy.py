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
