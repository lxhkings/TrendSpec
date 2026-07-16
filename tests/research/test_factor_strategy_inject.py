import datetime as dt

import trendspec.factors  # noqa: F401
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars
from trendspec.data.universe import get_universe
from trendspec.engine.backtest_engine import BacktestEngine
from trendspec.engine.base_engine import EngineConfig
from trendspec.combo import compute_combo_scores
from trendspec.strategy.factor_strategy import FactorStrategy

SPEC = {
    "market": "us",
    "factors": [
        {"name": "momentum", "params": {"period": 10}, "direction": "high", "weight": 1.0}
    ],
    "top_k": 5,
    "rebalance": 5,
}


def test_injected_scores_equal_self_computed():
    cfg = EngineConfig(
        market=Market.US,
        start_date=dt.date(2020, 1, 1),
        end_date=dt.date(2021, 1, 1),
    )
    baseline = BacktestEngine(cfg).run(FactorStrategy, params={"spec": SPEC})

    m = Market.US
    df = bars(market=m, start_date=cfg.start_date, end_date=cfg.end_date)
    uni = get_universe(m, cfg.root)
    scores = compute_combo_scores(df, SPEC["factors"], SPEC["market"], root=cfg.root)

    eng = BacktestEngine(cfg)
    eng.inject(data=df, universe=uni)
    injected = eng.run(
        FactorStrategy, params={"spec": SPEC, "precomputed_scores": scores}
    )
    assert injected.metrics == baseline.metrics
