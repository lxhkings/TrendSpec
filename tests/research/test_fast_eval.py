import datetime as dt

import trendspec.factors  # noqa: F401
import trendspec.strategy.factor_strategy  # noqa: F401

from trendspec.research.fast_eval import ResearchEvaluator
from trendspec.research.orchestrator import default_evaluate_fn

START, END = dt.date(2020, 1, 1), dt.date(2021, 1, 1)
CANDS = [
    {"market": "us", "factors": [{"name": "momentum", "params": {"period": 10}, "direction": "high", "weight": 1.0}], "top_k": 5, "rebalance": 5, "rationale": "x"},
    {"market": "us", "factors": [{"name": "momentum", "params": {"period": 10}, "direction": "high", "weight": 1.0}], "top_k": 10, "rebalance": 5, "rationale": "x"},
]


def test_batch_matches_per_candidate():
    fn = default_evaluate_fn("us", START, END, n_windows=4, capital=100000.0)
    expected = [fn(c) for c in CANDS]

    ev = ResearchEvaluator("us", START, END, n_windows=4, capital=100000.0, parallel=False)
    got = ev.evaluate_batch(CANDS)

    for g, e in zip(got, expected, strict=True):
        assert abs(g["oos_sharpe"] - e["oos_sharpe"]) < 1e-9
        assert abs(g["oos_max_drawdown"] - e["oos_max_drawdown"]) < 1e-9
        assert abs(g["oos_total_return"] - e["oos_total_return"]) < 1e-9