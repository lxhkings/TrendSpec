import datetime as dt
import trendspec.factors  # noqa: F401
import trendspec.strategy.factor_strategy  # noqa: F401
from trendspec.research.fast_eval import ResearchEvaluator

START, END = dt.date(2020, 1, 1), dt.date(2021, 1, 1)
CANDS = [
    {"market": "us", "factors": [{"name": "momentum", "params": {"period": p},
     "direction": "high", "weight": 1.0}], "top_k": k, "rebalance": 5, "rationale": "x"}
    for p in (10, 20) for k in (5, 10)
]


def test_parallel_equals_serial():
    serial = ResearchEvaluator("us", START, END, n_windows=4, parallel=False).evaluate_batch(CANDS)
    par = ResearchEvaluator("us", START, END, n_windows=4, parallel=True).evaluate_batch(CANDS)
    assert len(par) == len(serial)
    for p, s in zip(par, serial, strict=True):
        assert p["spec"] == s["spec"]  # 顺序确定性
        assert abs(p["oos_sharpe"] - s["oos_sharpe"]) < 1e-9
        assert abs(p["oos_total_return"] - s["oos_total_return"]) < 1e-9