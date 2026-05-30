"""研究评估基准：固定 grid，对比逐候选 vs ResearchEvaluator(串行/并行) 墙钟。"""

import time
from datetime import date

import trendspec.factors  # noqa: F401
import trendspec.strategy.factor_strategy  # noqa: F401
from trendspec.research.fast_eval import ResearchEvaluator
from trendspec.research.orchestrator import default_evaluate_fn

START, END = date(2018, 1, 1), date(2023, 12, 31)
CANDS = [
    {"market": "us", "factors": [{"name": "momentum", "params": {"period": p},
     "direction": "high", "weight": 1.0}], "top_k": k, "rebalance": r, "rationale": "b"}
    for p in (10, 20, 50) for k in (5, 10, 20) for r in (5, 10)
]  # 18 候选，6 unique combo


def _t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:28s} {dt:7.2f}s  n={len(out)}")
    return dt


if __name__ == "__main__":
    print(f"候选={len(CANDS)}  区间={START}..{END}")
    fn = default_evaluate_fn("us", START, END, n_windows=4, capital=100000.0)
    _t("baseline 逐候选", lambda: [fn(c) for c in CANDS])
    _t("evaluator 串行", lambda: ResearchEvaluator("us", START, END, parallel=False).evaluate_batch(CANDS))
    _t("evaluator 并行", lambda: ResearchEvaluator("us", START, END, parallel=True).evaluate_batch(CANDS))