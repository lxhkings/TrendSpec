"""hypothesis → 候选 FactorSpec 网格展开。"""

import itertools
import random
from typing import Any


def expand_grid(
    hypothesis: dict[str, Any],
    max_candidates: int = 200,
    rng_seed: int = 0,
) -> list[dict]:
    market = hypothesis["market"]
    rationale = hypothesis.get("rationale", "")
    factors = hypothesis["factors"]
    top_k_grid = hypothesis["top_k_grid"]
    rebalance_grid = hypothesis["rebalance_grid"]

    # 为每个因子展开其 param_grid → 该因子的 params 取值列表
    per_factor_options: list[list[dict]] = []
    for term in factors:
        grid = term.get("param_grid", {})
        if not grid:
            per_factor_options.append(
                [
                    {
                        "name": term["name"],
                        "params": {},
                        "direction": term["direction"],
                        "weight": term.get("weight", 1.0),
                    }
                ]
            )
            continue
        keys = list(grid.keys())
        opts = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            opts.append(
                {
                    "name": term["name"],
                    "params": dict(zip(keys, combo, strict=True)),
                    "direction": term["direction"],
                    "weight": term.get("weight", 1.0),
                }
            )
        per_factor_options.append(opts)

    candidates: list[dict] = []
    for factor_combo in itertools.product(*per_factor_options):
        for top_k in top_k_grid:
            for rebalance in rebalance_grid:
                candidates.append(
                    {
                        "market": market,
                        "factors": [dict(f) for f in factor_combo],
                        "top_k": top_k,
                        "rebalance": rebalance,
                        "rationale": rationale,
                    }
                )

    if len(candidates) > max_candidates:
        rng = random.Random(rng_seed)
        candidates = rng.sample(candidates, max_candidates)
    return candidates
