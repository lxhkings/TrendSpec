from trendspec.research.search import expand_grid


def _hypo():
    return {
        "market": "us",
        "factors": [{"name": "momentum", "direction": "high", "weight": 1.0,
                     "param_grid": {"period": [20, 60, 120]}}],
        "top_k_grid": [10, 20],
        "rebalance_grid": [5, 10],
        "rationale": "test",
    }


def test_expand_full_cartesian():
    cands = expand_grid(_hypo(), max_candidates=100, rng_seed=0)
    # 3 period × 2 top_k × 2 rebalance = 12
    assert len(cands) == 12
    one = cands[0]
    assert one["market"] == "us"
    assert one["factors"][0]["name"] == "momentum"
    assert "period" in one["factors"][0]["params"]
    assert one["top_k"] in (10, 20)
    assert one["rebalance"] in (5, 10)


def test_max_candidates_samples_deterministically():
    a = expand_grid(_hypo(), max_candidates=5, rng_seed=42)
    b = expand_grid(_hypo(), max_candidates=5, rng_seed=42)
    assert len(a) == 5
    assert a == b  # 同种子可复现


def test_each_candidate_is_valid_factorspec():
    from trendspec.research.spec import FactorSpec
    for c in expand_grid(_hypo(), max_candidates=100, rng_seed=0):
        FactorSpec(**c)  # 不抛异常即合法
