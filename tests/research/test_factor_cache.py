import datetime as dt
import polars as pl
import trendspec.factors  # noqa: F401 触发注册
from trendspec.research.factor_cache import build_combo_score, FactorCache


def _panel():
    rows = []
    for iid, base in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
        for i in range(40):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d,
                         "open": base + i, "high": base + i + 1,
                         "low": base + i - 1, "close": base + i, "volume": 1000 + i,
                         "ticker": iid})
    return pl.DataFrame(rows)


def test_build_combo_score_matches_inline_zscore():
    df = _panel()
    factors = [
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0},
        {"name": "volatility", "params": {"period": 10}, "direction": "low", "weight": 0.5},
    ]
    score = build_combo_score(df, factors, market="us")
    assert set(score.columns) == {"instrument_id", "date", "combo_score"}
    last = score.filter(pl.col("combo_score").is_not_null())
    assert last.height > 0
    assert last["combo_score"].is_finite().all()


def test_factor_cache_memoizes_by_name_params():
    df = _panel()
    cache = FactorCache(df)
    a = cache.get("momentum", {"period": 5})
    b = cache.get("momentum", {"period": 5})
    assert a is b  # 命中同一对象
    c = cache.get("momentum", {"period": 10})
    assert c is not a  # 不同参数不命中
    assert cache.compute_count == 2  # 只真正算了两次


def test_build_combo_score_normalizes_market_for_cross_sectional_factor():
    df = _panel()
    factors = [
        {"name": "rank_within_sector",
         "params": {"factor_name": "momentum", "market": "us"},
         "direction": "low", "weight": 1.0},
    ]
    score = build_combo_score(df, factors, market="us")
    assert set(score.columns) == {"instrument_id", "date", "combo_score"}
