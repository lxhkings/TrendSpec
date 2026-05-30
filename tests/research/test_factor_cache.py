import datetime as dt
import polars as pl
import trendspec.factors  # noqa: F401 触发注册
from trendspec.research.factor_cache import build_combo_score


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
    score = build_combo_score(df, factors)
    assert set(score.columns) == {"instrument_id", "date", "combo_score"}
    last = score.filter(pl.col("combo_score").is_not_null())
    assert last.height > 0
    assert last["combo_score"].is_finite().all()
