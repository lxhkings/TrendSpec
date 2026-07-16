import datetime as dt

import polars as pl

import trendspec.factors  # noqa: F401 触发注册
import trendspec.research.factor_cache as factor_cache_module
from trendspec.research.factor_cache import compute_combo_scores


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


def test_compute_combo_scores_returns_finite_scores():
    df = _panel()
    factors = [
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0},
        {"name": "volatility", "params": {"period": 10}, "direction": "low", "weight": 0.5},
    ]
    score = compute_combo_scores(df, factors, market="us")
    assert set(score.columns) == {"instrument_id", "date", "_group", "combo_score"}
    last = score.filter(pl.col("combo_score").is_not_null())
    assert last.height > 0
    assert last["combo_score"].is_finite().all()


def test_compute_combo_scores_normalizes_market_for_cross_sectional_factor():
    df = _panel()
    factors = [
        {"name": "rank_within_sector",
         "params": {"factor_name": "momentum", "market": "us"},
         "direction": "low", "weight": 1.0},
    ]
    score = compute_combo_scores(df, factors, market="us")
    assert set(score.columns) == {"instrument_id", "date", "_group", "combo_score"}


def _panel_with_margin():
    """A/B/C/D 四只股票，op_margin: A=10.0(过), B=-5.0(不过), C=null(缺失,不过),
    D=8.0(过)。过滤后须留 ≥2 名存活者(A,D)，否则单一存活者会撞上
    compute_combo_scores 既有的"单成员分组 std 无定义即剔除"规则（该规则本身
    正确，见 test_factor_strategy.py::test_init_single_member_group_excludes_from_ranking），
    与本测试想验证的"过滤剔除不达标/缺失行"语义无关，会造成误报。不复用共享
    _panel()（只有 A/B/C），自建含 D 的四股面板。"""
    df = _panel()
    extra_rows = []
    for i in range(40):
        d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
        extra_rows.append({"instrument_id": "D", "date": d,
                            "open": 40.0 + i, "high": 40.0 + i + 1,
                            "low": 40.0 + i - 1, "close": 40.0 + i, "volume": 1000 + i,
                            "ticker": "D"})
    df = pl.concat([df, pl.DataFrame(extra_rows)])
    margin = {"A": 10.0, "B": -5.0, "C": None, "D": 8.0}
    return df.with_columns(
        pl.col("instrument_id").replace_strict(margin, default=None).alias("op_margin")
    )


def test_filters_exclude_failing_and_missing_rows():
    df = _panel_with_margin()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    filters = [{"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0}]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    survivors = set(score["instrument_id"].unique().to_list())
    assert survivors == {"A", "D"}  # B 不达标剔除；C 缺失值也剔除


def test_filters_none_keeps_existing_behavior():
    df = _panel_with_margin()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    a = compute_combo_scores(df, factors, market="cn")
    b = compute_combo_scores(df, factors, market="cn", filters=None)
    assert a.equals(b)


def test_filters_zscore_stats_computed_on_survivors():
    """过滤在 z-score 之前生效：幸存者 z 统计只含幸存者。

    A/B 存活（op_margin>0=[10,5]），C 剔除。单因子 fund_op_margin 排序，
    两名幸存者 z 分应互为相反数（均值中心化后对称）。
    """
    df = _panel().with_columns(
        pl.col("instrument_id").replace_strict(
            {"A": 10.0, "B": 5.0, "C": -1.0}, default=None
        ).alias("op_margin")
    )
    factors = [{"name": "fund_op_margin", "params": {}, "direction": "high", "weight": 1.0}]
    filters = [{"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0}]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    one_day = score.filter(pl.col("date") == score["date"].min()).sort("instrument_id")
    assert one_day["instrument_id"].to_list() == ["A", "B"]
    vals = one_day["combo_score"].to_list()
    assert abs(vals[0] + vals[1]) < 1e-9


def test_apply_filters_same_factor_twice_computes_once(monkeypatch):
    """两个 filter 使用同一 name+params 时，filter 阶段只 compute_full 一次。"""
    df = _panel_with_margin()
    calls: list[tuple] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append((name, tuple(sorted((params or {}).items())), id(frame)))
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    filters = [
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0},
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": -1.0},  # 同 key，更松
    ]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    assert score.height > 0
    filter_calls = [c for c in calls if c[0] == "fund_op_margin"]
    assert len(filter_calls) == 1, f"expected 1 filter-stage compute, got {filter_calls}"


def test_score_stage_duplicate_factors_compute_once(monkeypatch):
    """factors 列表两个相同 name+params 时，score 阶段只 compute_full 一次。"""
    df = _panel()
    calls: list[str] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append(name)
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 0.5},
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 0.5},
    ]
    score = compute_combo_scores(df, factors, market="us")
    assert score.height > 0
    assert calls.count("momentum") == 1


def test_filter_and_score_same_factor_may_compute_twice(monkeypatch):
    """同因子既在 filters 又在 factors：允许每阶段各一次（共 2），禁止跨阶段强制 1 次。"""
    df = _panel_with_margin()
    calls: list[str] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append(name)
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [
        {"name": "fund_op_margin", "params": {}, "direction": "high", "weight": 1.0},
    ]
    filters = [
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0},
    ]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    assert score.height > 0
    # 行为冻结下允许 2；实现 memo 后仍应为 2，不能误改成 1 若那样会改截面语义
    assert calls.count("fund_op_margin") == 2
