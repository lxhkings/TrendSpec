import datetime as dt

import polars as pl
import pytest

import trendspec.factors  # noqa: F401 触发因子注册
from trendspec.combo import compute_combo_scores
from trendspec.research.factor_eval import (
    _attach_forward_returns,
    compute_quantile_returns,
    compute_rank_ic,
    compute_top_minus_bottom,
    summarize_ic,
)


def _panel() -> pl.DataFrame:
    """2支股票，20天，close = 10 + i（等差数列，方便手算前瞻收益）。"""
    rows = []
    for iid, base in [("A", 10.0), ("B", 100.0)]:
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": base + i})
    return pl.DataFrame(rows)


def test_attach_forward_returns_computes_shifted_ratio():
    df = _panel()
    out = _attach_forward_returns(df, horizon=5)
    row = out.filter((pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1)))
    close_t0 = 10.0
    close_t5 = 10.0 + 5
    expected = close_t5 / close_t0 - 1
    assert row["fwd_ret_5d"][0] == pytest.approx(expected)


def test_attach_forward_returns_tail_is_null():
    df = _panel()
    out = _attach_forward_returns(df, horizon=5)
    last_row = out.filter(
        (pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1) + dt.timedelta(days=19))
    )
    assert last_row["fwd_ret_5d"][0] is None


def test_attach_forward_returns_does_not_cross_instruments():
    """A 的最后一行不应该拿 B 的 close 算前瞻收益。"""
    df = _panel()
    out = _attach_forward_returns(df, horizon=1)
    second_last_a = out.filter(
        (pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1) + dt.timedelta(days=18))
    )
    # A 第19天(index18) close=28, 第20天(index19) close=29, 不应等于 B 的 close
    assert second_last_a["fwd_ret_1d"][0] == pytest.approx(29.0 / 28.0 - 1)


def test_attach_forward_returns_handles_shuffled_input():
    """回归测试：即使输入 panel 行序被打乱（非按日期排序），也应计算出正确的前瞻收益。

    这验证函数内部有防御性的排序，不依赖调用方已排序的假设。
    .over() 不会自动排序，只是分组；如果没有先排序，shift 会作用于错误的行序。"""
    df = _panel()
    # 打乱行序（反向排列）
    shuffled = df.reverse()
    out = _attach_forward_returns(shuffled, horizon=5)

    # 验证结果与未打乱的输入一致
    row = out.filter((pl.col("instrument_id") == "A") & (pl.col("date") == dt.date(2020, 1, 1)))
    close_t0 = 10.0
    close_t5 = 10.0 + 5
    expected = close_t5 / close_t0 - 1
    assert row["fwd_ret_5d"][0] == pytest.approx(expected)


def _panel_with_monotonic_relation() -> pl.DataFrame:
    """5支股票 x 30天：close 走势让 momentum 因子分与未来收益完全同向，
    构造出 RankIC 应该接近 1 的数据。价格按 instrument 分层次线性增长，
    增长越快的股票 momentum 分越高，未来收益也越高。"""
    rows = []
    slopes = {"A": 0.1, "B": 0.5, "C": 1.0, "D": 2.0, "E": 4.0}
    for iid, slope in slopes.items():
        price = 100.0
        for i in range(30):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": price})
            price *= (1 + slope / 100)
    return pl.DataFrame(rows)


def test_compute_rank_ic_returns_date_and_rank_ic_columns():
    df = _panel_with_monotonic_relation()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    ic_df = compute_rank_ic(df, factors, market="cn", horizon=5)
    assert set(ic_df.columns) == {"date", "rank_ic"}
    assert ic_df.height > 0


def test_compute_rank_ic_high_for_monotonic_relation():
    df = _panel_with_monotonic_relation()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    ic_df = compute_rank_ic(df, factors, market="cn", horizon=5)
    assert ic_df["rank_ic"].mean() > 0.9


def test_summarize_ic_computes_mean_std_ir_win_rate():
    ic_df = pl.DataFrame({
        "date": [dt.date(2020, 1, i) for i in range(1, 6)],
        "rank_ic": [0.2, 0.4, -0.1, 0.3, 0.1],
    })
    summary = summarize_ic(ic_df)
    assert summary["ic_mean"] == pytest.approx(0.18)
    assert summary["ic_win_rate"] == pytest.approx(0.8)  # 4/5 为正
    assert summary["ir"] == pytest.approx(summary["ic_mean"] / ic_df["rank_ic"].std())


def test_summarize_ic_empty_returns_none():
    empty = pl.DataFrame({"date": [], "rank_ic": []}, schema={"date": pl.Date, "rank_ic": pl.Float64})
    summary = summarize_ic(empty)
    assert summary == {"ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None}


def test_compute_quantile_returns_columns_and_labels():
    df = _panel_with_monotonic_relation()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    qr = compute_quantile_returns(df, factors, market="cn", horizon=5, n_quantiles=5)
    assert set(qr.columns) == {"date", "quantile", "avg_fwd_return"}
    assert set(qr["quantile"].unique().to_list()) <= {"0", "1", "2", "3", "4"}


def test_compute_quantile_returns_is_monotonic_for_monotonic_relation():
    df = _panel_with_monotonic_relation()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    qr = compute_quantile_returns(df, factors, market="cn", horizon=5, n_quantiles=5)
    avg_by_q = (
        qr.group_by("quantile")
        .agg(pl.col("avg_fwd_return").mean().alias("mean_ret"))
        .sort("quantile")
    )
    rets = avg_by_q["mean_ret"].to_list()
    assert rets == sorted(rets)  # 分位越高，平均前瞻收益越高（单调递增）


def test_compute_top_minus_bottom_positive_for_monotonic_relation():
    df = _panel_with_monotonic_relation()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    qr = compute_quantile_returns(df, factors, market="cn", horizon=5, n_quantiles=5)
    tmb = compute_top_minus_bottom(qr, n_quantiles=5)
    assert set(tmb.columns) == {"date", "top_minus_bottom"}
    assert tmb["top_minus_bottom"].mean() > 0


def test_compute_quantile_returns_per_date_qcut_regression():
    """Regression test: verify .over("date") is used in qcut, not global qcut.

    Constructs a 4-instrument panel where factor ranking changes significantly
    across dates:
    - Days 0-14: Instruments ranked by momentum as A > B > C > D
    - Days 15-29: Instruments ranked by momentum as D > C > B > A (reversed!)

    Per-date qcut (correct, with .over("date")): Each date bucketing reflects
    that date's cross-sectional ranking. E.g., A is in high quantile early, low
    quantile late.

    Global qcut (buggy, no .over("date")): All dates pooled for bucketing.
    A's high average momentum score across all dates locks it into high quantile
    throughout, even though its per-date ranking reverses—hiding the key fact
    that good momentum doesn't always coincide with good forward returns on all dates.

    This test extracts the per-date quantile assignment from the bucketed data
    and asserts the correct and buggy approaches assign instruments differently.
    It will FAIL if .over("date") is accidentally removed from compute_quantile_returns.
    """
    rows = []

    # Four instruments with distinct, inverted momentum patterns
    for i in range(30):
        d = dt.date(2020, 1, 1) + dt.timedelta(days=i)

        # A: high momentum days 0-14, flat 15-29
        price_a = 100.0 * (1.04 ** min(i, 14))
        rows.append({"instrument_id": "A", "date": d, "close": price_a})

        # B: medium momentum days 0-14, medium 15-29
        price_b = 100.0 * (1.02 ** i)
        rows.append({"instrument_id": "B", "date": d, "close": price_b})

        # C: low momentum days 0-14, medium 15-29
        price_c = 100.0 * (1.01 ** min(i, 14)) * (1.02 ** max(0, i - 14))
        rows.append({"instrument_id": "C", "date": d, "close": price_c})

        # D: flat days 0-14, high momentum 15-29
        price_d = 100.0 * (1.04 ** max(0, i - 14))
        rows.append({"instrument_id": "D", "date": d, "close": price_d})

    df = pl.DataFrame(rows)
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]

    # Correct result using compute_quantile_returns (uses .over("date"))
    correct_result = compute_quantile_returns(df, factors, market="cn", horizon=5, n_quantiles=2)

    # Manually compute what GLOBAL qcut would produce (without .over("date"))
    scores = compute_combo_scores(df, factors, market="cn", group_by=None, winsorize_pct=0.01, root=None)
    fwd = _attach_forward_returns(df, horizon=5)
    ret_col = "fwd_ret_5d"

    joined = scores.join(
        fwd.select(["instrument_id", "date", ret_col]),
        on=["instrument_id", "date"],
        how="inner",
    ).filter(pl.col("combo_score").is_not_null() & pl.col(ret_col).is_not_null())

    # Simulate buggy version: global qcut (no .over("date"))
    labels = ["0", "1"]
    buggy_bucketed = joined.with_columns(
        pl.col("combo_score").qcut(2, labels=labels).alias("quantile")  # No .over("date")!
    )

    buggy_result = (
        buggy_bucketed.group_by(["date", "quantile"])
        .agg(pl.col(ret_col).mean().alias("avg_fwd_return"))
        .with_columns(pl.col("quantile").cast(pl.String))
        .sort(["date", "quantile"])
    )

    # Core assertion: per-date qcut and global qcut must produce DIFFERENT results.
    # With 4 instruments and 2 quantiles, when ranking flips across dates, the
    # bucketing should differ—global qcut locks A into high quantile (its average
    # is highest), while per-date qcut puts A into low quantile on days 15-29.
    assert not correct_result.equals(buggy_result), (
        "Per-date qcut (correct) should differ from global qcut (buggy) when factor "
        "ranking changes across dates. If this assertion fails, .over('date') may have "
        "been accidentally removed from the qcut call in compute_quantile_returns."
    )


def _panel_all_identical() -> pl.DataFrame:
    """3支股票 close 序列完全相同 → momentum 截面每天全相等 → 组内 std=0。"""
    rows = []
    for iid in ["A", "B", "C"]:
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 10.0 + i})
    return pl.DataFrame(rows)


def test_combo_scores_zero_std_cross_section_rows_dropped():
    """截面全相等 → std=0 → z-score 非有限,整行剔除;不得漏出 NaN/inf combo_score。

    回归:2026-07-16 round,fund_revenue_cagr_3y/ema_alignment 因此出 IC均值=nan。"""
    df = _panel_all_identical()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    scores = compute_combo_scores(df, factors, "cn")
    assert scores.is_empty()


def test_combo_scores_partial_ties_all_finite():
    """部分并列(4 同 + 1 异)截面 std>0,行保留且 combo_score 全部有限。"""
    rows = []
    slopes = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 3.0}
    for iid, slope in slopes.items():
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 100.0 + slope * i})
    df = pl.DataFrame(rows)
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    scores = compute_combo_scores(df, factors, "cn")
    assert scores.height > 0
    assert scores["combo_score"].is_finite().all()


def test_summarize_ic_ignores_non_finite_rank_ic():
    """一颗 NaN 不得毒化整个均值(回归:IC均值=nan 但胜率有值)。"""
    ic_df = pl.DataFrame({
        "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)],
        "rank_ic": [0.5, float("nan"), 0.3],
    })
    s = summarize_ic(ic_df)
    assert s["ic_mean"] == pytest.approx(0.4)
    assert s["ic_win_rate"] == pytest.approx(1.0)


def test_summarize_ic_all_nan_returns_none():
    ic_df = pl.DataFrame({"date": [dt.date(2020, 1, 1)], "rank_ic": [float("nan")]})
    assert summarize_ic(ic_df) == {
        "ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None,
    }


def _panel_with_flat_tail() -> pl.DataFrame:
    """前 10 天 3 支股票斜率不同(momentum 有区分度),之后全部横盘——
    横盘段前瞻收益全为 0,收益秩零方差,corr 在这些日期产出 NaN。"""
    rows = []
    slopes = {"A": 0.5, "B": 1.0, "C": 2.0}
    for iid, slope in slopes.items():
        price = 100.0
        for i in range(25):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            if i < 10:
                price += slope
            rows.append({"instrument_id": iid, "date": d, "close": price})
    return pl.DataFrame(rows)


def test_compute_rank_ic_excludes_degenerate_dates():
    df = _panel_with_flat_tail()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    ic_df = compute_rank_ic(df, factors, "cn", horizon=5)
    assert ic_df.height > 0
    assert ic_df["rank_ic"].is_finite().all()


def test_compute_quantile_returns_handles_tied_scores_without_panic():
    """6支股票 4 支因子值并列:5 分位边界重复,未加 allow_duplicates 时
    polars qcut 直接 PanicException(回归:20260716 H1 分层)。"""
    rows = []
    slopes = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 2.0, "F": 3.0}
    for iid, slope in slopes.items():
        for i in range(25):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 100.0 + slope * i})
    df = pl.DataFrame(rows)
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    qr = compute_quantile_returns(df, factors, "cn", horizon=5, n_quantiles=5)
    assert not qr.is_empty()
