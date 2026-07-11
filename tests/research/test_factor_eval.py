import datetime as dt

import polars as pl
import pytest

import trendspec.factors  # noqa: F401 触发因子注册
from trendspec.research.factor_eval import _attach_forward_returns, compute_rank_ic, summarize_ic


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
