from datetime import date

import polars as pl
import pytest

import trendspec.factors  # noqa: F401  (触发注册)
from trendspec.factors.fundamental.trend import _quarterly_series, _asof_join_quarterly_result, _quarterly_shift_compute
from trendspec.factors.registry import get_factor


def _daily_df() -> pl.DataFrame:
    """AAA 3 个季度（Q1=100 @2020-03-31 生效4-15~7-14, Q2=110 @2020-06-30 生效7-15~10-14,
    Q3=121 @2020-09-30 生效10-15起），BBB 只有 1 个季度（50 @2020-03-31）。"""
    rows = []

    def add(iid, d, end_date, rev):
        rows.append({"instrument_id": iid, "date": d, "end_date": end_date, "total_revenue": rev})

    for d in [date(2020, 4, 15), date(2020, 5, 1), date(2020, 7, 14)]:
        add("AAA", d, date(2020, 3, 31), 100.0)
    for d in [date(2020, 7, 15), date(2020, 8, 1), date(2020, 10, 14)]:
        add("AAA", d, date(2020, 6, 30), 110.0)
    for d in [date(2020, 10, 15), date(2020, 11, 1)]:
        add("AAA", d, date(2020, 9, 30), 121.0)
    for d in [date(2020, 4, 20), date(2020, 5, 5)]:
        add("BBB", d, date(2020, 3, 31), 50.0)

    return pl.DataFrame(rows).sort("date")


def test_quarterly_series_extracts_change_points_only():
    df = _daily_df()
    q = _quarterly_series(df, "total_revenue")
    assert q.height == 4  # AAA 3 季 + BBB 1 季
    aaa = q.filter(pl.col("instrument_id") == "AAA").sort("date")
    assert aaa["total_revenue"].to_list() == [100.0, 110.0, 121.0]
    assert aaa["date"].to_list() == [date(2020, 4, 15), date(2020, 7, 15), date(2020, 10, 15)]


def test_quarterly_series_missing_column_returns_empty_with_schema():
    df = _daily_df().drop("total_revenue")
    q = _quarterly_series(df, "total_revenue")
    assert q.is_empty()
    assert set(q.columns) == {"instrument_id", "date", "end_date", "total_revenue"}


def test_asof_join_quarterly_result_preserves_row_order():
    df = _daily_df()
    quarterly_result = pl.DataFrame({
        "instrument_id": ["AAA", "AAA", "AAA", "BBB"],
        "date": [date(2020, 4, 15), date(2020, 7, 15), date(2020, 10, 15), date(2020, 4, 20)],
        "result": [None, 0.1, 0.1, None],
    })
    series = _asof_join_quarterly_result(df, quarterly_result)
    assert series.len() == df.height
    out = df.with_columns(series.alias("qoq"))
    row = out.filter(
        (pl.col("instrument_id") == "AAA") & (pl.col("date") == date(2020, 8, 1))
    ).row(0, named=True)
    assert row["qoq"] == pytest.approx(0.1)  # 8-1 落在 Q2 生效期内，asof 取 Q2 的 0.1
    row_before = out.filter(
        (pl.col("instrument_id") == "AAA") & (pl.col("date") == date(2020, 5, 1))
    ).row(0, named=True)
    assert row_before["qoq"] is None  # 5-1 落在 Q1 生效期，Q1 没有更早季度对比，null


def test_quarterly_shift_compute_ratio_mode():
    df = _daily_df()
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
    )
    aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
    assert aaa["result"].to_list() == [None, pytest.approx(0.1), pytest.approx(0.1)]
    bbb = result.filter(pl.col("instrument_id") == "BBB")
    assert bbb["result"].to_list() == [None]  # 只有 1 季，没有上一季可比


def test_quarterly_shift_compute_diff_mode():
    df = _daily_df()
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="diff",
    )
    aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
    assert aaa["result"].to_list() == [None, pytest.approx(10.0), pytest.approx(11.0)]


def test_quarterly_shift_compute_cagr_mode_requires_positive_base():
    rows = [
        {"instrument_id": "EEE", "date": date(2020, 4, 15), "end_date": date(2020, 3, 31), "rev": -50.0},
        {"instrument_id": "EEE", "date": date(2023, 4, 15), "end_date": date(2023, 3, 31), "rev": 100.0},
    ]
    df = pl.DataFrame(rows).rename({"rev": "total_revenue"}).sort("date")
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=34, gap_max_months=38, mode="cagr", cagr_years=3.0,
    )
    assert result.filter(pl.col("date") == date(2023, 4, 15))["result"].to_list() == [None]


def test_quarterly_shift_compute_gap_out_of_range_yields_null():
    """跳过一季（gap=6个月）时，用 QoQ 的 2~4 月容忍区间应判 null，不能拿相隔两季的
    数值硬算成"环比"。"""
    rows = [
        {"instrument_id": "CCC", "date": date(2020, 4, 15), "end_date": date(2020, 3, 31), "total_revenue": 100.0},
        {"instrument_id": "CCC", "date": date(2020, 10, 15), "end_date": date(2020, 9, 30), "total_revenue": 130.0},
    ]
    df = pl.DataFrame(rows).sort("date")
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
    )
    assert result["result"].to_list() == [None, None]


def test_quarterly_shift_compute_missing_column_returns_empty():
    df = _daily_df().drop("total_revenue")
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
    )
    assert result.is_empty()


def test_quarterly_shift_compute_anchor_shift_zero_matches_default():
    df = _daily_df()
    default = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
    )
    explicit = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
        anchor_shift=0,
    )
    assert default["result"].to_list() == explicit["result"].to_list()


def test_quarterly_shift_compute_anchor_shift_one_gives_prior_quarter_growth():
    """AAA 3 季营收 100→110→121。anchor_shift=1 时看的是"上一季度的环比"：
    在 Q3 那一行(10-15)，上一季度环比 = Q2 vs Q1 = (110-100)/100 = 0.1。
    Q1/Q2 两行都没有足够早的季度可比，应为 None。"""
    df = _daily_df()
    result = _quarterly_shift_compute(
        df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
        anchor_shift=1,
    )
    aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
    assert aaa["result"].to_list() == [None, None, pytest.approx(0.1)]
    bbb = result.filter(pl.col("instrument_id") == "BBB")
    assert bbb["result"].to_list() == [None]  # 只有 1 季，anchor_shift=1 后更没有可比对象


def test_fund_revenue_qoq_passthrough():
    df = _daily_df()
    res = get_factor("fund_revenue_qoq").compute_full(df)
    aaa = res.values.filter(pl.col("instrument_id") == "AAA").sort("date")
    row = aaa.filter(pl.col("date") == date(2020, 8, 1)).row(0, named=True)
    assert row["fund_revenue_qoq"] == pytest.approx(0.1)


def test_fund_revenue_qoq_prev_passthrough():
    df = _daily_df()
    res = get_factor("fund_revenue_qoq_prev").compute_full(df)
    aaa = res.values.filter(pl.col("instrument_id") == "AAA").sort("date")
    row = aaa.filter(pl.col("date") == date(2020, 11, 1)).row(0, named=True)
    assert row["fund_revenue_qoq_prev"] == pytest.approx(0.1)  # Q2 vs Q1 = (110-100)/100


def test_fund_net_income_qoq_missing_column_yields_null():
    df = _daily_df()  # 没有 net_income 列
    res = get_factor("fund_net_income_qoq").compute_full(df)
    assert res.values["fund_net_income_qoq"].null_count() == df.height


def _cagr_df() -> pl.DataFrame:
    """DDD：13 个季度末，营收从 100 按年化 10% 复利增长（12 季度=3年间隔）。"""
    rows = []
    rev = 100.0
    ends = [
        date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
        date(2021, 3, 31), date(2021, 6, 30), date(2021, 9, 30), date(2021, 12, 31),
        date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31),
        date(2023, 3, 31),
    ]
    for end in ends:
        rows.append({"instrument_id": "DDD", "date": end, "end_date": end, "total_revenue": rev})
        rev *= 1.1 ** 0.25
    return pl.DataFrame(rows).sort("date")


def test_fund_revenue_cagr_3y_matches_expected_rate():
    df = _cagr_df()
    res = get_factor("fund_revenue_cagr_3y").compute_full(df)
    last = res.values.sort("date").tail(1).row(0, named=True)
    assert last["fund_revenue_cagr_3y"] == pytest.approx(0.10, abs=0.01)


def test_fund_revenue_cagr_3y_missing_column_yields_null():
    df = _daily_df()  # 没有跨 3 年的数据，也没有额外列缺失场景在这里测；用 _daily_df 只验证接口不炸
    res = get_factor("fund_revenue_cagr_3y").compute_full(df)
    assert res.values["fund_revenue_cagr_3y"].null_count() == df.height


def _roe_df() -> pl.DataFrame:
    """FFF：5 个季度 ROE 依次 10, 11, 12, 13, 18（第5季相对第1季 +8）。"""
    ends = [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30),
            date(2020, 12, 31), date(2021, 3, 31)]
    roes = [10.0, 11.0, 12.0, 13.0, 18.0]
    rows = [{"instrument_id": "FFF", "date": e, "end_date": e, "roe": r}
            for e, r in zip(ends, roes, strict=True)]
    return pl.DataFrame(rows).sort("date")


def test_fund_roe_trend_4q_is_point_diff_not_ratio():
    df = _roe_df()
    res = get_factor("fund_roe_trend_4q").compute_full(df)
    last = res.values.sort("date").tail(1).row(0, named=True)
    assert last["fund_roe_trend_4q"] == pytest.approx(8.0)  # 18 - 10，不是比率


def test_fund_roe_trend_4q_missing_column_yields_null():
    df = _daily_df()  # 没有 roe 列
    res = get_factor("fund_roe_trend_4q").compute_full(df)
    assert res.values["fund_roe_trend_4q"].null_count() == df.height
