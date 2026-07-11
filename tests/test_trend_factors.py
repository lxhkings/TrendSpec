from datetime import date

import polars as pl
import pytest

from trendspec.factors.fundamental.trend import _quarterly_series, _asof_join_quarterly_result


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
