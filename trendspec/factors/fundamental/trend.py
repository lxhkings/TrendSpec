"""CN 成长趋势类基本面因子：环比(QoQ)、多年复合增速(CAGR)、ROE 趋势。

跨期计算型因子（tushare 原始数据没有现成字段），跟 quality.py/growth.py 的
_ColumnFactor 直通模式不同——需要从日频 df 里按 end_date 变化点还原出真正的
季度序列再做 shift 对比。三个模块级函数（_quarterly_series /
_quarterly_shift_compute / _asof_join_quarterly_result）被本文件全部 4 个
Factor 共用，不要在 Factor.compute() 里各自重新实现一遍这套逻辑。
"""

from typing import ClassVar, Literal

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.registry import register


def _quarterly_series(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """从日频 df（季度值已被前向填充到每天重复）按 end_date 变化点提取季度序列。

    返回 (instrument_id, date, end_date, value_col)，按 (instrument_id, date)
    排序，每个 instrument_id 每季度恰好一行（取该季度第一次出现的那一天，即
    公告生效日）。end_date 或 value_col 不在 df 里时返回同 schema 的空表。
    """
    empty_schema = {
        "instrument_id": pl.Utf8, "date": pl.Date,
        "end_date": pl.Date, value_col: pl.Float64,
    }
    if "end_date" not in df.columns or value_col not in df.columns:
        return pl.DataFrame(schema=empty_schema)

    d = df.select(["instrument_id", "date", "end_date", value_col]).sort(
        ["instrument_id", "date"]
    )
    is_change = (
        pl.col("end_date") != pl.col("end_date").shift(1).over("instrument_id")
    ).fill_null(True)
    return d.filter(is_change)


def _asof_join_quarterly_result(
    df: pl.DataFrame, quarterly_result: pl.DataFrame
) -> pl.Series:
    """把 quarterly_result (instrument_id, date, result) 前向广播回日频 df 的
    每一行（asof backward，同 data/fundamentals.py 的 merge 语义），返回的
    Series 长度、行顺序与 df 完全一致（可直接喂给 with_columns）。
    """
    indexed = df.with_row_index("_row_idx").select(["_row_idx", "instrument_id", "date"])
    joined = indexed.sort(["instrument_id", "date"]).join_asof(
        quarterly_result.sort(["instrument_id", "date"]),
        on="date", by="instrument_id", strategy="backward",
    ).sort("_row_idx")
    return joined["result"]
