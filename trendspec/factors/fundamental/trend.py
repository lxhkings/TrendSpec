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


def _quarterly_shift_compute(
    df: pl.DataFrame,
    value_col: str,
    n: int,
    gap_min_months: int,
    gap_max_months: int,
    mode: Literal["ratio", "cagr", "diff"],
    cagr_years: float | None = None,
) -> pl.DataFrame:
    """季度序列上 shift(n) 对比，返回 (instrument_id, date, result)。

    mode="ratio": (cur-base)/|base|；mode="cagr": (cur/base)**(1/cagr_years)-1
    （要求 base>0，cagr_years 必填）；mode="diff": cur-base（原始差值）。

    end_date 与 shift(n) 那行 end_date 的月份差不落在 [gap_min_months,
    gap_max_months] 内时判 null——防止漏报导致 shift(n) 实际跳过了缺的那季，
    比出一个语义错位的"环比"/"同比"。
    """
    q = _quarterly_series(df, value_col)
    empty = pl.DataFrame(schema={"instrument_id": pl.Utf8, "date": pl.Date, "result": pl.Float64})
    if q.is_empty():
        return empty

    cur = pl.col(value_col)
    base = pl.col(value_col).shift(n).over("instrument_id")
    base_end_date = pl.col("end_date").shift(n).over("instrument_id")
    gap_months = (
        (pl.col("end_date").dt.year() - base_end_date.dt.year()) * 12
        + (pl.col("end_date").dt.month() - base_end_date.dt.month())
    )
    gap_ok = gap_months.is_between(gap_min_months, gap_max_months)

    if mode == "cagr":
        raw = (cur / base) ** (1.0 / cagr_years) - 1.0
        valid = gap_ok & base.is_not_null() & (base > 0)
    elif mode == "ratio":
        raw = (cur - base) / base.abs()
        valid = gap_ok & base.is_not_null() & (base != 0)
    else:
        raw = cur - base
        valid = gap_ok & base.is_not_null()

    q = q.with_columns(pl.when(valid).then(raw).otherwise(None).alias("result"))
    return q.select(["instrument_id", "date", "result"])


@register("fund_revenue_qoq")
class FundRevenueQoQ(Factor):
    description: ClassVar[str] = "Revenue QoQ growth (quarter vs immediately prior quarter, PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or "total_revenue" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df, "total_revenue", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)


@register("fund_net_income_qoq")
class FundNetIncomeQoQ(Factor):
    description: ClassVar[str] = "Net income QoQ growth (quarter vs immediately prior quarter, PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or "net_income" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df, "net_income", n=1, gap_min_months=2, gap_max_months=4, mode="ratio",
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)


@register("fund_revenue_cagr_3y")
class FundRevenueCagr3Y(Factor):
    description: ClassVar[str] = "Revenue 3-year CAGR (12 quarters back, PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or "total_revenue" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df, "total_revenue", n=12, gap_min_months=34, gap_max_months=38,
            mode="cagr", cagr_years=3.0,
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)


@register("fund_roe_trend_4q")
class FundRoeTrend4Q(Factor):
    description: ClassVar[str] = "ROE change vs 4 quarters ago, absolute points not ratio (PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or "roe" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df, "roe", n=4, gap_min_months=10, gap_max_months=14, mode="diff",
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)
