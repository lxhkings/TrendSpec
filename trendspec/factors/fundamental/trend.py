"""CN 成长趋势类基本面因子：环比(QoQ)、多年复合增速(CAGR)、ROE 趋势。

跨期计算型因子（tushare 原始数据没有现成字段），跟 quality.py/growth.py 的
_ColumnFactor 直通模式不同——需要从日频 df 里按 end_date 变化点还原出真正的
季度序列再做 shift 对比。三个模块级函数（_quarterly_series /
_quarterly_shift_compute / _asof_join_quarterly_result）与
_QuarterlyShiftFactor 基类被本文件全部 Factor 共用，不要在 Factor.compute()
里各自重新实现一遍这套逻辑。
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
    anchor_shift: int = 0,
) -> pl.DataFrame:
    """季度序列上 shift(n) 对比，返回 (instrument_id, date, result)。

    mode="ratio": (cur-base)/|base|；mode="cagr": (cur/base)**(1/cagr_years)-1
    （要求 base>0，cagr_years 必填）；mode="diff": cur-base（原始差值）。

    anchor_shift: 把"当前季度"锚点往前挪 N 季（默认 0 = 最新季度）。
    anchor_shift=1 时比较的是 t-1 vs t-1-n，用来算"上一季度的环比"而不是
    "最新季度的环比"。默认值 0 与旧行为完全一致。

    end_date 与 shift(n) 那行 end_date 的月份差不落在 [gap_min_months,
    gap_max_months] 内时判 null——防止漏报导致 shift(n) 实际跳过了缺的那季，
    比出一个语义错位的"环比"/"同比"。
    """
    q = _quarterly_series(df, value_col)
    empty = pl.DataFrame(schema={"instrument_id": pl.Utf8, "date": pl.Date, "result": pl.Float64})
    if q.is_empty():
        return empty

    cur = pl.col(value_col).shift(anchor_shift).over("instrument_id")
    base = pl.col(value_col).shift(anchor_shift + n).over("instrument_id")
    cur_end_date = pl.col("end_date").shift(anchor_shift).over("instrument_id")
    base_end_date = pl.col("end_date").shift(anchor_shift + n).over("instrument_id")
    gap_months = (
        (cur_end_date.dt.year() - base_end_date.dt.year()) * 12
        + (cur_end_date.dt.month() - base_end_date.dt.month())
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


class _QuarterlyShiftFactor(Factor):
    """跨季 shift 因子共用壳：ClassVar 描述列与 gap，compute 只写一次。"""

    category: ClassVar[str] = "fundamental"
    value_col: ClassVar[str] = ""
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    cagr_years: ClassVar[float | None] = None
    anchor_shift: ClassVar[int] = 0

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or self.value_col not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df,
            self.value_col,
            n=self.n,
            gap_min_months=self.gap_min_months,
            gap_max_months=self.gap_max_months,
            mode=self.mode,
            cagr_years=self.cagr_years,
            anchor_shift=self.anchor_shift,
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)


@register("fund_revenue_qoq")
class FundRevenueQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Revenue QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "total_revenue"


@register("fund_revenue_qoq_prev")
class FundRevenueQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Revenue QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "total_revenue"
    anchor_shift: ClassVar[int] = 1


@register("fund_net_income_qoq")
class FundNetIncomeQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Net income QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "net_income"


@register("fund_net_income_qoq_prev")
class FundNetIncomeQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Net income QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "net_income"
    anchor_shift: ClassVar[int] = 1


@register("fund_revenue_cagr_3y")
class FundRevenueCagr3Y(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Revenue 3-year CAGR (12 quarters back, PIT)"
    value_col: ClassVar[str] = "total_revenue"
    n: ClassVar[int] = 12
    gap_min_months: ClassVar[int] = 34
    gap_max_months: ClassVar[int] = 38
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "cagr"
    cagr_years: ClassVar[float | None] = 3.0


@register("fund_roe_trend_4q")
class FundRoeTrend4Q(_QuarterlyShiftFactor):
    description: ClassVar[str] = "ROE change vs 4 quarters ago, absolute points not ratio (PIT)"
    value_col: ClassVar[str] = "roe"
    n: ClassVar[int] = 4
    gap_min_months: ClassVar[int] = 10
    gap_max_months: ClassVar[int] = 14
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "diff"
