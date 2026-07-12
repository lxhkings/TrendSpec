"""Growth fundamental factors: year-over-year deltas (precomputed at ingest)."""

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.fundamental.quality import _ColumnFactor
from trendspec.factors.registry import register


@register("fund_revenue_yoy")
class FundRevenueYoY(_ColumnFactor):
    description = "Revenue YoY growth (quarterly vs same quarter prior year, PIT)"
    column = "revenue_yoy"


@register("fund_net_income_yoy")
class FundNetIncomeYoY(_ColumnFactor):
    description = "Net income YoY growth (quarterly vs same quarter prior year, PIT)"
    column = "net_income_yoy"


@register("fund_revenue_yoy_band")
class FundRevenueYoYBand(Factor):
    """增速适中度：-|revenue_yoy - center|，越贴近理想增速带分越高。

    来自《投资分析框架》：收入增速 15-20% 理想，<10% 动力不足，
    >40-50% 难持续——单调排序 yoy 会奖励不可持续的极端增速，此因子
    以距 center 的绝对偏差取负修正。center 单位跟随 revenue_yoy 列：
    CN（tushare tr_yoy）为百分数，默认 17.5 即 17.5%。
    """

    description = "Revenue YoY closeness to ideal growth band (higher = closer to center)"
    category = "fundamental"

    def __init__(self, center: float = 17.5) -> None:
        self.params = {"center": center}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if "revenue_yoy" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return -(pl.col("revenue_yoy") - self.params["center"]).abs()
