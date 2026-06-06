"""Growth fundamental factors: year-over-year deltas (precomputed at ingest)."""

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
