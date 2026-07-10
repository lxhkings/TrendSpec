"""Cash-flow-quality fundamental factors: pass-through of precomputed ratios."""

from trendspec.factors.fundamental.quality import _ColumnFactor
from trendspec.factors.registry import register


@register("fund_ocf_to_debt")
class FundOcfToDebt(_ColumnFactor):
    description = "Operating cash flow to total debt (precomputed, quarterly, PIT)"
    column = "ocf_to_debt"


@register("fund_ocf_to_shortdebt")
class FundOcfToShortdebt(_ColumnFactor):
    description = "Operating cash flow to short-term debt (precomputed, quarterly, PIT)"
    column = "ocf_to_shortdebt"


@register("fund_q_ocf_to_sales")
class FundQOcfToSales(_ColumnFactor):
    description = "Quarterly operating cash flow to revenue (precomputed, PIT)"
    column = "q_ocf_to_sales"


@register("fund_fcff")
class FundFcff(_ColumnFactor):
    description = "Free cash flow to firm, absolute (precomputed, quarterly, PIT)"
    column = "fcff"
