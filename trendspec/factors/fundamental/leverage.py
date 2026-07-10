"""Leverage/solvency fundamental factors: pass-through of precomputed ratios."""

from trendspec.factors.fundamental.quality import _ColumnFactor
from trendspec.factors.registry import register


@register("fund_debt_to_assets")
class FundDebtToAssets(_ColumnFactor):
    description = "Debt-to-assets ratio (precomputed, quarterly, PIT)"
    column = "debt_to_assets"


@register("fund_current_ratio")
class FundCurrentRatio(_ColumnFactor):
    description = "Current ratio (precomputed, quarterly, PIT)"
    column = "current_ratio"


@register("fund_quick_ratio")
class FundQuickRatio(_ColumnFactor):
    description = "Quick ratio (precomputed, quarterly, PIT)"
    column = "quick_ratio"


@register("fund_debt_to_eqt")
class FundDebtToEqt(_ColumnFactor):
    description = "Debt-to-equity ratio (precomputed, quarterly, PIT)"
    column = "debt_to_eqt"
