"""Quality fundamental factors: pass-through of precomputed ratios."""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.registry import register


class _ColumnFactor(Factor):
    """Base for factors that pass through a single fundamental column.

    Returns a null Float64 expr when the column is absent (e.g. fundamentals
    dataset not merged), so factor_combo degrades gracefully instead of raising.
    """

    column: ClassVar[str] = ""
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if self.column not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return pl.col(self.column)


@register("fund_roe")
class FundROE(_ColumnFactor):
    description = "Return on equity (precomputed, quarterly, PIT)"
    column = "roe"


@register("fund_roic")
class FundROIC(_ColumnFactor):
    description = "Return on invested capital (precomputed, quarterly, PIT)"
    column = "roic"


@register("fund_net_margin")
class FundNetMargin(_ColumnFactor):
    description = "Net profit margin (precomputed, quarterly, PIT)"
    column = "net_margin"


@register("fund_op_margin")
class FundOpMargin(_ColumnFactor):
    description = "Operating margin (precomputed, quarterly, PIT)"
    column = "op_margin"
