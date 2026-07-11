"""Size and liquidity fundamental factors: pass-through of daily valuation-snapshot columns."""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.registry import register


class _ColumnFactor(Factor):
    """Base for factors that pass through a single valuation-snapshot column.

    Kept separate from fundamental.quality._ColumnFactor: this module's
    source columns refresh daily (valuation dataset), quality's refresh
    quarterly (fundamentals dataset) — independent even though the
    pass-through logic is currently identical.
    """

    column: ClassVar[str] = ""
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if self.column not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return pl.col(self.column)


@register("fund_total_mv")
class FundTotalMv(_ColumnFactor):
    description = "Total market cap (PIT, from daily valuation snapshot)"
    column = "total_mv"


@register("fund_circ_mv")
class FundCircMv(_ColumnFactor):
    description = "Circulating (free-float) market cap (PIT)"
    column = "circ_mv"


@register("fund_turnover_rate")
class FundTurnoverRate(_ColumnFactor):
    description = "Turnover rate, daily (PIT)"
    column = "turnover_rate"
