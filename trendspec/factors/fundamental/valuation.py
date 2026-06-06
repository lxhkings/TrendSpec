"""Valuation fundamental factors. PE only (shares outstanding unavailable)."""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.registry import register


@register("fund_pe_ttm")
class FundPETTM(Factor):
    """Trailing PE = close / eps_ttm. Null when eps_ttm <= 0 or columns absent."""

    description: ClassVar[str] = "Trailing PE (close / TTM diluted EPS, PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if "eps_ttm" not in df.columns or "close" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return (
            pl.when(pl.col("eps_ttm") > 0)
            .then(pl.col("close") / pl.col("eps_ttm"))
            .otherwise(None)
        )
