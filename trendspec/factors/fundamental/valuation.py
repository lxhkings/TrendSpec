"""Valuation fundamental factors."""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor
from trendspec.factors.registry import register


@register("fund_pe_ttm")
class FundPETTM(Factor):
    """Trailing PE. Prefers a directly-sourced "pe_ttm" column (e.g. CN
    Tushare daily_basic, official TTM calc) when present; otherwise falls
    back to close / eps_ttm (US, derived at ingest from quarterly EPS).
    Null when no source is available or the computed ratio is <= 0.
    """

    description: ClassVar[str] = "Trailing PE (PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if "pe_ttm" in df.columns:
            return (
                pl.when(pl.col("pe_ttm") > 0)
                .then(pl.col("pe_ttm"))
                .otherwise(None)
            )
        if "eps_ttm" not in df.columns or "close" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return (
            pl.when(pl.col("eps_ttm") > 0)
            .then(pl.col("close") / pl.col("eps_ttm"))
            .otherwise(None)
        )


@register("fund_pb")
class FundPB(Factor):
    """Price-to-book (precomputed, e.g. CN Tushare daily_basic "pb"). Null
    when the column is absent (e.g. US, not currently sourced) or <= 0.
    """

    description: ClassVar[str] = "Price-to-book (PIT)"
    category: ClassVar[str] = "fundamental"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        if "pb" not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        return (
            pl.when(pl.col("pb") > 0)
            .then(pl.col("pb"))
            .otherwise(None)
        )
