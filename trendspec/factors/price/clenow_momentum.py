"""Clenow momentum factor for TrendSpec.

Wraps `trendspec.strategy.indicators.clenow_score` (annualized exponential
regression slope x R^2) as a factor_combo-compatible Factor, so factor_combo
specs can rank on trend quality instead of raw N-day return.
"""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult
from trendspec.factors.registry import register
from trendspec.strategy.indicators import clenow_score


@register("clenow_momentum")
class ClenowMomentumFactor(Factor):
    """Annualized regression-slope x R^2 momentum (Clenow).

    Only smooth, consistent uptrends score high — a single-day gap or a
    choppy round-trip both get penalized via R^2, unlike raw N-day return.
    """

    name: ClassVar[str] = "clenow_momentum"
    description: ClassVar[str] = "Annualized exponential regression slope x R^2"
    category: ClassVar[str] = "momentum"

    def __init__(self, period: int = 90) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        raise NotImplementedError("ClenowMomentumFactor overrides compute_full instead")

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        period = self.params.get("period", 90)
        col_name = f"clenow_momentum_{period}"

        scored = clenow_score(df.sort(["instrument_id", "date"]), period=period)
        result_df = scored.select(
            ["instrument_id", "date", pl.col(f"CLENOW_SCORE_{period}").alias(col_name)]
        )

        return FactorResult(
            values=result_df,
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )
