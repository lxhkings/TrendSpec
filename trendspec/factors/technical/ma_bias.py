"""
Technical indicator factors for TrendSpec.

Factors:
- MABiasFactor: Price / MA - 1 (moving average bias)

These factors use technical indicators derived from price data.
"""

from typing import Any, ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("ma_bias")
class MABiasFactor(Factor):
    """
    Moving average bias factor - Price / MA - 1.

    Measures how far the current price is from its moving average.
    Positive values indicate price above MA, negative indicates below.

    This is useful for:
    - Mean reversion strategies (extreme deviations)
    - Trend confirmation (price above MA = uptrend)
    - Entry timing (reversion to MA)

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (technical)

    Parameters:
        period: Moving average period (default: 20)
        ma_type: Moving average type - "SMA" or "EMA" (default: "SMA")

    Example:
        >>> factor = MABiasFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains ma_bias_20 column
        >>> # Values > 0: price above MA
        >>> # Values < 0: price below MA
    """

    name: ClassVar[str] = "ma_bias"
    description: ClassVar[str] = "Moving average bias - (price / MA - 1)"
    category: ClassVar[str] = "technical"

    def __init__(self, period: int = 20, ma_type: str = "SMA") -> None:
        """
        Initialize MA bias factor.

        Args:
            period: Moving average period
            ma_type: Moving average type - "SMA" or "EMA"
        """
        self.params = {"period": period, "ma_type": ma_type}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute MA bias expression.

        Args:
            df: DataFrame with close price

        Returns:
            Polars expression for MA bias column
        """
        period = self.params.get("period", 20)
        ma_type = self.params.get("ma_type", "SMA")

        if ma_type == "EMA":
            # Exponential moving average
            ma = pl.col("close").ewm_mean(half_life=period).over("instrument_id")
        else:
            # Simple moving average
            ma = pl.col("close").rolling_mean(window_size=period).over("instrument_id")

        return pl.col("close") / ma - 1

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute MA bias for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed MA bias values
        """
        df_sorted = df.sort("date")
        period = self.params.get("period", 20)
        ma_type = self.params.get("ma_type", "SMA")
        col_name = f"ma_bias_{period}_{ma_type.lower()}"

        expr = self.compute(df_sorted)
        df_result = df_sorted.with_columns(expr.alias(col_name))

        result_df = df_result.select(["instrument_id", "date", col_name])

        return FactorResult(
            values=result_df,
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )