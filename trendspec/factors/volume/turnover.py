"""
Volume-based factors for TrendSpec.

Factors:
- TurnoverFactor: Volume / shares outstanding (turnover rate)
- VolumeRatioFactor: Volume / average volume

These factors use volume data to measure trading activity and liquidity.
"""

from typing import Any, ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("turnover_rate")
class TurnoverFactor(Factor):
    """
    Turnover rate factor - Volume / shares outstanding.

    Computes the turnover rate which measures trading activity relative
    to the total shares outstanding. Higher turnover indicates more
    active trading.

    This factor requires shares outstanding data to be present in the DataFrame.
    If not available, it returns a placeholder value.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (volume)

    Parameters:
        None (uses volume and shares_outstanding columns from data)

    Example:
        >>> factor = TurnoverFactor()
        >>> result = factor.compute_full(df)
        >>> # Result contains turnover_rate column
        >>> # Requires 'shares_outstanding' column in data
    """

    name: ClassVar[str] = "turnover_rate"
    description: ClassVar[str] = "Turnover rate - volume / shares outstanding"
    category: ClassVar[str] = "volume"

    def __init__(self) -> None:
        """Initialize turnover factor."""
        self.params = {}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute turnover rate expression.

        Args:
            df: DataFrame with volume and shares_outstanding columns

        Returns:
            Polars expression for turnover rate column
        """
        # Check if shares_outstanding column exists
        if "shares_outstanding" in df.columns:
            return pl.col("volume") / pl.col("shares_outstanding")
        # Placeholder if shares outstanding data not available
        return pl.lit(None, dtype=pl.Float64)

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute turnover rate for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed turnover values
        """
        df_sorted = df.sort("date")
        col_name = "turnover_rate"

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
                "requires_shares_outstanding": True,
            },
        )


@register("volume_ratio")
class VolumeRatioFactor(Factor):
    """
    Volume ratio factor - Volume / average volume.

    Computes the ratio of current volume to average volume over a period.
    Higher ratio indicates above-average trading activity.

    This is useful for detecting unusual trading activity or
    confirming price moves with volume.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (volume)

    Parameters:
        period: Number of days to compute average volume over (default: 20)

    Example:
        >>> factor = VolumeRatioFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains volume_ratio_20 column
        >>> # Values > 1 indicate above-average volume
    """

    name: ClassVar[str] = "volume_ratio"
    description: ClassVar[str] = "Volume ratio - volume / average volume"
    category: ClassVar[str] = "volume"

    def __init__(self, period: int = 20) -> None:
        """
        Initialize volume ratio factor.

        Args:
            period: Number of days to compute average volume over
        """
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute volume ratio expression.

        Args:
            df: DataFrame with volume column

        Returns:
            Polars expression for volume ratio column
        """
        period = self.params.get("period", 20)

        # Compute rolling average volume
        avg_volume = pl.col("volume").rolling_mean(window_size=period).over("instrument_id")

        return pl.col("volume") / avg_volume

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute volume ratio for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed volume ratio values
        """
        df_sorted = df.sort("date")
        period = self.params.get("period", 20)
        col_name = f"volume_ratio_{period}"

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