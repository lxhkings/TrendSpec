"""
Price-based factors for TrendSpec.

Factors:
- MomentumFactor: N-day returns (percentage change)
- MomentumRankFactor: Cross-sectional momentum rank

These factors use price data (open, high, low, close) to compute
momentum-related metrics.
"""

from typing import Any, ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("price_momentum")
class MomentumFactor(Factor):
    """
    Price momentum factor - N-day percentage returns.

    Computes the percentage return over a specified period.
    This is a fundamental momentum factor used in trend-following strategies.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (momentum)

    Parameters:
        period: Number of days to compute momentum over (default: 10)

    Example:
        >>> factor = MomentumFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains momentum_20 column with percentage returns
    """

    name: ClassVar[str] = "price_momentum"
    description: ClassVar[str] = "Price momentum - percentage change over N days"
    category: ClassVar[str] = "momentum"

    def __init__(self, period: int = 10) -> None:
        """
        Initialize momentum factor.

        Args:
            period: Number of days to compute momentum over
        """
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute momentum expression.

        Returns percentage change over the specified period.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for momentum column
        """
        period = self.params.get("period", 10)
        # Sort by date to ensure correct shift behavior
        return (
            (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
            * 100
        )

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute momentum for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed momentum values
        """
        df_sorted = df.sort("date")
        period = self.params.get("period", 10)
        col_name = f"momentum_{period}"

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


@register("momentum_rank")
class MomentumRankFactor(Factor):
    """
    Cross-sectional momentum rank factor.

    Ranks stocks by momentum at each date, returning percentile (0-1).
    Higher percentile = higher momentum relative to peers.

    This is useful for selecting the strongest/weakest performers
    in cross-sectional strategies.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (momentum)

    Parameters:
        period: Number of days to compute momentum over (default: 10)
        ascending: Rank in ascending order (default: False, higher momentum = higher rank)

    Example:
        >>> factor = MomentumRankFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains momentum_rank_20 column with percentile values (0-1)
    """

    name: ClassVar[str] = "momentum_rank"
    description: ClassVar[str] = "Cross-sectional momentum rank - percentile ranking"
    category: ClassVar[str] = "momentum"

    def __init__(self, period: int = 10, ascending: bool = False) -> None:
        """
        Initialize momentum rank factor.

        Args:
            period: Number of days to compute momentum over
            ascending: If True, lower momentum gets higher rank percentile
        """
        self.params = {"period": period, "ascending": ascending}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute momentum rank expression.

        Note: This factor requires cross-sectional computation,
        so the compute() method returns the raw momentum.
        Ranking is done in compute_full() across all instruments.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for momentum (ranking done separately)
        """
        period = self.params.get("period", 10)
        return (
            (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
            * 100
        )

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute momentum rank for entire DataFrame.

        Computes momentum first, then ranks within each date.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed rank values
        """
        period = self.params.get("period", 10)
        ascending = self.params.get("ascending", False)
        col_name = f"momentum_rank_{period}"

        # First compute momentum
        df_sorted = df.sort("date")
        momentum_col = f"_momentum_{period}"

        df_with_momentum = df_sorted.with_columns(
            (
                (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
                * 100
            ).alias(momentum_col)
        )

        # Rank within each date
        df_ranked = df_with_momentum.with_columns(
            (
                pl.col(momentum_col)
                .rank(method="average")
                .over("date")
                / pl.len().over("date")
            ).alias(col_name)
        )

        if ascending:
            # Invert the rank if ascending
            df_ranked = df_ranked.with_columns(
                (1 - pl.col(col_name)).alias(col_name)
            )

        result_df = df_ranked.select(["instrument_id", "date", col_name])

        return FactorResult(
            values=result_df,
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )