"""
Volatility factors for TrendSpec.

Factors:
- VolatilityFactor: N-day rolling standard deviation of returns
- VolatilityRankFactor: Cross-sectional volatility rank

These factors measure price volatility, useful for risk management
and volatility-adjusted position sizing.
"""

from typing import Any, ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("price_volatility")
class VolatilityFactor(Factor):
    """
    Volatility factor - N-day rolling standard deviation of returns.

    Computes the annualized rolling standard deviation of daily returns.
    This is essential for risk management and volatility-adjusted strategies.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (volatility)

    Parameters:
        period: Number of days to compute volatility over (default: 20)

    Example:
        >>> factor = VolatilityFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains volatility_20 column with annualized volatility
    """

    name: ClassVar[str] = "price_volatility"
    description: ClassVar[str] = "Rolling volatility - annualized std of returns"
    category: ClassVar[str] = "volatility"

    def __init__(self, period: int = 20) -> None:
        """
        Initialize volatility factor.

        Args:
            period: Number of days to compute volatility over
        """
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute volatility expression.

        Returns annualized rolling standard deviation of daily returns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for volatility column
        """
        period = self.params.get("period", 20)

        # Calculate daily returns first
        returns_expr = (
            pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1
        )

        # Rolling std, annualized (252 trading days for most markets)
        return (
            returns_expr
            .rolling_std(window_size=period)
            .over("instrument_id")
            * (252 ** 0.5)
        )

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute volatility for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed volatility values
        """
        df_sorted = df.sort("date")
        period = self.params.get("period", 20)
        col_name = f"volatility_{period}"

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


@register("volatility_rank")
class VolatilityRankFactor(Factor):
    """
    Cross-sectional volatility rank factor.

    Ranks stocks by volatility at each date, returning percentile (0-1).
    Higher percentile = higher volatility relative to peers.

    This is useful for filtering high/low volatility stocks or
    volatility-adjusted position sizing.

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (volatility)

    Parameters:
        period: Number of days to compute volatility over (default: 20)
        ascending: Rank in ascending order (default: False, higher volatility = higher rank)

    Example:
        >>> factor = VolatilityRankFactor(period=20)
        >>> result = factor.compute_full(df)
        >>> # Result contains volatility_rank_20 column with percentile values (0-1)
    """

    name: ClassVar[str] = "volatility_rank"
    description: ClassVar[str] = "Cross-sectional volatility rank - percentile ranking"
    category: ClassVar[str] = "volatility"

    def __init__(self, period: int = 20, ascending: bool = False) -> None:
        """
        Initialize volatility rank factor.

        Args:
            period: Number of days to compute volatility over
            ascending: If True, lower volatility gets higher rank percentile
        """
        self.params = {"period": period, "ascending": ascending}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute volatility expression (raw volatility, ranking done separately).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for volatility
        """
        period = self.params.get("period", 20)

        returns_expr = (
            pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1
        )

        return (
            returns_expr
            .rolling_std(window_size=period)
            .over("instrument_id")
            * (252 ** 0.5)
        )

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute volatility rank for entire DataFrame.

        Computes volatility first, then ranks within each date.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed rank values
        """
        period = self.params.get("period", 20)
        ascending = self.params.get("ascending", False)
        col_name = f"volatility_rank_{period}"

        # First compute volatility
        df_sorted = df.sort("date")
        vol_col = f"_volatility_{period}"

        # Calculate returns
        returns_expr = (
            pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1
        )

        df_with_vol = df_sorted.with_columns(
            (
                returns_expr
                .rolling_std(window_size=period)
                .over("instrument_id")
                * (252 ** 0.5)
            ).alias(vol_col)
        )

        # Rank within each date
        df_ranked = df_with_vol.with_columns(
            (
                pl.col(vol_col)
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