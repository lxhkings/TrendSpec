"""
Factor abstract base class for TrendSpec.

Factors represent computed values derived from market data.
Examples:
- Momentum factor: Return over N days
- Volatility factor: Rolling std of returns
- Fundamental factor: PE ratio, market cap
- Sector factor: Sector-relative return

Factors are:
- Named (for lookup)
- Computed via Polars expressions (vectorized)
- Cached per-date for fast access

Two modes:
- Cross-sectional: Factor values for all instruments at a date
- Time-series: Factor values for one instrument over time
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl


@dataclass
class FactorResult:
    """
    Result of factor computation.

    Attributes:
        values: DataFrame with factor values (instrument_id, date, factor_name)
        name: Factor name
        metadata: Additional metadata (description, units, etc.)
    """

    values: pl.DataFrame
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def cross_sectional(self, as_of_date: pl.Date) -> pl.DataFrame:
        """
        Get factor values for all instruments at a date.

        Args:
            as_of_date: Date to query

        Returns:
            DataFrame with (instrument_id, factor_value) for that date
        """
        return self.values.filter(pl.col("date") == as_of_date)

    def time_series(self, instrument_id: str) -> pl.DataFrame:
        """
        Get factor values for one instrument over time.

        Args:
            instrument_id: Instrument ID

        Returns:
            DataFrame with (date, factor_value) for that instrument
        """
        return self.values.filter(pl.col("instrument_id") == instrument_id)

    def rank(self, as_of_date: pl.Date, ascending: bool = True) -> pl.DataFrame:
        """
        Rank instruments by factor value at a date.

        Args:
            as_of_date: Date to query
            ascending: Sort order

        Returns:
            DataFrame with (instrument_id, rank) for that date
        """
        return (
            self.cross_sectional(as_of_date)
            .sort(self.name, descending=not ascending)
            .with_columns(pl.arange(1, pl.len() + 1).alias("rank"))
        )


class Factor(ABC):
    """
    Abstract base class for factors.

    Factors are computed values derived from market data.
    They are registered in the factor registry and accessed via ctx.factor().

    Attributes:
        name: Factor name (used for registry lookup and column name)
        description: Human-readable description
        category: Factor category (momentum, volatility, fundamental, etc.)

    Methods to implement:
        compute(df): Return Polars expression for factor computation

    Example:
        >>> class MomentumFactor(Factor):
        ...     name = "momentum_10"
        ...     description = "10-day momentum"
        ...     category = "momentum"
        ...
        ...     def compute(self, df: pl.DataFrame) -> pl.Expr:
        ...         return (
        ...             pl.col("close") - pl.col("close").shift(10)
        ...         ).over("instrument_id")
    """

    name: ClassVar[str] = "base_factor"
    description: ClassVar[str] = "Base factor class"
    category: ClassVar[str] = "general"

    # Instance attributes
    params: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute factor expression.

        Returns a Polars expression that computes the factor.
        The expression is applied over (instrument_id, date) grouping.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for factor column

        Example:
            >>> def compute(self, df):
            ...     period = self.params.get("period", 10)
            ...     return (
            ...         (pl.col("close") / pl.col("close").shift(period) - 1) * 100
            ...     ).over("instrument_id").alias(self.name)
        """
        pass

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute factor for entire DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed values
        """
        df_sorted = df.sort("date")
        expr = self.compute(df_sorted)

        if isinstance(expr, pl.Expr):
            df_result = df_sorted.with_columns(expr.alias(self.name))
        else:
            df_result = df_sorted.with_columns(expr)

        # Select relevant columns
        result_df = df_result.select(["instrument_id", "date", self.name])

        return FactorResult(
            values=result_df,
            name=self.name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )

    def validate_params(self) -> None:
        """Validate factor parameters. Override in subclasses."""
        pass

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate_params()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name}, params={self.params})"


# =============================================================================
# Common Factor Categories
# =============================================================================

FACTOR_CATEGORIES: dict[str, str] = {
    "momentum": "Price momentum factors",
    "volatility": "Volatility factors",
    "volume": "Volume-related factors",
    "trend": "Trend-following factors",
    "fundamental": "Fundamental factors (PE, market cap, etc.)",
    "technical": "Technical indicator factors",
    "sector": "Sector-relative factors",
    "macro": "Macro factors (interest rates, inflation)",
}


# =============================================================================
# Built-in Factor Types
# =============================================================================


class MomentumFactor(Factor):
    """Base class for momentum factors."""

    category = "momentum"

    def __init__(self, period: int = 10) -> None:
        self.params = {"period": period}


class VolatilityFactor(Factor):
    """Base class for volatility factors."""

    category = "volatility"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}


class VolumeFactor(Factor):
    """Base class for volume factors."""

    category = "volume"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}