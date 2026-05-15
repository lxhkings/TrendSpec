"""
Sector momentum factors for TrendSpec.

Factors:
- SectorMomentumFactor: Sector overall N-day returns (aggregate all stocks in sector)

These factors depend on data/sectors.py for PIT sector lookup.
"""

from datetime import date as DateType
from typing import Any, ClassVar

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index, SectorIndex, get_all_sectors
from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("sector_momentum")
class SectorMomentumFactor(Factor):
    """
    Sector momentum factor - aggregate N-day returns for all stocks in sector.

    Computes the overall sector momentum by aggregating returns of all stocks
    in that sector at a given date. Uses PIT sector lookup to get the correct
    sector assignments for each date.

    This is useful for:
    - Sector rotation strategies
    - Identifying strong/weak sectors
    - Sector-relative stock selection

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (sector)

    Parameters:
        period: Number of days to compute momentum over (default: 10)
        market: Market for sector lookup (default: Market.CN_A)
        aggregation: How to aggregate stock returns - "mean" or "median" (default: "mean")

    Example:
        >>> factor = SectorMomentumFactor(period=10, market=Market.CN_A)
        >>> result = factor.compute_full(df)
        >>> # Result contains sector_momentum_10 column
        >>> # Each stock has the momentum of its sector at that date
    """

    name: ClassVar[str] = "sector_momentum"
    description: ClassVar[str] = "Sector momentum - aggregate returns for sector"
    category: ClassVar[str] = "sector"

    def __init__(
        self,
        period: int = 10,
        market: Market = Market.CN_A,
        aggregation: str = "mean",
        root: str | None = None,
    ) -> None:
        """
        Initialize sector momentum factor.

        Args:
            period: Number of days to compute momentum over
            market: Market for sector lookup
            aggregation: How to aggregate stock returns
            root: Root directory for data_lake
        """
        self.params = {
            "period": period,
            "market": market,
            "aggregation": aggregation,
            "root": root,
        }

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute stock momentum expression (sector aggregation done in compute_full).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Polars expression for individual stock momentum
        """
        period = self.params.get("period", 10)
        return (
            (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
            * 100
        )

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute sector momentum for entire DataFrame.

        First computes individual stock momentum, then aggregates by sector
        using PIT sector lookup.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed sector momentum values
        """
        period = self.params.get("period", 10)
        market = self.params.get("market", Market.CN_A)
        aggregation = self.params.get("aggregation", "mean")
        root = self.params.get("root")
        col_name = f"sector_momentum_{period}"

        # Get sector index
        try:
            sector_index = get_sector_index(market, root)
        except Exception:
            # If sector index not available, return empty result with same structure as input
            df_sorted = df.sort("date")
            return FactorResult(
                values=df_sorted.select(["instrument_id", "date"]).with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(col_name)
                ),
                name=col_name,
                metadata={
                    "description": self.description,
                    "category": self.category,
                    "params": self.params,
                    "error": "Sector index not available",
                },
            )

        # First compute individual stock momentum
        df_sorted = df.sort("date")
        momentum_col = f"_momentum_{period}"

        df_with_momentum = df_sorted.with_columns(
            (
                (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
                * 100
            ).alias(momentum_col)
        )

        # For each date, compute sector momentum by aggregating stock momentum
        # This requires iterating through dates to get PIT sector assignments
        dates = df_sorted["date"].unique().sort()

        # Build sector momentum lookup: (date, sector) -> momentum
        sector_momentum_values: list[dict] = []

        for as_of_date in dates:
            # Get data for this date
            date_data = df_with_momentum.filter(pl.col("date") == as_of_date)

            # Get all sector assignments at this date
            sectors_at_date = sector_index.all_sectors_at_date(as_of_date)

            for sector_code, instruments in sectors_at_date.items():
                # Filter data for instruments in this sector
                sector_data = date_data.filter(pl.col("instrument_id").is_in(instruments))

                if sector_data.is_empty():
                    continue

                # Aggregate momentum for this sector
                momentum_values = sector_data[momentum_col].drop_nulls()

                if momentum_values.is_empty():
                    continue

                if aggregation == "median":
                    agg_momentum = momentum_values.median()
                else:
                    agg_momentum = momentum_values.mean()

                # Store for each instrument in sector
                for instrument_id in instruments:
                    sector_momentum_values.append({
                        "instrument_id": instrument_id,
                        "date": as_of_date,
                        col_name: agg_momentum,
                        "sector": sector_code,
                    })

        # Create DataFrame from sector momentum values
        if sector_momentum_values:
            sector_df = pl.DataFrame(sector_momentum_values)
        else:
            # Empty result
            sector_df = pl.DataFrame({
                "instrument_id": df_sorted["instrument_id"],
                "date": df_sorted["date"],
                col_name: [None] * len(df_sorted),
                "sector": [None] * len(df_sorted),
            })

        # Join back to original data
        result_df = df_sorted.join(
            sector_df.select(["instrument_id", "date", col_name]),
            on=["instrument_id", "date"],
            how="left",
        ).select(["instrument_id", "date", col_name])

        return FactorResult(
            values=result_df,
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )