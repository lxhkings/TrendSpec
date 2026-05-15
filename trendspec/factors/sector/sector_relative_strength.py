"""
Sector relative strength factors for TrendSpec.

Factors:
- SectorRelativeStrengthFactor: Stock N-day returns - Sector N-day returns

Measures excess performance vs sector, useful for identifying
sector outperformers/underperformers.

Depends on data/sectors.py for PIT sector lookup.
"""

from datetime import date as DateType
from typing import Any, ClassVar

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index, SectorIndex
from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register


@register("sector_relative_strength")
class SectorRelativeStrengthFactor(Factor):
    """
    Sector relative strength factor - Stock returns - Sector returns.

    Measures how much a stock outperforms or underperforms its sector.
    Positive values indicate outperformance, negative indicates underperformance.

    Uses PIT sector lookup to ensure correct sector assignments.

    This is useful for:
    - Identifying sector leaders/laggards
    - Stock selection within sectors
    - Factor neutralization (remove sector beta)

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (sector)

    Parameters:
        period: Number of days to compute returns over (default: 10)
        market: Market for sector lookup (default: Market.CN)
        aggregation: How to aggregate sector returns - "mean" or "median" (default: "mean")

    Example:
        >>> factor = SectorRelativeStrengthFactor(period=20, market=Market.CN)
        >>> result = factor.compute_full(df)
        >>> # Result contains sector_relative_strength_20 column
        >>> # Positive: stock outperforming its sector
        >>> # Negative: stock underperforming its sector
    """

    name: ClassVar[str] = "sector_relative_strength"
    description: ClassVar[str] = "Sector relative strength - stock return minus sector return"
    category: ClassVar[str] = "sector"

    def __init__(
        self,
        period: int = 10,
        market: Market = Market.CN,
        aggregation: str = "mean",
        root: str | None = None,
    ) -> None:
        """
        Initialize sector relative strength factor.

        Args:
            period: Number of days to compute returns over
            market: Market for sector lookup
            aggregation: How to aggregate sector returns
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
        Compute stock momentum expression (relative strength computed in compute_full).

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
        Compute sector relative strength for entire DataFrame.

        First computes individual stock momentum, then computes sector momentum,
        then subtracts sector momentum from stock momentum.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed relative strength values
        """
        period = self.params.get("period", 10)
        market = self.params.get("market", Market.CN)
        aggregation = self.params.get("aggregation", "mean")
        root = self.params.get("root")
        col_name = f"sector_relative_strength_{period}"

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
        stock_momentum_col = f"_stock_momentum_{period}"
        sector_momentum_col = f"_sector_momentum_{period}"

        df_with_stock_momentum = df_sorted.with_columns(
            (
                (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1)
                * 100
            ).alias(stock_momentum_col)
        )

        # For each date, compute sector momentum and relative strength
        dates = df_sorted["date"].unique().sort()

        relative_strength_values: list[dict] = []

        for as_of_date in dates:
            # Get data for this date
            date_data = df_with_stock_momentum.filter(pl.col("date") == as_of_date)

            # Get all sector assignments at this date
            sectors_at_date = sector_index.all_sectors_at_date(as_of_date)

            for sector_code, instruments in sectors_at_date.items():
                # Filter data for instruments in this sector
                sector_data = date_data.filter(pl.col("instrument_id").is_in(instruments))

                if sector_data.is_empty():
                    continue

                # Compute sector momentum
                momentum_values = sector_data[stock_momentum_col].drop_nulls()

                if momentum_values.is_empty():
                    continue

                if aggregation == "median":
                    sector_momentum = momentum_values.median()
                else:
                    sector_momentum = momentum_values.mean()

                # Compute relative strength for each stock
                for row in sector_data.iter_rows(named=True):
                    stock_momentum = row.get(stock_momentum_col)
                    if stock_momentum is not None:
                        relative_strength = stock_momentum - sector_momentum
                        relative_strength_values.append({
                            "instrument_id": row["instrument_id"],
                            "date": as_of_date,
                            col_name: relative_strength,
                        })

        # Create DataFrame from relative strength values
        if relative_strength_values:
            rs_df = pl.DataFrame(relative_strength_values)
        else:
            # Empty result
            rs_df = pl.DataFrame({
                "instrument_id": df_sorted["instrument_id"],
                "date": df_sorted["date"],
                col_name: [None] * len(df_sorted),
            })

        # Join back to original data
        result_df = df_sorted.join(
            rs_df,
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