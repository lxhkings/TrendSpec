"""
Cross-sectional neutralization factors for TrendSpec.

Factors:
- DemeanBySectorFactor: Subtract sector mean/median from each stock

Removes sector beta exposure, useful for sector-neutral strategies.
"""

from datetime import date as DateType
from typing import Any, ClassVar

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index, SectorIndex
from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register, get_factor


@register("demean_by_sector")
class DemeanBySectorFactor(Factor):
    """
    Demean by sector factor - subtract sector mean/median from each stock.

    Removes sector beta exposure by subtracting the sector average
    from each stock's factor value. This neutralizes sector effects.

    Uses PIT sector lookup to ensure correct sector assignments.

    This is useful for:
    - Sector-neutral strategies
    - Removing sector beta exposure
    - Identifying sector-independent alpha

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (cross_sectional)

    Parameters:
        factor_name: Name of the factor to demean (must be registered)
        market: Market for sector lookup (default: Market.CN_A)
        method: How to compute sector average - "mean" or "median" (default: "mean")

    Example:
        >>> factor = DemeanBySectorFactor(factor_name="momentum", market=Market.CN_A)
        >>> result = factor.compute_full(df)
        >>> # Result contains demean_momentum column
        >>> # Values are stock momentum minus sector average momentum
    """

    name: ClassVar[str] = "demean_by_sector"
    description: ClassVar[str] = "Demean factor by sector - subtract sector mean/median"
    category: ClassVar[str] = "cross_sectional"

    def __init__(
        self,
        factor_name: str = "momentum",
        market: Market = Market.CN_A,
        method: str = "mean",
        root: str | None = None,
    ) -> None:
        """
        Initialize demean by sector factor.

        Args:
            factor_name: Name of the factor to demean
            market: Market for sector lookup
            method: How to compute sector average - "mean" or "median"
            root: Root directory for data_lake
        """
        self.params = {
            "factor_name": factor_name,
            "market": market,
            "method": method,
            "root": root,
        }

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute placeholder expression (demeaning done in compute_full).

        Args:
            df: DataFrame with factor values

        Returns:
            Placeholder expression
        """
        return pl.lit(None, dtype=pl.Float64)

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute demeaned values for entire DataFrame.

        First computes the base factor, then subtracts sector average.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed demeaned values
        """
        factor_name = self.params.get("factor_name", "momentum")
        market = self.params.get("market", Market.CN_A)
        method = self.params.get("method", "mean")
        root = self.params.get("root")
        col_name = f"demean_{factor_name}"

        # Get sector index
        try:
            sector_index = get_sector_index(market, root)
        except Exception:
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

        # Get the base factor and compute it
        base_factor = get_factor(factor_name)
        if base_factor is None:
            return FactorResult(
                values=pl.DataFrame({
                    "instrument_id": df["instrument_id"],
                    "date": df["date"],
                    col_name: [None] * len(df),
                }),
                name=col_name,
                metadata={
                    "description": self.description,
                    "category": self.category,
                    "params": self.params,
                    "error": f"Factor '{factor_name}' not found",
                },
            )

        # Compute base factor
        base_result = base_factor.compute_full(df)
        factor_col = base_result.name

        # Join factor values back to sorted data
        df_sorted = df.sort("date")
        df_with_factor = df_sorted.join(
            base_result.values,
            on=["instrument_id", "date"],
            how="left",
        )

        # For each date, compute sector average and subtract
        dates = df_sorted["date"].unique().sort()

        demeaned_values: list[dict] = []

        for as_of_date in dates:
            date_data = df_with_factor.filter(pl.col("date") == as_of_date)

            sectors_at_date = sector_index.all_sectors_at_date(as_of_date)

            for sector_code, instruments in sectors_at_date.items():
                sector_data = date_data.filter(pl.col("instrument_id").is_in(instruments))

                if sector_data.is_empty():
                    continue

                # Get factor values for this sector
                factor_values = sector_data[factor_col].drop_nulls()

                if factor_values.is_empty():
                    continue

                # Compute sector average
                if method == "median":
                    sector_avg = factor_values.median()
                else:
                    sector_avg = factor_values.mean()

                # Subtract sector average from each stock
                for row in sector_data.iter_rows(named=True):
                    stock_value = row.get(factor_col)
                    if stock_value is not None:
                        demeaned = stock_value - sector_avg
                        demeaned_values.append({
                            "instrument_id": row["instrument_id"],
                            "date": as_of_date,
                            col_name: demeaned,
                        })

        # Create DataFrame from demeaned values
        if demeaned_values:
            demeaned_df = pl.DataFrame(demeaned_values)
        else:
            demeaned_df = pl.DataFrame({
                "instrument_id": df_sorted["instrument_id"],
                "date": df_sorted["date"],
                col_name: [None] * len(df_sorted),
            })

        # Join back to original data
        result_df = df_sorted.join(
            demeaned_df,
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