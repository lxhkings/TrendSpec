"""
Cross-sectional rank factors for TrendSpec.

Factors:
- RankWithinSectorFactor: Rank stocks within sector by factor value

Returns percentile (0-1) within sector, useful for sector-relative selection.
"""

from datetime import date as DateType
from typing import Any, ClassVar

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index, SectorIndex
from trendspec.factors.base import Factor, FactorResult, FACTOR_CATEGORIES
from trendspec.factors.registry import register, get_factor


@register("rank_within_sector")
class RankWithinSectorFactor(Factor):
    """
    Rank within sector factor - percentile rank of stock within its sector.

    Ranks stocks by a factor value within their sector, returning percentile (0-1).
    Higher percentile = higher factor value relative to sector peers.

    Uses PIT sector lookup to ensure correct sector assignments.

    This is useful for:
    - Sector-relative stock selection
    - Controlling for sector effects
    - Identifying sector leaders

    Attributes:
        name: Factor name for registry lookup
        description: Human-readable description
        category: Factor category (cross_sectional)

    Parameters:
        factor_name: Name of the factor to rank (must be registered)
        market: Market for sector lookup (default: Market.CN)
        ascending: Rank in ascending order (default: False, higher factor = higher rank)

    Example:
        >>> factor = RankWithinSectorFactor(factor_name="momentum", market=Market.CN)
        >>> result = factor.compute_full(df)
        >>> # Result contains rank_within_sector_momentum column
        >>> # Values are percentile (0-1) within each sector
    """

    name: ClassVar[str] = "rank_within_sector"
    description: ClassVar[str] = "Rank stocks within sector by factor value - percentile"
    category: ClassVar[str] = "cross_sectional"

    def __init__(
        self,
        factor_name: str = "momentum",
        market: Market = Market.CN,
        ascending: bool = False,
        root: str | None = None,
    ) -> None:
        """
        Initialize rank within sector factor.

        Args:
            factor_name: Name of the factor to rank
            market: Market for sector lookup
            ascending: If True, lower factor value gets higher rank
            root: Root directory for data_lake
        """
        self.params = {
            "factor_name": factor_name,
            "market": market,
            "ascending": ascending,
            "root": root,
        }

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        """
        Compute placeholder expression (ranking done in compute_full).

        Args:
            df: DataFrame with factor values

        Returns:
            Placeholder expression
        """
        # Ranking is done in compute_full with sector context
        return pl.lit(None, dtype=pl.Float64)

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        """
        Compute rank within sector for entire DataFrame.

        First computes the base factor, then ranks within each sector.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            FactorResult with computed rank values
        """
        factor_name = self.params.get("factor_name", "momentum")
        market = self.params.get("market", Market.CN)
        ascending = self.params.get("ascending", False)
        root = self.params.get("root")
        col_name = f"rank_within_sector_{factor_name}"

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

        # For each date, rank within sector
        dates = df_sorted["date"].unique().sort()

        rank_values: list[dict] = []

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

                n_stocks = len(factor_values)

                # Rank within sector
                sector_data_ranked = sector_data.with_columns(
                    (
                        pl.col(factor_col)
                        .rank(method="average", descending=not ascending)
                        / n_stocks
                    ).alias(col_name)
                )

                for row in sector_data_ranked.iter_rows(named=True):
                    rank_values.append({
                        "instrument_id": row["instrument_id"],
                        "date": as_of_date,
                        col_name: row.get(col_name),
                    })

        # Create DataFrame from rank values
        if rank_values:
            rank_df = pl.DataFrame(rank_values)
        else:
            rank_df = pl.DataFrame({
                "instrument_id": df_sorted["instrument_id"],
                "date": df_sorted["date"],
                col_name: [None] * len(df_sorted),
            })

        # Join back to original data
        result_df = df_sorted.join(
            rank_df,
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