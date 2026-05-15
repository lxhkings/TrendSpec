"""
Sector Relative Momentum Strategy.

Factor-based strategy demonstrating:
- Sector lookup via PIT
- Cross-sectional ranking
- Sector neutral risk rule usage
- Factor computation

Strategy logic:
- Compute sector-relative momentum factor
- Rank stocks within each sector by momentum
- Buy stocks with rank in top 10%
- Use sector neutral risk rule to balance positions

Parameters:
- momentum_period: Lookback period for momentum (default: 20)
- top_pct: Percentage threshold for top rank (default: 0.1, i.e., top 10%)

Example:
    >>> from trendspec.strategy.examples import SectorMomentumStrategy
    >>> strategy = SectorMomentumStrategy(params={"momentum_period": 20, "top_pct": 0.15})
"""

from datetime import date as DateType

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("sector_momentum")
class SectorMomentumStrategy(BaseStrategy):
    """
    Sector Relative Momentum Strategy.

    A factor-based strategy that selects stocks with the highest momentum
    relative to their sector peers.

    Parameters:
        momentum_period: Lookback period for momentum calculation (default: 20)
        top_pct: Top percentage threshold for ranking (default: 0.1 = top 10%)

    Signals:
        BUY: Stock's sector-relative momentum rank is in top top_pct

    This strategy demonstrates:
        - PIT sector lookup via ctx.sector()
        - Cross-sectional ranking within sectors
        - Factor computation (momentum)
        - Sector neutral risk rule integration

    Example:
        >>> strategy = SectorMomentumStrategy(params={"momentum_period": 20, "top_pct": 0.15})
    """

    name = "sector_momentum"
    version = "1.0.0"
    params = {"momentum_period": 20, "top_pct": 0.1}

    def _validate_dict_params(self) -> None:
        """Validate strategy parameters."""
        momentum_period = self.get_param("momentum_period", 20)
        top_pct = self.get_param("top_pct", 0.1)

        if momentum_period < 1:
            raise ValueError(f"momentum_period ({momentum_period}) must be >= 1")

        if top_pct <= 0 or top_pct >= 1:
            raise ValueError(f"top_pct ({top_pct}) must be between 0 and 1")

    def init(self, ctx: StrategyContext) -> None:
        """
        Precompute momentum and set up sector tracking.

        Called once before the backtest/screening starts.

        Args:
            ctx: StrategyContext with full data access
        """
        momentum_period = self.get_param("momentum_period", 20)

        # Precompute momentum (ROC - Rate of Change)
        self._momentum_df = ctx.precompute_indicator("ROC", period=momentum_period)

        # Store parameters
        self._momentum_period = momentum_period
        self._top_pct = self.get_param("top_pct", 0.1)

        # Store full data for cross-sectional ranking
        self._data = ctx._data

        # Log initialization
        ctx.strategy.log(
            f"Initialized with momentum_period={momentum_period}, top_pct={self._top_pct}"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Compute sector-relative momentum and generate signals.

        Called for each bar during backtest, or once for latest date during screening.

        Logic:
        1. Get current universe at PIT date
        2. Get sector for each instrument
        3. Compute momentum factor for all instruments
        4. Rank within each sector
        5. Select top top_pct in each sector

        Args:
            ctx: StrategyContext with current bar data
        """
        current_date = ctx.date

        # Get current universe (PIT lookup)
        universe_ids = ctx.pit_universe(current_date)

        if not universe_ids:
            return

        # Get momentum values for all instruments at current date
        momentum_col = f"ROC_{self._momentum_period}"

        # Filter data for current date and universe
        current_data = self._data.filter(
            (pl.col("date") == current_date) &
            (pl.col("instrument_id").is_in(universe_ids))
        )

        if current_data.is_empty():
            return

        # Join with momentum values
        momentum_data = self._momentum_df.filter(pl.col("date") == current_date)
        current_with_momentum = current_data.join(
            momentum_data.select(["instrument_id", momentum_col]),
            on="instrument_id",
            how="left"
        )

        if momentum_col not in current_with_momentum.columns:
            return

        # Group by sector and rank momentum within each sector
        sector_ranks: dict[str, dict[str, float]] = {}  # sector -> {instrument_id: rank}

        for row in current_with_momentum.iter_rows(named=True):
            instrument_id = row["instrument_id"]
            momentum_val = row.get(momentum_col)

            if momentum_val is None:
                continue

            # Get sector for this instrument at PIT date
            sector = ctx.sector(instrument_id, current_date)

            if sector is None:
                # No sector assigned, skip
                continue

            if sector not in sector_ranks:
                sector_ranks[sector] = {}

            sector_ranks[sector][instrument_id] = momentum_val

        # Rank within each sector and select top top_pct
        selected_instruments: list[str] = []

        for sector, instruments_momentum in sector_ranks.items():
            n_instruments = len(instruments_momentum)
            if n_instruments < 3:
                # Skip sectors with too few instruments
                continue

            # Sort by momentum descending
            sorted_instruments = sorted(
                instruments_momentum.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Select top top_pct
            n_top = max(1, int(n_instruments * self._top_pct))
            top_instruments = [inst for inst, _ in sorted_instruments[:n_top]]
            selected_instruments.extend(top_instruments)

        # Generate BUY signals for selected instruments
        if ctx.instrument_id in selected_instruments:
            if not ctx.has_position(ctx.instrument_id):
                momentum_val = ctx.indicator_value(
                    "ROC", ctx.instrument_id, current_date, period=self._momentum_period
                )
                sector = ctx.sector(ctx.instrument_id, current_date)

                ctx.signal(
                    "BUY",
                    ctx.instrument_id,
                    ctx.close,
                    trigger_value=momentum_val,
                    note=f"Top {self._top_pct*100:.0f}% momentum in sector {sector}",
                )