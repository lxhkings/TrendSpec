"""
Sector neutral risk rule for TrendSpec.

Rule:
- SectorNeutralRule: Align portfolio sector weights to benchmark

Forces sector neutrality by rejecting signals that cause sector deviation.
"""

from datetime import date as DateType
from typing import Any, ClassVar
from dataclasses import dataclass

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index, get_all_sectors
from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


@dataclass
class SectorWeights:
    """
    Sector weights container.

    Attributes:
        weights: Dict mapping sector code to weight (0-1)
        total: Total weight (should be ~1.0 for full portfolio)
    """

    weights: dict[str, float]
    total: float = 1.0


class SectorNeutralRule(RiskRule):
    """
    Sector neutral rule.

    Aligns portfolio sector weights to benchmark sector weights.
    Rejects signals that would cause sector deviation > threshold.

    This is useful for:
    - Sector-neutral strategies
    - Benchmark-relative risk control
    - Pure alpha generation (no sector beta)

    Attributes:
        name: Rule name
        priority: Rule priority (45)

    Parameters:
        max_deviation: Maximum sector deviation from benchmark (default: 0.05 = 5%)
        market: Market for sector lookup (default: Market.CN)
        benchmark_weights: Benchmark sector weights (dict, optional)
        auto_calculate_benchmark: Auto-calculate benchmark from universe (default: False)

    Example:
        >>> # With explicit benchmark weights
        >>> benchmark = {"tech": 0.30, "finance": 0.20, ...}
        >>> rule = SectorNeutralRule(max_deviation=0.05, benchmark_weights=benchmark)
        >>> # Rejects signals causing >5% sector deviation from benchmark
        >>>
        >>> # Auto-calculated from market cap
        >>> rule = SectorNeutralRule(max_deviation=0.03, auto_calculate_benchmark=True)
    """

    name: ClassVar[str] = "sector_neutral"
    priority: ClassVar[int] = 45

    def __init__(
        self,
        max_deviation: float = 0.05,
        market: Market = Market.CN,
        benchmark_weights: dict[str, float] | None = None,
        auto_calculate_benchmark: bool = False,
        root: str | None = None,
    ) -> None:
        """
        Initialize sector neutral rule.

        Args:
            max_deviation: Maximum sector deviation from benchmark
            market: Market for sector lookup
            benchmark_weights: Benchmark sector weights dict (sector_code -> weight)
            auto_calculate_benchmark: Auto-calculate benchmark weights
            root: Root directory for data_lake
        """
        self.params = {
            "max_deviation": max_deviation,
            "market": market,
            "benchmark_weights": benchmark_weights,
            "auto_calculate_benchmark": auto_calculate_benchmark,
            "root": root,
        }

        # Store computed benchmark weights
        self._benchmark_weights: dict[str, float] = benchmark_weights or {}

    def _calculate_portfolio_sector_weights(
        self,
        portfolio: Portfolio,
        ctx: StrategyContext,
        market: Market,
        root: str | None,
    ) -> SectorWeights:
        """
        Calculate current portfolio sector weights.

        Args:
            portfolio: Current portfolio state
            ctx: Strategy context
            market: Market for sector lookup
            root: Root directory for data_lake

        Returns:
            SectorWeights with current weights
        """
        weights: dict[str, float] = {}

        if portfolio.equity <= 0:
            return SectorWeights(weights, 0.0)

        try:
            sector_index = get_sector_index(market, root)
        except Exception:
            return SectorWeights(weights, 0.0)

        # Sum position values by sector
        for instrument_id, quantity in portfolio.positions.items():
            if quantity <= 0:
                continue

            position_value = portfolio.position_value(instrument_id)
            sector = sector_index.sector(instrument_id, ctx.date)

            if sector:
                weights[sector] = weights.get(sector, 0.0) + position_value / portfolio.equity

        return SectorWeights(weights, sum(weights.values()))

    def _calculate_benchmark_sector_weights(
        self,
        ctx: StrategyContext,
        market: Market,
        root: str | None,
    ) -> dict[str, float]:
        """
        Calculate benchmark sector weights from universe.

        If market cap data available, weight by market cap.
        Otherwise, weight equally across all sectors.

        Args:
            ctx: Strategy context
            market: Market for sector lookup
            root: Root directory for data_lake

        Returns:
            Dict mapping sector code to weight
        """
        # If benchmark weights provided, use them
        if self._benchmark_weights:
            return self._benchmark_weights

        # Auto-calculate from universe
        try:
            sector_index = get_sector_index(market, root)
            all_sectors = get_all_sectors(market)
        except Exception:
            return {}

        # Get all instruments in universe at current date
        universe = ctx.pit_universe()

        # Count instruments per sector
        sector_counts: dict[str, int] = {}

        for instrument_id in universe:
            sector = sector_index.sector(instrument_id, ctx.date)
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Equal weight across sectors with instruments
        if sector_counts:
            n_sectors = len(sector_counts)
            return {s: 1.0 / n_sectors for s in sector_counts}

        return {}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if sector weights align with benchmark.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            Allow or Reject result
        """
        # Sell signals always pass (reduce deviation)
        if signal.is_sell():
            return Allow(self.name)

        max_deviation = self.params.get("max_deviation", 0.05)
        market = self.params.get("market", Market.CN)
        root = self.params.get("root")

        # Get sector index
        try:
            sector_index = get_sector_index(market, root)
        except Exception:
            return Allow(self.name)

        # Get signal's sector
        signal_sector = sector_index.sector(signal.instrument_id, ctx.date)

        if signal_sector is None:
            return Allow(self.name)

        # Get current portfolio sector weights
        portfolio_weights = self._calculate_portfolio_sector_weights(
            portfolio, ctx, market, root
        )

        # Get benchmark weights
        benchmark_weights = self._calculate_benchmark_sector_weights(
            ctx, market, root
        )

        if not benchmark_weights:
            # No benchmark weights available - use equal weight
            # Get all sectors at this date
            all_sectors_at_date = sector_index.all_sectors_at_date(ctx.date)
            n_sectors = len(all_sectors_at_date)
            if n_sectors > 0:
                benchmark_weights = {s: 1.0 / n_sectors for s in all_sectors_at_date}

        if not benchmark_weights:
            return Allow(self.name)

        # Calculate current deviation for signal's sector
        current_portfolio_weight = portfolio_weights.weights.get(signal_sector, 0.0)
        benchmark_weight = benchmark_weights.get(signal_sector, 0.0)

        # Calculate proposed weight change
        order_size = ctx.get_param("order_size", 100)
        proposed_value = signal.price * order_size

        if portfolio.equity > 0:
            proposed_weight_change = proposed_value / portfolio.equity
        else:
            proposed_weight_change = 0.0

        proposed_portfolio_weight = current_portfolio_weight + proposed_weight_change

        # Check deviation
        proposed_deviation = abs(proposed_portfolio_weight - benchmark_weight)

        if proposed_deviation > max_deviation:
            return Reject(
                self.name,
                f"Sector {signal_sector} deviation {proposed_deviation:.1%} exceeds max {max_deviation:.1%}",
                {
                    "sector": signal_sector,
                    "current_portfolio_weight": current_portfolio_weight,
                    "proposed_portfolio_weight": proposed_portfolio_weight,
                    "benchmark_weight": benchmark_weight,
                    "deviation": proposed_deviation,
                    "max_deviation": max_deviation,
                    "instrument_id": signal.instrument_id,
                },
            )

        return Allow(self.name)


# Register rule in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["sector_neutral"] = SectorNeutralRule