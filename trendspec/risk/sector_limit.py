"""
Sector concentration limit risk rule for TrendSpec.

Rule:
- SectorConcentrationLimit: Single sector max % of portfolio

Depends on data/sectors.py for PIT sector lookup.
"""

from datetime import date as DateType
from typing import Any, ClassVar

from trendspec.data.markets import Market
from trendspec.data.sectors import get_sector_index
from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


class SectorConcentrationLimit(RiskRule):
    """
    Sector concentration limit rule.

    Limits exposure to a single sector to prevent over-concentration.
    Uses PIT sector lookup to get correct sector assignments.

    This is critical for:
    - Risk management (avoid sector blow-ups)
    - Portfolio diversification
    - Benchmark-relative risk

    Attributes:
        name: Rule name
        priority: Rule priority (40)

    Parameters:
        max_sector_pct: Maximum sector weight (default: 0.30 = 30%)
        market: Market for sector lookup (default: Market.CN_A)
        count_existing_position: Count existing positions in sector (default: True)

    Example:
        >>> rule = SectorConcentrationLimit(max_sector_pct=0.25, market=Market.CN_A)
        >>> # Rejects signals that would cause sector weight > 25%
    """

    name: ClassVar[str] = "sector_concentration_limit"
    priority: ClassVar[int] = 40

    def __init__(
        self,
        max_sector_pct: float = 0.30,
        market: Market = Market.CN_A,
        count_existing_position: bool = True,
        root: str | None = None,
    ) -> None:
        """
        Initialize sector concentration limit rule.

        Args:
            max_sector_pct: Maximum sector weight (e.g., 0.30 = 30%)
            market: Market for sector lookup
            count_existing_position: Include existing position in sector weight
            root: Root directory for data_lake
        """
        self.params = {
            "max_sector_pct": max_sector_pct,
            "market": market,
            "count_existing_position": count_existing_position,
            "root": root,
        }

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if sector concentration would exceed limit.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            Allow or Reject result
        """
        # Sell signals always pass
        if signal.is_sell():
            return Allow(self.name)

        max_sector_pct = self.params.get("max_sector_pct", 0.30)
        market = self.params.get("market", Market.CN_A)
        count_existing = self.params.get("count_existing_position", True)
        root = self.params.get("root")

        # Get sector index
        try:
            sector_index = get_sector_index(market, root)
        except Exception:
            # If sector index not available, allow by default
            return Allow(self.name)

        # Get sector for signal instrument at current date
        signal_sector = sector_index.sector(signal.instrument_id, ctx.date)

        if signal_sector is None:
            # No sector info - allow by default
            return Allow(self.name)

        # Calculate current sector weight
        if portfolio.equity <= 0:
            return Allow(self.name)

        # Sum up position values in this sector
        sector_value = 0.0
        sector_positions: list[str] = []

        for instrument_id, quantity in portfolio.positions.items():
            if quantity <= 0:
                continue

            # Check if this instrument is in the same sector
            instrument_sector = sector_index.sector(instrument_id, ctx.date)
            if instrument_sector == signal_sector:
                position_value = portfolio.position_value(instrument_id)
                sector_value += position_value
                sector_positions.append(instrument_id)

        # If existing position in signal's instrument, don't count as new sector exposure
        if count_existing and portfolio.has_position(signal.instrument_id):
            # Adding to existing position - just check if already over limit
            current_sector_pct = sector_value / portfolio.equity
            if current_sector_pct > max_sector_pct:
                return Reject(
                    self.name,
                    f"Sector {signal_sector} already exceeds limit ({current_sector_pct:.1%} > {max_sector_pct:.1%})",
                    {
                        "sector": signal_sector,
                        "current_weight": current_sector_pct,
                        "max_weight": max_sector_pct,
                        "sector_positions": sector_positions,
                        "instrument_id": signal.instrument_id,
                    },
                )
            return Allow(self.name)

        # Calculate proposed additional sector weight
        order_size = ctx.get_param("order_size", 100)
        proposed_value = signal.price * order_size
        proposed_sector_pct = (sector_value + proposed_value) / portfolio.equity

        if proposed_sector_pct > max_sector_pct:
            return Reject(
                self.name,
                f"Sector {signal_sector} concentration would exceed {max_sector_pct:.1%}",
                {
                    "sector": signal_sector,
                    "current_weight": sector_value / portfolio.equity,
                    "proposed_weight": proposed_sector_pct,
                    "proposed_value": proposed_value,
                    "max_weight": max_sector_pct,
                    "sector_positions": sector_positions,
                    "instrument_id": signal.instrument_id,
                },
            )

        return Allow(self.name)


# Register rule in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["sector_concentration_limit"] = SectorConcentrationLimit