"""
Position limit risk rules for TrendSpec.

Rules:
- MaxSinglePositionSize: Single stock max % of portfolio
- MaxPositionsCount: Max number of positions

These rules control portfolio concentration and position limits.
"""

from typing import Any, ClassVar

from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


class MaxSinglePositionSize(RiskRule):
    """
    Maximum single position size rule.

    Limits the size of any single position relative to portfolio equity.
    Prevents over-concentration in a single stock.

    Attributes:
        name: Rule name
        priority: Rule priority (10, runs early)

    Parameters:
        max_pct: Maximum position size as % of equity (default: 0.10 = 10%)

    Example:
        >>> rule = MaxSinglePositionSize(max_pct=0.15)
        >>> result = rule.check(signal, portfolio, ctx)
        >>> # Rejects if position would exceed 15% of equity
    """

    name: ClassVar[str] = "max_single_position_size"
    priority: ClassVar[int] = 10

    def __init__(self, max_pct: float = 0.10) -> None:
        """
        Initialize max single position size rule.

        Args:
            max_pct: Maximum position size as % of equity (e.g., 0.10 = 10%)
        """
        self.params = {"max_pct": max_pct}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if position size would exceed limit.

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

        max_pct = self.params.get("max_pct", 0.10)

        # Get order size from context
        order_size = ctx.get_param("order_size", 100)
        proposed_value = signal.price * order_size

        # Check against equity
        if portfolio.equity <= 0:
            return Allow(self.name)  # No equity to check against

        # Get current position value for this instrument
        current_value = portfolio.position_value(signal.instrument_id)

        # Calculate proposed position size as % of equity
        proposed_pct = (current_value + proposed_value) / portfolio.equity

        if proposed_pct > max_pct:
            return Reject(
                self.name,
                f"Position size would exceed {max_pct:.1%} of equity",
                {
                    "current_pct": current_value / portfolio.equity if portfolio.equity > 0 else 0,
                    "proposed_pct": proposed_pct,
                    "max_pct": max_pct,
                    "instrument_id": signal.instrument_id,
                },
            )

        return Allow(self.name)


class MaxPositionsCount(RiskRule):
    """
    Maximum number of positions rule.

    Limits the number of positions in the portfolio.
    Prevents over-diversification or portfolio complexity.

    Attributes:
        name: Rule name
        priority: Rule priority (20)

    Parameters:
        max_positions: Maximum number of positions (default: 10)

    Example:
        >>> rule = MaxPositionsCount(max_positions=15)
        >>> result = rule.check(signal, portfolio, ctx)
        >>> # Rejects buy signals when portfolio has 15 positions
    """

    name: ClassVar[str] = "max_positions_count"
    priority: ClassVar[int] = 20

    def __init__(self, max_positions: int = 10) -> None:
        """
        Initialize max positions count rule.

        Args:
            max_positions: Maximum number of positions allowed
        """
        self.params = {"max_positions": max_positions}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if portfolio has reached max positions.

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

        max_positions = self.params.get("max_positions", 10)
        current_count = portfolio.position_count()

        # If position already exists, allow (not adding new position)
        if portfolio.has_position(signal.instrument_id):
            return Allow(self.name)

        if current_count >= max_positions:
            return Reject(
                self.name,
                f"Max positions ({max_positions}) reached",
                {
                    "current_count": current_count,
                    "max_positions": max_positions,
                    "instrument_id": signal.instrument_id,
                },
            )

        return Allow(self.name)


# Register rules in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["max_single_position_size"] = MaxSinglePositionSize
_RULE_REGISTRY["max_positions_count"] = MaxPositionsCount