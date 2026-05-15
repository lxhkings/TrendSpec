"""
Risk rule abstract base class for TrendSpec.

Risk rules filter signals before broker execution.
Each rule checks a signal against portfolio/context and returns Allow or Reject.

Key design:
- Serial pipeline: Rules run in order, any rejection drops signal
- Context-aware: Rules access portfolio, capital, positions
- Logging: Reject reasons are logged for analysis

Built-in rules:
- MaxPositionSize: Limit position size
- MaxPositions: Limit number of positions
- MinCapital: Require minimum available capital
- SectorConcentration: Limit sector exposure
- LiquidityFilter: Filter low-volume stocks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


# =============================================================================
# Risk Check Results
# =============================================================================


@dataclass
class Allow:
    """
    Signal allowed through risk pipeline.

    Indicates the signal passed the risk check.

    Attributes:
        rule_name: Name of the rule that allowed (for logging)
        modified_signal: Optionally modified signal (e.g., with stop-loss added)
    """

    rule_name: str
    modified_signal: Signal | None = None

    def is_allowed(self) -> bool:
        return True

    def is_rejected(self) -> bool:
        return False


@dataclass
class Reject:
    """
    Signal rejected by risk rule.

    Indicates the signal failed the risk check.

    Attributes:
        rule_name: Name of the rule that rejected
        reason: Human-readable reason for rejection
        details: Additional details for logging/analysis
    """

    rule_name: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def is_allowed(self) -> bool:
        return False

    def is_rejected(self) -> bool:
        return True


# Type alias for risk check result
RiskResult = Allow | Reject


# =============================================================================
# Portfolio State (for risk checks)
# =============================================================================


@dataclass
class Portfolio:
    """
    Portfolio state for risk checks.

    Passed to risk rules along with signal and context.

    Attributes:
        positions: Dict of instrument_id -> quantity
        cash: Available cash
        equity: Total equity (cash + position values)
        positions_value: Total value of positions
        sector_weights: Sector weights (if available)
    """

    positions: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    equity: float = 0.0
    positions_value: float = 0.0
    sector_weights: dict[str, float] = field(default_factory=dict)
    position_prices: dict[str, float] = field(default_factory=dict)  # Entry prices

    def position_count(self) -> int:
        """Count number of positions."""
        return len([q for q in self.positions.values() if q > 0])

    def position_value(self, instrument_id: str) -> float:
        """Get position value for an instrument."""
        qty = self.positions.get(instrument_id, 0.0)
        price = self.position_prices.get(instrument_id, 0.0)
        return qty * price

    def has_position(self, instrument_id: str) -> bool:
        """Check if position exists."""
        return self.positions.get(instrument_id, 0.0) > 0

    def sector_weight(self, sector: str) -> float:
        """Get sector weight."""
        return self.sector_weights.get(sector, 0.0)


# =============================================================================
# Risk Rule Base Class
# =============================================================================


class RiskRule(ABC):
    """
    Abstract base class for risk rules.

    Risk rules check signals before broker execution.
    They access portfolio state and context to make decisions.

    Attributes:
        name: Rule name (for logging)
        priority: Rule priority (lower = runs first)
        enabled: Whether rule is active

    Methods to implement:
        check(signal, portfolio, ctx): Return Allow or Reject

    Example:
        >>> class MaxPositions(RiskRule):
        ...     name = "max_positions"
        ...     max_positions = 10
        ...
        ...     def check(self, signal, portfolio, ctx):
        ...         if signal.is_buy() and portfolio.position_count() >= self.max_positions:
        ...             return Reject(self.name, f"Max positions ({self.max_positions}) reached")
        ...         return Allow(self.name)
    """

    name: ClassVar[str] = "base_rule"
    priority: ClassVar[int] = 100
    enabled: ClassVar[bool] = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """
        Initialize risk rule with parameters.

        Args:
            params: Rule parameters
        """
        self.params: dict[str, Any] = params or {}

    @abstractmethod
    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check signal against portfolio and context.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context with data access

        Returns:
            Allow or Reject result

        Example:
            >>> def check(self, signal, portfolio, ctx):
            ...     if signal.is_buy() and portfolio.cash < signal.price * 100:
            ...         return Reject(self.name, "Insufficient capital")
            ...     return Allow(self.name)
        """
        pass

    def is_enabled(self) -> bool:
        """Check if rule is enabled."""
        return self.enabled

    def get_priority(self) -> int:
        """Get rule priority."""
        return self.priority

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority})"


# =============================================================================
# Built-in Risk Rules
# =============================================================================


class MaxPositionSize(RiskRule):
    """
    Maximum position size rule.

    Rejects buy signals that would exceed max position size.
    """

    name = "max_position_size"
    priority = 10

    def __init__(self, max_pct: float = 0.10) -> None:
        self.params = {"max_pct": max_pct}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        if signal.is_sell():
            return Allow(self.name)

        max_pct = self.params.get("max_pct", 0.10)
        position_value = portfolio.position_value(signal.instrument_id)
        proposed_value = signal.price * ctx.get_param("order_size", 100)

        # Check if new position would exceed max pct of equity
        if portfolio.equity > 0 and (position_value + proposed_value) / portfolio.equity > max_pct:
            return Reject(
                self.name,
                f"Position size would exceed {max_pct:.1%} of equity",
                {"current_pct": position_value / portfolio.equity, "max_pct": max_pct},
            )

        return Allow(self.name)


class MaxPositions(RiskRule):
    """
    Maximum number of positions rule.

    Rejects buy signals when max positions reached.
    """

    name = "max_positions"
    priority = 20

    def __init__(self, max_positions: int = 10) -> None:
        self.params = {"max_positions": max_positions}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        if signal.is_sell():
            return Allow(self.name)

        max_positions = self.params.get("max_positions", 10)
        current_count = portfolio.position_count()

        if current_count >= max_positions:
            return Reject(
                self.name,
                f"Max positions ({max_positions}) reached",
                {"current_count": current_count, "max_positions": max_positions},
            )

        return Allow(self.name)


class MinCapital(RiskRule):
    """
    Minimum available capital rule.

    Rejects buy signals when insufficient capital.
    """

    name = "min_capital"
    priority = 30

    def __init__(self, min_capital: float = 1000.0) -> None:
        self.params = {"min_capital": min_capital}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        if signal.is_sell():
            return Allow(self.name)

        min_capital = self.params.get("min_capital", 1000.0)
        order_size = ctx.get_param("order_size", 100)
        required_capital = signal.price * order_size

        if portfolio.cash < required_capital:
            return Reject(
                self.name,
                f"Insufficient capital: need {required_capital:.2f}, have {portfolio.cash:.2f}",
                {"required": required_capital, "available": portfolio.cash},
            )

        if portfolio.cash - required_capital < min_capital:
            return Reject(
                self.name,
                f"Would leave less than min capital ({min_capital:.2f})",
                {"remaining": portfolio.cash - required_capital, "min_capital": min_capital},
            )

        return Allow(self.name)


class SectorConcentration(RiskRule):
    """
    Sector concentration rule.

    Limits exposure to single sector.
    """

    name = "sector_concentration"
    priority = 40

    def __init__(self, max_sector_pct: float = 0.30) -> None:
        self.params = {"max_sector_pct": max_sector_pct}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        if signal.is_sell():
            return Allow(self.name)

        max_sector_pct = self.params.get("max_sector_pct", 0.30)

        # Get sector for signal instrument
        signal_sector = ctx.sector(signal.instrument_id)
        if signal_sector is None:
            return Allow(self.name)  # No sector info, allow

        # Get current sector weight
        current_weight = portfolio.sector_weight(signal_sector)
        order_size = ctx.get_param("order_size", 100)
        proposed_value = signal.price * order_size

        if portfolio.equity > 0:
            proposed_weight = proposed_value / portfolio.equity
            if current_weight + proposed_weight > max_sector_pct:
                return Reject(
                    self.name,
                    f"Sector {signal_sector} concentration would exceed {max_sector_pct:.1%}",
                    {
                        "sector": signal_sector,
                        "current_weight": current_weight,
                        "proposed_weight": proposed_weight,
                        "max_weight": max_sector_pct,
                    },
                )

        return Allow(self.name)


class LiquidityFilter(RiskRule):
    """
    Liquidity filter rule.

    Filters low-volume stocks to ensure tradability.
    """

    name = "liquidity_filter"
    priority = 50

    def __init__(self, min_volume: int = 100000) -> None:
        self.params = {"min_volume": min_volume}

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        min_volume = self.params.get("min_volume", 100000)

        # Check current volume
        try:
            current_volume = ctx.volume
            if current_volume < min_volume:
                return Reject(
                    self.name,
                    f"Volume {current_volume} below minimum {min_volume}",
                    {"volume": current_volume, "min_volume": min_volume},
                )
        except RuntimeError:
            # No volume data available, allow
            pass

        return Allow(self.name)


class DuplicatePosition(RiskRule):
    """
    Duplicate position filter.

    Prevents buying when position already exists.
    """

    name = "duplicate_position"
    priority = 5

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        if signal.is_sell():
            return Allow(self.name)

        if portfolio.has_position(signal.instrument_id):
            return Reject(
                self.name,
                "Position already exists",
                {"instrument_id": signal.instrument_id},
            )

        return Allow(self.name)


class UniverseMembership(RiskRule):
    """
    Universe membership check.

    Ensures signal instrument is in the PIT universe.
    """

    name = "universe_membership"
    priority = 1

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        # Check if instrument is in universe at current date
        universe = ctx.pit_universe()

        if signal.instrument_id not in universe:
            return Reject(
                self.name,
                f"Instrument {signal.instrument_id} not in universe",
                {"instrument_id": signal.instrument_id, "universe_size": len(universe)},
            )

        return Allow(self.name)


# =============================================================================
# Risk Rule Registry
# =============================================================================

_RULE_REGISTRY: dict[str, type[RiskRule]] = {}


def register_rule(name: str) -> callable:
    """
    Decorator to register a risk rule class.

    Args:
        name: Rule name for registry lookup

    Returns:
        Decorator function
    """
    def decorator(cls: type[RiskRule]) -> type[RiskRule]:
        _RULE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_rule(name: str, params: dict[str, Any] | None = None) -> RiskRule | None:
    """Get a risk rule instance by name."""
    cls = _RULE_REGISTRY.get(name)
    if cls is None:
        return None

    if params:
        return cls(**params)
    return cls()


def list_rules() -> list[str]:
    """Get list of registered rule names."""
    return sorted(_RULE_REGISTRY.keys())


# Register built-in rules
_RULE_REGISTRY["max_position_size"] = MaxPositionSize
_RULE_REGISTRY["max_positions"] = MaxPositions
_RULE_REGISTRY["min_capital"] = MinCapital
_RULE_REGISTRY["sector_concentration"] = SectorConcentration
_RULE_REGISTRY["liquidity_filter"] = LiquidityFilter
_RULE_REGISTRY["duplicate_position"] = DuplicatePosition
_RULE_REGISTRY["universe_membership"] = UniverseMembership