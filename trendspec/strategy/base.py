"""
Base strategy class for TrendSpec strategy framework.

BaseStrategy is the single extension point for all strategies.
Users inherit from BaseStrategy and implement init() and next().

Key design principles:
- DRY: BaseStrategy is the only extension point
- Vectorized init(): Precompute indicators once with Polars
- Dual-mode: Same next() works for backtest and screening
- PIT access: Context provides date-parametrized universe/sector/factor

Example:
    >>> class MACrossStrategy(BaseStrategy):
    ...     params = {"fast_period": 10, "slow_period": 20}
    ...
    ...     def init(self, ctx):
    ...         # Precompute indicators (vectorized)
    ...         self.ma_fast = ctx.precompute_indicator("MA", period=self.params["fast_period"])
    ...         self.ma_slow = ctx.precompute_indicator("MA", period=self.params["slow_period"])
    ...
    ...     def next(self, ctx):
    ...         # Per-bar logic
    ...         fast = ctx.indicator_value("MA", ctx.instrument_id, period=self.params["fast_period"])
    ...         slow = ctx.indicator_value("MA", ctx.instrument_id, period=self.params["slow_period"])
    ...
    ...         if fast > slow and not ctx.has_position():
    ...             ctx.signal("BUY", ctx.instrument_id, ctx.close)
    ...         elif fast < slow and ctx.has_position():
    ...             ctx.signal("SELL", ctx.instrument_id, ctx.close)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, ClassVar

from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


@dataclass
class StrategyParams:
    """
    Base class for strategy parameters.

    Strategies should define their own params dataclass inheriting from this.
    Provides type-safe parameter definition with validation.

    Example:
        >>> @dataclass
        ... class MACrossParams(StrategyParams):
        ...     fast_period: int = 10
        ...     slow_period: int = 20
        ...     stop_loss_pct: float = 0.05
        ...
        ...     def validate(self) -> None:
        ...         if self.fast_period >= self.slow_period:
        ...             raise ValueError("fast_period must be < slow_period")
    """

    def validate(self) -> None:
        """Validate parameters. Override in subclasses."""
        pass

    def __post_init__(self) -> None:
        """Call validate after initialization."""
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    This is the single extension point for TrendSpec strategies.
    Users implement init() and next() methods.

    Lifecycle:
    1. Strategy instantiated with params
    2. init() called once with full data for precomputation
    3. next() called per-bar (backtest) or once for latest date (screening)
    4. Signals collected and processed through risk pipeline

    Key methods to implement:
    - init(ctx): Precompute indicators, set up internal state
    - next(ctx): Per-bar logic, generate signals

    Optional overrides:
    - on_signal(sig): Custom signal handling (risk filter, etc.)
    - validate_params(): Custom parameter validation

    Attributes:
        name: Strategy name (used for logging, reporting)
        params: Strategy parameters (dict or StrategyParams dataclass)
        version: Strategy version for tracking

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     name = "my_strategy"
        ...     params = {"period": 20}
        ...
        ...     def init(self, ctx: StrategyContext) -> None:
        ...         self.ma = ctx.precompute_indicator("MA", period=self.params["period"])
        ...
        ...     def next(self, ctx: StrategyContext) -> None:
        ...         ma_val = ctx.indicator_value("MA", ctx.instrument_id, period=self.params["period"])
        ...         if ctx.close > ma_val:
        ...             ctx.signal("BUY", ctx.instrument_id, ctx.close)
    """

    # Class attributes
    name: ClassVar[str] = "base_strategy"
    version: ClassVar[str] = "1.0.0"

    def __init__(
        self,
        params: dict[str, Any] | StrategyParams | None = None,
    ) -> None:
        """
        Initialize strategy with parameters.

        Args:
            params: Strategy parameters (dict or StrategyParams dataclass)
        """
        self.params: dict[str, Any] | StrategyParams = params or {}
        self._context: StrategyContext | None = None
        self._initialized: bool = False

        # Validate params
        if isinstance(self.params, StrategyParams):
            self.params.validate()
        elif isinstance(self.params, dict):
            self._validate_dict_params()

    def _validate_dict_params(self) -> None:
        """Validate dict params. Override for custom validation."""
        pass

    # =========================================================================
    # Core Methods (Must Implement)
    # =========================================================================

    @abstractmethod
    def init(self, ctx: StrategyContext) -> None:
        """
        One-time initialization with full data.

        Called once before backtest/screening starts.
        Use for precomputing indicators (vectorized with Polars).

        Args:
            ctx: StrategyContext with full data access

        Example:
            >>> def init(self, ctx):
            ...     # Precompute MA for entire dataset
            ...     self.ma20 = ctx.precompute_indicator("MA", period=20)
            ...     # Store data for later use
            ...     self._data = ctx._data
        """
        pass

    @abstractmethod
    def next(self, ctx: StrategyContext) -> None:
        """
        Per-bar trigger (backtest) or latest date (screening).

        Called for each bar during backtest, or once for latest date during screening.
        Access current bar data via ctx.close, ctx.date, etc.
        Generate signals via ctx.signal().

        Args:
            ctx: StrategyContext updated with current bar data

        Example:
            >>> def next(self, ctx):
            ...     # Check if price above MA
            ...     ma_val = ctx.indicator_value("MA", ctx.instrument_id, period=20)
            ...     if ctx.close > ma_val and not ctx.has_position():
            ...         ctx.signal("BUY", ctx.instrument_id, ctx.close, note="Above MA20")
        """
        pass

    # =========================================================================
    # Optional Methods (Can Override)
    # =========================================================================

    def on_signal(self, sig: Signal) -> Signal | None:
        """
        Signal handler - called before risk pipeline.

        Default behavior: Forward signal to broker unchanged.
        Override for custom signal filtering/modification.

        Use cases:
        - Add risk checks at strategy level
        - Modify signal parameters
        - Cancel signal based on internal state
        - Log signals for debugging

        Args:
            sig: Signal generated by strategy

        Returns:
            Modified signal, or None to cancel

        Example:
            >>> def on_signal(self, sig):
            ...     # Add stop-loss to buy signals
            ...     if sig.is_buy():
            ...         sig.stop_loss = sig.price * 0.95
            ...     return sig
        """
        # Default: forward unchanged
        return sig

    def on_bar_end(self, ctx: StrategyContext) -> None:
        """
        Called at end of each bar after signal processing.

        Use for:
        - Updating internal tracking state
        - Logging bar results
        - Post-processing

        Args:
            ctx: StrategyContext for current bar
        """
        pass

    def on_backtest_end(self, ctx: StrategyContext) -> None:
        """
        Called at end of backtest.

        Use for:
        - Final summary calculations
        - Cleanup
        - Reporting

        Args:
            ctx: StrategyContext with final state
        """
        pass

    # =========================================================================
    # State Management
    # =========================================================================

    def set_context(self, ctx: StrategyContext) -> None:
        """Set the strategy context (called by engine)."""
        self._context = ctx

    def get_context(self) -> StrategyContext | None:
        """Get current context."""
        return self._context

    def mark_initialized(self) -> None:
        """Mark strategy as initialized."""
        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if strategy has been initialized."""
        return self._initialized

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message from the strategy.

        Args:
            message: Log message
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        # For now, just print. Later integrate with logging system.
        print(f"[{self.name}] [{level}] {message}")

    def resolve_screening_date(self, requested_date: date) -> date:
        """Return the effective screening date for a given requested date.

        Override in strategies with periodic rebalancing to return the last
        rebalance date at or before requested_date, so screening always
        produces signals regardless of which day the user requests.
        """
        return requested_date

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value."""
        if isinstance(self.params, StrategyParams):
            return getattr(self.params, key, default)
        return self.params.get(key, default)

    def __repr__(self) -> str:
        """Return string representation."""
        params_str = str(self.params) if self.params else "{}"
        return f"{self.__class__.__name__}(name={self.name}, params={params_str})"


# =============================================================================
# Strategy Registry
# =============================================================================

_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register_strategy(name: str) -> Callable:
    """
    Decorator to register a strategy class.

    Args:
        name: Strategy name for lookup

    Returns:
        Decorator function
    """
    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> type[BaseStrategy] | None:
    """Get strategy class by name."""
    return _STRATEGY_REGISTRY.get(name)


def list_strategies() -> list[str]:
    """Get list of registered strategy names."""
    return sorted(_STRATEGY_REGISTRY.keys())


def create_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """
    Create a strategy instance by name.

    Args:
        name: Strategy name
        params: Strategy parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy not found
    """
    cls = get_strategy(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")

    return cls(params=params or {})