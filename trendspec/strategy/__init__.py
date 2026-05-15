"""
TrendSpec strategy module.

Provides the strategy framework for writing custom trading strategies.
Key components:
- BaseStrategy: Abstract base class for all strategies
- StrategyContext: Context with data access, indicators, PIT lookup
- Signal: Trading signal dataclass
- Indicators: Pre-built indicator functions (Polars expressions)

Design principles:
- DRY: BaseStrategy is the only extension point
- Vectorized init(): Precompute indicators once with Polars
- Dual-mode: Same next() for backtest and screening
- PIT access: Context provides date-parametrized universe/sector/factor

Example strategy:
    >>> from trendspec.strategy import BaseStrategy, StrategyContext
    ...
    >>> class MACrossStrategy(BaseStrategy):
    ...     name = "ma_cross"
    ...     params = {"fast": 10, "slow": 20}
    ...
    ...     def init(self, ctx: StrategyContext):
    ...         ctx.precompute_indicator("MA", period=self.params["fast"])
    ...         ctx.precompute_indicator("MA", period=self.params["slow"])
    ...
    ...     def next(self, ctx: StrategyContext):
    ...         fast = ctx.indicator_value("MA", period=self.params["fast"])
    ...         slow = ctx.indicator_value("MA", period=self.params["slow"])
    ...         if fast > slow:
    ...             ctx.signal("BUY", ctx.instrument_id, ctx.close)
"""

from trendspec.strategy.base import (
    BaseStrategy,
    StrategyParams,
    create_strategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.indicators import (
    compute_indicator,
    get_indicator,
    indicator_info,
    list_indicators,
    register_indicator,
)
from trendspec.strategy.signal import Signal, SignalBatch

__all__ = [
    # Base Strategy
    "BaseStrategy",
    "StrategyParams",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "create_strategy",
    # Context
    "StrategyContext",
    # Signal
    "Signal",
    "SignalBatch",
    # Indicators
    "compute_indicator",
    "get_indicator",
    "indicator_info",
    "list_indicators",
    "register_indicator",
]