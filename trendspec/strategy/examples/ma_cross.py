"""
Dual Moving Average Crossover Strategy.

Classic trend-following strategy demonstrating:
- Indicator computation (MA)
- Cross signal generation
- Parameter configuration

Strategy logic:
- Buy when short MA crosses above long MA
- Sell when short MA crosses below long MA

Parameters:
- short_period: Short MA period (default: 20)
- long_period: Long MA period (default: 60)

Example:
    >>> from trendspec.strategy.examples import MACrossStrategy
    >>> strategy = MACrossStrategy(params={"short_period": 10, "long_period": 30})
"""

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("ma_cross")
class MACrossStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.

    A classic trend-following strategy that generates signals based on
    the crossover of two moving averages.

    Parameters:
        short_period: Period for the short moving average (default: 20)
        long_period: Period for the long moving average (default: 60)

    Signals:
        BUY: Short MA crosses above Long MA
        SELL: Short MA crosses below Long MA

    Example:
        >>> strategy = MACrossStrategy(params={"short_period": 10, "long_period": 30})
        >>> # Run via backtest engine for historical simulation
        >>> # Run via screening engine for latest signals
    """

    name = "ma_cross"
    version = "1.0.0"
    params = {"short_period": 20, "long_period": 60}

    def _validate_dict_params(self) -> None:
        """Validate strategy parameters."""
        short = self.get_param("short_period", 20)
        long = self.get_param("long_period", 60)

        if short >= long:
            raise ValueError(f"short_period ({short}) must be < long_period ({long})")

        if short < 1:
            raise ValueError(f"short_period ({short}) must be >= 1")

        if long < 1:
            raise ValueError(f"long_period ({long}) must be >= 1")

    def init(self, ctx: StrategyContext) -> None:
        """
        Precompute moving averages for all instruments.

        Called once before the backtest/screening starts.
        Uses vectorized computation for efficiency.

        Args:
            ctx: StrategyContext with full data access
        """
        short_period = self.get_param("short_period", 20)
        long_period = self.get_param("long_period", 60)

        # Precompute short MA
        self._short_ma_df = ctx.precompute_indicator("MA", period=short_period)

        # Precompute long MA
        self._long_ma_df = ctx.precompute_indicator("MA", period=long_period)

        # Store periods for later lookup
        self._short_period = short_period
        self._long_period = long_period

        # Track previous MA relationship for crossover detection
        self._prev_above: dict[str, bool] = {}

        # Log initialization
        ctx.strategy.log(f"Initialized with short_period={short_period}, long_period={long_period}")

    def next(self, ctx: StrategyContext) -> None:
        """
        Check for MA crossover and generate signals.

        Called for each bar during backtest, or once for latest date during screening.

        Crossover logic:
        - BUY: Short MA was below Long MA, now above
        - SELL: Short MA was above Long MA, now below

        Args:
            ctx: StrategyContext with current bar data
        """
        # Get current MA values
        short_ma = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=self._short_period)
        long_ma = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=self._long_period)

        if short_ma is None or long_ma is None:
            # Not enough data for MA calculation
            return

        # Determine current relationship
        currently_above = short_ma > long_ma
        instrument_id = ctx.instrument_id

        # Get previous relationship
        prev_above = self._prev_above.get(instrument_id)

        # Check for crossover
        if prev_above is not None:
            # Crossover detected
            if currently_above and not prev_above:
                # Short MA crossed above Long MA -> BUY signal
                if not ctx.has_position(instrument_id):
                    ctx.signal(
                        "BUY",
                        instrument_id,
                        ctx.close,
                        trigger_value=short_ma,
                        note=f"MA{self._short_period} ({short_ma:.2f}) crossed above MA{self._long_period} ({long_ma:.2f})",
                    )
            elif not currently_above and prev_above:
                # Short MA crossed below Long MA -> SELL signal
                if ctx.has_position(instrument_id):
                    ctx.signal(
                        "SELL",
                        instrument_id,
                        ctx.close,
                        trigger_value=short_ma,
                        note=f"MA{self._short_period} ({short_ma:.2f}) crossed below MA{self._long_period} ({long_ma:.2f})",
                    )

        # Store current relationship for next bar
        self._prev_above[instrument_id] = currently_above