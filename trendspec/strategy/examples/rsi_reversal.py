"""
RSI Oversold/Overbought Reversal Strategy.

Counter-trend strategy demonstrating:
- RSI indicator usage
- Oversold/overbought thresholds
- Signal filtering

Strategy logic:
- Buy when RSI < oversold threshold (e.g., 30)
- Sell when RSI > overbought threshold (e.g., 70)

Parameters:
- rsi_period: RSI calculation period (default: 14)
- oversold: Oversold threshold (default: 30)
- overbought: Overbought threshold (default: 70)

Example:
    >>> from trendspec.strategy.examples import RSIReversalStrategy
    >>> strategy = RSIReversalStrategy(params={"rsi_period": 14, "oversold": 25})
"""

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("rsi_reversal")
class RSIReversalStrategy(BaseStrategy):
    """
    RSI Oversold/Overbought Reversal Strategy.

    A counter-trend strategy that buys when RSI indicates oversold conditions
    and sells when RSI indicates overbought conditions.

    Parameters:
        rsi_period: Period for RSI calculation (default: 14)
        oversold: RSI threshold for oversold condition (default: 30)
        overbought: RSI threshold for overbought condition (default: 70)

    Signals:
        BUY: RSI drops below oversold threshold
        SELL: RSI rises above overbought threshold

    Example:
        >>> strategy = RSIReversalStrategy(params={"rsi_period": 14, "oversold": 25, "overbought": 75})
    """

    name = "rsi_reversal"
    version = "1.0.0"
    params = {"rsi_period": 14, "oversold": 30, "overbought": 70}

    def _validate_dict_params(self) -> None:
        """Validate strategy parameters."""
        rsi_period = self.get_param("rsi_period", 14)
        oversold = self.get_param("oversold", 30)
        overbought = self.get_param("overbought", 70)

        if rsi_period < 1:
            raise ValueError(f"rsi_period ({rsi_period}) must be >= 1")

        if oversold < 0 or oversold > 50:
            raise ValueError(f"oversold ({oversold}) must be between 0 and 50")

        if overbought < 50 or overbought > 100:
            raise ValueError(f"overbought ({overbought}) must be between 50 and 100")

        if oversold >= overbought:
            raise ValueError(f"oversold ({oversold}) must be < overbought ({overbought})")

    def init(self, ctx: StrategyContext) -> None:
        """
        Precompute RSI for all instruments.

        Called once before the backtest/screening starts.
        Uses vectorized computation for efficiency.

        Args:
            ctx: StrategyContext with full data access
        """
        rsi_period = self.get_param("rsi_period", 14)

        # Precompute RSI
        self._rsi_df = ctx.precompute_indicator("RSI", period=rsi_period)

        # Store period for later lookup
        self._rsi_period = rsi_period

        # Get thresholds
        self._oversold = self.get_param("oversold", 30)
        self._overbought = self.get_param("overbought", 70)

        # Log initialization
        ctx.strategy.log(
            f"Initialized with rsi_period={rsi_period}, "
            f"oversold={self._oversold}, overbought={self._overbought}"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Check RSI levels and generate signals.

        Called for each bar during backtest, or once for latest date during screening.

        Signal logic:
        - BUY: RSI below oversold threshold and no position
        - SELL: RSI above overbought threshold and has position

        Args:
            ctx: StrategyContext with current bar data
        """
        # Get current RSI value
        rsi = ctx.indicator_value("RSI", ctx.instrument_id, ctx.date, period=self._rsi_period)

        if rsi is None:
            # Not enough data for RSI calculation
            return

        instrument_id = ctx.instrument_id

        # Check for oversold condition (BUY signal)
        if rsi < self._oversold:
            if not ctx.has_position(instrument_id):
                ctx.signal(
                    "BUY",
                    instrument_id,
                    ctx.close,
                    trigger_value=rsi,
                    note=f"RSI ({rsi:.2f}) below oversold threshold ({self._oversold})",
                )

        # Check for overbought condition (SELL signal)
        elif rsi > self._overbought:
            if ctx.has_position(instrument_id):
                ctx.signal(
                    "SELL",
                    instrument_id,
                    ctx.close,
                    trigger_value=rsi,
                    note=f"RSI ({rsi:.2f}) above overbought threshold ({self._overbought})",
                )