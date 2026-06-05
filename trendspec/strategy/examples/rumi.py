"""
RUMI Trend-Following Strategy (David Imgraben).

Classic RUMI: smoothed DIFF of fast SMA and slow WMA.
Signal: state-based zero-line comparison.
  RUMI > 0 and no position → BUY
  RUMI < 0 and has position → SELL (exit long)
"""

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("rumi")
class RumiStrategy(BaseStrategy):
    name = "rumi"
    version = "1.0.0"
    params = {
        "fast_period": 3,
        "slow_period": 50,
        "signal_period": 30,
    }

    def _validate_dict_params(self) -> None:
        self.params = {**self.__class__.params, **self.params}
        fast = self.get_param("fast_period")
        slow = self.get_param("slow_period")
        signal = self.get_param("signal_period")
        if fast >= slow:
            raise ValueError(f"fast_period ({fast}) must be < slow_period ({slow})")
        if signal < 1:
            raise ValueError(f"signal_period ({signal}) must be >= 1")

    def init(self, ctx: StrategyContext) -> None:
        self._fast = self.get_param("fast_period")
        self._slow = self.get_param("slow_period")
        self._signal = self.get_param("signal_period")
        ctx.precompute_indicator(
            "RUMI",
            fast_period=self._fast,
            slow_period=self._slow,
            signal_period=self._signal,
        )

    def next(self, ctx: StrategyContext) -> None:
        rumi = ctx.indicator_value(
            "RUMI",
            ctx.instrument_id,
            ctx.date,
            fast_period=self._fast,
            slow_period=self._slow,
            signal_period=self._signal,
        )
        if rumi is None:
            return

        if rumi > 0 and not ctx.has_position():
            ctx.signal(
                "BUY",
                ctx.instrument_id,
                ctx.close,
                trigger_value=rumi,
                note=f"RUMI={rumi:.4f}",
            )
        elif rumi < 0 and ctx.has_position():
            ctx.signal(
                "SELL",
                ctx.instrument_id,
                ctx.close,
                trigger_value=rumi,
                note=f"RUMI={rumi:.4f}",
            )
