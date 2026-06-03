"""
Relative Strength EMA Cross strategy (rs_ema_cross).

对每只股票计算其相对基准 (QQQ) 的比值 ratio = close / benchmark_close，
在 ratio 序列上取 EMA60/EMA120：
  BUY  = EMA_short > EMA_long  且 空仓
  SELL = EMA_short <= EMA_long 且 持仓
状态型语义，全美股池，仓位走引擎默认。
"""

from typing import Any

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext

_DEFAULTS = {
    "benchmark_id": "QQQ",
    "ema_short": 60,
    "ema_long": 120,
}


@register_strategy("rs_ema_cross")
class RelativeStrengthEMACross(BaseStrategy):
    """股票/基准比值的 EMA 短长周期交叉。"""

    name = "rs_ema_cross"
    version = "1.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

    def init(self, ctx: StrategyContext) -> None:  # noqa: D401
        self._rs_short: dict[tuple, float] = {}
        self._rs_long: dict[tuple, float] = {}

    def next(self, ctx: StrategyContext) -> None:  # noqa: D401
        return