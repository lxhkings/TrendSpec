"""
Episodic Pivot strategy (Chris Flanders EP).

Catalyst proxy: gap-up + volume spike + close-in-range, filtered by trend
(EMA50 > EMA200), base compression (ATR10 < 0.7 × ATR30), and liquidity (ADV20 ≥ $20M).

T+1 next_open entry. Pivot-day-low hard stop + EMA10 close-cross-down trailing exit.
"""

from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


_DEFAULTS: dict[str, Any] = {
    # Catalyst / pivot detection
    "gap_pct": 0.05,
    "volume_multiplier": 3.0,
    "close_in_range_min": 0.80,

    # Trend filter
    "trend_ma_short": 50,
    "trend_ma_long": 200,

    # Base compression filter
    "base_atr_short": 10,
    "base_atr_long": 30,
    "base_compression_ratio": 0.70,

    # Liquidity filter
    "adv_lookback": 20,
    "adv_dollar_threshold": 20_000_000,

    # Exit
    "trail_ema_period": 10,

    # Sizing
    "max_positions": 10,
    "position_pct": 0.10,
}


@register_strategy("episodic_pivot")
class EpisodicPivot(BaseStrategy):
    """Chris Flanders Episodic Pivot — gap+volume catalyst, T+1 entry, EMA10 trail."""

    name = "episodic_pivot"
    version = "1.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

        self._pivot_low: dict[str, float] = {}
        self._entry_date: dict[str, DateType] = {}
        self._iid_dates: dict[str, pl.Series] = {}
        self._iid_ohlcv: dict[str, dict[DateType, dict[str, float]]] = {}

    def init(self, ctx: StrategyContext) -> None:
        """Precompute indicators (filled in later tasks)."""
        # Placeholder — Task 2 fills in indicator precompute + per-iid caches
        pass

    def next(self, ctx: StrategyContext) -> None:
        """Per-bar signal generation (filled in later tasks)."""
        pass
