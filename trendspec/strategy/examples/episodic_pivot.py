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
        """Vectorized precompute of indicators + per-iid lookup caches."""
        # Precompute indicators (cached by StrategyContext, looked up via indicator_value)
        ctx.precompute_indicator("ADV", period=self.get_param("adv_lookback"))
        ctx.precompute_indicator("ATR", period=self.get_param("base_atr_short"))
        ctx.precompute_indicator("ATR", period=self.get_param("base_atr_long"))
        ctx.precompute_indicator("EMA", period=self.get_param("trail_ema_period"))
        ctx.precompute_indicator("EMA", period=self.get_param("trend_ma_short"))
        ctx.precompute_indicator("EMA", period=self.get_param("trend_ma_long"))

        # Build per-iid sorted date arrays + OHLCV dict for O(1) T-1 lookup
        if ctx._data is None or ctx._data.is_empty():
            return

        for (iid,), grp in ctx._data.sort(["instrument_id", "date"]).group_by(
            ["instrument_id"], maintain_order=True
        ):
            self._iid_dates[iid] = grp["date"]
            self._iid_ohlcv[iid] = {
                row["date"]: {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for row in grp.iter_rows(named=True)
            }

    def _prev_bar(self, iid: str, as_of_date: DateType) -> dict[str, float] | None:
        """Return OHLCV dict for the trading day immediately before `as_of_date`, or None."""
        dates = self._iid_dates.get(iid)
        if dates is None:
            return None
        idx = dates.search_sorted(as_of_date, side="left")
        if idx < 1:
            return None
        prev_date = dates[idx - 1]
        return self._iid_ohlcv.get(iid, {}).get(prev_date)

    def _check_buy(self, ctx: StrategyContext, iid: str, as_of: DateType) -> bool:
        """Return True iff all 6 BUY conditions fire on bar `as_of`."""
        # Current bar (T)
        t_bar = self._iid_ohlcv.get(iid, {}).get(as_of)
        if t_bar is None:
            return False

        # Previous bar (T-1)
        prev = self._prev_bar(iid, as_of)
        if prev is None:
            return False

        # Get date index for T-1 indicator lookups
        dates = self._iid_dates.get(iid)
        if dates is None:
            return False
        idx = dates.search_sorted(as_of, side="left")
        if idx < 1:
            return False
        prev_date = dates[idx - 1]

        # Indicators at T and T-1
        adv_prev = ctx.indicator_value("ADV", iid, prev_date, period=self.get_param("adv_lookback"))
        atr_short_prev = ctx.indicator_value("ATR", iid, prev_date, period=self.get_param("base_atr_short"))
        atr_long_prev = ctx.indicator_value("ATR", iid, prev_date, period=self.get_param("base_atr_long"))
        ema_short = ctx.indicator_value("EMA", iid, as_of, period=self.get_param("trend_ma_short"))
        ema_long = ctx.indicator_value("EMA", iid, as_of, period=self.get_param("trend_ma_long"))

        if any(v is None for v in (adv_prev, atr_short_prev, atr_long_prev, ema_short, ema_long)):
            return False

        # Cond 1: gap-up
        gap = (t_bar["open"] / prev["close"]) - 1.0
        if gap < self.get_param("gap_pct"):
            return False

        # Cond 2: dollar-volume spike (T's dollar volume >= volume_multiplier x ADV20[T-1])
        t_dollar_volume = t_bar["close"] * t_bar["volume"]
        if t_dollar_volume < self.get_param("volume_multiplier") * adv_prev:
            return False

        # Cond 3: close in upper portion of T's range
        t_range = t_bar["high"] - t_bar["low"]
        if t_range <= 0:
            return False
        close_in_range = (t_bar["close"] - t_bar["low"]) / t_range
        if close_in_range < self.get_param("close_in_range_min"):
            return False

        # Cond 4: trend (close > EMA50 > EMA200)
        if not (t_bar["close"] > ema_short > ema_long):
            return False

        # Cond 5: base compression (ATR10[T-1] < ratio x ATR30[T-1])
        if atr_short_prev >= self.get_param("base_compression_ratio") * atr_long_prev:
            return False

        # Cond 6: liquidity (ADV20[T-1] >= $20M)
        if adv_prev < self.get_param("adv_dollar_threshold"):
            return False

        return True

    def _check_sell(
        self, ctx: StrategyContext, iid: str, as_of: DateType
    ) -> tuple[bool, str | None]:
        """Return (triggered, reason). Reason in {'hard_stop', 'trail_ema', None}."""
        bar = self._iid_ohlcv.get(iid, {}).get(as_of)
        if bar is None:
            return (False, None)

        # Hard stop: today's low <= pivot_day_low
        pivot_low = self._pivot_low.get(iid)
        if pivot_low is not None and bar["low"] <= pivot_low:
            return (True, "hard_stop")

        # Trailing: close < EMA10
        ema10 = ctx.indicator_value("EMA", iid, as_of, period=self.get_param("trail_ema_period"))
        if ema10 is not None and bar["close"] < ema10:
            return (True, "trail_ema")

        return (False, None)

    def next(self, ctx: StrategyContext) -> None:
        """Per-bar signal generation (filled in later tasks)."""
        pass
