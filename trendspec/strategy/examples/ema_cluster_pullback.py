"""
EMA Cluster Pullback Strategy.

Signal logic (see spec for full details):
  BUY  = 日 EMA20/60/120 密集缠绕 + 周线股价回踩 EMA20 + 多头趋势向上
       ∧ 指数过滤 + ADV20 ≥ 阈值
       (连续 confirmation_days 日满足)

  SELL = 收盘 < 日 EMA60 连续 confirmation_days 日
       ∨ 硬止损: 收盘 ≤ entry_price * (1 - stop_loss_pct)
"""

from collections import deque
from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


_DEFAULTS = {
    "ema_short": 20,
    "ema_mid": 60,
    "ema_long": 120,
    "daily_cluster_threshold": 0.04,
    "weekly_proximity_threshold": 0.025,
    "weekly_ema_period": 20,
    "ema_long_slope_lookback": 20,
    "adv_lookback": 20,
    "adv_threshold_us": 5_000_000,
    "adv_threshold_cn": 50_000_000,
    "market_index_id_us": "SP500",
    "market_index_id_cn": "CSI800",
    "market_ema_period": 200,
    "market_filter_enabled": True,
    "confirmation_days": 2,
    "stop_loss_pct": 0.08,
    "sell_ma_period": 60,
}


@register_strategy("ema_cluster_pullback")
class EMAClusterPullback(BaseStrategy):
    """日线 EMA 密集缠绕 + 周线 EMA20 回踩 + 多头趋势确认."""

    name = "ema_cluster_pullback"
    version = "1.0.0"
    params: dict[str, Any] = dict(_DEFAULTS)

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Merge user params over defaults so get_param() never sees missing keys."""
        merged = dict(_DEFAULTS)
        if params:
            merged.update(params)
        super().__init__(params=merged)

    def init(self, ctx: StrategyContext) -> None:
        """Vectorized precompute of all indicators."""
        s = self.get_param("ema_short")
        m = self.get_param("ema_mid")
        l = self.get_param("ema_long")
        w = self.get_param("weekly_ema_period")

        ctx.precompute_indicator("EMA", period=s)
        ctx.precompute_indicator("EMA", period=m)
        ctx.precompute_indicator("EMA", period=l)
        ctx.precompute_weekly_indicator("EMA", period=w)

        # ADV20 = rolling mean of close*volume (precompute for fast lookup)
        self._adv20_fast = self._compute_adv20_fast(
            ctx._data, lookback=self.get_param("adv_lookback")
        )

        self._market_ema_cache: dict[tuple, float | None] = {}
        self._entry_price: dict[str, float] = {}
        self._buy_pass_history: dict[str, deque] = {}
        self._sell_break_history: dict[str, deque] = {}

        self._full_data = ctx._data

    @staticmethod
    def _compute_adv20_fast(
        df: pl.DataFrame | None, lookback: int
    ) -> dict[tuple, float]:
        """Build {(iid, date): adv} dict for fast O(1) lookup."""
        if df is None or df.is_empty():
            return {}
        with_adv = df.sort("date").with_columns(
            (pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=lookback)
            .over("instrument_id")
            .alias("_adv")
        )
        return {
            (iid, dt): val
            for iid, dt, val in with_adv.select(
                ["instrument_id", "date", "_adv"]
            ).iter_rows()
            if val is not None
        }

    def next(self, ctx: StrategyContext) -> None:
        """Implemented in subsequent tasks."""
        raise NotImplementedError("next() implemented in Tasks 8-9")