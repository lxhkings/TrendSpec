"""
Minervini Trend Template Strategy.

Pure-technical implementation of Mark Minervini's 6-criteria Trend Template.

Buy logic: event-driven -- when a stock passes all 6 criteria for
`confirmation_days` consecutive days and is not already held.
Sell logic: when a held stock fails the criteria for `confirmation_days`
consecutive days.
"""

from collections import deque
from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("minervini_trend")
class MinerviniTrendTemplate(BaseStrategy):
    """Minervini Trend Template -- 6-criteria pure-technical momentum filter."""

    name = "minervini_trend"
    version = "1.0.0"
    params: dict[str, Any] = {
        "ma_short": 50,
        "ma_mid": 150,
        "ma_long": 200,
        "ma_slope_lookback": 20,
        "high_low_lookback": 252,
        "low_distance_min": 0.30,
        "high_distance_max": 0.25,
        "rs_period": 252,
        "rs_threshold": 70.0,
        "market_index_id": "SP500",
        "market_ma_short": 50,
        "market_ma_long": 200,
        "confirmation_days": 2,
    }

    def _validate_dict_params(self) -> None:
        rs = self.get_param("rs_threshold", 70.0)
        conf = self.get_param("confirmation_days", 2)
        if not (0 <= rs <= 100):
            raise ValueError(f"rs_threshold ({rs}) must be 0-100")
        if conf < 1:
            raise ValueError(f"confirmation_days ({conf}) must be >= 1")

    def init(self, ctx: StrategyContext) -> None:
        """Precompute all stock-level indicators over the full dataset."""
        ma_s = self.get_param("ma_short", 50)
        ma_m = self.get_param("ma_mid", 150)
        ma_l = self.get_param("ma_long", 200)
        hl = self.get_param("high_low_lookback", 252)
        rs_p = self.get_param("rs_period", 252)

        ctx.precompute_indicator("MA", period=ma_s)
        ctx.precompute_indicator("MA", period=ma_m)
        ctx.precompute_indicator("MA", period=ma_l)
        ctx.precompute_indicator("HH", period=hl)
        ctx.precompute_indicator("LL", period=hl)
        ctx.precompute_indicator("RS_RATING", period=rs_p)

        self._ma_s = ma_s
        self._ma_m = ma_m
        self._ma_l = ma_l
        self._ma_slope_lb = self.get_param("ma_slope_lookback", 20)
        self._hl = hl
        self._low_min = self.get_param("low_distance_min", 0.30)
        self._high_max = self.get_param("high_distance_max", 0.25)
        self._rs_p = rs_p
        self._rs_thr = self.get_param("rs_threshold", 70.0)
        self._index_id = self.get_param("market_index_id", "SP500")
        self._idx_ma_s = self.get_param("market_ma_short", 50)
        self._idx_ma_l = self.get_param("market_ma_long", 200)
        self._conf = self.get_param("confirmation_days", 2)

        # Pass history: instrument_id -> deque of booleans (last conf days)
        self._pass_history: dict[str, deque] = {}

        # Cache index MA series
        self._index_ma_cache: dict[tuple, float | None] = {}

        self._full_data = ctx._data
        ctx.strategy.log(
            f"Initialized: conf={self._conf}, rs>={self._rs_thr}, idx={self._index_id}"
        )

    def _market_passes(self, ctx: StrategyContext, current_date: DateType) -> bool:
        """Check market filter: index close > index MA50 AND > index MA200."""
        idx_close = ctx.index_close(self._index_id, current_date)
        if idx_close is None:
            return True  # if no data, skip market filter

        def _index_ma(lookback: int) -> float | None:
            cache_key = (self._index_id, current_date, lookback)
            if cache_key in self._index_ma_cache:
                return self._index_ma_cache[cache_key]
            if not hasattr(ctx, "_indices_cache") or ctx._indices_cache is None:
                return None
            series = (
                ctx._indices_cache
                .filter(pl.col("instrument_id") == self._index_id)
                .sort("date")
                .filter(pl.col("date") <= current_date)
                .tail(lookback)["close"]
            )
            val = float(series.mean()) if len(series) >= lookback else None
            self._index_ma_cache[cache_key] = val
            return val

        ma_s = _index_ma(self._idx_ma_s)
        ma_l = _index_ma(self._idx_ma_l)

        passes = True
        if ma_s is not None and idx_close <= ma_s:
            passes = False
        if ma_l is not None and idx_close <= ma_l:
            passes = False
        return passes

    def next(self, ctx: StrategyContext) -> None:
        """Check Trend Template criteria and emit BUY/SELL on confirmed state changes."""
        iid = ctx.instrument_id
        current_date = ctx.date
        d = current_date

        # --- Stock-level criteria ---
        ma_s = ctx.indicator_value("MA", iid, d, period=self._ma_s)
        ma_m = ctx.indicator_value("MA", iid, d, period=self._ma_m)
        ma_l = ctx.indicator_value("MA", iid, d, period=self._ma_l)
        hh = ctx.indicator_value("HH", iid, d, period=self._hl)
        ll = ctx.indicator_value("LL", iid, d, period=self._hl)
        rs = ctx.indicator_value("RS_RATING", iid, d, period=self._rs_p)
        close = ctx.close

        if any(v is None for v in [ma_s, ma_m, ma_l, hh, ll, rs]):
            self._pass_history.setdefault(iid, deque(maxlen=self._conf)).append(False)
            return

        # Rule 1: Price alignment
        rule1 = close > ma_s > ma_m > ma_l

        # Rule 3: Distance from 52-week low
        rule3 = ll > 0 and (close - ll) / ll >= self._low_min

        # Rule 4: Distance from 52-week high
        rule4 = hh > 0 and (hh - close) / hh <= self._high_max

        # Rule 5: RS Rating threshold
        rule5 = rs >= self._rs_thr

        # Rule 2: MA200 slope
        day_data_prev = self._full_data.filter(
            pl.col("instrument_id") == iid
        ).sort("date")
        today_idx = day_data_prev["date"].search_sorted(d, side="left")
        if today_idx >= self._ma_slope_lb:
            prev_date = day_data_prev["date"][today_idx - self._ma_slope_lb]
            ma_l_prev = ctx.indicator_value(
                "MA", iid, prev_date, period=self._ma_l
            )
            rule2 = ma_l_prev is not None and ma_l > ma_l_prev
        else:
            rule2 = False

        # Rule 6: Market filter
        rule6 = self._market_passes(ctx, d)

        passes = rule1 and rule2 and rule3 and rule4 and rule5 and rule6

        history = self._pass_history.setdefault(iid, deque(maxlen=self._conf))
        history.append(passes)

        if len(history) < self._conf:
            return

        all_pass = all(history)
        all_fail = not any(history)

        if all_pass and not ctx.has_position(iid):
            ctx.signal(
                "BUY", iid, close, trigger_value=rs,
                note=f"TT pass: RS={rs:.0f}, MA stack ok"
            )
        elif all_fail and ctx.has_position(iid):
            ctx.signal("SELL", iid, close, note="TT fail: criteria lost")
