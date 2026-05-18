"""
Qullamaggie Momentum Breakout Strategy.

Based on Kristjan Kullamägi's ("Qullamaggie") episodic-pivot / momentum-breakout
playbook: trade high-ADR stocks already in a strong move, wait for tight
consolidation, buy the volume-confirmed breakout, partial sell after a few bars,
trail the rest with a short MA.

See `docs/superpowers/specs/2026-05-17-qullamaggie-momentum-spec.md` for full
design.
"""

from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("qullamaggie_momentum")
class QullamaggieMomentumStrategy(BaseStrategy):
    """Qullamaggie-style momentum breakout with partial sell + MA trailing exit."""

    name = "qullamaggie_momentum"
    version = "1.0.0"
    params: dict[str, Any] = {
        "ma_short_period": 10,
        "ma_mid_period": 20,
        "ma_long_period": 50,
        "roc_period": 60,
        "prior_move_threshold": 0.30,
        "adr_period": 20,
        "adr_pct_min": 0.04,
        "dollar_volume_min": 5_000_000,
        "consolidation_days": 5,
        "consolidation_tightness": 1.5,
        "volume_mult": 1.5,
        "partial_sell_after_days": 4,
        "partial_sell_fraction": 0.5,
        "trail_ma_period": 10,
        "risk_pct": 0.005,
    }

    def _validate_dict_params(self) -> None:
        risk_pct = self.get_param("risk_pct", 0.005)
        psf = self.get_param("partial_sell_fraction", 0.5)
        adr_min = self.get_param("adr_pct_min", 0.04)
        cons_days = self.get_param("consolidation_days", 5)

        if not (0 < risk_pct < 1):
            raise ValueError(f"risk_pct ({risk_pct}) must be in (0, 1)")
        if not (0 <= psf <= 1):
            raise ValueError(f"partial_sell_fraction ({psf}) must be in [0, 1]")
        if adr_min < 0:
            raise ValueError(f"adr_pct_min ({adr_min}) must be >= 0")
        if cons_days < 2:
            raise ValueError(f"consolidation_days ({cons_days}) must be >= 2")

    def init(self, ctx: StrategyContext) -> None:
        """Precompute every indicator the strategy reads in next() and cache params."""
        ma_short = self.get_param("ma_short_period", 10)
        ma_mid = self.get_param("ma_mid_period", 20)
        ma_long = self.get_param("ma_long_period", 50)
        roc_period = self.get_param("roc_period", 60)
        adr_period = self.get_param("adr_period", 20)
        cons_days = self.get_param("consolidation_days", 5)
        trail_ma = self.get_param("trail_ma_period", 10)

        # Precompute indicators
        ctx.precompute_indicator("MA", period=ma_short)
        ctx.precompute_indicator("MA", period=ma_mid)
        ctx.precompute_indicator("MA", period=ma_long)
        if trail_ma not in (ma_short, ma_mid, ma_long):
            ctx.precompute_indicator("MA", period=trail_ma)
        ctx.precompute_indicator("ROC", period=roc_period)
        ctx.precompute_indicator("VMA", period=ma_mid)
        ctx.precompute_indicator("ADR_PCT", period=adr_period)
        ctx.precompute_indicator("HH", period=cons_days)
        ctx.precompute_indicator("LL", period=cons_days)

        # Cache resolved param values for fast lookup in next()
        self._ma_short = ma_short
        self._ma_mid = ma_mid
        self._ma_long = ma_long
        self._roc_period = roc_period
        self._prior_move_threshold = self.get_param("prior_move_threshold", 0.30)
        self._adr_period = adr_period
        self._adr_pct_min = self.get_param("adr_pct_min", 0.04)
        self._dollar_volume_min = self.get_param("dollar_volume_min", 5_000_000)
        self._consolidation_days = cons_days
        self._consolidation_tightness = self.get_param("consolidation_tightness", 1.5)
        self._volume_mult = self.get_param("volume_mult", 1.5)
        self._partial_sell_after_days = self.get_param("partial_sell_after_days", 4)
        self._partial_sell_fraction = self.get_param("partial_sell_fraction", 0.5)
        self._trail_ma_period = trail_ma
        self._risk_pct = self.get_param("risk_pct", 0.005)

        # Per-instrument runtime state
        self._position_state: dict[str, dict[str, Any]] = {}

        # Cache full data for cross-bar history lookups
        self._full_data = ctx._data

        ctx.strategy.log(
            f"Initialized: ma=({ma_short}/{ma_mid}/{ma_long}), trail_ma={trail_ma}, "
            f"adr>={self._adr_pct_min}, cons_days={cons_days}, "
            f"partial_sell_after={self._partial_sell_after_days} bars"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Per-instrument event-driven entry/exit.

        Task 4: BUY path complete. SELL paths (Tasks 5-7) come next.
        """
        iid = ctx.instrument_id
        d = ctx.date

        # ---- Exit path (Tasks 5-7) ----
        if ctx.has_position(iid):
            self._handle_exit(ctx, iid, d)
            return

        # ---- Entry path ----
        if not self._passes_universe_filter(ctx, iid, d):
            return

        cons = self._evaluate_consolidation(ctx, iid, d)
        if cons is None:
            return

        if not self._breakout_today(ctx, iid, d, cons):
            return

        entry_price = ctx.close
        stop = cons["low"]
        if stop <= 0 or entry_price <= stop:
            return

        # Risk-based sizing: shares = nav * risk_pct / (entry - stop)
        nav = ctx.available_capital
        for held_iid, qty in ctx.positions.items():
            held_close = self._get_close(held_iid, d)
            if held_close is not None:
                nav += qty * held_close

        risk_per_share = entry_price - stop
        shares = int(nav * self._risk_pct / risk_per_share)
        if shares < 1:
            return

        sig = ctx.signal(
            "BUY",
            iid,
            entry_price,
            trigger_value=cons["range_pct"],
            note=(
                f"Q-BO: cons_low={stop:.2f}, cons_high={cons['high']:.2f}, "
                f"range_pct={cons['range_pct']:.4f}, shares={shares}"
            ),
        )
        sig.shares = float(shares)

        self._position_state[iid] = {
            "entry_price": entry_price,
            "entry_date": d,
            "shares": shares,
            "initial_shares": shares,
            "half_sold": False,
            "stop": stop,
            "bars_since_entry": 0,
        }

    # =========================================================================
    # Entry helpers
    # =========================================================================

    def _passes_universe_filter(
        self, ctx: StrategyContext, iid: str, d: DateType
    ) -> bool:
        close = ctx.close
        volume = ctx.volume

        ma_short = ctx.indicator_value("MA", iid, d, period=self._ma_short)
        ma_mid = ctx.indicator_value("MA", iid, d, period=self._ma_mid)
        ma_long = ctx.indicator_value("MA", iid, d, period=self._ma_long)
        roc = ctx.indicator_value("ROC", iid, d, period=self._roc_period)
        adr = ctx.indicator_value("ADR_PCT", iid, d, period=self._adr_period)

        if any(v is None for v in (ma_short, ma_mid, ma_long, roc, adr)):
            return False
        if close <= ma_long:
            return False
        if not (ma_short > ma_mid > ma_long):
            return False
        if roc < self._prior_move_threshold * 100:
            # ROC indicator output is in percent (close/close_n - 1) * 100
            return False
        if adr < self._adr_pct_min:
            return False
        if close * volume < self._dollar_volume_min:
            return False
        return True

    def _evaluate_consolidation(
        self, ctx: StrategyContext, iid: str, d: DateType
    ) -> dict | None:
        """
        Inspect the prior `consolidation_days` bars (excluding today).

        Returns a dict {high, low, range_pct} when the window is tight enough
        AND no bar in the window violated the MA20 floor; else None.
        """
        df = self._full_data.filter(pl.col("instrument_id") == iid).sort("date")
        today_idx = df["date"].search_sorted(d, side="left")
        start = today_idx - self._consolidation_days
        if start < 0:
            return None

        window = df.slice(start, self._consolidation_days)
        if len(window) < self._consolidation_days:
            return None

        cons_high = float(window["high"].max())
        cons_low = float(window["low"].min())
        close = ctx.close
        if close <= 0:
            return None
        range_pct = (cons_high - cons_low) / close

        adr = ctx.indicator_value("ADR_PCT", iid, d, period=self._adr_period)
        if adr is None or adr <= 0:
            return None
        if range_pct > self._consolidation_tightness * adr:
            return None

        ma_mid = ctx.indicator_value("MA", iid, d, period=self._ma_mid)
        if ma_mid is None:
            return None
        if cons_low <= ma_mid:
            return None

        return {"high": cons_high, "low": cons_low, "range_pct": range_pct}

    def _breakout_today(
        self,
        ctx: StrategyContext,
        iid: str,
        d: DateType,
        cons: dict,
    ) -> bool:
        close = ctx.close
        if close <= cons["high"]:
            return False

        vma = ctx.indicator_value("VMA", iid, d, period=self._ma_mid)
        if vma is None or ctx.volume <= self._volume_mult * vma:
            return False

        # Prior close — load directly from data
        df = self._full_data.filter(pl.col("instrument_id") == iid).sort("date")
        today_idx = df["date"].search_sorted(d, side="left")
        if today_idx < 1:
            return False
        prev_close = float(df["close"][today_idx - 1])
        return close > prev_close

    # =========================================================================
    # Exit handler (filled in by Tasks 5-7)
    # =========================================================================

    def _handle_exit(self, ctx: StrategyContext, iid: str, d: DateType) -> None:
        """
        Per-bar exit handler for a held position.

        Order of evaluation each bar:
          1. Increment bars_since_entry.
          2. If bars_since_entry >= partial_sell_after_days and not half_sold yet,
             emit a SELL for partial_sell_fraction of remaining shares.
          3. (Task 7 will add trailing exit + stop-loss here.)
        """
        st = self._position_state.get(iid)
        if st is None:
            return

        st["bars_since_entry"] += 1

        if (
            not st["half_sold"]
            and st["bars_since_entry"] >= self._partial_sell_after_days
        ):
            sell_qty = int(st["shares"] * self._partial_sell_fraction)
            if sell_qty >= 1:
                sig = ctx.signal(
                    "SELL",
                    iid,
                    ctx.close,
                    trigger_value=float(st["bars_since_entry"]),
                    note=f"Q partial 1/{int(1/self._partial_sell_fraction)}",
                )
                sig.shares = float(sell_qty)
                st["shares"] -= sell_qty
                st["half_sold"] = True

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_close(self, iid: str, d: DateType) -> float | None:
        rows = self._full_data.filter(
            (pl.col("instrument_id") == iid) & (pl.col("date") == d)
        )
        if rows.is_empty():
            return None
        return float(rows["close"].item())
