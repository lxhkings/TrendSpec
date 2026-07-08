"""
Clenow Quantitative Momentum Strategy.

Based on Andreas Clenow's "Stocks on the Move".

Strategy logic:
- Score = annualized exponential regression slope × R² over 90-day window
- Filters: price > SMA(200), no single day drop > 15% in 90 days, score > 0
- Rank all qualifying universe stocks by score descending
- Weekly rebalance (default: Wednesday):
    SELL: current positions that dropped below 200 SMA, or rank fell out of top 80%
    BUY:  top-ranked stocks not yet held (ATR-based position sizing)
- Position size: int(total_equity × risk_factor / ATR(20))

Parameters:
    sma_period (int): Trend filter SMA period. Default 200.
    atr_period (int): ATR period for position sizing. Default 20.
    score_period (int): Regression lookback in trading days. Default 90.
    gap_period (int): Window for gap filter (matches score_period). Default 90.
    risk_factor (float): Equity fraction per ATR unit. Default 0.001.
    rebalance_weekday (int): 0=Mon … 4=Fri. Default 2 (Wednesday).
    top_n (int): Max number of positions held. Default 20.
    sell_rank_mult (float): Exit buffer — a held position is only sold for
        rank reasons once its rank exceeds top_n * sell_rank_mult. Default 1.5.
    cash_buffer (float): Fraction of available capital held back from new
        BUYs, absorbing execution slippage. Default 0.02.
    max_gap (float): Maximum allowed single-day drop (negative). Default -0.15.
"""

from datetime import date as DateType

import polars as pl

from trendspec.data.sectors import sector as sector_lookup
from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("clenow_momentum")
class ClenowMomentumStrategy(BaseStrategy):
    """
    Clenow quantitative momentum strategy (Stocks on the Move).

    Ranks universe stocks by exponential regression slope × R² and holds
    the top top_n names, sized by ATR-based risk parity within a strict
    cash budget (zero leverage). Rebalances weekly.

    Parameters:
        sma_period: Trend filter period (default: 200)
        atr_period: ATR period for position sizing (default: 20)
        score_period: Regression lookback in trading days (default: 90)
        gap_period: Gap filter lookback window (default: 90)
        risk_factor: Equity fraction per ATR unit (default: 0.001)
        rebalance_weekday: 0=Mon…4=Fri (default: 2 = Wednesday)
        top_n: Max number of positions held (default: 20)
        sell_rank_mult: Exit buffer multiplier on top_n (default: 1.5)
        cash_buffer: Fraction of capital held back from new BUYs (default: 0.02)
        max_gap: Max allowed single-day drop, e.g. -0.15 (default: -0.15)
    """

    name = "clenow_momentum"
    version = "1.0.0"
    params = {
        "sma_period": 200,
        "atr_period": 20,
        "score_period": 90,
        "gap_period": 90,
        "risk_factor": 0.001,
        "rebalance_weekday": 2,
        "top_n": 20,
        "sell_rank_mult": 1.5,
        "cash_buffer": 0.02,
        "max_gap": -0.15,
        "max_per_sector": 0,       # 0 = 不限；1 = 每行业最多1只
        # Display-only fields (do not affect entry/exit logic)
        "atr_stop_k": 3.0,
        "drawdown_period": 63,
        "volume_avg_period": 50,
        "warn_deviation_max": 40.0,
        "warn_vol_mult_low": 1.0,
        "warn_vol_mult_high": 3.0,
        "warn_drawdown_max": -15.0,
    }

    def _validate_dict_params(self) -> None:
        # Merge class-level defaults into instance dict so get_param() works
        # without callers needing to supply the default value.
        self.params = {**self.params}
        for key, value in self.__class__.params.items():
            self.params.setdefault(key, value)

        top_n = self.get_param("top_n", 20)
        sell_rank_mult = self.get_param("sell_rank_mult", 1.5)
        cash_buffer = self.get_param("cash_buffer", 0.02)
        risk_factor = self.get_param("risk_factor", 0.001)
        rebalance_weekday = self.get_param("rebalance_weekday", 2)

        if top_n < 1:
            raise ValueError(f"top_n ({top_n}) must be >= 1")
        if sell_rank_mult < 1.0:
            raise ValueError(f"sell_rank_mult ({sell_rank_mult}) must be >= 1.0")
        if not (0 <= cash_buffer < 1):
            raise ValueError(f"cash_buffer ({cash_buffer}) must be in [0, 1)")
        if risk_factor <= 0:
            raise ValueError(f"risk_factor ({risk_factor}) must be > 0")
        if rebalance_weekday not in range(5):
            raise ValueError(f"rebalance_weekday ({rebalance_weekday}) must be 0-4 (Mon-Fri)")

        atr_stop_k = self.get_param("atr_stop_k", 3.0)
        drawdown_period = self.get_param("drawdown_period", 63)
        volume_avg_period = self.get_param("volume_avg_period", 50)
        warn_deviation_max = self.get_param("warn_deviation_max", 40.0)
        warn_vol_mult_low = self.get_param("warn_vol_mult_low", 1.0)
        warn_vol_mult_high = self.get_param("warn_vol_mult_high", 3.0)
        warn_drawdown_max = self.get_param("warn_drawdown_max", -15.0)

        if atr_stop_k <= 0:
            raise ValueError(f"atr_stop_k ({atr_stop_k}) must be > 0")
        if drawdown_period < 2:
            raise ValueError(f"drawdown_period ({drawdown_period}) must be >= 2")
        if volume_avg_period < 2:
            raise ValueError(f"volume_avg_period ({volume_avg_period}) must be >= 2")
        if warn_deviation_max <= 0:
            raise ValueError(f"warn_deviation_max ({warn_deviation_max}) must be > 0")
        if warn_drawdown_max >= 0:
            raise ValueError(f"warn_drawdown_max ({warn_drawdown_max}) must be < 0")
        if warn_vol_mult_low >= warn_vol_mult_high:
            raise ValueError(
                f"warn_vol_mult_low ({warn_vol_mult_low}) must be < "
                f"warn_vol_mult_high ({warn_vol_mult_high})"
            )

    def init(self, ctx: StrategyContext) -> None:
        """Precompute all indicators once over the full dataset."""
        sma_period = self.get_param("sma_period", 200)
        atr_period = self.get_param("atr_period", 20)
        score_period = self.get_param("score_period", 90)
        gap_period = self.get_param("gap_period", 90)

        ctx.precompute_indicator("MA", period=sma_period)
        ctx.precompute_indicator("ATR", period=atr_period)
        ctx.precompute_indicator("CLENOW_SCORE", period=score_period)
        ctx.precompute_indicator("MIN_DAILY_RETURN", period=gap_period)
        ctx.precompute_indicator("HH", period=self.get_param("drawdown_period", 63))
        ctx.precompute_indicator("SMA_VOLUME", period=self.get_param("volume_avg_period", 50))
        ctx.precompute_indicator("CLENOW_R2", period=score_period)

        self._sma_period = sma_period
        self._atr_period = atr_period
        self._score_period = score_period
        self._gap_period = gap_period
        self._risk_factor = self.get_param("risk_factor", 0.001)
        self._rebalance_weekday = self.get_param("rebalance_weekday", 2)
        self._top_n = int(self.get_param("top_n", 20))
        self._sell_rank_mult = self.get_param("sell_rank_mult", 1.5)
        self._cash_buffer = self.get_param("cash_buffer", 0.02)
        self._max_gap = self.get_param("max_gap", -0.15)
        self._max_per_sector = int(self.get_param("max_per_sector", 0))
        self._drawdown_period = self.get_param("drawdown_period", 63)
        self._volume_avg_period = self.get_param("volume_avg_period", 50)
        self._atr_stop_k = self.get_param("atr_stop_k", 3.0)
        self._warn_deviation_max = self.get_param("warn_deviation_max", 40.0)
        self._warn_vol_mult_low = self.get_param("warn_vol_mult_low", 1.0)
        self._warn_vol_mult_high = self.get_param("warn_vol_mult_high", 3.0)
        self._warn_drawdown_max = self.get_param("warn_drawdown_max", -15.0)

        self._last_rebalance_date: DateType | None = None
        self._full_data = ctx._data

        ctx.strategy.log(
            f"Initialized: sma={sma_period}, atr={atr_period}, "
            f"score_period={score_period}, weekday={self._rebalance_weekday}, "
            f"top_n={self._top_n}"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Weekly rebalancing via cross-sectional momentum ranking.

        Only runs on the configured weekday. The first instrument call of a
        rebalance day does all the work; subsequent calls return immediately.
        """
        current_date = ctx.date

        if not ctx.is_screening and current_date.weekday() != self._rebalance_weekday:
            return

        if current_date == self._last_rebalance_date:
            return

        self._last_rebalance_date = current_date

        day_data = self._full_data.filter(pl.col("date") == current_date)
        if day_data.is_empty():
            return

        def get_close(instrument_id: str) -> float | None:
            rows = day_data.filter(pl.col("instrument_id") == instrument_id)
            return rows["close"].item() if not rows.is_empty() else None

        def get_ticker(instrument_id: str) -> str:
            rows = day_data.filter(pl.col("instrument_id") == instrument_id)
            return rows["ticker"].item() if not rows.is_empty() else instrument_id

        # --- Score qualifying universe instruments ---
        universe_ids = ctx.pit_universe(current_date)
        scores: dict[str, float] = {}

        for iid in universe_ids:
            sma = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            score = ctx.indicator_value("CLENOW_SCORE", iid, current_date, period=self._score_period)
            min_ret = ctx.indicator_value("MIN_DAILY_RETURN", iid, current_date, period=self._gap_period)
            close = get_close(iid)

            if sma is None or score is None or min_ret is None or close is None:
                continue
            if close <= sma:
                continue
            if min_ret < self._max_gap:
                continue
            if score <= 0:
                continue

            scores[iid] = score

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)

        if self._max_per_sector > 0:
            seen: dict[str, int] = {}
            deduped = []
            for iid in ranked:
                sec = sector_lookup(ctx.market, iid, current_date) or ""
                if seen.get(sec, 0) < self._max_per_sector:
                    deduped.append(iid)
                    seen[sec] = seen.get(sec, 0) + 1
            ranked = deduped

        # Rank map (1-based) over the full qualifying+deduped list — used both
        # to decide sells (buffer band) and to label BUY signal extras["rank"].
        rank_of = {iid: pos for pos, iid in enumerate(ranked, start=1)}
        entry_candidates = ranked[: self._top_n]
        exit_rank_threshold = int(self._top_n * self._sell_rank_mult)
        retain_set = set(ranked[:exit_rank_threshold])

        # Compute total equity for position sizing
        nav = ctx.available_capital
        for iid, qty in ctx.positions.items():
            close = get_close(iid)
            if close is not None:
                nav += qty * close

        # SELL: below trend filter, no longer qualifying, or rank fell outside
        # the buffer band (top_n * sell_rank_mult). Always sells the FULL
        # position — no partial-exit residue.
        for iid in list(ctx.positions.keys()):
            sma = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            close = get_close(iid)

            sell_reason = None
            if close is None:
                # No price data (halt/delist). Cannot sell at unknown price.
                # Position held until price returns. Log for visibility.
                ctx.strategy.log(f"WARN: no price for held position {iid} on {current_date} — skipping SELL")
                continue
            elif sma is not None and close <= sma:
                sell_reason = f"below SMA{self._sma_period}"
            elif iid not in rank_of:
                sell_reason = "no longer qualifies"
            elif iid not in retain_set:
                sell_reason = f"rank out of top {exit_rank_threshold}"

            if sell_reason:
                sig = ctx.signal("SELL", iid, close, note=sell_reason)
                sig.ticker = get_ticker(iid)
                sig.shares = float(ctx.positions[iid])

        # BUY: top-ranked stocks not already held, sized by ATR risk parity and
        # capped by both open slots (top_n) and remaining cash budget — the
        # combination guarantees zero leverage (sum of buys never exceeds
        # available cash) and a hard position-count ceiling.
        slots = max(0, self._top_n - len(ctx.positions))
        available = ctx.available_capital * (1 - self._cash_buffer)
        filled = 0

        for iid in entry_candidates:
            if filled >= slots:
                break
            if ctx.has_position(iid):
                continue

            atr = ctx.indicator_value("ATR", iid, current_date, period=self._atr_period)
            close = get_close(iid)

            if atr is None or atr <= 0 or close is None or close <= 0:
                continue

            target_shares = int(nav * self._risk_factor / atr)
            if ctx.is_screening:
                shares = target_shares
            else:
                affordable_shares = int(available / close)
                shares = min(target_shares, affordable_shares)
            if shares < 1:
                continue

            ma200 = ctx.indicator_value("MA", iid, current_date, period=self._sma_period)
            hh = ctx.indicator_value("HH", iid, current_date, period=self._drawdown_period)
            vol_avg = ctx.indicator_value(
                "SMA_VOLUME", iid, current_date, period=self._volume_avg_period
            )
            r2 = ctx.indicator_value("CLENOW_R2", iid, current_date, period=self._score_period)

            day_rows = day_data.filter(pl.col("instrument_id") == iid)
            today_vol = day_rows["volume"].item() if not day_rows.is_empty() else None

            # Display-only fields: use 0.0 fallback — missing data never blocks BUY signal
            deviation_pct = (close - ma200) / ma200 * 100 if (ma200 is not None and ma200 > 0) else 0.0
            drawdown_pct = (close - hh) / hh * 100 if (hh is not None and hh > 0) else 0.0
            vol_mult_valid = vol_avg is not None and vol_avg > 0 and today_vol is not None
            vol_mult = float(today_vol) / float(vol_avg) if vol_mult_valid else 0.0  # type: ignore[arg-type]
            r2_val = float(r2) if r2 is not None else 0.0
            stop_loss = close - self._atr_stop_k * atr

            alerts: list[str] = []
            if deviation_pct > self._warn_deviation_max:
                alerts.append("均线乖离过大")
            if vol_mult_valid and vol_mult < self._warn_vol_mult_low:
                alerts.append("量能萎缩")
            if vol_mult_valid and vol_mult > self._warn_vol_mult_high:
                alerts.append("放量过快")
            if drawdown_pct < self._warn_drawdown_max:
                alerts.append("回撤过深")

            sector_code = sector_lookup(ctx.market, iid, current_date)

            sig = ctx.signal(
                "BUY",
                iid,
                close,
                trigger_value=scores[iid],
                note=f"score={scores[iid]:.2f}, atr={atr:.2f}, shares={shares}",
            )
            sig.ticker = get_ticker(iid)
            sig.shares = float(shares)
            sig.extras = {
                "sector": sector_code,
                "rank": rank_of[iid],
                "r2": r2_val,
                "deviation_pct": float(deviation_pct),
                "drawdown_pct": float(drawdown_pct),
                "vol_mult": float(vol_mult),
                "stop_loss": float(stop_loss),
                "alerts": alerts,
            }

            available -= shares * close
            filled += 1
