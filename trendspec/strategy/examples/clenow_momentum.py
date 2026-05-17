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
    top_pct (float): Fraction of ranked universe to hold. Default 0.8 (top 80%).
    max_gap (float): Maximum allowed single-day drop (negative). Default -0.15.
"""

from datetime import date as DateType

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("clenow_momentum")
class ClenowMomentumStrategy(BaseStrategy):
    """
    Clenow quantitative momentum strategy (Stocks on the Move).

    Ranks universe stocks by exponential regression slope × R² and holds
    the top fraction, sized by ATR-based risk parity. Rebalances weekly.

    Parameters:
        sma_period: Trend filter period (default: 200)
        atr_period: ATR period for position sizing (default: 20)
        score_period: Regression lookback in trading days (default: 90)
        gap_period: Gap filter lookback window (default: 90)
        risk_factor: Equity fraction per ATR unit (default: 0.001)
        rebalance_weekday: 0=Mon…4=Fri (default: 2 = Wednesday)
        top_pct: Fraction of ranked universe to hold (default: 0.8)
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
        "top_pct": 0.8,
        "max_gap": -0.15,
    }

    def _validate_dict_params(self) -> None:
        top_pct = self.get_param("top_pct", 0.8)
        risk_factor = self.get_param("risk_factor", 0.001)
        rebalance_weekday = self.get_param("rebalance_weekday", 2)

        if not (0 < top_pct < 1):
            raise ValueError(f"top_pct ({top_pct}) must be between 0 and 1 exclusive")
        if risk_factor <= 0:
            raise ValueError(f"risk_factor ({risk_factor}) must be > 0")
        if rebalance_weekday not in range(5):
            raise ValueError(f"rebalance_weekday ({rebalance_weekday}) must be 0-4 (Mon-Fri)")

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

        self._sma_period = sma_period
        self._atr_period = atr_period
        self._score_period = score_period
        self._gap_period = gap_period
        self._risk_factor = self.get_param("risk_factor", 0.001)
        self._rebalance_weekday = self.get_param("rebalance_weekday", 2)
        self._top_pct = self.get_param("top_pct", 0.8)
        self._max_gap = self.get_param("max_gap", -0.15)

        self._last_rebalance_date: DateType | None = None
        self._full_data = ctx._data

        ctx.strategy.log(
            f"Initialized: sma={sma_period}, atr={atr_period}, "
            f"score_period={score_period}, weekday={self._rebalance_weekday}, "
            f"top_pct={self._top_pct}"
        )

    def next(self, ctx: StrategyContext) -> None:
        """
        Weekly rebalancing via cross-sectional momentum ranking.

        Only runs on the configured weekday. The first instrument call of a
        rebalance day does all the work; subsequent calls return immediately.
        """
        current_date = ctx.date

        if current_date.weekday() != self._rebalance_weekday:
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
        n_keep = max(1, int(len(ranked) * self._top_pct))
        top_set = set(ranked[:n_keep])

        # Compute total equity for position sizing
        nav = ctx.available_capital
        for iid, qty in ctx.positions.items():
            close = get_close(iid)
            if close is not None:
                nav += qty * close

        # SELL: positions no longer in top set or below trend filter
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
            elif iid not in top_set:
                sell_reason = "rank out of top qualifying universe"

            if sell_reason:
                sig = ctx.signal("SELL", iid, close, note=sell_reason)
                sig.ticker = get_ticker(iid)

        # BUY: top-ranked stocks not already held
        for iid in ranked[:n_keep]:
            if ctx.has_position(iid):
                continue

            atr = ctx.indicator_value("ATR", iid, current_date, period=self._atr_period)
            close = get_close(iid)

            if atr is None or atr <= 0 or close is None or close <= 0:
                continue

            shares = int(nav * self._risk_factor / atr)
            if shares < 1:
                continue

            sig = ctx.signal(
                "BUY",
                iid,
                close,
                trigger_value=scores[iid],
                note=f"score={scores[iid]:.2f}, atr={atr:.2f}, shares={shares}",
            )
            sig.ticker = get_ticker(iid)
            sig.shares = float(shares)
