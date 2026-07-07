"""
Tests for TrendSpec example strategies.

Tests:
- ClenowMomentumStrategy initialization and signal generation
- Signal.shares field behavior
- Indicator tests: CLENOW_SCORE, MIN_DAILY_RETURN, ADR_PCT, RS_RATING
- Indices loader and index_close context helper
- Strategy comparison analyzer
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.engine.backtest_engine import BacktestEngine
from trendspec.engine.base_engine import EngineConfig
from trendspec.strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    create_strategy,
    get_strategy,
    list_strategies,
    register_strategy,
)


# =============================================================================
# Signal.shares Tests
# =============================================================================


def test_signal_shares_field() -> None:
    """Signal.shares defaults to None and can be set after creation."""
    sig = Signal(direction="BUY", ticker="AAPL", instrument_id="AAPL", price=150.0)
    assert sig.shares is None

    sig.shares = 42.0
    assert sig.shares == 42.0


def test_signal_shares_not_in_repr() -> None:
    """Signal.shares is excluded from repr (like timestamp)."""
    sig = Signal(direction="BUY", ticker="AAPL", instrument_id="AAPL", price=150.0, shares=10.0)
    assert "shares" not in repr(sig)


# =============================================================================
# BacktestEngine signal.shares Tests
# =============================================================================


@pytest.mark.parametrize("custom_shares,expected_shares", [(7, 7), (None, 100)])
def test_backtest_engine_uses_signal_shares(custom_shares, expected_shares) -> None:
    """Engine uses signal.shares when set; falls back to order_size=100 otherwise."""
    from datetime import date
    from unittest.mock import MagicMock, patch

    import polars as pl

    from trendspec.risk.pipeline import RiskPipeline

    day_data = pl.DataFrame({
        "instrument_id": ["AAPL"],
        "ticker": ["AAPL"],
        "date": [date(2024, 1, 2)],
        "open": [180.0], "high": [185.0], "low": [178.0],
        "close": [182.0], "volume": [50_000_000], "adj_factor": [1.0],
    })

    @register_strategy("_test_signal_shares")
    class SharesTestStrategy(BaseStrategy):
        name = "_test_signal_shares"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            if not ctx.has_position(ctx.instrument_id):
                sig = ctx.signal("BUY", ctx.instrument_id, ctx.close)
                if custom_shares is not None:
                    sig.shares = float(custom_shares)

    config = EngineConfig(
        market=Market.US,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        initial_capital=100_000.0,
        order_size=100,
        costs_model="none",
        root="/tmp/nonexistent",
        risk_pipeline=RiskPipeline([]),  # no rules → all signals pass
    )

    engine = BacktestEngine(config)
    engine._data = day_data
    def _ticker_list(_d):
        return ["AAPL"]

    engine._universe = MagicMock(tickers=_ticker_list)

    with (
        patch.object(engine, "load_data"),
        patch.object(engine, "load_universe"),
    ):
        result = engine.run(SharesTestStrategy)

    assert len(result.trades) >= 1, "Expected at least one trade"
    assert all(t.shares == expected_shares for t in result.trades), (
        f"Expected {expected_shares} shares per trade, got: {[t.shares for t in result.trades]}"
    )


def test_backtest_engine_rejects_buy_exceeding_cash() -> None:
    """Engine's cash pre-check skips a BUY that would overdraw cash, even
    though each individual order looks affordable considered in isolation.
    """
    from datetime import date
    from unittest.mock import MagicMock, patch

    import polars as pl

    from trendspec.risk.pipeline import RiskPipeline

    day_data = pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT"],
        "ticker": ["AAPL", "MSFT"],
        "date": [date(2024, 1, 2), date(2024, 1, 2)],
        "open": [100.0, 100.0], "high": [105.0, 105.0], "low": [95.0, 95.0],
        "close": [100.0, 100.0], "volume": [50_000_000, 50_000_000], "adj_factor": [1.0, 1.0],
    })

    @register_strategy("_test_cash_guard")
    class CashGuardTestStrategy(BaseStrategy):
        name = "_test_cash_guard"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            if not ctx.has_position(ctx.instrument_id):
                sig = ctx.signal("BUY", ctx.instrument_id, ctx.close)
                sig.shares = 80.0  # 80 * 100 = 8000 per order; only 1 fits in 10,000 cash

    config = EngineConfig(
        market=Market.US,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        initial_capital=10_000.0,
        order_size=100,
        costs_model="none",
        root="/tmp/nonexistent",
        risk_pipeline=RiskPipeline([]),  # no rules → all signals pass to cash pre-check
    )

    engine = BacktestEngine(config)
    engine._data = day_data

    def _ticker_list(_d):
        return ["AAPL", "MSFT"]

    engine._universe = MagicMock(tickers=_ticker_list)

    with (
        patch.object(engine, "load_data"),
        patch.object(engine, "load_universe"),
    ):
        result = engine.run(CashGuardTestStrategy)

    assert len(result.trades) == 1, (
        f"Expected exactly 1 trade (cash exhausted after first), got {len(result.trades)}"
    )


def test_backtest_engine_credits_same_day_sell_for_buy() -> None:
    """Cash pre-check must credit same-day SELL proceeds before evaluating a
    later BUY in the signal list, mirroring how ClenowMomentumStrategy emits
    SELLs (exits) before BUYs (new entries) within a single rebalance day.
    """
    from datetime import date
    from unittest.mock import MagicMock, patch

    import polars as pl

    from trendspec.risk.pipeline import RiskPipeline

    day_data = pl.DataFrame({
        "instrument_id": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 3)],
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [100.0, 100.0, 100.0, 100.0],
        "low": [100.0, 100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 100.0, 100.0],
        "volume": [50_000_000, 50_000_000, 50_000_000, 50_000_000],
        "adj_factor": [1.0, 1.0, 1.0, 1.0],
    })

    @register_strategy("_test_sell_funds_buy")
    class SellFundsBuyTestStrategy(BaseStrategy):
        name = "_test_sell_funds_buy"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            if ctx.date == date(2024, 1, 2):
                if ctx.instrument_id == "AAPL" and not ctx.has_position("AAPL"):
                    sig = ctx.signal("BUY", "AAPL", ctx.close)
                    sig.shares = 50.0  # 50 * 100 = 5000, exactly the full starting capital
            elif ctx.date == date(2024, 1, 3):
                if ctx.instrument_id == "AAPL" and ctx.has_position("AAPL"):
                    sig = ctx.signal("SELL", "AAPL", ctx.close)
                    sig.shares = 50.0  # frees 5000 in proceeds
                if ctx.instrument_id == "MSFT" and not ctx.has_position("MSFT"):
                    sig = ctx.signal("BUY", "MSFT", ctx.close)
                    sig.shares = 40.0  # 40 * 100 = 4000

    config = EngineConfig(
        market=Market.US,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        initial_capital=5000.0,
        order_size=100,
        costs_model="none",
        root="/tmp/nonexistent",
        risk_pipeline=RiskPipeline([]),  # no rules → all signals pass to cash pre-check
    )

    engine = BacktestEngine(config)
    engine._data = day_data

    def _ticker_list(_d):
        return ["AAPL", "MSFT"]

    engine._universe = MagicMock(tickers=_ticker_list)

    with (
        patch.object(engine, "load_data"),
        patch.object(engine, "load_universe"),
    ):
        result = engine.run(SellFundsBuyTestStrategy)

    assert len(result.trades) == 3, (
        f"Expected 3 trades (day-1 BUY AAPL, day-2 SELL AAPL, day-2 BUY MSFT), "
        f"got {len(result.trades)}: "
        f"{[(t.instrument_id, t.direction, t.shares) for t in result.trades]}"
    )
    msft_trades = [t for t in result.trades if t.instrument_id == "MSFT"]
    assert len(msft_trades) == 1, "Expected exactly one MSFT trade (the day-2 BUY)"
    assert msft_trades[0].direction == "BUY"
    assert msft_trades[0].shares == 40.0


# =============================================================================
# CLENOW_SCORE and MIN_DAILY_RETURN Indicator Tests
# =============================================================================


def _make_price_df(n_days: int = 150) -> pl.DataFrame:
    """Synthetic OHLCV data for two instruments over n_days."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for inst in ["AAA", "BBB"]:
        price = 100.0
        for i in range(n_days):
            price *= 1 + rng.normal(0.001, 0.015)
            rows.append({
                "instrument_id": inst,
                "ticker": inst,
                "date": date(2023, 1, 1) + timedelta(days=i),
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1_000_000,
                "adj_factor": 1.0,
            })
    return pl.DataFrame(rows)


class TestClenowScoreIndicator:
    from trendspec.strategy.indicators import compute_indicator, list_indicators

    def test_registered(self) -> None:
        from trendspec.strategy.indicators import list_indicators
        assert "CLENOW_SCORE" in list_indicators()

    def test_columns_added(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        assert "CLENOW_SCORE_90" in result.columns
        assert "CLENOW_SLOPE_90" in result.columns
        assert "CLENOW_R2_90" in result.columns

    def test_null_before_lookback(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
        assert aaa["CLENOW_SCORE_90"][:89].is_null().all()

    def test_r2_bounded(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(150)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        r2 = result["CLENOW_R2_90"].drop_nulls()
        assert (r2 >= 0).all() and (r2 <= 1).all()

    def test_uptrend_scores_positive(self) -> None:
        """Monotonically increasing prices → positive slope → positive score."""
        from trendspec.strategy.indicators import compute_indicator
        rows = [
            {"instrument_id": "UP", "ticker": "UP",
             "date": date(2023, 1, 1) + timedelta(days=i),
             "open": 100 + i, "high": 101 + i, "low": 99 + i,
             "close": 100.0 + i, "volume": 1_000_000, "adj_factor": 1.0}
            for i in range(120)
        ]
        df = pl.DataFrame(rows)
        result = compute_indicator(df, "CLENOW_SCORE", period=90)
        last = result.filter(pl.col("instrument_id") == "UP").sort("date").tail(1)
        assert last["CLENOW_SCORE_90"].item() > 0


class TestMinDailyReturnIndicator:
    def test_registered(self) -> None:
        from trendspec.strategy.indicators import list_indicators
        assert "MIN_DAILY_RETURN" in list_indicators()

    def test_column_added(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(150)
        result = compute_indicator(df, "MIN_DAILY_RETURN", period=90)
        assert "MIN_DAILY_RETURN_90" in result.columns

    def test_gap_detected(self) -> None:
        """A 20% single-day drop must appear in MIN_DAILY_RETURN < -0.15."""
        from trendspec.strategy.indicators import compute_indicator
        rows = []
        price = 100.0
        for i in range(150):
            if i == 100:
                price *= 0.80  # 20% gap down
            rows.append({
                "instrument_id": "G", "ticker": "G",
                "date": date(2023, 1, 1) + timedelta(days=i),
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 1_000_000, "adj_factor": 1.0,
            })
        df = pl.DataFrame(rows)
        result = compute_indicator(df, "MIN_DAILY_RETURN", period=90)
        g = result.filter(pl.col("instrument_id") == "G").sort("date")
        post_gap = g.filter(pl.col("date") >= date(2023, 1, 1) + timedelta(days=101))
        assert (post_gap["MIN_DAILY_RETURN_90"].drop_nulls() < -0.15).any()


class TestAdrPctIndicator:
    def test_registered(self) -> None:
        from trendspec.strategy.indicators import list_indicators
        assert "ADR_PCT" in list_indicators()

    def test_column_added(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(60)
        result = compute_indicator(df, "ADR_PCT", period=20)
        assert "ADR_PCT_20" in result.columns

    def test_null_before_lookback(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(60)
        result = compute_indicator(df, "ADR_PCT", period=20)
        aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
        # First 19 rows should be null
        assert aaa["ADR_PCT_20"][:19].is_null().all()
        # Row at index 19 should be non-null
        assert aaa["ADR_PCT_20"][19] is not None

    def test_known_values(self) -> None:
        """A constant 5% daily H-L range should produce ADR_PCT approx 0.05."""
        from trendspec.strategy.indicators import compute_indicator
        rows = []
        for i in range(40):
            close = 100.0
            rows.append({
                "instrument_id": "X", "ticker": "X",
                "date": date(2023, 1, 1) + timedelta(days=i),
                "open": close, "high": close * 1.025, "low": close * 0.975,
                "close": close, "volume": 1_000_000, "adj_factor": 1.0,
            })
        df = pl.DataFrame(rows)
        result = compute_indicator(df, "ADR_PCT", period=20)
        last_val = result.sort("date").tail(1)["ADR_PCT_20"].item()
        # (high - low) / close = (102.5 - 97.5) / 100 = 0.05
        assert abs(last_val - 0.05) < 1e-9

    def test_per_instrument_independent(self) -> None:
        """Two instruments are computed independently (no leakage)."""
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(60)
        result = compute_indicator(df, "ADR_PCT", period=20)
        aaa = result.filter(pl.col("instrument_id") == "AAA").sort("date")
        bbb = result.filter(pl.col("instrument_id") == "BBB").sort("date")
        # Both should have null prefix of 19 rows
        assert aaa["ADR_PCT_20"][:19].is_null().all()
        assert bbb["ADR_PCT_20"][:19].is_null().all()


# =============================================================================
# ClenowMomentumStrategy Tests
# =============================================================================


class TestClenowMomentumStrategyInit:
    def test_strategy_registration(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        assert get_strategy("clenow_momentum") is ClenowMomentumStrategy

    def test_default_params(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        strategy = ClenowMomentumStrategy()
        assert strategy.get_param("sma_period", 200) == 200
        assert strategy.get_param("atr_period", 20) == 20
        assert strategy.get_param("score_period", 90) == 90
        assert strategy.get_param("gap_period", 90) == 90
        assert strategy.get_param("risk_factor", 0.001) == 0.001
        assert strategy.get_param("rebalance_weekday", 2) == 2
        assert strategy.get_param("top_pct", 0.8) == 0.8

    def test_invalid_top_pct(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="top_pct"):
            ClenowMomentumStrategy(params={"top_pct": 1.5})

    def test_invalid_risk_factor(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="risk_factor"):
            ClenowMomentumStrategy(params={"risk_factor": -0.001})

    def test_invalid_rebalance_weekday(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="rebalance_weekday"):
            ClenowMomentumStrategy(params={"rebalance_weekday": 7})

    def test_in_list_strategies(self) -> None:
        import trendspec.strategy.examples  # noqa: F401 — trigger registration
        assert "clenow_momentum" in list_strategies()

    def test_new_display_param_defaults(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        s = ClenowMomentumStrategy()
        assert s.get_param("atr_stop_k", None) == 3.0
        assert s.get_param("drawdown_period", None) == 63
        assert s.get_param("volume_avg_period", None) == 50
        assert s.get_param("warn_deviation_max", None) == 40.0
        assert s.get_param("warn_vol_mult_low", None) == 1.0
        assert s.get_param("warn_vol_mult_high", None) == 3.0
        assert s.get_param("warn_drawdown_max", None) == -15.0

    def test_invalid_atr_stop_k(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="atr_stop_k"):
            ClenowMomentumStrategy(params={"atr_stop_k": 0})
        with pytest.raises(ValueError, match="atr_stop_k"):
            ClenowMomentumStrategy(params={"atr_stop_k": -1.0})

    def test_invalid_drawdown_period(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="drawdown_period"):
            ClenowMomentumStrategy(params={"drawdown_period": 1})

    def test_invalid_volume_avg_period(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="volume_avg_period"):
            ClenowMomentumStrategy(params={"volume_avg_period": 1})

    def test_invalid_vol_mult_threshold_ordering(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_vol_mult"):
            ClenowMomentumStrategy(params={
                "warn_vol_mult_low": 3.0, "warn_vol_mult_high": 1.0,
            })

    def test_invalid_warn_drawdown_max_nonneg(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_drawdown_max"):
            ClenowMomentumStrategy(params={"warn_drawdown_max": 0.0})

    def test_invalid_warn_deviation_max_nonpos(self) -> None:
        from trendspec.strategy.examples import ClenowMomentumStrategy
        with pytest.raises(ValueError, match="warn_deviation_max"):
            ClenowMomentumStrategy(params={"warn_deviation_max": 0.0})


class TestClenowMomentumStrategySignals:
    """Integration: init() precomputes indicators without error."""

    def _make_trending_df(self, n_days: int = 300) -> pl.DataFrame:
        import numpy as np
        rng = np.random.default_rng(0)
        rows = []
        for inst, trend in [("UP1", 0.002), ("UP2", 0.0015), ("DOWN", -0.003)]:
            price = 100.0
            for i in range(n_days):
                price = max(1.0, price * (1 + trend + rng.normal(0, 0.005)))
                rows.append({
                    "instrument_id": inst, "ticker": inst,
                    "date": date(2022, 1, 1) + timedelta(days=i),
                    "open": price * 0.995, "high": price * 1.005,
                    "low": price * 0.990, "close": price,
                    "volume": 1_000_000, "adj_factor": 1.0,
                })
        return pl.DataFrame(rows)

    def test_init_precomputes_indicators(self) -> None:
        from trendspec.strategy.context import StrategyContext
        from trendspec.strategy.examples import ClenowMomentumStrategy

        df = self._make_trending_df(300)
        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        cache_keys = list(ctx._indicator_cache.keys())
        assert any("CLENOW_SCORE" in k for k in cache_keys)
        assert any("MIN_DAILY_RETURN" in k for k in cache_keys)
        assert any("ATR" in k for k in cache_keys)
        assert any("MA" in k for k in cache_keys)

    def test_init_precomputes_display_indicators(self) -> None:
        """init() also precomputes HH, SMA_VOLUME, CLENOW_R2 for display fields."""
        from trendspec.strategy.context import StrategyContext
        from trendspec.strategy.examples import ClenowMomentumStrategy

        df = self._make_trending_df(300)
        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
            "drawdown_period": 20, "volume_avg_period": 20,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        cache_keys = list(ctx._indicator_cache.keys())
        assert any("HH" in k for k in cache_keys), f"HH not precomputed: {cache_keys}"
        assert any("SMA_VOLUME" in k for k in cache_keys), f"SMA_VOLUME not precomputed: {cache_keys}"
        assert any("CLENOW_R2" in k for k in cache_keys), f"CLENOW_R2 not precomputed: {cache_keys}"

    def test_next_generates_buy_with_shares_on_rebalance_day(self) -> None:
        """On a rebalance day, strategy generates BUY signals with positive shares."""
        from unittest.mock import MagicMock

        from trendspec.strategy.context import StrategyContext
        from trendspec.strategy.examples import ClenowMomentumStrategy

        df = self._make_trending_df(300)
        instrument_ids = df["instrument_id"].unique().to_list()

        # Find a Wednesday in the data range
        all_dates = df["date"].unique().sort()
        wednesdays = [d for d in all_dates.to_list() if d.weekday() == 2]
        assert wednesdays, "No Wednesdays in synthetic data"
        rebalance_date = wednesdays[-1]

        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50,
            "score_period": 30,
            "gap_period": 30,
            "atr_period": 10,
            "rebalance_weekday": 2,
            "risk_factor": 0.001,
            "top_pct": 0.8,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        # Mock pit_universe to return instruments from synthetic data (no data lake needed)
        mock_universe = MagicMock()
        mock_universe.tickers.return_value = instrument_ids
        ctx.set_universe(mock_universe)

        # Simulate engine: update positions (empty) + available capital
        ctx.update_positions({}, 100_000.0)

        # Feed all instruments for the rebalance date
        def _mock_sector(_m, _iid, _dt):
            return None

        with patch(
            "trendspec.strategy.examples.clenow_momentum.sector_lookup",
            side_effect=_mock_sector,
        ):
            for iid in instrument_ids:
                row = df.filter(
                    (pl.col("instrument_id") == iid) & (pl.col("date") == rebalance_date)
                )
                if row.is_empty():
                    continue
                ctx.update_bar(rebalance_date, iid, row["ticker"].item(), df)
                strategy.next(ctx)

        signals = ctx.pending_signals()
        buy_signals = [s for s in signals if s.is_buy()]

        # With 300 days of uptrending data, at least some stocks should qualify
        assert len(buy_signals) > 0, "Expected BUY signals on rebalance day with uptrending data"
        # All BUY signals must have computed shares (ATR-based)
        for sig in buy_signals:
            assert sig.shares is not None, f"Signal for {sig.instrument_id} missing shares"
            assert sig.shares >= 1.0, f"Signal shares must be >= 1, got {sig.shares}"

    def test_next_no_signals_on_non_rebalance_day(self) -> None:
        """On a non-rebalance weekday, next() returns immediately with no signals."""
        from unittest.mock import MagicMock

        from trendspec.strategy.context import StrategyContext
        from trendspec.strategy.examples import ClenowMomentumStrategy

        df = self._make_trending_df(300)
        instrument_ids = df["instrument_id"].unique().to_list()

        # Find a Monday (weekday=0) — not the default rebalance day (Wednesday=2)
        all_dates = df["date"].unique().sort()
        mondays = [d for d in all_dates.to_list() if d.weekday() == 0]
        assert mondays
        monday = mondays[-1]

        strategy = ClenowMomentumStrategy(params={
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
        })
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        strategy.init(ctx)

        mock_universe = MagicMock()
        mock_universe.tickers.return_value = instrument_ids
        ctx.set_universe(mock_universe)

        ctx.update_positions({}, 100_000.0)

        for iid in instrument_ids:
            row = df.filter(
                (pl.col("instrument_id") == iid) & (pl.col("date") == monday)
            )
            if row.is_empty():
                continue
            ctx.update_bar(monday, iid, row["ticker"].item(), df)
            strategy.next(ctx)

        assert ctx.pending_signals() == [], "Expected no signals on non-rebalance day"

    def _run_strategy_and_get_buys(
        self,
        df: pl.DataFrame,
        rebalance_date: date,
        sector_index_mock=None,
        params_override: dict | None = None,
    ) -> list:
        """Helper: init + manually invoke next() for one rebalance day, return BUY signals."""
        from trendspec.strategy.context import StrategyContext
        from trendspec.strategy.examples import ClenowMomentumStrategy

        params = {
            "sma_period": 50, "score_period": 30, "gap_period": 30, "atr_period": 10,
            "drawdown_period": 20, "volume_avg_period": 20,
            "rebalance_weekday": rebalance_date.weekday(),
        }
        if params_override:
            params.update(params_override)

        strategy = ClenowMomentumStrategy(params=params)
        ctx = StrategyContext(market=Market.US, strategy=strategy, data=df)
        ctx._universe = MagicMock()
        ctx._universe.tickers = MagicMock(return_value=list(df["instrument_id"].unique()))
        ids = list(df["instrument_id"].unique())

        def _pit_universe(_d):
            return ids

        ctx.pit_universe = _pit_universe
        ctx._current_date = rebalance_date
        ctx.update_positions({}, 1_000_000.0)
        strategy.init(ctx)

        collected: list = []
        original_signal = ctx.signal

        def capture_signal(*args, **kwargs):
            sig = original_signal(*args, **kwargs)
            collected.append(sig)
            return sig

        ctx.signal = capture_signal

        # Always mock sector_lookup to avoid DB config dependency;
        # use a default passthrough that returns None when no custom mock is given.
        if sector_index_mock is None:

            def _mock_sector(_m, _iid, _dt):
                return None

            sector_index_mock = _mock_sector

        with patch(
            "trendspec.strategy.examples.clenow_momentum.sector_lookup",
            side_effect=sector_index_mock,
        ):
            strategy.next(ctx)

        return [s for s in collected if s.is_buy()]

    def test_buy_signal_has_full_extras_schema(self) -> None:
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        assert len(buys) >= 1
        for sig in buys:
            keys = set(sig.extras.keys())
            assert keys == {
                "sector", "rank", "r2", "deviation_pct",
                "drawdown_pct", "vol_mult", "stop_loss", "alerts",
            }
            assert isinstance(sig.extras["rank"], int)
            assert sig.extras["rank"] >= 1
            assert isinstance(sig.extras["r2"], float)
            assert 0.0 <= sig.extras["r2"] <= 1.0
            assert isinstance(sig.extras["deviation_pct"], float)
            assert isinstance(sig.extras["drawdown_pct"], float)
            assert isinstance(sig.extras["vol_mult"], float)
            assert isinstance(sig.extras["stop_loss"], float)
            assert sig.extras["stop_loss"] > 0
            assert isinstance(sig.extras["alerts"], list)

    def test_buy_rank_monotonic_top_first(self) -> None:
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        ranks = [s.extras["rank"] for s in buys]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_buy_sector_lookup_returned(self) -> None:
        """sector() mocked → extras['sector'] reflects mock value."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]

        def mock_sector(_market, iid, _dt):
            return {"UP1": "Technology", "UP2": "Financials", "DOWN": "Energy"}.get(iid)

        buys = self._run_strategy_and_get_buys(df, rebalance_date, sector_index_mock=mock_sector)
        sectors = {s.instrument_id: s.extras["sector"] for s in buys}
        assert sectors.get("UP1") == "Technology" or sectors.get("UP2") == "Financials"

    def test_buy_sector_missing_returns_none(self) -> None:
        """sector() returns None → extras['sector'] is None, signal still emitted."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date, sector_index_mock=lambda *_a: None)
        assert len(buys) >= 1
        assert all(s.extras["sector"] is None for s in buys)

    def test_stop_loss_formula(self) -> None:
        """stop_loss == close - atr_stop_k * ATR(20)"""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(df, rebalance_date)
        for sig in buys:
            assert sig.extras["stop_loss"] < sig.price
            implied_atr = (sig.price - sig.extras["stop_loss"]) / 3.0
            assert implied_atr > 0

    def test_alerts_normal_when_no_threshold_hit(self) -> None:
        """trending smooth df → no alerts expected."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": 0.0,
                "warn_vol_mult_high": 9999.0,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all(s.extras["alerts"] == [] for s in buys)

    def test_alerts_deviation_trigger(self) -> None:
        """warn_deviation_max=0.01 → any positive deviation triggers."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 0.01,
                "warn_vol_mult_low": -1.0,
                "warn_vol_mult_high": 9999.0,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert any("均线乖离过大" in s.extras["alerts"] for s in buys)

    def test_alerts_vol_low_trigger(self) -> None:
        """warn_vol_mult_low extremely high → volume shrink alert."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": 9999.0,
                "warn_vol_mult_high": 99999.0,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all("量能萎缩" in s.extras["alerts"] for s in buys)

    def test_alerts_vol_high_trigger(self) -> None:
        """warn_vol_mult_high extremely low → volume spike alert."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": -1.0,
                "warn_vol_mult_high": 0.001,
                "warn_drawdown_max": -9999.0,
            },
        )
        assert all("放量过快" in s.extras["alerts"] for s in buys)

    def test_alerts_drawdown_trigger(self) -> None:
        """warn_drawdown_max close to 0 → almost any drawdown triggers."""
        df = self._make_trending_df(200)
        rebalance_date = df.sort("date")["date"].to_list()[-1]
        buys = self._run_strategy_and_get_buys(
            df, rebalance_date,
            params_override={
                "warn_deviation_max": 9999.0,
                "warn_vol_mult_low": -1.0,
                "warn_vol_mult_high": 9999.0,
                "warn_drawdown_max": -0.001,
            },
        )
        has_dd_alert = any("回撤过深" in s.extras["alerts"] for s in buys)
        assert has_dd_alert


# =============================================================================
# RS_RATING Indicator Tests
# =============================================================================


class TestRSRatingIndicator:
    def test_registered(self) -> None:
        from trendspec.strategy.indicators import list_indicators
        assert "RS_RATING" in list_indicators()

    def test_column_added(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(300)
        result = compute_indicator(df, "RS_RATING", period=252)
        assert "RS_RATING_252" in result.columns

    def test_values_in_range(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = _make_price_df(300)
        result = compute_indicator(df, "RS_RATING", period=252)
        values = result["RS_RATING_252"].drop_nulls()
        assert len(values) > 0
        assert (values >= 0).all() and (values <= 100).all()


# =============================================================================
# Indices Loader Tests
# =============================================================================


class TestIndicesLoader:
    def test_read_indices_returns_dataframe(self, tmp_path) -> None:
        from trendspec.data.parquet_loader import read_indices
        from trendspec.ingest.writer import write_parquet

        df = pl.DataFrame({
            "instrument_id": ["SP500", "SP500", "SP500"],
            "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "close": [4800.0, 4820.0, 4810.0],
        })
        write_parquet(df, Market.US, "indices", str(tmp_path))

        result = read_indices(Market.US, root=str(tmp_path))
        assert "instrument_id" in result.columns
        assert "date" in result.columns
        assert "close" in result.columns
        assert len(result) == 3


def test_context_index_close_returns_price(tmp_path) -> None:
    """ctx.index_close() returns close price for a known index+date."""
    from trendspec.ingest.writer import write_parquet
    from trendspec.strategy.context import StrategyContext

    df = pl.DataFrame({
        "instrument_id": ["SP500", "SP500"],
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "close": [4800.0, 4820.0],
    })
    write_parquet(df, Market.US, "indices", str(tmp_path))

    from unittest.mock import MagicMock
    strategy = MagicMock(spec=BaseStrategy)
    strategy.log = MagicMock()
    ctx = StrategyContext(market=Market.US, strategy=strategy, root=str(tmp_path))
    ctx._current_date = date(2024, 1, 2)

    assert ctx.index_close("SP500") == 4800.0
    assert ctx.index_close("SP500", date(2024, 1, 3)) == 4820.0
    assert ctx.index_close("SP500", date(2000, 1, 1)) is None


# =============================================================================
# Strategy Comparison Tests
# =============================================================================


class TestStrategyComparison:
    def test_comparison_row_fields(self) -> None:
        from trendspec.analyzer.strategy_comparison import ComparisonRow
        row = ComparisonRow(
            strategy_name="test", total_return=0.1, annualized_return=0.05,
            max_drawdown=0.1, sharpe_ratio=1.2, total_trades=10,
            final_nav=110000.0, elapsed_seconds=0.5,
        )
        assert row.strategy_name == "test"
        assert row.error is None

    def test_comparison_row_with_error(self) -> None:
        from trendspec.analyzer.strategy_comparison import ComparisonRow
        row = ComparisonRow(
            strategy_name="broken", total_return=0.0, annualized_return=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, total_trades=0,
            final_nav=0.0, elapsed_seconds=0.0, error="data missing",
        )
        assert row.error == "data missing"

    def test_comparison_report_sort(self) -> None:
        from trendspec.analyzer.strategy_comparison import ComparisonReport, ComparisonRow
        rows = [
            ComparisonRow("low", 0.1, 0.05, 0.1, 0.5, 10, 105000, 1.0),
            ComparisonRow("high", 0.3, 0.15, 0.05, 1.5, 20, 130000, 1.0),
        ]
        report = ComparisonReport(rows, "us", (date(2022, 1, 1), date(2024, 1, 1)))
        sorted_rows = report._sorted_rows("sharpe")
        assert sorted_rows[0].strategy_name == "high"

    def test_comparison_report_csv_export(self, tmp_path) -> None:
        from trendspec.analyzer.strategy_comparison import ComparisonReport, ComparisonRow
        rows = [ComparisonRow("s1", 0.1, 0.05, 0.1, 1.0, 5, 110000, 0.5)]
        report = ComparisonReport(rows, "us", (date(2022, 1, 1), date(2024, 1, 1)))
        path = report.export("csv", tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "s1" in content
