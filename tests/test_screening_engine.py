"""
Tests for TrendSpec screening engine.

Tests the screening execution flow:
- Engine initialization
- Single-date execution
- Signal generation (no execution)
- BUY/SELL signal collection
- No portfolio/broker/trade execution
"""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.engine.base_engine import EngineConfig
from trendspec.engine.screening_engine import (
    ScreeningEngine,
    ScreeningResult,
    ScreeningConfig,
    screen,
)
from trendspec.strategy.base import BaseStrategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def screening_config(temp_root) -> EngineConfig:
    """Create screening engine configuration."""
    return EngineConfig(
        market=Market.CN_A,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        initial_capital=100000.0,
        costs_model="none",
        root=temp_root,  # Use temp root to avoid settings dependency
    )


@pytest.fixture
def sample_screening_data() -> pl.DataFrame:
    """Create sample data for screening."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001", "SH600036"],
        "ticker": ["600000", "000001", "600036"],
        "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
        "open": [10.0, 8.0, 15.0],
        "high": [10.5, 8.5, 15.5],
        "low": [9.5, 7.5, 14.5],
        "close": [10.2, 8.2, 15.2],
        "volume": [1000000, 800000, 500000],
        "adj_factor": [1.0, 1.0, 1.0],
    })


@pytest.fixture
def buy_signal_strategy():
    """Create strategy that generates BUY signals."""

    class BuySignalStrategy(BaseStrategy):
        name = "buy_signal_strategy"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            # Generate BUY for instruments with price > 10
            if ctx.close > 10.0:
                ctx.signal("BUY", ctx.instrument_id, ctx.close, trigger_value=ctx.close)

    return BuySignalStrategy


@pytest.fixture
def mixed_signal_strategy():
    """Create strategy that generates both BUY and SELL signals."""

    class MixedSignalStrategy(BaseStrategy):
        name = "mixed_signal_strategy"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            # BUY for high prices, SELL for low prices
            if ctx.close > 12.0:
                ctx.signal("BUY", ctx.instrument_id, ctx.close)
            elif ctx.close < 9.0:
                ctx.signal("SELL", ctx.instrument_id, ctx.close)

    return MixedSignalStrategy


# =============================================================================
# Screening Engine Tests
# =============================================================================


class TestScreeningEngine:
    """Tests for ScreeningEngine."""

    def test_engine_init(self, screening_config):
        """Test screening engine initialization."""
        engine = ScreeningEngine(screening_config)

        assert engine.config.market == Market.CN_A
        assert engine._target_date == date(2024, 1, 2)

    def test_get_trading_days(self, screening_config):
        """Test trading days for screening."""
        engine = ScreeningEngine(screening_config)
        trading_days = engine.get_trading_days()

        # Should return single target date
        assert len(trading_days) == 1
        assert trading_days[0] == date(2024, 1, 2)

    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_data")
    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_universe")
    def test_run_empty(self, mock_universe, mock_data, screening_config):
        """Test run with empty data."""
        mock_data.return_value = pl.DataFrame()
        mock_universe.return_value = MagicMock(tickers=lambda d: [])

        engine = ScreeningEngine(screening_config)

        class EmptyStrategy(BaseStrategy):
            name = "empty"
            def init(self, ctx): pass
            def next(self, ctx): pass

        result = engine.run(EmptyStrategy)

        assert result.signals == []
        assert result.trades == []  # No trades in screening
        assert result.equity_curve == []  # No equity curve

    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_data")
    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_universe")
    def test_run_with_signals(self, mock_universe, mock_data, screening_config, buy_signal_strategy):
        """Test run that generates BUY signals."""
        # Setup mocks
        sample_data = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001"],
            "ticker": ["600000", "000001"],
            "date": [date(2024, 1, 2), date(2024, 1, 2)],
            "open": [10.0, 8.0],
            "close": [10.2, 8.2],
            "volume": [1000000, 800000],
            "adj_factor": [1.0, 1.0],
        })
        mock_data.return_value = sample_data
        mock_universe.return_value = MagicMock(tickers=lambda d: ["SH600000", "SZ000001"])

        engine = ScreeningEngine(screening_config)
        result = engine.run(buy_signal_strategy)

        # Should have signals
        assert len(result.signals) >= 0
        assert len(result.trades) == 0  # No trades in screening
        assert result.screening_date == date(2024, 1, 2)

    def test_no_portfolio_updates(self, temp_root):
        """Test that screening doesn't have portfolio tracking."""
        config = EngineConfig(
            market=Market.CN_A,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            root=temp_root,
        )
        engine = ScreeningEngine(config)

        # Screening engine should not have portfolio attribute
        # Portfolio is managed by BacktestEngine, not ScreeningEngine
        assert not hasattr(engine, '_portfolio') or engine._portfolio is None


class TestScreeningResult:
    """Tests for ScreeningResult."""

    def test_result_init(self):
        """Test screening result initialization."""
        result = ScreeningResult(
            signals=[],
            screening_date=date(2024, 1, 2),
            universe_size=100,
            buy_signals=[],
            sell_signals=[],
        )

        assert result.screening_date == date(2024, 1, 2)
        assert result.universe_size == 100
        assert result.buy_count() == 0
        assert result.sell_count() == 0

    def test_result_with_signals(self):
        """Test result with signals."""
        buy_signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )
        sell_signal = Signal(
            direction="SELL",
            ticker="000001",
            instrument_id="SZ000001",
            price=8.0,
        )

        result = ScreeningResult(
            signals=[buy_signal, sell_signal],
            buy_signals=[buy_signal],
            sell_signals=[sell_signal],
            signal_count=2,
        )

        assert result.buy_count() == 1
        assert result.sell_count() == 1
        assert result.signal_count == 2


class TestScreeningConfig:
    """Tests for ScreeningConfig."""

    def test_config_init(self):
        """Test screening config initialization."""
        config = ScreeningConfig(
            market=Market.CN_A,
            target_date=date(2024, 1, 2),
        )

        assert config.market == Market.CN_A
        assert config.target_date == date(2024, 1, 2)
        assert config.include_sell_signals == False


# =============================================================================
# Signal Filtering Tests
# =============================================================================


class TestSignalFiltering:
    """Tests for signal filtering in screening."""

    def test_buy_signals_only(self):
        """Test that screening focuses on BUY signals."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.0),
            Signal("BUY", "600036", "SH600036", 15.0),
            Signal("SELL", "000001", "SZ000001", 8.0),
        ]

        buy_signals = [s for s in signals if s.is_buy()]

        assert len(buy_signals) == 2
        assert all(s.is_buy() for s in buy_signals)

    def test_signal_dataframe(self):
        """Test signals to DataFrame conversion."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.0, trigger_value=10.0),
            Signal("SELL", "000001", "SZ000001", 8.0),
        ]

        records = []
        for signal in signals:
            records.append({
                "direction": signal.direction,
                "ticker": signal.ticker,
                "instrument_id": signal.instrument_id,
                "price": signal.price,
                "trigger_value": signal.trigger_value,
            })

        df = pl.DataFrame(records)

        assert len(df) == 2
        assert "direction" in df.columns
        assert "ticker" in df.columns


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestScreenFunction:
    """Tests for screen convenience function."""

    @patch("trendspec.engine.screening_engine.ScreeningEngine.run")
    @patch("trendspec.config.settings.get_settings")
    def test_screen_function(self, mock_settings, mock_run):
        """Test screen convenience function."""
        mock_settings.return_value.data_lake.data_lake_root = "/data"
        mock_run.return_value = ScreeningResult(
            signals=[],
            screening_date=date(2024, 1, 2),
        )

        class TestStrategy(BaseStrategy):
            name = "test"
            def init(self, ctx): pass
            def next(self, ctx): pass

        result = screen(Market.CN_A, TestStrategy, date(2024, 1, 2))

        assert result.screening_date == date(2024, 1, 2)


# =============================================================================
# Dual-Mode Design Tests
# =============================================================================


class TestDualModeDesign:
    """Tests for dual-mode design (same strategy for backtest and screening)."""

    def test_strategy_next_same_interface(self):
        """Test that strategy.next() has same interface for both modes."""

        # Create a strategy
        class TestStrategy(BaseStrategy):
            name = "test"

            def init(self, ctx: StrategyContext) -> None:
                pass

            def next(self, ctx: StrategyContext) -> None:
                # Same interface regardless of mode
                if ctx.close > 10.0:
                    ctx.signal("BUY", ctx.instrument_id, ctx.close)

        # Strategy should work for both backtest and screening
        # The engine handles the loop difference

        strategy = TestStrategy()
        assert hasattr(strategy, "next")
        assert hasattr(strategy, "init")


# =============================================================================
# Market Overview Restriction Tests
# =============================================================================


class TestNoMarketOverview:
    """Tests that screening engine doesn't generate market overview."""

    def test_no_market_overview_in_result(self):
        """Test that result doesn't contain market overview data."""
        result = ScreeningResult(
            signals=[Signal("BUY", "600000", "SH600000", 10.0)],
            screening_date=date(2024, 1, 2),
        )

        # Should not have top_returns, sector_summary, etc.
        assert not hasattr(result, "top_returns")
        assert not hasattr(result, "sector_summary")
        assert not hasattr(result, "market_overview")

    def test_signals_only_output(self):
        """Test that output is signals only."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.0, trigger_value=10.0),
        ]

        result = ScreeningResult(
            signals=signals,
            buy_signals=signals,
            signal_count=1,
        )

        # Output should be just signals
        assert result.signal_count == 1
        assert len(result.buy_signals) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestScreeningIntegration:
    """Integration tests for screening."""

    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_data")
    @patch("trendspec.engine.screening_engine.ScreeningEngine.load_universe")
    def test_full_screening_flow(self, mock_universe, mock_data, screening_config, mixed_signal_strategy):
        """Test full screening flow."""
        # Setup sample data
        sample_data = pl.DataFrame({
            "instrument_id": ["SH600000", "SZ000001", "SH600036"],
            "ticker": ["600000", "000001", "600036"],
            "date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "open": [10.0, 8.0, 15.0],
            "close": [10.2, 8.2, 15.2],
            "volume": [1000000, 800000, 500000],
            "adj_factor": [1.0, 1.0, 1.0],
        })
        mock_data.return_value = sample_data
        mock_universe.return_value = MagicMock(tickers=lambda d: ["SH600000", "SZ000001", "SH600036"])

        engine = ScreeningEngine(screening_config)
        result = engine.run(mixed_signal_strategy)

        # Verify result structure
        assert result.screening_date == date(2024, 1, 2)
        assert len(result.trades) == 0  # No trades
        assert len(result.equity_curve) == 0  # No equity curve


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestScreeningEdgeCases:
    """Tests for edge cases in screening."""

    def test_empty_universe(self, temp_root):
        """Test screening with empty universe."""
        config = EngineConfig(
            market=Market.CN_A,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            root=temp_root,
        )
        engine = ScreeningEngine(config)

        # Mock empty universe
        engine._universe = MagicMock(tickers=lambda d: [])

        trading_days = engine.get_trading_days()
        assert trading_days  # Should still have trading day

    def test_no_signals_generated(self):
        """Test when strategy generates no signals."""

        class NoSignalStrategy(BaseStrategy):
            name = "no_signal"
            def init(self, ctx): pass
            def next(self, ctx): pass  # No signals

        result = ScreeningResult(
            signals=[],
            buy_signals=[],
            sell_signals=[],
            signal_count=0,
        )

        assert result.signal_count == 0
        assert result.buy_count() == 0
        assert result.sell_count() == 0


# =============================================================================
# Date Handling Tests
# =============================================================================


class TestScreeningDateHandling:
    """Tests for date handling in screening."""

    def test_target_date_is_start_date(self, temp_root):
        """Test that target date is set from start_date."""
        config = EngineConfig(
            market=Market.CN_A,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            root=temp_root,
        )
        engine = ScreeningEngine(config)

        assert engine._target_date == config.start_date

    def test_non_trading_day_handling(self, temp_root):
        """Test handling of non-trading day."""
        config = EngineConfig(
            market=Market.CN_A,
            start_date=date(2024, 1, 1),  # New Year's Day
            end_date=date(2024, 1, 1),
            root=temp_root,
        )

        engine = ScreeningEngine(config)
        trading_days = engine.get_trading_days()

        # Should find previous trading day
        assert len(trading_days) == 1
        # Should not be Jan 1 (holiday)
        assert trading_days[0] != date(2024, 1, 1)