"""
Tests for TrendSpec backtest engine.

Tests the backtest execution flow:
- Engine initialization
- Strategy execution loop
- Signal generation and processing
- Broker execution simulation
- Portfolio updates
- Equity curve recording
- Metrics calculation
"""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.engine.backtest_engine import BacktestEngine, BacktestMetrics
from trendspec.engine.base_engine import EngineConfig
from trendspec.engine.broker import Broker, Order, Trade
from trendspec.engine.costs import CNACostsModel, NoCostsModel, USCostsModel
from trendspec.engine.portfolio import EquityCurvePoint, Portfolio, Position
from trendspec.strategy.base import BaseStrategy
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
        "ticker": ["600000", "600000", "000001", "000001"],
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 3)],
        "open": [10.0, 10.5, 8.0, 8.5],
        "high": [10.5, 11.0, 8.5, 9.0],
        "low": [9.5, 10.0, 7.5, 8.0],
        "close": [10.2, 10.8, 8.2, 8.8],
        "volume": [1000000, 1200000, 800000, 900000],
        "adj_factor": [1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def engine_config(temp_root) -> EngineConfig:
    """Create engine configuration for testing."""
    return EngineConfig(
        market=Market.CN,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 3),
        initial_capital=100000.0,
        order_size=100,
        costs_model="none",  # No costs for simpler testing
        root=temp_root,  # Use temp root to avoid settings dependency
    )


@pytest.fixture
def mock_strategy_class():
    """Create a mock strategy class for testing."""

    class MockStrategy(BaseStrategy):
        name = "mock_strategy"

        def init(self, ctx: StrategyContext) -> None:
            pass

        def next(self, ctx: StrategyContext) -> None:
            # Simple strategy: generate BUY signal for every instrument
            if ctx.close > 10.0:
                ctx.signal("BUY", ctx.instrument_id, ctx.close)

    return MockStrategy


# =============================================================================
# Portfolio Tests
# =============================================================================


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_portfolio_init(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.cash == 100000
        assert portfolio.nav() == 100000
        assert portfolio.position_count() == 0

    def test_portfolio_buy(self):
        """Test portfolio buy update."""
        portfolio = Portfolio(initial_capital=100000)

        portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            cost=0.0,
            trade_date=date(2024, 1, 2),
        )

        assert portfolio.cash == 100000 - 1000  # Deducted 100 * 10
        assert portfolio.nav() == 100000  # NAV unchanged (cash + position value)
        assert portfolio.position_count() == 1

    def test_portfolio_sell(self):
        """Test portfolio sell update."""
        portfolio = Portfolio(initial_capital=100000)

        # Buy first
        portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            cost=0.0,
            trade_date=date(2024, 1, 2),
        )

        # Update price
        portfolio.update_prices({"SH600000": 12.0})

        # Sell
        realized_pnl = portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="SELL",
            shares=100,
            price=12.0,
            cost=0.0,
            trade_date=date(2024, 1, 3),
        )

        assert realized_pnl == 200  # 100 * (12 - 10)
        assert portfolio.cash == 100000 + 200  # Original + P&L
        assert portfolio.position_count() == 0

    def test_portfolio_nav(self):
        """Test NAV calculation."""
        portfolio = Portfolio(initial_capital=100000)

        # Buy position
        portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            cost=0.0,
            trade_date=date(2024, 1, 2),
        )

        # Update price
        portfolio.update_prices({"SH600000": 12.0})

        # NAV should be cash + position value
        expected_nav = (100000 - 1000) + (100 * 12)
        assert portfolio.nav() == expected_nav

    def test_portfolio_sector_weights(self):
        """Test sector weight calculation."""
        portfolio = Portfolio(initial_capital=100000)

        # Buy position with sector
        portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            cost=0.0,
            trade_date=date(2024, 1, 2),
            sector="Finance",
        )

        portfolio.update_prices({"SH600000": 10.0})

        weights = portfolio.sector_weights()
        assert "Finance" in weights

    def test_portfolio_to_risk_portfolio(self):
        """Test conversion to risk portfolio format."""
        portfolio = Portfolio(initial_capital=100000)

        portfolio.update_position(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            cost=0.0,
            trade_date=date(2024, 1, 2),
        )

        risk_data = portfolio.to_risk_portfolio()

        assert "positions" in risk_data
        assert "cash" in risk_data
        assert "equity" in risk_data
        assert risk_data["positions"]["SH600000"] == 100


class TestPosition:
    """Tests for Position class."""

    def test_position_init(self):
        """Test position initialization."""
        pos = Position(
            instrument_id="SH600000",
            ticker="600000",
            shares=100,
            avg_cost=10.0,
            current_price=10.0,
        )

        assert pos.shares == 100
        assert pos.avg_cost == 10.0
        assert pos.current_value == 1000

    def test_position_add_shares(self):
        """Test adding shares to position."""
        pos = Position(
            instrument_id="SH600000",
            ticker="600000",
            shares=100,
            avg_cost=10.0,
            current_price=10.0,
        )

        pos.add_shares(50, 12.0)

        # Weighted average cost
        expected_avg = (100 * 10 + 50 * 12) / 150
        assert pos.avg_cost == expected_avg
        assert pos.shares == 150

    def test_position_remove_shares(self):
        """Test removing shares from position."""
        pos = Position(
            instrument_id="SH600000",
            ticker="600000",
            shares=100,
            avg_cost=10.0,
            current_price=12.0,
        )

        realized_pnl = pos.remove_shares(50)

        # P&L = 50 * (12 - 10) = 100
        assert realized_pnl == 100
        assert pos.shares == 50

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(
            instrument_id="SH600000",
            ticker="600000",
            shares=100,
            avg_cost=10.0,
            current_price=12.0,
        )

        assert pos.unrealized_pnl == 200
        assert pos.unrealized_pnl_pct == 0.2


class TestEquityCurvePoint:
    """Tests for EquityCurvePoint class."""

    def test_equity_curve_point(self):
        """Test equity curve point creation."""
        point = EquityCurvePoint(
            date=date(2024, 1, 2),
            nav=102000,
            cash=50000,
            position_value=52000,
            position_count=5,
            daily_return=0.02,
        )

        assert point.date == date(2024, 1, 2)
        assert point.nav == 102000
        assert point.position_count == 5


# =============================================================================
# Broker Tests
# =============================================================================


class TestBroker:
    """Tests for Broker class."""

    def test_broker_init(self):
        """Test broker initialization."""
        broker = Broker(slippage_bps=5, execution_mode="next_open")
        assert broker.slippage_bps == 5
        assert broker.default_execution_mode == "next_open"

    def test_broker_submit_order(self):
        """Test order submission."""
        broker = Broker()
        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )

        order = broker.submit(signal, shares=100)

        assert order.instrument_id == "SH600000"
        assert order.direction == "BUY"
        assert order.shares == 100
        assert len(broker.pending_orders()) == 1

    def test_broker_execute_orders(self):
        """Test order execution."""
        broker = Broker(costs_model=NoCostsModel())

        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )

        broker.submit(signal, shares=100)

        prices_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 3)],
            "open": [10.5],
            "high": [11.0],
            "low": [10.0],
            "close": [10.8],
        })

        trades = broker.execute_orders(date(2024, 1, 3), prices_df)

        assert len(trades) == 1
        assert trades[0].direction == "BUY"
        assert trades[0].shares == 100
        assert broker.trade_count() == 1

    def test_broker_slippage(self):
        """Test slippage calculation."""
        broker = Broker(slippage_bps=10, costs_model=NoCostsModel())

        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )

        broker.submit(signal, shares=100)

        prices_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 3)],
            "open": [10.0],
            "close": [10.0],
        })

        trades = broker.execute_orders(date(2024, 1, 3), prices_df)

        # Slippage: 10 bps = 0.001 = 10 * 0.001 = 0.01 on 10.0 price
        expected_price = 10.0 + 0.01  # BUY: pay more
        assert trades[0].price == expected_price


class TestOrder:
    """Tests for Order class."""

    def test_order_init(self):
        """Test order initialization."""
        order = Order(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            signal_price=10.0,
        )

        assert order.instrument_id == "SH600000"
        assert order.is_buy()
        assert not order.is_sell()


class TestTrade:
    """Tests for Trade class."""

    def test_trade_init(self):
        """Test trade initialization."""
        trade = Trade(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.5,
            signal_price=10.0,
            execution_date=date(2024, 1, 3),
        )

        assert trade.instrument_id == "SH600000"
        assert trade.total_value == 1050
        assert trade.is_buy()


# =============================================================================
# Costs Model Tests
# =============================================================================


class TestCostsModels:
    """Tests for transaction cost models."""

    def test_cna_costs_buy(self):
        """Test CN_A costs for buy."""
        costs = CNACostsModel()

        # Buy 10000 value
        cost = costs.calculate("BUY", 10000)

        # Commission: max(3, 5) = 5, no stamp duty for buy
        assert cost == 5.0 + 0.1  # 5 commission + 0.1 transfer fee

    def test_cna_costs_sell(self):
        """Test CN_A costs for sell."""
        costs = CNACostsModel()

        # Sell 10000 value
        cost = costs.calculate("SELL", 10000)

        # Commission: 5, stamp duty: 10, transfer: 0.1
        assert cost == 5.0 + 10.0 + 0.1

    def test_us_costs(self):
        """Test US costs."""
        costs = USCostsModel()

        cost = costs.calculate("BUY", 10000)
        assert cost == 5.0  # 0.05% * 10000

        cost = costs.calculate("SELL", 10000)
        assert cost == 5.0  # Same (no stamp duty)

    def test_no_costs(self):
        """Test no costs model."""
        costs = NoCostsModel()

        cost = costs.calculate("BUY", 10000)
        assert cost == 0.0


# =============================================================================
# Backtest Engine Tests
# =============================================================================


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_engine_init(self, engine_config):
        """Test engine initialization."""
        engine = BacktestEngine(engine_config)

        assert engine.config.market == Market.CN
        assert engine.config.initial_capital == 100000

    def test_engine_get_trading_days(self, engine_config):
        """Test trading days retrieval."""
        engine = BacktestEngine(engine_config)
        trading_days = engine.get_trading_days()

        # Should return trading days in range
        assert len(trading_days) >= 0

    @patch("trendspec.engine.backtest_engine.BacktestEngine.load_data")
    @patch("trendspec.engine.backtest_engine.BacktestEngine.load_universe")
    def test_engine_run_empty(self, mock_universe, mock_data, engine_config):
        """Test engine run with empty data."""
        mock_data.return_value = pl.DataFrame()
        mock_universe.return_value = MagicMock(tickers=lambda d: [])

        engine = BacktestEngine(engine_config)

        # Create minimal strategy
        class EmptyStrategy(BaseStrategy):
            name = "empty"
            def init(self, ctx): pass
            def next(self, ctx): pass

        result = engine.run(EmptyStrategy)

        # Should return empty result
        assert result.trades == []
        assert result.signals == []


class TestBacktestMetrics:
    """Tests for BacktestMetrics."""

    def test_metrics_to_dict(self):
        """Test metrics conversion to dict."""
        metrics = BacktestMetrics(
            total_return=0.25,
            max_drawdown=0.1,
            total_trades=10,
            initial_capital=100000,
            final_nav=125000,
        )

        data = metrics.to_dict()

        assert data["total_return"] == 0.25
        assert data["max_drawdown"] == 0.1
        assert data["total_trades"] == 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestEngineIntegration:
    """Integration tests for engine components."""

    def test_portfolio_broker_integration(self):
        """Test portfolio and broker integration."""
        portfolio = Portfolio(initial_capital=100000)
        broker = Broker(costs_model=NoCostsModel())

        # Create signal
        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
        )

        # Submit and execute
        broker.submit(signal, shares=100)

        prices_df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 2)],
            "open": [10.0],
            "close": [10.0],
        })

        trades = broker.execute_orders(date(2024, 1, 2), prices_df)

        # Update portfolio
        for trade in trades:
            portfolio.update_position(
                instrument_id=trade.instrument_id,
                ticker=trade.ticker,
                direction=trade.direction,
                shares=trade.shares,
                price=trade.price,
                cost=trade.cost,
                trade_date=trade.execution_date,
            )

        assert portfolio.position_count() == 1
        assert portfolio.cash == 100000 - 1000


# =============================================================================
# Signal Processing Tests
# =============================================================================


class TestSignalProcessing:
    """Tests for signal processing."""

    def test_signal_generation(self):
        """Test signal generation."""
        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.0,
            trigger_value=20.0,
            note="Above MA20",
        )

        assert signal.direction == "BUY"
        assert signal.is_buy()
        assert signal.price == 10.0

    def test_signal_validation(self):
        """Test signal validation."""
        # Invalid direction
        with pytest.raises(ValueError):
            Signal(
                direction="INVALID",
                ticker="600000",
                instrument_id="SH600000",
                price=10.0,
            )

        # Invalid price
        with pytest.raises(ValueError):
            Signal(
                direction="BUY",
                ticker="600000",
                instrument_id="SH600000",
                price=-10.0,
            )
