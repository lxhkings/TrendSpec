"""
Tests for TrendSpec analyzer metrics.

Tests performance metrics calculation:
- Total return, Annualized return
- Max drawdown, Drawdown duration
- Sharpe ratio
- Win rate, Trade count
- Profit/Loss ratio
"""

from datetime import date

import pytest

from trendspec.analyzer.equity_curve import DrawdownPoint, EquityCurve
from trendspec.analyzer.metrics import (
    PerformanceMetrics,
    calculate_metrics,
)
from trendspec.analyzer.trade_log import TradeLogAnalyzer
from trendspec.engine.broker import Trade
from trendspec.engine.portfolio import EquityCurvePoint

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_equity_curve() -> list[EquityCurvePoint]:
    """Create sample equity curve for testing."""
    return [
        EquityCurvePoint(
            date=date(2024, 1, 2),
            nav=100000,
            cash=100000,
            position_value=0,
            position_count=0,
            daily_return=0.0,
            cumulative_return=0.0,
        ),
        EquityCurvePoint(
            date=date(2024, 1, 3),
            nav=102000,
            cash=50000,
            position_value=52000,
            position_count=5,
            daily_return=0.02,
            cumulative_return=0.02,
        ),
        EquityCurvePoint(
            date=date(2024, 1, 4),
            nav=98000,  # Drawdown
            cash=50000,
            position_value=48000,
            position_count=5,
            daily_return=-0.0392,
            cumulative_return=-0.02,
        ),
        EquityCurvePoint(
            date=date(2024, 1, 5),
            nav=105000,
            cash=45000,
            position_value=60000,
            position_count=6,
            daily_return=0.0714,
            cumulative_return=0.05,
        ),
        EquityCurvePoint(
            date=date(2024, 1, 6),
            nav=110000,
            cash=40000,
            position_value=70000,
            position_count=7,
            daily_return=0.0476,
            cumulative_return=0.10,
        ),
    ]


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    return [
        Trade(
            instrument_id="SH600000",
            ticker="600000",
            direction="BUY",
            shares=100,
            price=10.0,
            signal_price=10.0,
            slippage=0.0,
            cost=5.0,
            execution_date=date(2024, 1, 3),
        ),
        Trade(
            instrument_id="SH600000",
            ticker="600000",
            direction="SELL",
            shares=100,
            price=12.0,  # Profit
            signal_price=10.0,
            slippage=0.0,
            cost=5.0,
            execution_date=date(2024, 1, 5),
        ),
        Trade(
            instrument_id="SZ000001",
            ticker="000001",
            direction="BUY",
            shares=50,
            price=8.0,
            signal_price=8.0,
            slippage=0.0,
            cost=2.0,
            execution_date=date(2024, 1, 3),
        ),
        Trade(
            instrument_id="SZ000001",
            ticker="000001",
            direction="SELL",
            shares=50,
            price=7.0,  # Loss
            signal_price=8.0,
            slippage=0.0,
            cost=2.0,
            execution_date=date(2024, 1, 6),
        ),
    ]


# =============================================================================
# PerformanceMetrics Tests
# =============================================================================


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_metrics_init(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            max_drawdown=0.10,
        )
        assert metrics.total_return == 0.25
        assert metrics.max_drawdown == 0.10

    def test_metrics_to_dict(self):
        """Test metrics conversion to dict."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            total_trades=10,
            win_rate=0.6,
        )
        data = metrics.to_dict()
        assert data["total_return"] == 0.25
        assert data["total_trades"] == 10
        assert data["win_rate"] == 0.6

    def test_metrics_to_chinese_dict(self):
        """Test metrics conversion to Chinese dict."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            max_drawdown=0.10,
        )
        data = metrics.to_chinese_dict()
        assert "总收益率" in data
        assert "最大回撤" in data
        assert data["总收益率"] == 0.25

    def test_metrics_format_percentage(self):
        """Test percentage formatting."""
        metrics = PerformanceMetrics()
        assert metrics.format_percentage(0.25) == "25.00%"
        assert metrics.format_percentage(-0.10) == "-10.00%"

    def test_metrics_format_money(self):
        """Test money formatting."""
        metrics = PerformanceMetrics()
        assert metrics.format_money(100000) == "100,000.00"

    def test_chinese_names_mapping(self):
        """Test Chinese names mapping."""
        names = PerformanceMetrics.CHINESE_NAMES
        assert names["total_return"] == "总收益率"
        assert names["max_drawdown"] == "最大回撤"
        assert names["sharpe_ratio"] == "夏普比率"


# =============================================================================
# calculate_metrics Tests
# =============================================================================


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with empty data."""
        metrics = calculate_metrics(
            equity_curve=[],
            trades=[],
            initial_capital=100000,
        )
        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0

    def test_calculate_metrics_basic(self, sample_equity_curve, sample_trades):
        """Test basic metrics calculation."""
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=100000,
        )

        assert metrics.total_return == 0.10  # 10% return
        assert metrics.final_nav == 110000
        assert metrics.trading_days == 5
        assert metrics.total_trades == 4
        assert metrics.total_costs == 14.0  # 5 + 5 + 2 + 2

    def test_calculate_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=[],
            initial_capital=100000,
        )

        # Max drawdown should be around 3.9% (from 102000 to 98000)
        assert metrics.max_drawdown > 0.0
        assert metrics.max_drawdown < 0.05

    def test_calculate_win_rate(self, sample_equity_curve, sample_trades):
        """Test win rate calculation."""
        # Note: calculate_metrics doesn't use position_costs directly,
        # win rate is calculated from sell trades using signal_price as proxy
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=100000,
        )

        # Win rate should be between 0 and 1
        assert metrics.win_rate >= 0.0
        assert metrics.win_rate <= 1.0


# =============================================================================
# EquityCurve Tests
# =============================================================================


class TestEquityCurve:
    """Tests for EquityCurve class."""

    def test_equity_curve_init(self, sample_equity_curve):
        """Test equity curve initialization."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        assert curve.initial_capital == 100000
        assert len(curve.points) == 5

    def test_drawdown_series(self, sample_equity_curve):
        """Test drawdown series calculation."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        drawdowns = curve.drawdown_series()

        assert len(drawdowns) == 5
        assert drawdowns[0].drawdown == 0.0  # No drawdown at start

        # Check max drawdown point
        max_dd = max(d.drawdown for d in drawdowns)
        assert max_dd > 0.0

    def test_returns_series(self, sample_equity_curve):
        """Test returns series."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        returns = curve.returns_series()

        assert len(returns) == 5
        assert returns[0] == 0.0  # First day return is 0

    def test_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        max_dd = curve.max_drawdown()

        assert max_dd > 0.0
        assert max_dd < 0.05

    def test_underwater_periods(self, sample_equity_curve):
        """Test underwater periods detection."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        periods = curve.underwater_periods()

        # Should have one underwater period
        assert len(periods) >= 1

    def test_summary(self, sample_equity_curve):
        """Test equity curve summary."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        summary = curve.summary()

        assert summary["total_points"] == 5
        assert summary["initial_nav"] == 100000
        assert summary["final_nav"] == 110000

    def test_to_dataframe(self, sample_equity_curve):
        """Test conversion to DataFrame."""
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        df = curve.to_dataframe()

        assert not df.is_empty()
        assert df.height == 5
        assert "nav" in df.columns
        assert "drawdown" in df.columns


# =============================================================================
# TradeLogAnalyzer Tests
# =============================================================================


class TestTradeLogAnalyzer:
    """Tests for TradeLogAnalyzer class."""

    def test_analyzer_init(self, sample_trades):
        """Test analyzer initialization."""
        analyzer = TradeLogAnalyzer(sample_trades)
        assert len(analyzer.trades) == 4

    def test_buy_trades(self, sample_trades):
        """Test buy trades extraction."""
        analyzer = TradeLogAnalyzer(sample_trades)
        buys = analyzer.buy_trades()

        assert len(buys) == 2
        for trade in buys:
            assert trade.is_buy()

    def test_sell_trades(self, sample_trades):
        """Test sell trades extraction."""
        analyzer = TradeLogAnalyzer(sample_trades)
        sells = analyzer.sell_trades()

        assert len(sells) == 2
        for trade in sells:
            assert trade.is_sell()

    def test_winning_trades(self, sample_trades):
        """Test winning trades extraction."""
        analyzer = TradeLogAnalyzer(
            sample_trades,
            position_costs={"SH600000": 10.0, "SZ000001": 8.0},
        )
        winners = analyzer.winning_trades()

        # SH600000 sold at 12 > avg_cost 10 (winning)
        assert len(winners) == 1

    def test_losing_trades(self, sample_trades):
        """Test losing trades extraction."""
        analyzer = TradeLogAnalyzer(
            sample_trades,
            position_costs={"SH600000": 10.0, "SZ000001": 8.0},
        )
        losers = analyzer.losing_trades()

        # SZ000001 sold at 7 < avg_cost 8 (losing)
        assert len(losers) == 1

    def test_trade_summary(self, sample_trades):
        """Test trade summary."""
        analyzer = TradeLogAnalyzer(sample_trades)
        summary = analyzer.trade_summary()

        assert summary.total_trades == 4
        assert summary.buy_trades == 2
        assert summary.sell_trades == 2
        assert summary.total_costs == 14.0

    def test_trades_by_instrument(self, sample_trades):
        """Test grouping by instrument."""
        analyzer = TradeLogAnalyzer(sample_trades)
        grouped = analyzer.trades_by_instrument()

        assert "SH600000" in grouped
        assert "SZ000001" in grouped
        assert len(grouped["SH600000"]) == 2
        assert len(grouped["SZ000001"]) == 2

    def test_to_dataframe(self, sample_trades):
        """Test conversion to DataFrame."""
        analyzer = TradeLogAnalyzer(sample_trades)
        df = analyzer.to_dataframe()

        assert not df.is_empty()
        assert df.height == 4
        assert "direction" in df.columns
        assert "price" in df.columns

    def test_summary_dict_chinese(self, sample_trades):
        """Test summary dict with Chinese names."""
        analyzer = TradeLogAnalyzer(sample_trades)
        data = analyzer.summary_dict()

        assert "交易次数" in data
        assert "买入次数" in data
        assert data["交易次数"] == 4


# =============================================================================
# DrawdownPoint Tests
# =============================================================================


class TestDrawdownPoint:
    """Tests for DrawdownPoint dataclass."""

    def test_drawdown_point_init(self):
        """Test drawdown point initialization."""
        point = DrawdownPoint(
            date=date(2024, 1, 5),
            drawdown=0.05,
            peak_date=date(2024, 1, 4),
            duration_days=1,
        )
        assert point.drawdown == 0.05
        assert point.duration_days == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnalyzerIntegration:
    """Integration tests for analyzer components."""

    def test_full_metrics_flow(self, sample_equity_curve, sample_trades):
        """Test full metrics calculation flow."""
        # Calculate metrics
        metrics = calculate_metrics(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
            initial_capital=100000,
            risk_free_rate=0.03,
        )

        # Analyze equity curve
        curve = EquityCurve(sample_equity_curve, initial_capital=100000)
        max_dd = curve.max_drawdown()

        # Analyze trades
        analyzer = TradeLogAnalyzer(sample_trades)
        summary = analyzer.trade_summary()

        # Verify consistency
        assert metrics.max_drawdown == max_dd
        assert metrics.total_trades == summary.total_trades
        assert metrics.total_costs == summary.total_costs
