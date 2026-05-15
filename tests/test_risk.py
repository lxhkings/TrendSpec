"""
Tests for TrendSpec risk module.

Tests:
- Allow/Reject result types
- Portfolio state
- RiskRule base class
- RiskPipeline
- Built-in risk rules
- New risk rules (position limit, drawdown halt, liquidity, price limit, sector)
"""

from datetime import date

import polars as pl
import pytest

from trendspec.risk import (
    Allow,
    Reject,
    Portfolio,
    RiskRule,
    RiskPipeline,
    PipelineResult,
    PipelineStats,
    MaxPositionSize,
    MaxPositions,
    MinCapital,
    SectorConcentration,
    LiquidityFilter,
    DuplicatePosition,
    UniverseMembership,
    default_pipeline,
    get_rule,
    list_rules,
)
# New risk rule imports
from trendspec.risk.position_limit import MaxSinglePositionSize, MaxPositionsCount
from trendspec.risk.drawdown_halt import DrawdownHaltRule, DrawdownState
from trendspec.risk.liquidity import MinLiquidityRule
from trendspec.risk.price_limit import PriceLimitRule
from trendspec.risk.sector_limit import SectorConcentrationLimit
from trendspec.risk.sector_neutral import SectorNeutralRule, SectorWeights
from trendspec.strategy import Signal, StrategyContext, BaseStrategy
from trendspec.data.markets import Market


# =============================================================================
# Result Types Tests
# =============================================================================


class TestAllowReject:
    """Tests for Allow and Reject result types."""

    def test_allow_creation(self) -> None:
        """Test creating Allow result."""
        allow = Allow(rule_name="test_rule")
        assert allow.rule_name == "test_rule"
        assert allow.is_allowed()
        assert not allow.is_rejected()
        assert allow.modified_signal is None

    def test_allow_with_modified_signal(self) -> None:
        """Test Allow with modified signal."""
        signal = Signal("BUY", "600000", "SH600000", 10.5)
        allow = Allow(rule_name="test_rule", modified_signal=signal)
        assert allow.modified_signal == signal

    def test_reject_creation(self) -> None:
        """Test creating Reject result."""
        reject = Reject(rule_name="test_rule", reason="Too risky", details={"max": 100})
        assert reject.rule_name == "test_rule"
        assert reject.reason == "Too risky"
        assert reject.details["max"] == 100
        assert reject.is_rejected()
        assert not reject.is_allowed()


# =============================================================================
# Portfolio Tests
# =============================================================================


class TestPortfolio:
    """Tests for Portfolio state."""

    def test_empty_portfolio(self) -> None:
        """Test empty portfolio."""
        portfolio = Portfolio()
        assert portfolio.position_count() == 0
        assert portfolio.cash == 0.0
        assert portfolio.equity == 0.0

    def test_portfolio_with_positions(self) -> None:
        """Test portfolio with positions."""
        portfolio = Portfolio(
            positions={"SH600000": 100.0, "SZ000001": 50.0},
            cash=5000.0,
            equity=10000.0,
            position_prices={"SH600000": 10.5, "SZ000001": 20.0},
        )
        assert portfolio.position_count() == 2
        assert portfolio.has_position("SH600000")
        assert not portfolio.has_position("SH600036")
        assert portfolio.position_value("SH600000") == 1050.0
        assert portfolio.sector_weight("finance") == 0.0

    def test_portfolio_sector_weights(self) -> None:
        """Test portfolio sector weights."""
        portfolio = Portfolio(
            positions={"SH600000": 100.0},
            sector_weights={"finance": 0.3, "tech": 0.2},
        )
        assert portfolio.sector_weight("finance") == 0.3
        assert portfolio.sector_weight("tech") == 0.2


# =============================================================================
# Risk Rule Tests
# =============================================================================


class TestRiskRules:
    """Tests for risk rules."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(
            positions={},
            cash=10000.0,
            equity=10000.0,
        )

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_max_positions_rule_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MaxPositions rule allows when under limit."""
        rule = MaxPositions(max_positions=10)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_max_positions_rule_reject(self, signal: Signal, context: StrategyContext) -> None:
        """Test MaxPositions rule rejects when at limit."""
        portfolio = Portfolio(
            positions={f"SH{i:06d}": 100.0 for i in range(10)},
            cash=1000.0,
            equity=10000.0,
        )
        rule = MaxPositions(max_positions=10)
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()
        assert "Max positions" in result.reason

    def test_max_position_size_rule(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MaxPositionSize rule."""
        rule = MaxPositionSize(max_pct=0.15)  # Allow up to 15% of equity
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_min_capital_rule_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MinCapital rule allows when sufficient."""
        rule = MinCapital(min_capital=1000.0)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_min_capital_rule_reject(self, signal: Signal, context: StrategyContext) -> None:
        """Test MinCapital rule rejects when insufficient."""
        portfolio = Portfolio(cash=500.0, equity=500.0)
        rule = MinCapital(min_capital=1000.0)
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()
        assert "Insufficient capital" in result.reason

    def test_duplicate_position_rule_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test DuplicatePosition allows when no existing position."""
        rule = DuplicatePosition()
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_duplicate_position_rule_reject(self, signal: Signal, context: StrategyContext) -> None:
        """Test DuplicatePosition rejects when position exists."""
        portfolio = Portfolio(positions={"SH600000": 100.0})
        rule = DuplicatePosition()
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()
        assert "Position already exists" in result.reason

    def test_liquidity_filter_rule_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test LiquidityFilter allows when volume sufficient."""
        rule = LiquidityFilter(min_volume=100000)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_liquidity_filter_rule_reject(self, signal: Signal, portfolio: Portfolio) -> None:
        """Test LiquidityFilter rejects when volume insufficient."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy()
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [50000],  # Low volume
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)

        rule = LiquidityFilter(min_volume=100000)
        result = rule.check(signal, portfolio, ctx)
        assert result.is_rejected()
        assert "below minimum" in result.reason

    def test_sell_signals_always_allowed(self, context: StrategyContext) -> None:
        """Test sell signals are allowed by most rules."""
        sell_signal = Signal("SELL", "600000", "SH600000", 10.5)
        portfolio = Portfolio(positions={"SH600000": 100.0})

        # Most rules should allow sell signals
        max_pos = MaxPositions(max_positions=10)
        assert max_pos.check(sell_signal, portfolio, context).is_allowed()

        min_cap = MinCapital(min_capital=1000.0)
        assert min_cap.check(sell_signal, portfolio, context).is_allowed()

        dup_pos = DuplicatePosition()
        assert dup_pos.check(sell_signal, portfolio, context).is_allowed()


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestRiskPipeline:
    """Tests for RiskPipeline."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(cash=10000.0, equity=10000.0)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_empty_pipeline(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test empty pipeline allows all signals."""
        pipeline = RiskPipeline()
        result = pipeline.run(signal, portfolio, context)
        assert result.is_allowed()

    def test_pipeline_single_rule(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test pipeline with single rule."""
        pipeline = RiskPipeline([MaxPositions(10)])
        result = pipeline.run(signal, portfolio, context)
        assert result.is_allowed()

    def test_pipeline_rejection(self, context: StrategyContext) -> None:
        """Test pipeline rejection."""
        signal = Signal("BUY", "600000", "SH600000", 10.5)
        portfolio = Portfolio(
            positions={f"SH{i:06d}": 100.0 for i in range(10)},
            cash=1000.0,
            equity=10000.0,
        )
        pipeline = RiskPipeline([MaxPositions(10)])
        result = pipeline.run(signal, portfolio, context)
        assert result.is_rejected()
        assert result.rejection_reason is not None

    def test_pipeline_priority_order(self, signal: Signal, context: StrategyContext) -> None:
        """Test rules run in priority order."""
        # Create rules with different priorities
        rule1 = DuplicatePosition()  # priority 5
        rule2 = MaxPositions(5)       # priority 20

        portfolio = Portfolio(
            positions={"SH600000": 100.0},  # Has position
            cash=1000.0,
            equity=1000.0,
        )

        pipeline = RiskPipeline([rule2, rule1])  # Added in wrong order
        result = pipeline.run(signal, portfolio, context)

        # Should be rejected by DuplicatePosition (priority 5) first
        assert result.is_rejected()
        assert result.final_result.rule_name == "duplicate_position"

    def test_pipeline_add_remove_rule(self) -> None:
        """Test adding and removing rules."""
        pipeline = RiskPipeline()
        pipeline.add_rule(MaxPositions(10))
        assert len(pipeline.get_rules()) == 1

        pipeline.add_rule(MinCapital(1000.0))
        assert len(pipeline.get_rules()) == 2

        removed = pipeline.remove_rule("max_positions")
        assert removed
        assert len(pipeline.get_rules()) == 1

    def test_pipeline_batch(self, context: StrategyContext) -> None:
        """Test running batch of signals."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.5),
            Signal("BUY", "000001", "SZ000001", 20.5),
        ]
        portfolio = Portfolio(cash=10000.0, equity=10000.0)
        pipeline = RiskPipeline([MaxPositions(10)])

        results = pipeline.run_batch(signals, portfolio, context)
        assert len(results) == 2
        assert all(r.is_allowed() for r in results)

    def test_pipeline_filter_allowed(self, context: StrategyContext) -> None:
        """Test filtering allowed signals."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.5),
            Signal("BUY", "000001", "SZ000001", 20.5),
        ]
        portfolio = Portfolio(cash=10000.0, equity=10000.0)
        pipeline = RiskPipeline([MaxPositions(10)])

        allowed = pipeline.filter_allowed(signals, portfolio, context)
        assert len(allowed) == 2


class TestPipelineStats:
    """Tests for PipelineStats."""

    def test_stats_empty(self) -> None:
        """Test empty stats."""
        stats = PipelineStats()
        assert stats.total_signals == 0
        assert stats.rejection_rate() == 0.0

    def test_stats_record(self) -> None:
        """Test recording stats."""
        stats = PipelineStats()
        signal = Signal("BUY", "600000", "SH600000", 10.5)

        # Record allowed
        allowed_result = PipelineResult(
            final_result=Allow("test"),
            signal=signal,
        )
        stats.record_result(allowed_result)
        assert stats.allowed_count == 1

        # Record rejected
        rejected_result = PipelineResult(
            final_result=Reject("test", "Too many positions"),
            signal=signal,
            rejection_reason="Too many positions",
        )
        stats.record_result(rejected_result)
        assert stats.rejected_count == 1
        assert stats.rejection_by_rule["test"] == 1

    def test_stats_summary(self) -> None:
        """Test stats summary."""
        stats = PipelineStats(total_signals=100, allowed_count=80, rejected_count=20)
        summary = stats.summary()
        assert "Total signals: 100" in summary
        assert "Allowed: 80" in summary
        assert "Rejected: 20" in summary


# =============================================================================
# Default Pipeline Tests
# =============================================================================


class TestDefaultPipeline:
    """Tests for default pipeline."""

    def test_default_pipeline_creation(self) -> None:
        """Test creating default pipeline."""
        pipeline = default_pipeline()
        rules = pipeline.get_rules()
        assert len(rules) >= 5  # Should have multiple rules

    def test_default_pipeline_with_params(self) -> None:
        """Test default pipeline with custom params."""
        pipeline = default_pipeline(
            max_positions=5,
            max_position_pct=0.05,
            min_capital=500.0,
        )
        rules = pipeline.get_rules()
        assert len(rules) >= 5


# =============================================================================
# Registry Tests
# =============================================================================


class TestRuleRegistry:
    """Tests for rule registry."""

    def test_list_rules(self) -> None:
        """Test listing rules."""
        rules = list_rules()
        assert "max_positions" in rules
        assert "min_capital" in rules
        assert "duplicate_position" in rules

    def test_get_rule(self) -> None:
        """Test getting rule instance."""
        rule = get_rule("max_positions", {"max_positions": 5})
        assert rule is not None
        assert rule.name == "max_positions"
        assert rule.params.get("max_positions") == 5


# =============================================================================
# Position Limit Rule Tests
# =============================================================================


class TestPositionLimitRules:
    """Tests for new position limit rules."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(cash=10000.0, equity=10000.0)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_max_single_position_size_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MaxSinglePositionSize allows when under limit."""
        rule = MaxSinglePositionSize(max_pct=0.15)  # 15% max, order is ~10.5% of 10k
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_max_single_position_size_reject(self, signal: Signal, context: StrategyContext) -> None:
        """Test MaxSinglePositionSize rejects when over limit."""
        portfolio = Portfolio(cash=1000.0, equity=1000.0)
        rule = MaxSinglePositionSize(max_pct=0.10)  # 10% max
        # Order size 100 * price 10.5 = 1050, which is > 10% of 1000
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()

    def test_max_positions_count_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MaxPositionsCount allows when under limit."""
        rule = MaxPositionsCount(max_positions=10)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_max_positions_count_reject(self, signal: Signal, context: StrategyContext) -> None:
        """Test MaxPositionsCount rejects when at limit."""
        portfolio = Portfolio(
            positions={f"SH{i:06d}": 100.0 for i in range(10)},
            cash=1000.0,
            equity=10000.0,
        )
        rule = MaxPositionsCount(max_positions=10)
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()

    def test_max_positions_count_allows_existing_position(self, context: StrategyContext) -> None:
        """Test MaxPositionsCount allows adding to existing position."""
        signal = Signal("BUY", "600000", "SH600000", 10.5)
        portfolio = Portfolio(
            positions={"SH600000": 100.0},  # Existing position
            cash=1000.0,
            equity=10000.0,
        )
        rule = MaxPositionsCount(max_positions=1)
        # Should allow since we're not adding a new position
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()


# =============================================================================
# Drawdown Halt Rule Tests
# =============================================================================


class TestDrawdownHaltRule:
    """Tests for drawdown halt rule."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_drawdown_halt_init(self) -> None:
        """Test DrawdownHaltRule initialization."""
        rule = DrawdownHaltRule(halt_threshold=0.10, recovery_threshold=0.05)
        assert rule.params["halt_threshold"] == 0.10
        assert rule.params["recovery_threshold"] == 0.05
        assert not rule.is_halted()

    def test_drawdown_halt_invalid_thresholds(self) -> None:
        """Test DrawdownHaltRule rejects invalid thresholds."""
        with pytest.raises(ValueError):
            DrawdownHaltRule(halt_threshold=0.05, recovery_threshold=0.10)

    def test_drawdown_halt_allows_when_ok(self, signal: Signal, context: StrategyContext) -> None:
        """Test DrawdownHaltRule allows when drawdown is low."""
        portfolio = Portfolio(cash=10000.0, equity=10000.0)
        rule = DrawdownHaltRule(halt_threshold=0.10, recovery_threshold=0.05)
        rule._state.peak_equity = 10000.0  # Set peak
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_drawdown_halt_rejects_when_halted(self, signal: Signal, context: StrategyContext) -> None:
        """Test DrawdownHaltRule rejects when drawdown exceeds threshold."""
        portfolio = Portfolio(cash=9000.0, equity=9000.0)  # 10% below peak
        rule = DrawdownHaltRule(halt_threshold=0.05, recovery_threshold=0.02)
        rule._state.peak_equity = 10000.0  # Set peak
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()

    def test_drawdown_halt_allows_sell_even_when_halted(self, context: StrategyContext) -> None:
        """Test DrawdownHaltRule allows sell signals when halted."""
        sell_signal = Signal("SELL", "600000", "SH600000", 10.5)
        portfolio = Portfolio(cash=9000.0, equity=9000.0)
        rule = DrawdownHaltRule(halt_threshold=0.05, recovery_threshold=0.02, halt_buy_only=True)
        rule._state.peak_equity = 10000.0
        result = rule.check(sell_signal, portfolio, context)
        assert result.is_allowed()

    def test_drawdown_reset(self) -> None:
        """Test DrawdownHaltRule reset."""
        rule = DrawdownHaltRule()
        rule._state.peak_equity = 10000.0
        rule._state.halted = True
        rule.reset()
        assert rule._state.peak_equity == 0.0
        assert not rule._state.halted


# =============================================================================
# Min Liquidity Rule Tests
# =============================================================================


class TestMinLiquidityRule:
    """Tests for minimum liquidity rule."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(cash=10000.0, equity=10000.0)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [500000],  # 500K volume
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_min_liquidity_allow(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MinLiquidityRule allows when volume sufficient."""
        rule = MinLiquidityRule(min_volume=100000)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()

    def test_min_liquidity_reject(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test MinLiquidityRule rejects when volume insufficient."""
        rule = MinLiquidityRule(min_volume=1000000)  # 1M min
        result = rule.check(signal, portfolio, context)
        assert result.is_rejected()


# =============================================================================
# Price Limit Rule Tests
# =============================================================================


class TestPriceLimitRule:
    """Tests for price limit rule."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 11.0)  # Assume at 涨停

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(cash=10000.0, equity=10000.0)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [11.0],
            "low": [10.0],
            "close": [11.0],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_price_limit_rule_init(self) -> None:
        """Test PriceLimitRule initialization."""
        rule = PriceLimitRule(market=Market.CN, limit_pct=0.10)
        assert rule.params["market"] == Market.CN
        assert rule.params["limit_pct"] == 0.10

    def test_price_limit_us_market(self, signal: Signal, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test PriceLimitRule allows US signals (no daily limits)."""
        rule = PriceLimitRule(market=Market.US)
        result = rule.check(signal, portfolio, context)
        assert result.is_allowed()


# =============================================================================
# Sector Limit Rule Tests
# =============================================================================


class TestSectorLimitRules:
    """Tests for sector limit rules."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create a sample signal."""
        return Signal("BUY", "600000", "SH600000", 10.5)

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        """Create a sample portfolio."""
        return Portfolio(cash=10000.0, equity=10000.0)

    @pytest.fixture
    def context(self) -> StrategyContext:
        """Create a sample context."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy(params={"order_size": 100})
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)
        return ctx

    def test_sector_concentration_limit_init(self) -> None:
        """Test SectorConcentrationLimit initialization."""
        rule = SectorConcentrationLimit(max_sector_pct=0.30, market=Market.CN)
        assert rule.params["max_sector_pct"] == 0.30
        assert rule.params["market"] == Market.CN

    def test_sector_concentration_limit_allow_sell(self, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test SectorConcentrationLimit always allows sell signals."""
        sell_signal = Signal("SELL", "600000", "SH600000", 10.5)
        rule = SectorConcentrationLimit(max_sector_pct=0.30)
        result = rule.check(sell_signal, portfolio, context)
        assert result.is_allowed()

    def test_sector_neutral_rule_init(self) -> None:
        """Test SectorNeutralRule initialization."""
        rule = SectorNeutralRule(max_deviation=0.05, market=Market.CN)
        assert rule.params["max_deviation"] == 0.05
        assert rule.params["market"] == Market.CN

    def test_sector_neutral_rule_allow_sell(self, portfolio: Portfolio, context: StrategyContext) -> None:
        """Test SectorNeutralRule always allows sell signals."""
        sell_signal = Signal("SELL", "600000", "SH600000", 10.5)
        rule = SectorNeutralRule(max_deviation=0.05)
        result = rule.check(sell_signal, portfolio, context)
        assert result.is_allowed()

    def test_sector_weights_creation(self) -> None:
        """Test SectorWeights dataclass."""
        weights = SectorWeights(weights={"tech": 0.3, "finance": 0.2}, total=0.5)
        assert weights.weights["tech"] == 0.3
        assert weights.weights["finance"] == 0.2
        assert weights.total == 0.5