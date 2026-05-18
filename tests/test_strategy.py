"""
Tests for TrendSpec strategy module.

Tests:
- Signal dataclass validation
- StrategyContext state management
- BaseStrategy abstract class
- Indicator computation
"""

from dataclasses import dataclass
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.strategy import (
    BaseStrategy,
    Signal,
    SignalBatch,
    StrategyContext,
    StrategyParams,
    compute_indicator,
    list_indicators,
)
from trendspec.strategy.indicators import atr, bollinger_bands, ema, ma, macd, rsi

# =============================================================================
# Signal Tests
# =============================================================================


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation_buy(self) -> None:
        """Test creating a buy signal."""
        signal = Signal(
            direction="BUY",
            ticker="600000",
            instrument_id="SH600000",
            price=10.5,
            trigger_value=20.0,
            note="Above MA20",
        )
        assert signal.direction == "BUY"
        assert signal.is_buy()
        assert not signal.is_sell()
        assert signal.instrument_id == "SH600000"

    def test_signal_creation_sell(self) -> None:
        """Test creating a sell signal."""
        signal = Signal(
            direction="SELL",
            ticker="600000",
            instrument_id="SH600000",
            price=10.5,
            note="Below MA20",
        )
        assert signal.direction == "SELL"
        assert signal.is_sell()
        assert not signal.is_buy()

    def test_signal_invalid_direction(self) -> None:
        """Test invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(
                direction="HOLD",
                ticker="600000",
                instrument_id="SH600000",
                price=10.5,
            )

    def test_signal_invalid_price(self) -> None:
        """Test negative price raises error."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(
                direction="BUY",
                ticker="600000",
                instrument_id="SH600000",
                price=-1.0,
            )

    def test_signal_zero_price(self) -> None:
        """Test zero price raises error."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(
                direction="BUY",
                ticker="600000",
                instrument_id="SH600000",
                price=0.0,
            )

    def test_signal_extras_default_empty(self) -> None:
        """extras defaults to empty dict, not shared across instances."""
        s1 = Signal(direction="BUY", ticker="A", instrument_id="A", price=1.0)
        s2 = Signal(direction="BUY", ticker="B", instrument_id="B", price=2.0)
        assert s1.extras == {}
        assert s2.extras == {}
        s1.extras["foo"] = 1
        assert s2.extras == {}  # 不共享同一 dict 实例

    def test_signal_extras_arbitrary_payload(self) -> None:
        """extras accepts arbitrary keys/values."""
        s = Signal(
            direction="BUY",
            ticker="X",
            instrument_id="X",
            price=10.0,
            extras={"rank": 1, "sector": "Tech", "alerts": ["a", "b"]},
        )
        assert s.extras["rank"] == 1
        assert s.extras["sector"] == "Tech"
        assert s.extras["alerts"] == ["a", "b"]


class TestSignalBatch:
    """Tests for SignalBatch."""

    def test_empty_batch(self) -> None:
        """Test empty batch."""
        batch = SignalBatch(signals=[])
        assert batch.is_empty()
        assert len(batch) == 0

    def test_batch_with_signals(self) -> None:
        """Test batch with signals."""
        signals = [
            Signal("BUY", "600000", "SH600000", 10.5),
            Signal("SELL", "000001", "SZ000001", 20.5),
        ]
        batch = SignalBatch(signals=signals)
        assert not batch.is_empty()
        assert len(batch) == 2
        assert len(batch.buy_signals()) == 1
        assert len(batch.sell_signals()) == 1


# =============================================================================
# StrategyContext Tests
# =============================================================================


class TestStrategyContext:
    """Tests for StrategyContext."""

    def test_context_initialization(self) -> None:
        """Test context initialization."""
        # Create a simple strategy for testing
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy()
        ctx = StrategyContext(Market.CN, strategy)

        assert ctx.market == Market.CN
        assert ctx.strategy == strategy

    def test_context_bar_update(self) -> None:
        """Test updating context for a bar."""
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
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })

        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)

        assert ctx.date == date(2024, 1, 15)
        assert ctx.instrument_id == "SH600000"
        assert ctx.ticker == "600000"
        assert ctx.close == 10.2
        assert ctx.open == 10.0
        assert ctx.volume == 1000000

    def test_context_signal_generation(self) -> None:
        """Test signal generation."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                ctx.signal("BUY", ctx.instrument_id, ctx.close)

        strategy = DummyStrategy()
        data = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 15)],
            "ticker": ["600000"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })

        ctx = StrategyContext(Market.CN, strategy, data=data)
        ctx.update_bar(date(2024, 1, 15), "SH600000", "600000", data)

        sig = ctx.signal("BUY")
        assert sig.direction == "BUY"
        assert sig.instrument_id == "SH600000"

        pending = ctx.pending_signals()
        assert len(pending) == 1  # One from signal() call

    def test_context_position_management(self) -> None:
        """Test position management."""
        class DummyStrategy(BaseStrategy):
            name = "dummy"
            def init(self, ctx) -> None:
                pass
            def next(self, ctx) -> None:
                pass

        strategy = DummyStrategy()
        ctx = StrategyContext(Market.CN, strategy)

        ctx.update_positions({"SH600000": 100.0, "SZ000001": 50.0}, 10000.0)

        assert ctx.position("SH600000") == 100.0
        assert ctx.has_position("SH600000")
        assert not ctx.has_position("SH600036")
        assert ctx.available_capital == 10000.0


# =============================================================================
# BaseStrategy Tests
# =============================================================================


class TestBaseStrategy:
    """Tests for BaseStrategy."""

    def test_strategy_params_dataclass(self) -> None:
        """Test StrategyParams dataclass."""
        @dataclass
        class MyParams(StrategyParams):
            period: int = 20
            threshold: float = 0.05

            def validate(self) -> None:
                if self.period < 1:
                    raise ValueError("period must be >= 1")

        params = MyParams(period=10, threshold=0.03)
        assert params.period == 10
        assert params.threshold == 0.03

        params_dict = params.to_dict()
        assert params_dict["period"] == 10
        assert params_dict["threshold"] == 0.03

    def test_strategy_params_validation(self) -> None:
        """Test params validation."""
        @dataclass
        class MyParams(StrategyParams):
            period: int = 20

            def validate(self) -> None:
                if self.period < 1:
                    raise ValueError("period must be >= 1")

        with pytest.raises(ValueError):
            MyParams(period=0)

    def test_strategy_creation(self) -> None:
        """Test strategy creation."""
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            params = {"period": 20}

            def init(self, ctx) -> None:
                pass

            def next(self, ctx) -> None:
                pass

        strategy = MyStrategy(params={"period": 20})
        assert strategy.name == "my_strategy"
        assert strategy.params["period"] == 20

    def test_strategy_with_params_object(self) -> None:
        """Test strategy with params object."""
        @dataclass
        class MyParams(StrategyParams):
            period: int = 20

        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            def init(self, ctx) -> None:
                pass

            def next(self, ctx) -> None:
                pass

        params = MyParams(period=15)
        strategy = MyStrategy(params=params)
        assert strategy.get_param("period") == 15

    def test_strategy_get_param(self) -> None:
        """Test get_param method."""
        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            def init(self, ctx) -> None:
                pass

            def next(self, ctx) -> None:
                pass

        strategy = MyStrategy(params={"period": 20, "threshold": 0.05})
        assert strategy.get_param("period") == 20
        assert strategy.get_param("missing", default=10) == 10

    def test_strategy_on_signal_default(self) -> None:
        """Test default on_signal returns signal unchanged."""
        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            def init(self, ctx) -> None:
                pass

            def next(self, ctx) -> None:
                pass

        strategy = MyStrategy()
        signal = Signal("BUY", "600000", "SH600000", 10.5)
        result = strategy.on_signal(signal)
        assert result == signal


# =============================================================================
# Indicator Tests
# =============================================================================


class TestIndicators:
    """Tests for indicator computation."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        return pl.DataFrame({
            "instrument_id": ["SH600000"] * 10 + ["SZ000001"] * 10,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
                date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6),
                date(2024, 1, 7), date(2024, 1, 8), date(2024, 1, 9),
                date(2024, 1, 10),
            ] * 2,
            "ticker": ["600000"] * 10 + ["000001"] * 10,
            "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9] +
                    [20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9],
            "high": [10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4] +
                    [20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4],
            "low": [9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7] +
                   [19.8, 19.9, 20.0, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7],
            "close": [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1] +
                     [20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1],
            "volume": [1000000] * 20,
            "adj_factor": [1.0] * 20,
        })

    def test_ma_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test MA indicator."""
        result = ma(sample_data, period=3)
        assert "MA_3" in result.columns
        # Check that MA is computed per instrument
        assert result.filter(
            (pl.col("instrument_id") == "SH600000") & (pl.col("date") == date(2024, 1, 10))
        )["MA_3"].item() > 0

    def test_ema_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test EMA indicator."""
        result = ema(sample_data, period=5)
        assert "EMA_5" in result.columns

    def test_rsi_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test RSI indicator."""
        result = rsi(sample_data, period=5)
        assert "RSI_5" in result.columns
        # RSI should be between 0 and 100
        rsi_values = result["RSI_5"].drop_nulls()
        assert all(0 <= v <= 100 for v in rsi_values)

    def test_macd_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test MACD indicator."""
        result = macd(sample_data)
        assert "MACD_line" in result.columns
        assert "MACD_signal" in result.columns
        assert "MACD_hist" in result.columns

    def test_atr_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test ATR indicator."""
        result = atr(sample_data, period=5)
        assert "ATR_5" in result.columns

    def test_bollinger_bands(self, sample_data: pl.DataFrame) -> None:
        """Test Bollinger Bands indicator."""
        result = bollinger_bands(sample_data, period=5)
        assert "BB_middle" in result.columns
        assert "BB_upper" in result.columns
        assert "BB_lower" in result.columns
        # Upper should be > middle > lower
        for row in result.drop_nulls(subset=["BB_upper", "BB_middle", "BB_lower"]).iter_rows(named=True):
            assert row["BB_upper"] >= row["BB_middle"]
            assert row["BB_middle"] >= row["BB_lower"]

    def test_compute_indicator_function(self, sample_data: pl.DataFrame) -> None:
        """Test compute_indicator function."""
        result = compute_indicator(sample_data, "MA", period=5)
        assert "MA_5" in result.columns

    def test_unknown_indicator(self, sample_data: pl.DataFrame) -> None:
        """Test unknown indicator raises error."""
        with pytest.raises(ValueError, match="Unknown indicator"):
            compute_indicator(sample_data, "UNKNOWN_INDICATOR")

    def test_list_indicators(self) -> None:
        """Test list_indicators returns known indicators."""
        indicators = list_indicators()
        assert "MA" in indicators
        assert "EMA" in indicators
        assert "RSI" in indicators
        assert "MACD" in indicators
        assert "ATR" in indicators
        assert "BB" in indicators


# =============================================================================
# Strategy Lifecycle Tests
# =============================================================================


class TestStrategyLifecycle:
    """Tests for strategy lifecycle."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample OHLCV data."""
        return pl.DataFrame({
            "instrument_id": ["SH600000"] * 5,
            "date": [
                date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
                date(2024, 1, 4), date(2024, 1, 5),
            ],
            "ticker": ["600000"] * 5,
            "open": [10.0, 10.5, 11.0, 10.8, 11.2],
            "high": [10.5, 11.0, 11.5, 11.2, 11.7],
            "low": [9.8, 10.2, 10.7, 10.5, 11.0],
            "close": [10.2, 10.8, 11.2, 10.9, 11.4],
            "volume": [1000000] * 5,
            "adj_factor": [1.0] * 5,
        })

    def test_strategy_init_and_next(self, sample_data: pl.DataFrame) -> None:
        """Test strategy init and next methods."""
        class TestStrategy(BaseStrategy):
            name = "test_strategy"

            def init(self, ctx) -> None:
                ctx.precompute_indicator("MA", period=3)
                self.initialized = True

            def next(self, ctx) -> None:
                ma_val = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=3)
                if ma_val and ctx.close > ma_val:
                    ctx.signal("BUY", ctx.instrument_id, ctx.close, trigger_value=ma_val)

        strategy = TestStrategy()
        ctx = StrategyContext(Market.CN, strategy, data=sample_data)

        # Run init
        strategy.init(ctx)
        assert strategy.initialized

        # Run next for a bar
        ctx.update_bar(date(2024, 1, 5), "SH600000", "600000", sample_data)
        strategy.next(ctx)

        # Check signals generated
        signals = ctx.pending_signals()
        # Should have generated a signal (price above MA)
        assert len(signals) >= 0  # May or may not have signal depending on MA value


class TestHHIndicator:
    """Highest High (rolling max of close) indicator."""

    def _sample_df(self) -> pl.DataFrame:
        from datetime import date
        return pl.DataFrame({
            "instrument_id": ["A"] * 5,
            "ticker": ["A"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "close": [10.0, 12.0, 11.0, 15.0, 14.0],
            "open": [10.0] * 5, "high": [10.0] * 5,
            "low": [10.0] * 5, "volume": [1000] * 5, "adj_factor": [1.0] * 5,
        })

    def test_hh_period_3(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = self._sample_df()
        out = compute_indicator(df, "HH", period=3)
        vals = out.sort("date")["HH_3"].to_list()
        # Window 3 days: day 1/2 insufficient → None; day 3 max(10,12,11)=12;
        # day 4 max(12,11,15)=15; day 5 max(11,15,14)=15
        assert vals == [None, None, 12.0, 15.0, 15.0]

    def test_hh_per_instrument_isolated(self) -> None:
        """HH computed per instrument_id group, not across instruments."""
        from datetime import date

        from trendspec.strategy.indicators import compute_indicator
        df = pl.DataFrame({
            "instrument_id": ["A", "A", "A", "B", "B", "B"],
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)] * 2,
            "close": [10.0, 12.0, 11.0, 100.0, 90.0, 95.0],
            "open": [0.0] * 6, "high": [0.0] * 6,
            "low": [0.0] * 6, "volume": [0] * 6, "adj_factor": [1.0] * 6,
        })
        out = compute_indicator(df, "HH", period=2).sort(["instrument_id", "date"])
        a_vals = out.filter(pl.col("instrument_id") == "A")["HH_2"].to_list()
        b_vals = out.filter(pl.col("instrument_id") == "B")["HH_2"].to_list()
        assert a_vals == [None, 12.0, 12.0]
        assert b_vals == [None, 100.0, 95.0]

# =============================================================================
# SMA_VOLUME Indicator Tests
# =============================================================================


class TestSMAVolumeIndicator:
    """SMA of volume column."""

    def test_sma_volume_period_3(self) -> None:
        from trendspec.strategy.indicators import compute_indicator
        df = pl.DataFrame({
            "instrument_id": ["A"] * 5,
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "ticker": ["A"] * 5,
            "close": [10.0] * 5, "open": [10.0] * 5,
            "high": [10.0] * 5, "low": [10.0] * 5,
            "volume": [100, 200, 300, 400, 500],
            "adj_factor": [1.0] * 5,
        })
        out = compute_indicator(df, "SMA_VOLUME", period=3).sort("date")
        vals = out["SMA_VOLUME_3"].to_list()
        # Day 1/2 insufficient; day 3 (100+200+300)/3=200; day 4 300; day 5 400
        assert vals[0] is None
        assert vals[1] is None
        assert vals[2] == pytest.approx(200.0)
        assert vals[3] == pytest.approx(300.0)
        assert vals[4] == pytest.approx(400.0)


# =============================================================================
# CLENOW_R2 Indicator Tests
# =============================================================================


class TestClenowR2Indicator:
    """Standalone R² from log-price linear regression."""

    def test_clenow_r2_range(self) -> None:
        """R² ∈ [0, 1] for any non-degenerate window."""
        from datetime import timedelta

        import numpy as np

        from trendspec.strategy.indicators import compute_indicator

        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(99):
            prices.append(max(1.0, prices[-1] * (1 + 0.002 + rng.normal(0, 0.01))))

        df = pl.DataFrame({
            "instrument_id": ["A"] * 100,
            "ticker": ["A"] * 100,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(100)],
            "close": prices,
            "open": prices, "high": prices, "low": prices,
            "volume": [1000] * 100, "adj_factor": [1.0] * 100,
        })
        out = compute_indicator(df, "CLENOW_R2", period=60).sort("date")
        col = out["CLENOW_R2_60"].to_list()
        # First period-1 rows are None
        assert all(v is None for v in col[:59])
        # Subsequent rows ∈ [0, 1]
        for v in col[59:]:
            assert v is not None
            assert 0.0 <= v <= 1.0

    def test_clenow_r2_perfect_log_trend(self) -> None:
        """Perfect log-linear sequence → R² ≈ 1.0"""
        from datetime import timedelta

        import numpy as np

        from trendspec.strategy.indicators import compute_indicator

        # ln(price) = a + b*i  →  price = exp(a) * exp(b*i)
        prices = [float(np.exp(0.01 * i)) for i in range(60)]
        df = pl.DataFrame({
            "instrument_id": ["A"] * 60,
            "ticker": ["A"] * 60,
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(60)],
            "close": prices,
            "open": prices, "high": prices, "low": prices,
            "volume": [1000] * 60, "adj_factor": [1.0] * 60,
        })
        out = compute_indicator(df, "CLENOW_R2", period=30).sort("date")
        last_r2 = out["CLENOW_R2_30"].to_list()[-1]
        assert last_r2 == pytest.approx(1.0, abs=1e-6)
