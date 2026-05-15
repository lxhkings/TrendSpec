"""
Tests for TrendSpec factors module.

Tests:
- Factor base class
- Factor registry
- Factor computation
- FactorResult
"""

from datetime import date

import polars as pl
import pytest

from trendspec.factors import (
    Factor,
    FactorResult,
    MomentumFactor,
    VolatilityFactor,
    VolumeFactor,
    register,
    get_factor,
    get_factor_class,
    list_factors,
    factor_info,
    clear_registry,
)
from trendspec.factors.registry import (
    Momentum,
    Returns,
    Volatility,
    VolumeRatio,
    PriceRange,
)


# =============================================================================
# Factor Base Tests
# =============================================================================


class TestFactorBase:
    """Tests for Factor base class."""

    def test_factor_creation(self) -> None:
        """Test creating a custom factor."""
        class MyFactor(Factor):
            name = "my_factor"
            description = "My custom factor"
            category = "momentum"

            def compute(self, df: pl.DataFrame) -> pl.Expr:
                return pl.col("close") * 2

        factor = MyFactor()
        assert factor.name == "my_factor"
        assert factor.description == "My custom factor"
        assert factor.category == "momentum"

    def test_factor_with_params(self) -> None:
        """Test factor with parameters."""
        class MyFactor(Factor):
            name = "my_factor"

            def __init__(self, period: int = 10) -> None:
                self.params = {"period": period}

            def compute(self, df: pl.DataFrame) -> pl.Expr:
                period = self.params.get("period", 10)
                return pl.col("close").shift(period)

        factor = MyFactor(period=20)
        assert factor.params["period"] == 20

    def test_factor_compute_full(self) -> None:
        """Test factor compute_full method."""
        class MyFactor(Factor):
            name = "my_factor"

            def compute(self, df: pl.DataFrame) -> pl.Expr:
                return pl.col("close") * 2

        factor = MyFactor()
        df = pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [10.0, 11.0],
        })

        result = factor.compute_full(df)
        assert isinstance(result, FactorResult)
        assert result.name == "my_factor"
        assert "my_factor" in result.values.columns


class TestFactorResult:
    """Tests for FactorResult."""

    @pytest.fixture
    def factor_data(self) -> pl.DataFrame:
        """Create sample factor data."""
        return pl.DataFrame({
            "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
            "momentum": [5.0, 10.0, 3.0, 8.0],
        })

    def test_cross_sectional(self, factor_data: pl.DataFrame) -> None:
        """Test cross-sectional retrieval."""
        result = FactorResult(values=factor_data, name="momentum")
        cross = result.cross_sectional(date(2024, 1, 1))
        assert len(cross) == 2
        assert set(cross["instrument_id"]) == {"SH600000", "SZ000001"}

    def test_time_series(self, factor_data: pl.DataFrame) -> None:
        """Test time-series retrieval."""
        result = FactorResult(values=factor_data, name="momentum")
        ts = result.time_series("SH600000")
        assert len(ts) == 2
        assert set(ts["date"]) == {date(2024, 1, 1), date(2024, 1, 2)}

    def test_rank(self, factor_data: pl.DataFrame) -> None:
        """Test ranking."""
        result = FactorResult(values=factor_data, name="momentum")
        ranked = result.rank(date(2024, 1, 2), ascending=True)
        # SH600000 has momentum 10, SZ000001 has 8
        # So SH600000 should be rank 2 (ascending), SZ000001 rank 1
        assert "rank" in ranked.columns
        # Check rank column exists
        assert len(ranked) == 2


# =============================================================================
# Registry Tests
# =============================================================================


class TestFactorRegistry:
    """Tests for factor registry."""

    def test_register_factor(self) -> None:
        """Test registering a factor."""
        @register("test_factor")
        class TestFactor(Factor):
            name = "test_factor"

            def compute(self, df: pl.DataFrame) -> pl.Expr:
                return pl.col("close")

        assert "test_factor" in list_factors()

    def test_get_factor(self) -> None:
        """Test getting a factor instance."""
        # Built-in factors should be registered
        factor = get_factor("momentum", {"period": 10})
        assert factor is not None
        assert factor.name == "momentum"
        assert factor.params.get("period") == 10

    def test_get_factor_class(self) -> None:
        """Test getting factor class."""
        cls = get_factor_class("momentum")
        assert cls is not None
        assert cls.name == "momentum"

    def test_list_factors(self) -> None:
        """Test listing registered factors."""
        factors = list_factors()
        assert "momentum" in factors
        assert "returns" in factors
        assert "volatility" in factors
        assert "volume_ratio" in factors

    def test_factor_info(self) -> None:
        """Test factor info."""
        info = factor_info("momentum")
        assert info is not None
        assert info["name"] == "momentum"
        assert info["category"] == "momentum"

    def test_unknown_factor(self) -> None:
        """Test getting unknown factor."""
        factor = get_factor("nonexistent")
        assert factor is None


# =============================================================================
# Built-in Factor Tests
# =============================================================================


class TestBuiltInFactors:
    """Tests for built-in factors."""

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

    def test_momentum_factor(self, sample_data: pl.DataFrame) -> None:
        """Test momentum factor."""
        factor = Momentum(period=5)
        result = factor.compute_full(sample_data)
        assert "momentum" in result.values.columns

    def test_returns_factor(self, sample_data: pl.DataFrame) -> None:
        """Test returns factor."""
        factor = Returns()
        result = factor.compute_full(sample_data)
        assert "returns" in result.values.columns

    def test_volatility_factor(self, sample_data: pl.DataFrame) -> None:
        """Test volatility factor."""
        factor = Volatility(period=5)
        result = factor.compute_full(sample_data)
        assert "volatility" in result.values.columns

    def test_volume_ratio_factor(self, sample_data: pl.DataFrame) -> None:
        """Test volume ratio factor."""
        factor = VolumeRatio(period=5)
        result = factor.compute_full(sample_data)
        assert "volume_ratio" in result.values.columns

    def test_price_range_factor(self, sample_data: pl.DataFrame) -> None:
        """Test price range factor."""
        factor = PriceRange()
        result = factor.compute_full(sample_data)
        assert "price_range" in result.values.columns

    def test_factor_categories(self) -> None:
        """Test factor category classes."""
        # MomentumFactor is abstract, but we can check its category attribute
        assert MomentumFactor.category == "momentum"

        # VolatilityFactor is abstract, check its category
        assert VolatilityFactor.category == "volatility"

        # VolumeFactor is abstract, check its category
        assert VolumeFactor.category == "volume"

        # Use concrete factors for param testing
        mom = Momentum(period=10)
        assert mom.category == "momentum"
        assert mom.params.get("period") == 10

        vol = Volatility(period=20)
        assert vol.category == "volatility"


# =============================================================================
# Factor Integration Tests
# =============================================================================


class TestFactorIntegration:
    """Integration tests for factors."""

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

    def test_multiple_factors(self, sample_data: pl.DataFrame) -> None:
        """Test computing multiple factors."""
        momentum = Momentum(period=3)
        returns = Returns()
        vol = Volatility(period=3)

        mom_result = momentum.compute_full(sample_data)
        ret_result = returns.compute_full(sample_data)
        vol_result = vol.compute_full(sample_data)

        assert "momentum" in mom_result.values.columns
        assert "returns" in ret_result.values.columns
        assert "volatility" in vol_result.values.columns