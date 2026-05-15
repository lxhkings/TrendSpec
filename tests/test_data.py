"""Tests for TrendSpec data module - markets and schema."""

from datetime import date, datetime

import polars as pl
import pytest

from trendspec.data import (
    COLUMN_TYPES,
    OHLC_COLUMNS,
    PRIMARY_KEY,
    REQUIRED_COLUMNS,
    AdjustmentMode,
    Market,
    validate_dataframe_schema,
)
from trendspec.data.markets import (
    _MARKET_METADATA,
    CommissionRule,
    MarketMetadata,
    PriceLimitRule,
    TradingHours,
)
from trendspec.data.schema import (
    _is_compatible_type,
    get_primary_key_schema,
    get_schema,
)


class TestMarketEnum:
    """Tests for Market enum."""

    def test_market_values(self) -> None:
        """Market enum should have expected values."""
        assert Market.CN_A.value == "CN_A"
        assert Market.US.value == "US"
        assert Market.HK.value == "HK"

    def test_market_is_strenum(self) -> None:
        """Market should be a StrEnum for easy serialization."""
        assert isinstance(Market.CN_A, str)
        assert Market.CN_A == "CN_A"

    def test_all_markets_have_metadata(self) -> None:
        """All market enum values should have metadata."""
        for market in Market:
            assert market in _MARKET_METADATA
            metadata = _MARKET_METADATA[market]
            assert isinstance(metadata, MarketMetadata)


class TestMarketMetadataCN:
    """Tests for China A-share market metadata."""

    @pytest.fixture
    def cn_market(self) -> Market:
        """Return CN_A market."""
        return Market.CN_A

    def test_path(self, cn_market: Market) -> None:
        """CN_A path should be cn_a."""
        assert cn_market.path == "cn_a"

    def test_price_precision(self, cn_market: Market) -> None:
        """CN_A should have 2 decimal places for price."""
        assert cn_market.price_precision == 2

    def test_trading_calendar(self, cn_market: Market) -> None:
        """CN_A calendar should be SSE/SZSE."""
        assert cn_market.trading_calendar == "SSE/SZSE"

    def test_sector_classification(self, cn_market: Market) -> None:
        """CN_A should use Shenwan Level 1 classification."""
        assert cn_market.sector_classification == "Shenwan_L1"
        assert cn_market.sector_count == 28

    def test_currency(self, cn_market: Market) -> None:
        """CN_A currency should be CNY."""
        assert cn_market.currency == "CNY"

    def test_price_limit_rules(self, cn_market: Market) -> None:
        """CN_A should have price limits."""
        rules = cn_market.price_limit_rules
        assert isinstance(rules, PriceLimitRule)
        assert rules.has_limit is True
        assert rules.regular_limit_pct == 0.10
        assert rules.special_limit_pct == 0.20

    def test_commission_rules(self, cn_market: Market) -> None:
        """CN_A should have commission and stamp duty."""
        rules = cn_market.commission_rules
        assert isinstance(rules, CommissionRule)
        assert rules.commission_rate == 0.0003
        assert rules.commission_min == 5.0
        assert rules.stamp_duty == 0.001
        assert rules.stamp_duty_side == "sell"

    def test_trading_hours(self, cn_market: Market) -> None:
        """CN_A should have correct trading hours."""
        hours = cn_market.trading_hours
        assert isinstance(hours, TradingHours)
        assert hours.market_open == "09:30"
        assert hours.market_close == "11:30"
        assert hours.timezone == "Asia/Shanghai"

    def test_data_path(self, cn_market: Market) -> None:
        """data_path should construct correct path."""
        path = cn_market.data_path("/data/lake")
        assert path == "/data/lake/cn_a"

    def test_is_trading_day_weekend(self, cn_market: Market) -> None:
        """Weekend should not be trading day."""
        # Saturday
        assert cn_market.is_trading_day(date(2024, 1, 6)) is False
        # Sunday
        assert cn_market.is_trading_day(date(2024, 1, 7)) is False

    def test_is_trading_day_weekday(self, cn_market: Market) -> None:
        """Weekday should be trading day (placeholder)."""
        # Monday
        assert cn_market.is_trading_day(date(2024, 1, 8)) is True
        # Friday
        assert cn_market.is_trading_day(date(2024, 1, 12)) is True

    def test_is_trading_day_datetime(self, cn_market: Market) -> None:
        """is_trading_day should work with datetime objects."""
        assert cn_market.is_trading_day(datetime(2024, 1, 6)) is False  # Saturday
        assert cn_market.is_trading_day(datetime(2024, 1, 8)) is True  # Monday


class TestMarketMetadataUS:
    """Tests for US market metadata."""

    @pytest.fixture
    def us_market(self) -> Market:
        """Return US market."""
        return Market.US

    def test_path(self, us_market: Market) -> None:
        """US path should be us."""
        assert us_market.path == "us"

    def test_price_precision(self, us_market: Market) -> None:
        """US should have 4 decimal places for price."""
        assert us_market.price_precision == 4

    def test_sector_classification(self, us_market: Market) -> None:
        """US should use GICS Sector classification."""
        assert us_market.sector_classification == "GICS_Sector"
        assert us_market.sector_count == 8

    def test_currency(self, us_market: Market) -> None:
        """US currency should be USD."""
        assert us_market.currency == "USD"

    def test_price_limit_rules(self, us_market: Market) -> None:
        """US should not have daily price limits."""
        rules = us_market.price_limit_rules
        assert rules.has_limit is False
        assert rules.circuit_breaker_pct == 0.07

    def test_commission_rules(self, us_market: Market) -> None:
        """US should have no stamp duty."""
        rules = us_market.commission_rules
        assert rules.stamp_duty == 0.0
        assert rules.stamp_duty_side == "none"

    def test_trading_hours(self, us_market: Market) -> None:
        """US should have pre-market and after-hours."""
        hours = us_market.trading_hours
        assert hours.pre_market_open == "04:00"
        assert hours.market_open == "09:30"
        assert hours.market_close == "16:00"
        assert hours.after_market_close == "20:00"
        assert hours.timezone == "America/New_York"


class TestMarketMetadataHK:
    """Tests for Hong Kong market metadata (placeholder)."""

    @pytest.fixture
    def hk_market(self) -> Market:
        """Return HK market."""
        return Market.HK

    def test_path(self, hk_market: Market) -> None:
        """HK path should be hk."""
        assert hk_market.path == "hk"

    def test_currency(self, hk_market: Market) -> None:
        """HK currency should be HKD."""
        assert hk_market.currency == "HKD"

    def test_is_trading_day_not_implemented(self, hk_market: Market) -> None:
        """HK market should raise NotImplementedError for trading day check."""
        with pytest.raises(NotImplementedError, match="Hong Kong market calendar"):
            hk_market.is_trading_day(date(2024, 1, 8))


class TestSchemaConstants:
    """Tests for schema constants."""

    def test_primary_key(self) -> None:
        """Primary key should be (instrument_id, date)."""
        assert PRIMARY_KEY == ("instrument_id", "date")
        assert len(PRIMARY_KEY) == 2

    def test_required_columns(self) -> None:
        """Required columns should include all core columns."""
        assert "instrument_id" in REQUIRED_COLUMNS
        assert "date" in REQUIRED_COLUMNS
        assert "ticker" in REQUIRED_COLUMNS
        assert "open" in REQUIRED_COLUMNS
        assert "high" in REQUIRED_COLUMNS
        assert "low" in REQUIRED_COLUMNS
        assert "close" in REQUIRED_COLUMNS
        assert "volume" in REQUIRED_COLUMNS
        assert "adj_factor" in REQUIRED_COLUMNS

    def test_required_columns_count(self) -> None:
        """Should have exactly 9 required columns."""
        assert len(REQUIRED_COLUMNS) == 9

    def test_ohlc_columns(self) -> None:
        """OHLC columns should be correct."""
        assert {"open", "high", "low", "close"} == OHLC_COLUMNS

    def test_column_types(self) -> None:
        """Column types should map to Polars dtypes."""
        assert COLUMN_TYPES["instrument_id"] == pl.String
        assert COLUMN_TYPES["date"] == pl.Date
        assert COLUMN_TYPES["ticker"] == pl.String
        assert COLUMN_TYPES["close"] == pl.Float64
        assert COLUMN_TYPES["volume"] == pl.Int64
        assert COLUMN_TYPES["adj_factor"] == pl.Float64


class TestValidateDataFrameSchema:
    """Tests for schema validation."""

    def test_valid_schema(self) -> None:
        """Valid DataFrame should have no errors."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 0

    def test_missing_required_column(self) -> None:
        """Missing required column should produce error."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            # Missing ticker
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 1
        assert "ticker" in errors[0]

    def test_missing_multiple_columns(self) -> None:
        """Missing multiple columns should be reported."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            # Missing ticker, all OHLC, volume, adj_factor
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 1
        assert "ticker" in errors[0]
        assert "open" in errors[0]

    def test_wrong_type_string_for_numeric(self) -> None:
        """Wrong type for numeric column should produce error."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            "open": ["10.0"],  # String instead of float
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 1
        assert "open" in errors[0]

    def test_compatible_int_types(self) -> None:
        """Int32 should be compatible with Int64 for volume."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": pl.Series([1000000], dtype=pl.Int32),  # Int32 instead of Int64
            "adj_factor": [1.0],
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 0

    def test_compatible_float_types(self) -> None:
        """Float32 should be compatible with Float64 for prices."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            "open": pl.Series([10.0], dtype=pl.Float32),
            "high": pl.Series([10.5], dtype=pl.Float32),
            "low": pl.Series([9.8], dtype=pl.Float32),
            "close": pl.Series([10.2], dtype=pl.Float32),
            "volume": [1000000],
            "adj_factor": [1.0],
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 0

    def test_extra_columns_allowed(self) -> None:
        """Extra columns should not produce errors."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000000],
            "adj_factor": [1.0],
            "amount": [10200000.0],  # Extra column
            "sector": ["金融"],  # Extra column
        })
        errors = validate_dataframe_schema(df)
        assert len(errors) == 0

    def test_require_all_false(self) -> None:
        """With require_all=False, missing columns are OK."""
        df = pl.DataFrame({
            "instrument_id": ["SH600000"],
            "date": [date(2024, 1, 1)],
            "ticker": ["浦发银行"],
            # Missing OHLC, volume, adj_factor
        })
        errors = validate_dataframe_schema(df, require_all=False)
        assert len(errors) == 0


class TestTypeCompatibility:
    """Tests for type compatibility helper."""

    def test_exact_match(self) -> None:
        """Exact type match should be compatible."""
        assert _is_compatible_type(pl.Float64, pl.Float64)
        assert _is_compatible_type(pl.Int64, pl.Int64)
        assert _is_compatible_type(pl.String, pl.String)

    def test_int_compatibility(self) -> None:
        """Int types should be interchangeable."""
        assert _is_compatible_type(pl.Int32, pl.Int64)
        assert _is_compatible_type(pl.Int64, pl.Int32)
        assert _is_compatible_type(pl.UInt32, pl.Int64)
        assert _is_compatible_type(pl.UInt64, pl.Int64)

    def test_float_compatibility(self) -> None:
        """Float types should be interchangeable."""
        assert _is_compatible_type(pl.Float32, pl.Float64)
        assert _is_compatible_type(pl.Float64, pl.Float32)

    def test_string_compatibility(self) -> None:
        """String and Categorical should be interchangeable."""
        assert _is_compatible_type(pl.String, pl.Categorical)
        assert _is_compatible_type(pl.Categorical, pl.String)

    def test_incompatible_types(self) -> None:
        """Incompatible types should not pass."""
        assert not _is_compatible_type(pl.Float64, pl.Int64)
        assert not _is_compatible_type(pl.String, pl.Date)
        assert not _is_compatible_type(pl.Boolean, pl.Int64)


class TestSchemaHelpers:
    """Tests for schema helper functions."""

    def test_get_schema_all(self) -> None:
        """get_schema should return all required columns."""
        schema = get_schema()
        assert set(schema.keys()) == REQUIRED_COLUMNS

    def test_get_schema_subset(self) -> None:
        """get_schema should return subset when specified."""
        columns = frozenset({"instrument_id", "date", "close"})
        schema = get_schema(columns)
        assert set(schema.keys()) == columns
        assert schema["instrument_id"] == pl.String
        assert schema["date"] == pl.Date
        assert schema["close"] == pl.Float64

    def test_get_primary_key_schema(self) -> None:
        """get_primary_key_schema should return PK columns."""
        schema = get_primary_key_schema()
        assert set(schema.keys()) == {"instrument_id", "date"}
        assert schema["instrument_id"] == pl.String
        assert schema["date"] == pl.Date


class TestAdjustmentMode:
    """Tests for adjustment mode constants."""

    def test_modes_exist(self) -> None:
        """AdjustmentMode should have expected modes."""
        assert hasattr(AdjustmentMode, "RAW")
        assert hasattr(AdjustmentMode, "FORWARD")
        assert hasattr(AdjustmentMode, "BACKWARD")

    def test_modes_values(self) -> None:
        """AdjustmentMode values should be correct."""
        assert AdjustmentMode.RAW == "raw"
        assert AdjustmentMode.FORWARD == "forward"
        assert AdjustmentMode.BACKWARD == "backward"

    def test_all_modes(self) -> None:
        """ALL_MODES should contain all modes."""
        assert "raw" in AdjustmentMode.ALL_MODES
        assert "forward" in AdjustmentMode.ALL_MODES
        assert "backward" in AdjustmentMode.ALL_MODES


class TestDataclasses:
    """Tests for dataclasses used in market metadata."""

    def test_trading_hours_frozen(self) -> None:
        """TradingHours should be immutable."""
        hours = TradingHours(
            pre_market_open="09:00",
            market_open="09:30",
            market_close="16:00",
            after_market_close="20:00",
            timezone="America/New_York",
        )
        with pytest.raises(AttributeError):
            hours.market_open = "10:00"  # type: ignore[misc]

    def test_price_limit_rule_frozen(self) -> None:
        """PriceLimitRule should be immutable."""
        rule = PriceLimitRule(
            has_limit=True,
            regular_limit_pct=0.10,
            description="Test",
        )
        with pytest.raises(AttributeError):
            rule.has_limit = False  # type: ignore[misc]

    def test_commission_rule_frozen(self) -> None:
        """CommissionRule should be immutable."""
        rule = CommissionRule(
            commission_rate=0.001,
            commission_min=5.0,
            stamp_duty=0.0,
            stamp_duty_side="none",
            transfer_fee=0.0,
        )
        with pytest.raises(AttributeError):
            rule.commission_rate = 0.002  # type: ignore[misc]

    def test_market_metadata_frozen(self) -> None:
        """MarketMetadata should be immutable."""
        metadata = MarketMetadata(
            path="test",
            price_precision=2,
            trading_calendar="TEST",
            sector_classification="TEST",
            sector_count=10,
            currency="USD",
            price_limit_rules=PriceLimitRule(has_limit=False),
            commission_rules=CommissionRule(
                commission_rate=0.001,
                commission_min=0.0,
                stamp_duty=0.0,
                stamp_duty_side="none",
                transfer_fee=0.0,
            ),
            trading_hours=TradingHours(
                pre_market_open=None,
                market_open="09:30",
                market_close="16:00",
                after_market_close=None,
                timezone="UTC",
            ),
        )
        with pytest.raises(AttributeError):
            metadata.path = "new_path"  # type: ignore[misc]
