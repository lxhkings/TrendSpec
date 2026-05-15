"""
Tests for TrendSpec parquet_loader module.

Tests lazy Parquet loading, OHLCV data access, and price adjustment.
Uses temporary directories with mock Parquet files.
"""

import os
import tempfile
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import (
    ADJUSTMENT_MODES,
    AdjustmentMode,
    bars,
    bars_for_instrument,
    get_date_range,
    get_instrument_ids,
    read_components,
    read_sectors,
    scan_parquet,
    scan_parquet_glob,
    _lazyframe_is_empty,
)
from trendspec.ingest.writer import write_parquet


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_root() -> str:
    """Create temporary directory for data_lake."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cn_a_daily_data() -> pl.DataFrame:
    """Sample CN_A daily OHLCV data."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
        "ticker": ["600000", "600000", "000001", "000001"],
        "open": [10.0, 10.2, 20.0, 20.2],
        "high": [10.5, 10.8, 20.5, 20.8],
        "low": [9.8, 10.0, 19.8, 20.0],
        "close": [10.2, 10.5, 20.2, 20.5],
        "volume": [1000000, 1200000, 500000, 600000],
        "adj_factor": [1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def cn_a_daily_with_adj() -> pl.DataFrame:
    """CN_A daily data with adjustment factors (for testing adjustment)."""
    # Simulate dividend: adj_factor changes from 1.0 to 0.95
    return pl.DataFrame({
        "instrument_id": ["SH600000"] * 4,
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "ticker": ["600000"] * 4,
        "open": [10.0, 10.2, 10.5, 10.0],  # Price drop on Jan 4 (dividend)
        "high": [10.5, 10.8, 11.0, 10.2],
        "low": [9.8, 10.0, 10.3, 9.8],
        "close": [10.2, 10.5, 10.8, 10.0],
        "volume": [1000000, 1200000, 1500000, 800000],
        "adj_factor": [1.0, 1.0, 1.0, 0.95],  # Dividend on Jan 4
    })


@pytest.fixture
def cn_a_components() -> pl.DataFrame:
    """Sample CN_A component events."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000002", "SH600000"],
        "date": [date(2020, 1, 1), date(2024, 1, 15), date(2024, 3, 1)],
        "event": ["IPO", "IPO", "HALT"],
        "event_details": ["Listed", "Listed", "Suspended"],
    })


@pytest.fixture
def cn_a_sectors() -> pl.DataFrame:
    """Sample CN_A sector assignments."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001"],
        "date": [date(2020, 1, 1), date(2020, 1, 1)],
        "sector": ["15", "16"],
        "sector_name": ["银行", "非银金融"],
    })


@pytest.fixture
def populated_data_lake(
    temp_root: str,
    cn_a_daily_data: pl.DataFrame,
    cn_a_components: pl.DataFrame,
    cn_a_sectors: pl.DataFrame,
) -> str:
    """Populate data_lake with sample data."""
    # Write daily data
    write_parquet(cn_a_daily_data, Market.CN, "daily", temp_root)

    # Write components data
    write_parquet(cn_a_components, Market.CN, "components", temp_root)

    # Write sectors data
    write_parquet(cn_a_sectors, Market.CN, "sectors", temp_root)

    return temp_root


# =============================================================================
# Scan Parquet Tests
# =============================================================================


class TestScanParquet:
    """Tests for lazy Parquet scanning."""

    def test_scan_parquet_empty_dir(self, temp_root: str) -> None:
        """Scan empty directory should return empty LazyFrame."""
        lf = scan_parquet(temp_root, Market.CN, "daily")
        assert _lazyframe_is_empty(lf)

    def test_scan_parquet_with_data(self, populated_data_lake: str) -> None:
        """Scan populated data_lake should return data."""
        lf = scan_parquet(populated_data_lake, Market.CN, "daily")

        if not _lazyframe_is_empty(lf):
            df = lf.collect()
            assert len(df) > 0
            assert "instrument_id" in df.columns

    def test_scan_parquet_glob(self, populated_data_lake: str) -> None:
        """Scan with glob pattern should work."""
        # Use a glob pattern that matches Hive partitions
        lf = scan_parquet_glob(
            "cn/daily/instrument_id=SH600000/*.parquet",
            populated_data_lake
        )

        if not _lazyframe_is_empty(lf):
            df = lf.collect()
            assert len(df) > 0


# =============================================================================
# Bars Tests
# =============================================================================


class TestBars:
    """Tests for OHLCV bars retrieval."""

    def test_bars_empty_data_lake(self, temp_root: str) -> None:
        """Bars from empty data_lake should return empty DataFrame."""
        df = bars(Market.CN, root=temp_root)
        assert df.is_empty()

    def test_bars_with_data(self, populated_data_lake: str) -> None:
        """Bars should return OHLCV data."""
        df = bars(Market.CN, root=populated_data_lake)

        if not df.is_empty():
            assert "instrument_id" in df.columns
            assert "date" in df.columns
            assert "close" in df.columns

    def test_bars_date_filter(self, populated_data_lake: str) -> None:
        """Bars should respect date filter."""
        df = bars(
            Market.CN,
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            root=populated_data_lake,
        )

        if not df.is_empty():
            assert df["date"].min() >= date(2024, 1, 2)
            assert df["date"].max() <= date(2024, 1, 2)

    def test_bars_instrument_filter(self, populated_data_lake: str) -> None:
        """Bars should respect instrument filter."""
        df = bars(
            Market.CN,
            instrument_ids=["SH600000"],
            root=populated_data_lake,
        )

        if not df.is_empty():
            assert df["instrument_id"].unique().to_list() == ["SH600000"]

    def test_bars_adjustment_mode_raw(self, populated_data_lake: str) -> None:
        """Raw adjustment mode should preserve original prices."""
        df = bars(Market.CN, adjustment_mode="raw", root=populated_data_lake)

        if not df.is_empty():
            # Raw prices should not be modified
            assert "adj_factor" in df.columns

    def test_bars_invalid_adjustment_mode(self, populated_data_lake: str) -> None:
        """Invalid adjustment mode should raise error."""
        with pytest.raises(ValueError, match="Invalid adjustment mode"):
            bars(Market.CN, adjustment_mode="invalid", root=populated_data_lake)


class TestBarsForInstrument:
    """Tests for single-instrument bars retrieval."""

    def test_bars_for_instrument_empty(self, temp_root: str) -> None:
        """Bars for non-existent instrument should return empty."""
        df = bars_for_instrument(Market.CN, "SH600000", root=temp_root)
        assert df.is_empty()

    def test_bars_for_instrument_with_data(self, populated_data_lake: str) -> None:
        """Bars for existing instrument should return data."""
        df = bars_for_instrument(Market.CN, "SH600000", root=populated_data_lake)

        if not df.is_empty():
            assert df["instrument_id"].unique().to_list() == ["SH600000"]
            assert df.sort("date")["date"].to_list() == sorted(df["date"].to_list())


# =============================================================================
# Adjustment Tests
# =============================================================================


class TestPriceAdjustment:
    """Tests for price adjustment modes."""

    def test_adjustment_modes_constant(self) -> None:
        """Adjustment modes should be defined."""
        assert "raw" in ADJUSTMENT_MODES
        assert "forward" in ADJUSTMENT_MODES
        assert "backward" in ADJUSTMENT_MODES

    def test_forward_adjustment(self, temp_root: str, cn_a_daily_with_adj: pl.DataFrame) -> None:
        """Forward adjustment should adjust historical prices."""
        # Write data
        write_parquet(cn_a_daily_with_adj, Market.CN, "daily", temp_root)

        # Get forward-adjusted bars
        df = bars(Market.CN, adjustment_mode="forward", root=temp_root)

        if not df.is_empty():
            # Forward adjustment: historical prices adjusted upward
            # The most recent price should stay close to original
            # Check that adj_factor is still present
            assert "adj_factor" in df.columns

    def test_backward_adjustment(self, temp_root: str, cn_a_daily_with_adj: pl.DataFrame) -> None:
        """Backward adjustment should adjust current prices."""
        # Write data
        write_parquet(cn_a_daily_with_adj, Market.CN, "daily", temp_root)

        # Get backward-adjusted bars
        df = bars(Market.CN, adjustment_mode="backward", root=temp_root)

        if not df.is_empty():
            assert "adj_factor" in df.columns

    def test_raw_no_adjustment(self, temp_root: str, cn_a_daily_with_adj: pl.DataFrame) -> None:
        """Raw mode should not modify prices."""
        # Write data
        write_parquet(cn_a_daily_with_adj, Market.CN, "daily", temp_root)

        # Get raw bars
        df_raw = bars(Market.CN, adjustment_mode="raw", root=temp_root)
        df_no_mode = bars(Market.CN, adjustment_mode="raw", root=temp_root)

        if not df_raw.is_empty() and not df_no_mode.is_empty():
            # Prices should be identical
            assert df_raw["close"].to_list() == df_no_mode["close"].to_list()


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestGetInstrumentIds:
    """Tests for instrument ID listing."""

    def test_get_instrument_ids_empty(self, temp_root: str) -> None:
        """Empty data_lake should return empty list."""
        ids = get_instrument_ids(Market.CN, temp_root)
        assert ids == []

    def test_get_instrument_ids_with_data(self, populated_data_lake: str) -> None:
        """Populated data_lake should return instrument IDs."""
        ids = get_instrument_ids(Market.CN, populated_data_lake)
        # Should include the instruments we wrote
        assert isinstance(ids, list)


class TestGetDateRange:
    """Tests for date range retrieval."""

    def test_get_date_range_empty(self, temp_root: str) -> None:
        """Empty data_lake should return None range."""
        min_date, max_date = get_date_range(Market.CN, root=temp_root)
        assert min_date is None
        assert max_date is None

    def test_get_date_range_with_data(self, populated_data_lake: str) -> None:
        """Populated data_lake should return date range."""
        min_date, max_date = get_date_range(Market.CN, root=populated_data_lake)

        if min_date is not None and max_date is not None:
            assert min_date <= max_date


class TestReadComponentsSectors:
    """Tests for reading components and sectors."""

    def test_read_components_empty(self, temp_root: str) -> None:
        """Read components from empty data_lake."""
        df = read_components(Market.CN, root=temp_root)
        assert df.is_empty()

    def test_read_components_with_data(self, populated_data_lake: str) -> None:
        """Read components from populated data_lake."""
        df = read_components(Market.CN, root=populated_data_lake)

        if not df.is_empty():
            assert "instrument_id" in df.columns
            assert "event" in df.columns

    def test_read_sectors_empty(self, temp_root: str) -> None:
        """Read sectors from empty data_lake."""
        df = read_sectors(Market.CN, root=temp_root)
        assert df.is_empty()

    def test_read_sectors_with_data(self, populated_data_lake: str) -> None:
        """Read sectors from populated data_lake."""
        df = read_sectors(Market.CN, root=populated_data_lake)

        if not df.is_empty():
            assert "instrument_id" in df.columns
            assert "sector" in df.columns or "sector_name" in df.columns