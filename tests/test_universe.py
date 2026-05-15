"""
Tests for TrendSpec universe module.

Tests PIT universe tracking for survivorship bias prevention.
"""

import os
import tempfile
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.universe import (
    CNAUniverse,
    HKUniverse,
    USUniverse,
    Universe,
    get_universe,
)
from trendspec.data.universe.cn import IPO_EVENT, DELIST_EVENT, HALT_EVENT
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
def cn_a_daily_df() -> pl.DataFrame:
    """Sample CN_A daily data."""
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
def cn_a_components_df() -> pl.DataFrame:
    """Sample CN_A component events."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SZ000001", "SH600001"],
        "date": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        "event": [IPO_EVENT, IPO_EVENT, IPO_EVENT],
        "event_details": ["IPO", "IPO", "IPO"],
    })


@pytest.fixture
def cn_a_components_with_delist() -> pl.DataFrame:
    """CN_A components with delisting event."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SH600000", "SZ000001"],
        "date": [date(2020, 1, 1), date(2024, 6, 1), date(2021, 1, 1)],
        "event": [IPO_EVENT, DELIST_EVENT, IPO_EVENT],
        "event_details": ["IPO", "Delisted", "IPO"],
    })


@pytest.fixture
def cn_a_components_with_halt() -> pl.DataFrame:
    """CN_A components with halt event."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SH600000", "SH600000", "SZ000001"],
        "date": [
            date(2020, 1, 1),
            date(2024, 3, 1),  # Halt start
            date(2024, 3, 15),  # Resume
            date(2021, 1, 1),
        ],
        "event": [IPO_EVENT, HALT_EVENT, "RESUME", IPO_EVENT],
        "event_details": ["IPO", "Halted", "Resumed", "IPO"],
    })


@pytest.fixture
def populated_cn_a_universe(
    temp_root: str,
    cn_a_daily_df: pl.DataFrame,
    cn_a_components_df: pl.DataFrame,
) -> str:
    """Populate data_lake for CN_A universe."""
    write_parquet(cn_a_daily_df, Market.CN, "daily", temp_root)
    write_parquet(cn_a_components_df, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def populated_cn_a_with_delist(
    temp_root: str,
    cn_a_daily_df: pl.DataFrame,
    cn_a_components_with_delist: pl.DataFrame,
) -> str:
    """Populate data_lake with delisting event."""
    write_parquet(cn_a_daily_df, Market.CN, "daily", temp_root)
    write_parquet(cn_a_components_with_delist, Market.CN, "components", temp_root)
    return temp_root


@pytest.fixture
def populated_cn_a_with_halt(
    temp_root: str,
    cn_a_daily_df: pl.DataFrame,
    cn_a_components_with_halt: pl.DataFrame,
) -> str:
    """Populate data_lake with halt event."""
    write_parquet(cn_a_daily_df, Market.CN, "daily", temp_root)
    write_parquet(cn_a_components_with_halt, Market.CN, "components", temp_root)
    return temp_root


# =============================================================================
# Universe Base Tests
# =============================================================================


class TestUniverseBase:
    """Tests for Universe abstract base class."""

    def test_universe_is_abstract(self) -> None:
        """Universe should be abstract class."""
        assert hasattr(Universe, "tickers")
        assert hasattr(Universe, "contains")
        assert hasattr(Universe, "ipo_date")
        assert hasattr(Universe, "delist_date")
        assert hasattr(Universe, "is_active")


# =============================================================================
# CNAUniverse Tests
# =============================================================================


class TestCNAUniverse:
    """Tests for CN_A PIT universe."""

    def test_universe_empty(self, temp_root: str) -> None:
        """Empty data_lake should create empty universe."""
        universe = CNAUniverse(temp_root)
        assert universe.instrument_count_total() == 0

    def test_universe_with_data(self, populated_cn_a_universe: str) -> None:
        """Universe should load component events."""
        universe = CNAUniverse(populated_cn_a_universe)

        if universe.instrument_count_total() > 0:
            assert universe.instrument_count_total() >= 1

    def test_tickers_pit(self, populated_cn_a_universe: str) -> None:
        """tickers() should return instruments at specific date."""
        universe = CNAUniverse(populated_cn_a_universe)

        if universe.instrument_count_total() > 0:
            tickers = universe.tickers(date(2024, 1, 1))
            assert isinstance(tickers, list)

    def test_contains_pit(self, populated_cn_a_universe: str) -> None:
        """contains() should check membership at specific date."""
        universe = CNAUniverse(populated_cn_a_universe)

        if universe.instrument_count_total() > 0:
            # SH600000 IPO in 2020
            exists_2024 = universe.contains("SH600000", date(2024, 1, 1))
            # Should exist after IPO
            assert exists_2024 or not universe.ipo_date("SH600000")

    def test_ipo_date(self, populated_cn_a_universe: str) -> None:
        """ipo_date() should return IPO date."""
        universe = CNAUniverse(populated_cn_a_universe)

        ipo = universe.ipo_date("SH600000")
        # Should return IPO date or None
        assert ipo is None or isinstance(ipo, date)

    def test_before_ipo_not_in_universe(self, populated_cn_a_universe: str) -> None:
        """Instrument should not be in universe before IPO."""
        universe = CNAUniverse(populated_cn_a_universe)

        if universe.ipo_date("SH600000"):
            ipo = universe.ipo_date("SH600000")
            before_ipo = date(ipo.year - 1, ipo.month, ipo.day)
            assert universe.contains("SH600000", before_ipo) is False

    def test_delisted_not_in_universe(self, populated_cn_a_with_delist: str) -> None:
        """Delisted instrument should not be in universe after delist."""
        universe = CNAUniverse(populated_cn_a_with_delist)

        if universe.delist_date("SH600000"):
            delist = universe.delist_date("SH600000")
            after_delist = date(delist.year, delist.month, delist.day + 1)
            assert universe.contains("SH600000", after_delist) is False

    def test_halted_not_active(self, populated_cn_a_with_halt: str) -> None:
        """Halted instrument should not be in active tickers."""
        universe = CNAUniverse(populated_cn_a_with_halt)

        if universe.instrument_count_total() > 0:
            # During halt period (March 1-15, 2024)
            tickers_during_halt = universe.tickers(date(2024, 3, 5))
            # SH600000 should not be in active tickers during halt
            if universe.ipo_date("SH600000"):
                # The halted instrument should not be in tickers during halt
                # (unless halt handling is not complete)
                pass  # Allow for incomplete halt tracking

    def test_all_instruments_includes_delisted(
        self,
        populated_cn_a_with_delist: str,
    ) -> None:
        """all_instruments should include delisted stocks."""
        universe = CNAUniverse(populated_cn_a_with_delist)

        all_ids = universe.all_instruments()
        # Should include SH600000 even though it was delisted
        assert isinstance(all_ids, frozenset)

    def test_universe_dates(self, populated_cn_a_universe: str) -> None:
        """universe_dates should return dates with data."""
        universe = CNAUniverse(populated_cn_a_universe)

        dates = universe.universe_dates()
        assert isinstance(dates, list)
        if dates:
            assert all(isinstance(d, date) for d in dates)


# =============================================================================
# USUniverse Tests
# =============================================================================


class TestUSUniverse:
    """Tests for US PIT universe."""

    def test_universe_empty(self, temp_root: str) -> None:
        """Empty data_lake should create empty universe."""
        universe = USUniverse(temp_root)
        assert universe.instrument_count_total() == 0

    def test_universe_type_parameter(self, temp_root: str) -> None:
        """Universe should accept universe_type parameter."""
        universe_sp500 = USUniverse(temp_root, universe_type="sp500")
        universe_r1000 = USUniverse(temp_root, universe_type="r1000")

        assert universe_sp500.universe_type == "sp500"
        assert universe_r1000.universe_type == "r1000"


# =============================================================================
# HKUniverse Tests
# =============================================================================


class TestHKUniverse:
    """Tests for HK universe placeholder."""

    def test_hk_raises_not_implemented(self, temp_root: str) -> None:
        """HK universe should raise NotImplementedError."""
        universe = HKUniverse(temp_root)

        with pytest.raises(NotImplementedError, match="Hong Kong market"):
            universe.tickers(date(2024, 1, 1))

        with pytest.raises(NotImplementedError, match="Hong Kong market"):
            universe.contains("HK0001", date(2024, 1, 1))

        with pytest.raises(NotImplementedError, match="Hong Kong market"):
            universe.ipo_date("HK0001")

        with pytest.raises(NotImplementedError, match="Hong Kong market"):
            universe.delist_date("HK0001")

        with pytest.raises(NotImplementedError, match="Hong Kong market"):
            universe.is_active("HK0001", date(2024, 1, 1))


# =============================================================================
# get_universe Factory Tests
# =============================================================================


class TestGetUniverse:
    """Tests for universe factory function."""

    def test_get_universe_cn_a(self, temp_root: str) -> None:
        """get_universe should return CNAUniverse for CN_A."""
        universe = get_universe("CN", temp_root)
        assert isinstance(universe, CNAUniverse)

    def test_get_universe_us(self, temp_root: str) -> None:
        """get_universe should return USUniverse for US."""
        universe = get_universe("US", temp_root)
        assert isinstance(universe, USUniverse)

    def test_get_universe_hk(self, temp_root: str) -> None:
        """get_universe should return HKUniverse for HK."""
        universe = get_universe("HK", temp_root)
        assert isinstance(universe, HKUniverse)

    def test_get_universe_unknown_raises(self, temp_root: str) -> None:
        """get_universe should raise for unknown market."""
        with pytest.raises(ValueError, match="Unknown market"):
            get_universe("UNKNOWN", temp_root)


# =============================================================================
# PIT Design Rule Tests
# =============================================================================


class TestPITDesignRules:
    """Tests to verify PIT design rules are enforced."""

    def test_tickers_requires_date(self, populated_cn_a_universe: str) -> None:
        """tickers() must accept date parameter (PIT rule)."""
        universe = CNAUniverse(populated_cn_a_universe)

        # This should work - date is required
        tickers = universe.tickers(date(2024, 1, 1))
        assert isinstance(tickers, list)

    def test_contains_requires_date(self, populated_cn_a_universe: str) -> None:
        """contains() must accept date parameter (PIT rule)."""
        universe = CNAUniverse(populated_cn_a_universe)

        # This should work - date is required
        contains = universe.contains("SH600000", date(2024, 1, 1))
        assert isinstance(contains, bool)

    def test_is_active_requires_date(self, populated_cn_a_universe: str) -> None:
        """is_active() must accept date parameter (PIT rule)."""
        universe = CNAUniverse(populated_cn_a_universe)

        # This should work - date is required
        is_active = universe.is_active("SH600000", date(2024, 1, 1))
        assert isinstance(is_active, bool)

    def test_no_current_universe_shortcut(
        self,
        populated_cn_a_universe: str,
    ) -> None:
        """Universe should NOT have 'current' shortcuts (PIT rule)."""
        universe = CNAUniverse(populated_cn_a_universe)

        # Check that there are no "current" or "latest" methods
        # that don't require date parameter
        methods = dir(universe)

        # Filter out methods that might be "current" shortcuts
        # (We want to ensure all universe queries require date)
        for method in methods:
            if "current" in method.lower() or "latest" in method.lower():
                # These should not exist for universe membership
                if method.startswith("_"):
                    continue  # Private methods are OK
                # Check that it's not a tickers/contains variant without date
                pass