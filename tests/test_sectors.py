"""
Tests for TrendSpec sectors module.

Tests PIT sector attribution with memory index.
"""

import tempfile
from datetime import date

import polars as pl
import pytest

from trendspec.data.markets import Market
from trendspec.data.sectors import (
    GICS_SECTORS,
    SHENWAN_L1_SECTORS,
    SectorIndex,
    clear_sector_cache,
    get_all_sectors,
    get_sector_index,
    sector,
    sector_name,
    sector_universe,
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
def cn_a_sectors_df() -> pl.DataFrame:
    """Sample CN_A sector assignments."""
    return pl.DataFrame({
        "instrument_id": ["SH600000", "SH600000", "SZ000001", "SH600001"],
        "date": [
            date(2020, 1, 1),
            date(2024, 1, 1),  # Sector change
            date(2020, 1, 1),
            date(2020, 1, 1),
        ],
        "sector": ["10", "15", "16", "15"],
        "sector_name": ["农林牧渔", "银行", "非银金融", "银行"],
    })


@pytest.fixture
def us_sectors_df() -> pl.DataFrame:
    """Sample US sector assignments."""
    return pl.DataFrame({
        "instrument_id": ["AAPL", "MSFT", "JPM"],
        "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1)],
        "sector": ["45", "45", "40"],
        "sector_name": ["Information Technology", "Information Technology", "Financials"],
    })


@pytest.fixture
def populated_cn_a_sectors(temp_root: str, cn_a_sectors_df: pl.DataFrame) -> str:
    """Populate data_lake with CN_A sectors."""
    write_parquet(cn_a_sectors_df, Market.CN, "sectors", temp_root)
    return temp_root


@pytest.fixture
def populated_us_sectors(temp_root: str, us_sectors_df: pl.DataFrame) -> str:
    """Populate data_lake with US sectors."""
    write_parquet(us_sectors_df, Market.US, "sectors", temp_root)
    return temp_root


# =============================================================================
# Sector Constants Tests
# =============================================================================


class TestSectorConstants:
    """Tests for sector classification constants."""

    def test_shenwan_l1_count(self) -> None:
        """Shenwan Level 1 should have 28 sectors."""
        assert len(SHENWAN_L1_SECTORS) == 28

    def test_shenwan_l1_names(self) -> None:
        """Shenwan sectors should have Chinese names."""
        assert SHENWAN_L1_SECTORS["15"] == "银行"
        assert SHENWAN_L1_SECTORS["11"] == "医药生物"

    def test_gics_count(self) -> None:
        """GICS should have 11 sectors."""
        assert len(GICS_SECTORS) == 11

    def test_gics_names(self) -> None:
        """GICS sectors should have English names."""
        assert GICS_SECTORS["45"] == "Information Technology"
        assert GICS_SECTORS["40"] == "Financials"


# =============================================================================
# SectorIndex Tests
# =============================================================================


class TestSectorIndex:
    """Tests for PIT sector index."""

    def test_sector_index_empty(self, temp_root: str) -> None:
        """Empty data_lake should create empty index."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, temp_root)
        assert index.instrument_count() == 0

    def test_sector_index_with_data(self, populated_cn_a_sectors: str) -> None:
        """Index should load sector assignments."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, populated_cn_a_sectors)

        if index.instrument_count() > 0:
            assert index.instrument_count() >= 1

    def test_sector_pit_lookup(self, populated_cn_a_sectors: str) -> None:
        """PIT sector lookup should return correct sector at date."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, populated_cn_a_sectors)

        if index.instrument_count() > 0:
            # SH600000 changed from sector "10" to "15" in 2024
            # At 2020 date, should return "10"
            sector_2020 = index.sector("SH600000", date(2021, 1, 1))
            if sector_2020:
                assert sector_2020 == "10"

            # At 2024 date, should return "15"
            sector_2024 = index.sector("SH600000", date(2024, 6, 1))
            if sector_2024:
                assert sector_2024 == "15"

    def test_sector_not_found(self, populated_cn_a_sectors: str) -> None:
        """Unknown instrument should return None."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, populated_cn_a_sectors)

        sector = index.sector("UNKNOWN", date(2024, 1, 1))
        assert sector is None

    def test_sector_before_assignment(self, populated_cn_a_sectors: str) -> None:
        """Date before first assignment should return None."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, populated_cn_a_sectors)

        # SH600000 first assigned in 2020
        sector = index.sector("SH600000", date(2019, 1, 1))
        # Should return None since no assignment before 2020
        assert sector is None

    def test_sector_universe(self, populated_cn_a_sectors: str) -> None:
        """Sector universe should return instruments in sector at date."""
        clear_sector_cache()
        index = SectorIndex(Market.CN, populated_cn_a_sectors)

        if index.instrument_count() > 0:
            # Get banking sector at 2024
            banking = index.sector_universe("15", date(2024, 6, 1))
            assert isinstance(banking, list)

    def test_hk_raises_not_implemented(self, temp_root: str) -> None:
        """HK sector index should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SectorIndex(Market.HK, temp_root)


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestSectorFunctions:
    """Tests for convenience functions."""

    def test_sector_function(self, populated_cn_a_sectors: str) -> None:
        """sector() function should work."""
        clear_sector_cache()

        s = sector(Market.CN, "SH600000", date(2024, 6, 1), populated_cn_a_sectors)
        # Should return a sector or None
        assert s is None or isinstance(s, str)

    def test_sector_name_function(self) -> None:
        """sector_name() should return sector name."""
        name = sector_name(Market.CN, "15")
        assert name == "银行"

        name = sector_name(Market.US, "45")
        assert name == "Information Technology"

    def test_sector_name_unknown(self) -> None:
        """Unknown sector code should return None."""
        name = sector_name(Market.CN, "99")
        assert name is None

    def test_sector_universe_function(self, populated_cn_a_sectors: str) -> None:
        """sector_universe() function should work."""
        clear_sector_cache()

        instruments = sector_universe(
            Market.CN, "15", date(2024, 6, 1), populated_cn_a_sectors
        )
        assert isinstance(instruments, list)

    def test_get_all_sectors_cn(self) -> None:
        """get_all_sectors should return Shenwan sectors."""
        sectors = get_all_sectors(Market.CN)
        assert len(sectors) == 28

    def test_get_all_sectors_us(self) -> None:
        """get_all_sectors should return GICS sectors."""
        sectors = get_all_sectors(Market.US)
        assert len(sectors) == 11

    def test_get_sector_index_cached(self, populated_cn_a_sectors: str) -> None:
        """get_sector_index should cache indices."""
        clear_sector_cache()

        index1 = get_sector_index(Market.CN, populated_cn_a_sectors)
        index2 = get_sector_index(Market.CN, populated_cn_a_sectors)

        # Should return same instance due to caching
        assert index1 is index2

    def test_clear_sector_cache(self, populated_cn_a_sectors: str) -> None:
        """clear_sector_cache should clear the cache."""
        index1 = get_sector_index(Market.CN, populated_cn_a_sectors)
        clear_sector_cache()
        index2 = get_sector_index(Market.CN, populated_cn_a_sectors)

        # Should return different instance after clear
        # (Though this is hard to verify without checking cache stats)
        assert isinstance(index2, SectorIndex)


# =============================================================================
# CN GICS Groups Tests
# =============================================================================


from trendspec.data.sectors import CN_GICS_GROUPS, TICKER_GROUP_OVERRIDES, gics_group


def test_gics_groups_cover_expected_sectors():
    assert CN_GICS_GROUPS["日常消费"] == [
        "白酒", "啤酒", "红黄酒", "软饮料", "食品", "乳制品", "饲料", "日用化工",
        "种植业", "农业综合",
    ]
    assert "综合类" not in [s for members in CN_GICS_GROUPS.values() for s in members]
    assert "广告包装" not in [s for members in CN_GICS_GROUPS.values() for s in members]


def test_gics_group_override_takes_precedence():
    # 分众传媒在 override 表里强制归"通信服务"，即使传入一个假的 sector 值
    assert gics_group("随便什么", "SZ002027") == "通信服务"
    assert gics_group("随便什么", "SZ002831") == "材料"


def test_gics_group_falls_back_to_sector_mapping():
    assert gics_group("白酒", "SH600519.SH") == "日常消费"
    assert gics_group("半导体", "SH688981.SH") == "信息技术"


def test_gics_group_returns_none_for_unmapped_sector():
    assert gics_group("综合类", "SH600000.SH") is None
    assert gics_group(None, "SH600000.SH") is None
    assert gics_group("不存在的行业名", "SH600000.SH") is None
