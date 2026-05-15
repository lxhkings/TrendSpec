"""
PIT sector attribution for TrendSpec.

Pre-built memory index for O(1) sector lookup.
Key design:
- Must accept date parameter - no "current sector" shortcuts
- Primary key is (instrument_id, date) - ticker can change
- ~150MB memory acceptable for pre-built index

Sector classifications:
- CN_A: Shenwan Level 1 (28 sectors)
- US: GICS Sector (11 sectors, often grouped to 8 for backtesting)
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Final

import polars as pl

from trendspec.config.settings import get_settings
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import scan_parquet, _lazyframe_is_empty

# =============================================================================
# Sector Classifications
# =============================================================================

# Shenwan Level 1 sectors (CN_A) - 28 sectors
SHENWAN_L1_SECTORS: Final[dict[str, str]] = {
    "01": "农林牧渔",
    "02": "采掘",
    "03": "化工",
    "04": "钢铁",
    "05": "有色金属",
    "06": "电子",
    "07": "家用电器",
    "08": "食品饮料",
    "09": "纺织服饰",
    "10": "轻工制造",
    "11": "医药生物",
    "12": "公用事业",
    "13": "交通运输",
    "14": "房地产",
    "15": "银行",
    "16": "非银金融",
    "17": "综合",
    "18": "建筑建材",
    "19": "建筑装饰",
    "20": "电气设备",
    "21": "机械设备",
    "22": "国防军工",
    "23": "计算机",
    "24": "传媒",
    "25": "通信",
    "26": "商贸零售",
    "27": "社会服务",
    "28": "汽车",
}

# GICS sectors (US) - 11 sectors
GICS_SECTORS: Final[dict[str, str]] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}

# GICS sectors grouped to 8 for backtesting (common grouping)
GICS_SECTORS_8: Final[dict[str, str]] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer",  # Discretionary + Staples combined
    "35": "Health Care",
    "40": "Financials",  # Financials + Real Estate combined
    "45": "Technology",
    "50": "Communication",  # Communication Services
}


# =============================================================================
# PIT Sector Memory Index
# =============================================================================

class SectorIndex:
    """
    Pre-built memory index for PIT sector lookup.

    Structure: dict[instrument_id, dict[date, sector]]

    Memory estimate: ~150MB for full CN_A + US universe
    Provides O(1) lookup for sector at any point in time.

    Critical for survivorship bias prevention - uses historical
    sector assignments, not current classifications.
    """

    def __init__(self, market: Market, root: str | None = None) -> None:
        """
        Initialize sector index for a market.

        Loads all sector assignments into memory for O(1) lookup.

        Args:
            market: Market enum (CN_A, US, HK)
            root: Root directory for data_lake
        """
        self.market = market
        self.root = root or get_settings().data_lake.data_lake_root
        self._index: dict[str, dict[date, str]] = {}
        self._dates_by_instrument: dict[str, list[date]] = {}

        # Load index on initialization
        self._build_index()

    def _build_index(self) -> None:
        """
        Build memory index from Parquet sectors data.

        Reads all sector assignments and builds:
        - dict[instrument_id, dict[date, sector]]
        - dict[instrument_id, sorted_dates] for binary search
        """
        if self.market == Market.HK:
            raise NotImplementedError(
                "Hong Kong market sector index not yet implemented."
            )

        # Scan sectors Parquet
        lf = scan_parquet(self.root, self.market, "sectors")

        if _lazyframe_is_empty(lf):
            return

        # Collect all sector assignments
        df = lf.collect()

        if df.is_empty():
            return

        # Build index structure
        # Group by instrument_id and date -> sector
        for row in df.iter_rows(named=True):
            instrument_id = row.get("instrument_id")
            assign_date = row.get("date")
            sector = row.get("sector") or row.get("sector_code")

            if instrument_id and assign_date and sector:
                if instrument_id not in self._index:
                    self._index[instrument_id] = {}
                    self._dates_by_instrument[instrument_id] = []

                self._index[instrument_id][assign_date] = sector
                self._dates_by_instrument[instrument_id].append(assign_date)

        # Sort dates for each instrument for binary search
        for instrument_id in self._dates_by_instrument:
            self._dates_by_instrument[instrument_id].sort()

    def sector(self, instrument_id: str, as_of_date: date) -> str | None:
        """
        Get sector for an instrument at a specific date (PIT lookup).

        PIT design: as_of_date parameter is REQUIRED.
        No "current sector" shortcuts - prevents survivorship bias.

        Uses binary search on sorted dates for efficiency.

        Args:
            instrument_id: Instrument ID
            as_of_date: Date to check

        Returns:
            Sector code or None if instrument not in index

        Example:
            >>> index = SectorIndex(Market.CN)
            >>> index.sector("SH600000", date(2024, 1, 15))
            '15'  # Banking sector
        """
        if instrument_id not in self._index:
            return None

        # Get sorted dates for this instrument
        sorted_dates = self._dates_by_instrument[instrument_id]

        if not sorted_dates:
            return None

        # Binary search: find the latest date <= as_of_date
        # This is the sector assignment that was active at as_of_date
        left, right = 0, len(sorted_dates) - 1
        result_idx = -1

        while left <= right:
            mid = (left + right) // 2
            mid_date = sorted_dates[mid]

            if mid_date <= as_of_date:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        if result_idx == -1:
            # No sector assignment before as_of_date
            return None

        # Return the sector at the found date
        found_date = sorted_dates[result_idx]
        return self._index[instrument_id].get(found_date)

    def sector_universe(
        self,
        sector_code: str,
        as_of_date: date,
    ) -> list[str]:
        """
        Get all instruments in a sector at a specific date (PIT lookup).

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            sector_code: Sector code to filter
            as_of_date: Date to check

        Returns:
            List of instrument_ids in the sector at that date

        Example:
            >>> index = SectorIndex(Market.CN)
            >>> index.sector_universe("15", date(2024, 1, 15))
            ['SH600000', 'SH600016', ...]  # Banking stocks
        """
        instruments: list[str] = []

        for instrument_id in self._index:
            s = self.sector(instrument_id, as_of_date)
            if s == sector_code:
                instruments.append(instrument_id)

        return instruments

    def all_sectors_at_date(self, as_of_date: date) -> dict[str, list[str]]:
        """
        Get all sectors and their instruments at a specific date.

        PIT design: as_of_date parameter is REQUIRED.

        Args:
            as_of_date: Date to check

        Returns:
            Dict mapping sector code to list of instrument_ids
        """
        result: dict[str, list[str]] = {}

        for instrument_id in self._index:
            s = self.sector(instrument_id, as_of_date)
            if s:
                if s not in result:
                    result[s] = []
                result[s].append(instrument_id)

        return result

    def instrument_count(self) -> int:
        """Get total number of instruments in the index."""
        return len(self._index)


# =============================================================================
# Cached Sector Indices
# =============================================================================

# Global cache for sector indices
# Uses lru_cache to avoid rebuilding indices
@lru_cache(maxsize=4)
def get_sector_index(market: Market, root: str | None = None) -> SectorIndex:
    """
    Get cached sector index for a market.

    Indices are cached to avoid rebuilding on each call.

    Args:
        market: Market enum
        root: Root directory for data_lake

    Returns:
        SectorIndex instance
    """
    # Note: root is used in cache key, so None is converted to actual root
    actual_root = root or get_settings().data_lake.data_lake_root
    return SectorIndex(market, actual_root)


# =============================================================================
# Convenience Functions
# =============================================================================


def sector(
    market: Market,
    instrument_id: str,
    as_of_date: date,
    root: str | None = None,
) -> str | None:
    """
    Get sector for an instrument at a specific date (PIT lookup).

    PIT design: as_of_date parameter is REQUIRED.

    Args:
        market: Market enum
        instrument_id: Instrument ID
        as_of_date: Date to check
        root: Root directory for data_lake

    Returns:
        Sector code or None

    Example:
        >>> sector(Market.CN, "SH600000", date(2024, 1, 15))
        '15'
    """
    index = get_sector_index(market, root)
    return index.sector(instrument_id, as_of_date)


def sector_name(
    market: Market,
    sector_code: str,
) -> str | None:
    """
    Get sector name from sector code.

    Args:
        market: Market enum
        sector_code: Sector code

    Returns:
        Sector name or None if not found
    """
    if market == Market.CN:
        return SHENWAN_L1_SECTORS.get(sector_code)
    elif market == Market.US:
        return GICS_SECTORS.get(sector_code)
    elif market == Market.HK:
        return GICS_SECTORS.get(sector_code)
    return None


def sector_universe(
    market: Market,
    sector_code: str,
    as_of_date: date,
    root: str | None = None,
) -> list[str]:
    """
    Get all instruments in a sector at a specific date (PIT lookup).

    PIT design: as_of_date parameter is REQUIRED.

    Args:
        market: Market enum
        sector_code: Sector code
        as_of_date: Date to check
        root: Root directory for data_lake

    Returns:
        List of instrument_ids in the sector at that date

    Example:
        >>> sector_universe(Market.CN, "15", date(2024, 1, 15))
        ['SH600000', 'SH600016', ...]
    """
    index = get_sector_index(market, root)
    return index.sector_universe(sector_code, as_of_date)


def get_all_sectors(market: Market) -> dict[str, str]:
    """
    Get all sector codes and names for a market.

    Args:
        market: Market enum

    Returns:
        Dict mapping sector code to sector name
    """
    if market == Market.CN:
        return dict(SHENWAN_L1_SECTORS)
    elif market == Market.US:
        return dict(GICS_SECTORS)
    elif market == Market.HK:
        return dict(GICS_SECTORS)
    return {}


def clear_sector_cache() -> None:
    """
    Clear the sector index cache.

    Useful for testing or when sector data is updated.
    """
    get_sector_index.cache_clear()