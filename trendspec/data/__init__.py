"""
TrendSpec data module.

Provides market abstraction, schema definitions, data access utilities,
trading calendar, PIT sector tracking, and universe management.

Key design principles:
- Primary key is (instrument_id, date) - ticker can change
- PIT (point-in-time) design - prevents survivorship bias
- Every API must accept date parameter
"""

from trendspec.data.calendar import (
    count_trading_days,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import (
    bars,
    bars_for_instrument,
    read_components,
    read_sectors,
    scan_parquet,
    scan_parquet_glob,
)
from trendspec.data.schema import (
    COLUMN_TYPES,
    OHLC_COLUMNS,
    PRIMARY_KEY,
    REQUIRED_COLUMNS,
    AdjustmentMode,
    validate_dataframe_schema,
)
from trendspec.data.sectors import (
    SHENWAN_L1_SECTORS,
    GICS_SECTORS,
    SectorIndex,
    sector,
    sector_name,
    sector_universe,
    get_all_sectors,
    get_sector_index,
)
from trendspec.data.universe import (
    CNAUniverse,
    HKUniverse,
    USUniverse,
    Universe,
    get_universe,
)

__all__ = [
    # Markets
    "Market",
    # Schema
    "REQUIRED_COLUMNS",
    "COLUMN_TYPES",
    "PRIMARY_KEY",
    "OHLC_COLUMNS",
    "validate_dataframe_schema",
    # Calendar
    "is_trading_day",
    "trading_days_between",
    "next_trading_day",
    "previous_trading_day",
    "count_trading_days",
    # Parquet Loader
    "scan_parquet",
    "scan_parquet_glob",
    "bars",
    "bars_for_instrument",
    "read_components",
    "read_sectors",
    "AdjustmentMode",
    # Sectors
    "SHENWAN_L1_SECTORS",
    "GICS_SECTORS",
    "SectorIndex",
    "sector",
    "sector_name",
    "sector_universe",
    "get_all_sectors",
    "get_sector_index",
    # Universe
    "Universe",
    "CNAUniverse",
    "USUniverse",
    "HKUniverse",
    "get_universe",
]
