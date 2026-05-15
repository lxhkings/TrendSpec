"""
Sector assignments ingestor.

Pulls historical sector assignments (date, instrument_id, sector).
CN_A: Shenwan Level 1 (28 sectors)
US: GICS Sector (8 sectors for backtesting)
Used by data/sectors.py for PIT sector lookup.

This module provides market-agnostic sector ingestion.
"""

from datetime import date

import polars as pl
from sqlalchemy import Engine, text

from trendspec.data.markets import Market
from trendspec.ingest.incremental import get_full_date_range, rows_to_dataframe
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.schema_map import get_column_map, get_table_name
from trendspec.ingest.writer import write_parquet

# =============================================================================
# Sector Classifications
# =============================================================================

# Shenwan Level 1 sectors (CN_A)
SHENWAN_L1_SECTORS = {
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

# GICS sectors (US) - 8 sectors for backtesting (grouped from 11)
GICS_SECTORS_8 = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Technology",
    # Note: Real Estate (60) and Utilities (55) often grouped with Financials
}

# Full GICS sectors (11)
GICS_SECTORS_11 = {
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


# =============================================================================
# Generic Sector Ingestor
# =============================================================================


def ingest_sectors(
    engine: Engine,
    market: Market,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest historical sector assignments for a market.

    Args:
        engine: SQLAlchemy engine
        market: Market enum
        manifest: Manifest for tracking sync state
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with row_count, date_range, instrument_count
    """
    dataset = "sectors"
    table_name = get_table_name(market, dataset)
    column_map = get_column_map(market, dataset)

    # Get date range for sync
    if full_sync:
        last_date = None
    else:
        state = manifest.get_dataset_state(dataset)
        if state and "date_range" in state:
            last_date = state["date_range"]["end"]
        else:
            last_date = None

    # Build SQL query
    sql_columns = list(column_map.values())
    date_column = column_map.get("date", "assign_date")

    if last_date:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"WHERE {date_column} > '{last_date}' ORDER BY {date_column}"
        )
    else:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"ORDER BY {date_column}"
        )

    # Execute query
    with engine.connect() as conn:
        result = conn.execute(sql)
        rows = result.fetchall()
        column_names = list(result.keys())

    if not rows:
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    # Create DataFrame using helper function
    df = rows_to_dataframe(rows, column_names)

    # Rename columns
    rename_map = {sql_col: polars_col for polars_col, sql_col in column_map.items()}
    df = df.rename(rename_map)

    # Write to Parquet
    write_parquet(df, market, dataset, root)

    # Get summary
    date_range = get_full_date_range(df)
    instrument_count = df.select("instrument_id").n_unique()
    row_count = len(df)

    # Update manifest
    manifest.update_dataset_state(dataset, row_count, date_range, instrument_count)

    return {
        "row_count": row_count,
        "date_range": date_range,
        "instrument_count": instrument_count,
    }


# =============================================================================
# Sector Lookup Functions
# =============================================================================


def get_sector_at_date(
    df: pl.DataFrame,
    instrument_id: str,
    as_of_date: date,
) -> str | None:
    """
    Get sector assignment for an instrument as of a specific date.

    PIT (point-in-time) sector lookup for backtesting.

    Args:
        df: Sectors DataFrame
        instrument_id: Instrument ID
        as_of_date: Date to check

    Returns:
        Sector code/name or None if not found
    """
    # Filter for this instrument and dates <= as_of_date
    filtered = df.filter(
        (pl.col("instrument_id") == instrument_id) &
        (pl.col("date") <= as_of_date)
    )

    if filtered.is_empty():
        return None

    # Get latest assignment (sort by date descending and take first row)
    latest = filtered.sort("date", descending=True).head(1)

    if "sector" in latest.columns:
        return latest["sector"].item()
    elif "sector_name" in latest.columns:
        return latest["sector_name"].item()

    return None


def get_sector_name(market: Market, sector_code: str) -> str | None:
    """
    Get sector name from sector code for a market.

    Args:
        market: Market enum
        sector_code: Sector code

    Returns:
        Sector name or None if not found
    """
    if market == Market.CN_A:
        return SHENWAN_L1_SECTORS.get(sector_code)
    elif market == Market.US:
        return GICS_SECTORS_11.get(sector_code)
    else:
        return None


def get_all_sectors(market: Market) -> dict[str, str]:
    """
    Get all sector codes and names for a market.

    Args:
        market: Market enum

    Returns:
        Dict mapping sector code to sector name
    """
    if market == Market.CN_A:
        return SHENWAN_L1_SECTORS
    elif market == Market.US:
        return GICS_SECTORS_11
    elif market == Market.HK:
        return GICS_SECTORS_11  # HK uses GICS
    else:
        return {}


def get_sector_instruments(
    df: pl.DataFrame,
    sector: str,
    as_of_date: date,
) -> list[str]:
    """
    Get all instruments in a sector as of a specific date.

    Args:
        df: Sectors DataFrame
        sector: Sector code or name
        as_of_date: Date to check

    Returns:
        List of instrument_ids in the sector
    """
    # Filter for sector assignments before as_of_date
    historical = df.filter(pl.col("date") <= as_of_date)

    # Group by instrument and get latest sector
    # Only aggregate columns that exist in the DataFrame
    agg_exprs = []
    if "sector" in df.columns:
        agg_exprs.append(pl.col("sector").last().alias("current_sector"))
    if "sector_name" in df.columns:
        agg_exprs.append(pl.col("sector_name").last().alias("current_sector_name"))

    if not agg_exprs:
        return []

    latest_assignments = historical.group_by("instrument_id").agg(agg_exprs)

    # Filter for target sector
    if "current_sector" in latest_assignments.columns:
        in_sector = latest_assignments.filter(pl.col("current_sector") == sector)
    elif "current_sector_name" in latest_assignments.columns:
        in_sector = latest_assignments.filter(pl.col("current_sector_name") == sector)
    else:
        return []

    return in_sector["instrument_id"].to_list()


# =============================================================================
# Sector Data Access
# =============================================================================


def read_sectors(
    root: str,
    market: Market,
    instrument_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """
    Read sector assignments from Parquet cache.

    Args:
        root: Root directory for data_lake
        market: Market enum
        instrument_id: Optional instrument filter
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Polars DataFrame with sector assignments
    """
    import os

    dataset = "sectors"
    dataset_path = os.path.join(root, market.path, dataset)

    if not os.path.exists(dataset_path):
        return pl.DataFrame()

    # Read all Parquet files in dataset
    parquet_files = []
    for partition_dir in os.listdir(dataset_path):
        if partition_dir.startswith("instrument_id="):
            partition_path = os.path.join(dataset_path, partition_dir)
            for f in os.listdir(partition_path):
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(partition_path, f))

    if not parquet_files:
        return pl.DataFrame()

    df = pl.read_parquet(parquet_files)

    # Apply filters
    if instrument_id is not None:
        df = df.filter(pl.col("instrument_id") == instrument_id)

    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)

    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)

    return df.sort("date")
