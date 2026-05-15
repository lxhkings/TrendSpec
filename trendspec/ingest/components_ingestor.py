"""
Component changes ingestor.

Pulls historical component changes (IPO, delist, halt events).
Output: (date, instrument_id, event) long table.
Used by PIT Universe to avoid survivorship bias.

This module provides market-agnostic component ingestion.
Specific market ingestors (cn_a_ingestor, us_ingestor) use this.
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
# Event Types
# =============================================================================

# Standard event types for component tracking
EVENT_TYPES = {
    "IPO": "Initial public offering - stock added to exchange",
    "DELIST": "Delisting - stock removed from exchange",
    "HALT": "Trading halt - temporary suspension",
    "RESUME": "Trading resumed - after halt",
    "SP500_ADD": "Added to S&P 500 index",
    "SP500_DEL": "Removed from S&P 500 index",
    "R1000_ADD": "Added to Russell 1000 index",
    "R1000_DEL": "Removed from Russell 1000 index",
    "CSI300_ADD": "Added to CSI 300 index",
    "CSI300_DEL": "Removed from CSI 300 index",
}


# =============================================================================
# Generic Component Ingestor
# =============================================================================


def ingest_components(
    engine: Engine,
    market: Market,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest historical component changes for a market.

    Args:
        engine: SQLAlchemy engine
        market: Market enum
        manifest: Manifest for tracking sync state
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with row_count, date_range, instrument_count
    """
    dataset = "components"
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
    date_column = column_map.get("date", "event_date")

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

    # Validate event types
    valid_events = set(EVENT_TYPES.keys())
    if "event" in df.columns:
        unique_events = df.select("event").unique()["event"].to_list()
        unknown_events = set(unique_events) - valid_events
        if unknown_events:
            # Log warning but continue - new event types may be added
            pass

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
# Component Event Analysis
# =============================================================================


def get_ipo_instruments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get instruments that had IPO events.

    Args:
        df: Components DataFrame

    Returns:
        DataFrame with IPO events only
    """
    return df.filter(pl.col("event") == "IPO")


def get_delisted_instruments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get instruments that were delisted.

    Args:
        df: Components DataFrame

    Returns:
        DataFrame with delisting events only
    """
    return df.filter(pl.col("event") == "DELIST")


def get_index_changes(df: pl.DataFrame, index_name: str = "SP500") -> pl.DataFrame:
    """
    Get index addition/removal events.

    Args:
        df: Components DataFrame
        index_name: Index name (SP500, R1000, CSI300)

    Returns:
        DataFrame with index changes
    """
    add_event = f"{index_name}_ADD"
    del_event = f"{index_name}_DEL"
    return df.filter(pl.col("event").is_in([add_event, del_event]))


def get_active_instruments(df: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    """
    Get instruments that were active as of a specific date.

    Active = listed and not delisted/halted.

    Args:
        df: Components DataFrame
        as_of_date: Date to check

    Returns:
        DataFrame with active instrument_ids
    """
    # Filter events up to as_of_date
    historical = df.filter(pl.col("date") <= as_of_date)

    # Group by instrument_id and get latest event
    latest_events = historical.group_by("instrument_id").agg(
        pl.col("event").last().alias("latest_event"),
        pl.col("date").max().alias("last_event_date"),
    )

    # Active if latest event is not DELIST or HALT
    active = latest_events.filter(
        ~pl.col("latest_event").is_in(["DELIST", "HALT"])
    )

    return active.select(["instrument_id", "last_event_date"])


def get_delisted_before_date(df: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    """
    Get instruments that were delisted before a specific date.

    This is critical for PIT universe - need to exclude these from backtest.

    Args:
        df: Components DataFrame
        as_of_date: Date to check

    Returns:
        DataFrame with delisted instrument_ids and delist dates
    """
    delisted = df.filter(
        (pl.col("event") == "DELIST") & (pl.col("date") <= as_of_date)
    )

    return delisted.select(["instrument_id", "date"]).rename({"date": "delist_date"})


# =============================================================================
# Component Data Access
# =============================================================================


def read_components(
    root: str,
    market: Market,
    instrument_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """
    Read component events from Parquet cache.

    Args:
        root: Root directory for data_lake
        market: Market enum
        instrument_id: Optional instrument filter
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Polars DataFrame with component events
    """
    import os

    dataset = "components"
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
