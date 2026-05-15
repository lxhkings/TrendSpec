"""
Incremental sync logic for ETL pipeline.

Reads manifest to get last date per instrument_id.
SQL WHERE date > :last_date for incremental pull.
Track sync state per instrument.
"""

from datetime import date, datetime
from typing import Any

import polars as pl
from sqlalchemy import Engine, text

from trendspec.ingest.manifest import Manifest


def rows_to_dataframe(rows: list[Any], column_names: list[str]) -> pl.DataFrame:
    """
    Convert SQLAlchemy result rows to Polars DataFrame.

    SQLAlchemy 2.x Row objects are tuple-like, use integer indices.
    SQLite stores dates as strings - converts to proper date types.

    Args:
        rows: List of SQLAlchemy Row objects
        column_names: List of column names

    Returns:
        Polars DataFrame
    """
    # Create DataFrame with proper column indexing
    data_dict = {}
    for i, col in enumerate(column_names):
        data_dict[col] = [row[i] for row in rows]

    df = pl.DataFrame(data_dict)

    # Convert date columns from string to date type (SQLite stores dates as strings)
    date_columns = ["trade_date", "event_date", "assign_date", "date"]
    for col in df.columns:
        if col in date_columns:
            # Try to convert to date if it's a string
            if df[col].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col).str.to_date(format="%Y-%m-%d").alias(col)
                )

    return df


def get_instruments_to_sync(
    manifest: Manifest,
    dataset: str,
    all_instruments: list[str],
) -> dict[str, str | None]:
    """
    Determine which instruments need syncing and their last dates.

    Args:
        manifest: Manifest object
        dataset: Dataset name (daily, components, sectors)
        all_instruments: List of all available instrument_ids

    Returns:
        Dict mapping instrument_id to last_date (None for new instruments)
    """
    result: dict[str, str | None] = {}

    for instrument_id in all_instruments:
        last_date = manifest.get_last_date(dataset, instrument_id)
        result[instrument_id] = last_date

    return result


def build_incremental_where_clause(
    instrument_id: str,
    last_date: str | None,
    date_column: str = "trade_date",
) -> str:
    """
    Build SQL WHERE clause for incremental pull.

    Args:
        instrument_id: Instrument ID to sync
        last_date: Last synced date (YYYY-MM-DD), None for full sync
        date_column: SQL column name for date

    Returns:
        SQL WHERE clause string
    """
    if last_date is None:
        # Full sync - no date filter
        return f"instrument_id = '{instrument_id}'"
    else:
        # Incremental sync - pull data after last_date
        return f"instrument_id = '{instrument_id}' AND {date_column} > '{last_date}'"


def sync_instrument_incremental(
    engine: Engine,
    table_name: str,
    column_map: dict[str, str],
    instrument_id: str,
    last_date: str | None,
) -> pl.DataFrame:
    """
    Sync data for a single instrument incrementally.

    Args:
        engine: SQLAlchemy engine
        table_name: SQL table name
        column_map: Column mapping (polars -> sql)
        instrument_id: Instrument ID to sync
        last_date: Last synced date, None for full sync

    Returns:
        Polars DataFrame with synced data
    """
    # Build SQL columns
    sql_columns = list(column_map.values())

    # Build WHERE clause
    date_column = column_map.get("date", "trade_date")
    where_clause = build_incremental_where_clause(instrument_id, last_date, date_column)

    # Build SQL query
    sql = text(
        f"SELECT {', '.join(sql_columns)} FROM {table_name} "
        f"WHERE {where_clause} ORDER BY {date_column}"
    )

    # Execute query and convert to DataFrame
    with engine.connect() as conn:
        result = conn.execute(sql)
        rows = result.fetchall()
        column_names = list(result.keys())

    if not rows:
        return pl.DataFrame()

    # Create DataFrame using helper function (handles date conversion)
    df = rows_to_dataframe(rows, column_names)

    # Rename columns to Polars names
    rename_map = {sql_col: polars_col for polars_col, sql_col in column_map.items()}
    df = df.rename(rename_map)

    return df


def sync_batch_incremental(
    engine: Engine,
    table_name: str,
    column_map: dict[str, str],
    instruments_with_dates: dict[str, str | None],
    batch_size: int = 100,
) -> pl.DataFrame:
    """
    Sync data for multiple instruments in batches.

    Args:
        engine: SQLAlchemy engine
        table_name: SQL table name
        column_map: Column mapping
        instruments_with_dates: Dict of {instrument_id: last_date}
        batch_size: Number of instruments per batch query

    Returns:
        Combined Polars DataFrame with all synced data
    """
    # Build SQL columns
    sql_columns = list(column_map.values())
    date_column = column_map.get("date", "trade_date")

    # Separate new instruments (no last_date) from incremental ones
    new_instruments = [iid for iid, ld in instruments_with_dates.items() if ld is None]
    incremental_instruments = {
        iid: ld for iid, ld in instruments_with_dates.items() if ld is not None
    }

    all_dfs: list[pl.DataFrame] = []

    # Handle new instruments (full sync)
    if new_instruments:
        # Batch query for new instruments
        for i in range(0, len(new_instruments), batch_size):
            batch = new_instruments[i:i + batch_size]
            instrument_list = "','".join(batch)
            sql = text(
                f"SELECT {', '.join(sql_columns)} FROM {table_name} "
                f"WHERE instrument_id IN ('{instrument_list}') "
                f"ORDER BY instrument_id, {date_column}"
            )

            with engine.connect() as conn:
                result = conn.execute(sql)
                rows = result.fetchall()
                columns = result.keys()

            if rows:
                column_names = list(columns)
                # Create DataFrame using helper function (handles date conversion)
                df = rows_to_dataframe(rows, column_names)
                rename_map = {sql_col: polars_col for polars_col, sql_col in column_map.items()}
                df = df.rename(rename_map)
                all_dfs.append(df)

    # Handle incremental instruments
    for instrument_id, last_date in incremental_instruments.items():
        df = sync_instrument_incremental(
            engine, table_name, column_map, instrument_id, last_date
        )
        if not df.is_empty():
            all_dfs.append(df)

    # Combine all DataFrames
    if not all_dfs:
        return pl.DataFrame()

    return pl.concat(all_dfs)


def update_manifest_after_sync(
    manifest: Manifest,
    dataset: str,
    df: pl.DataFrame,
) -> None:
    """
    Update manifest with new sync state after data pull.

    Args:
        manifest: Manifest object
        dataset: Dataset name
        df: Polars DataFrame with synced data
    """
    if df.is_empty():
        return

    # Get max date per instrument_id
    max_dates = df.group_by("instrument_id").agg(
        pl.col("date").max().alias("last_date")
    )

    # Update manifest for each instrument
    for row in max_dates.iter_rows(named=True):
        instrument_id = row["instrument_id"]
        last_date = row["last_date"]
        if isinstance(last_date, date):
            last_date_str = last_date.isoformat()
        elif isinstance(last_date, datetime):
            last_date_str = last_date.date().isoformat()
        else:
            last_date_str = str(last_date)

        manifest.update_instrument_date(dataset, instrument_id, last_date_str)


def get_full_date_range(df: pl.DataFrame) -> tuple[str, str]:
    """
    Get the full date range from a DataFrame.

    Args:
        df: Polars DataFrame

    Returns:
        Tuple of (min_date, max_date) as strings
    """
    if df.is_empty():
        return ("", "")

    min_date = df.select(pl.col("date").min()).item()
    max_date = df.select(pl.col("date").max()).item()

    if isinstance(min_date, date):
        min_date_str = min_date.isoformat()
    elif isinstance(min_date, datetime):
        min_date_str = min_date.date().isoformat()
    else:
        min_date_str = str(min_date)

    if isinstance(max_date, date):
        max_date_str = max_date.isoformat()
    elif isinstance(max_date, datetime):
        max_date_str = max_date.date().isoformat()
    else:
        max_date_str = str(max_date)

    return (min_date_str, max_date_str)
