"""
US stock data ingestor.

Pulls daily OHLCV for SP500 + Russell 1000 (~1050 stocks).
Handles ticker changes and delisted stocks.
Uses incremental sync logic.

Datasets:
- daily: Daily OHLCV data
- components: Historical component changes (IPO, delist, index changes)
- sectors: Historical sector assignments (GICS Sector)
"""

from sqlalchemy import Engine, text

from trendspec.data.markets import Market
from trendspec.ingest.incremental import (
    get_full_date_range,
    rows_to_dataframe,
    sync_batch_incremental,
    update_manifest_after_sync,
)
from trendspec.ingest.manifest import Manifest
from trendspec.ingest.schema_map import (
    US_COMPONENTS_MAP,
    US_DAILY_MAP,
    US_SECTORS_MAP,
    get_table_name,
)
from trendspec.ingest.writer import write_parquet

# =============================================================================
# US Instrument Universe
# =============================================================================

# Target universe: SP500 + Russell 1000 (~1050 stocks)
# These are tracked in the components table

# Default universe sources
US_UNIVERSE_TABLES = ["sp500", "russell1000"]


def get_us_instrument_list(engine: Engine) -> list[str]:
    """
    Get list of US instrument_ids from database.

    Includes SP500 + Russell 1000 constituents, both current and historical.
    This ensures we capture delisted stocks for survivorship bias prevention.

    Args:
        engine: SQLAlchemy engine

    Returns:
        List of instrument_ids (e.g., ["AAPL", "MSFT", ...])
    """
    # Query distinct instrument_ids from all universe tables
    # Also include historical (delisted) instruments from components table
    daily_table = get_table_name(Market.US, "daily")
    components_table = get_table_name(Market.US, "components")
    sql = text(f"""
        SELECT DISTINCT instrument_id FROM (
            SELECT instrument_id FROM {daily_table}
            UNION
            SELECT instrument_id FROM {components_table} WHERE event_type IN ('IPO', 'DELIST')
        ) ORDER BY instrument_id
    """)

    with engine.connect() as conn:
        result = conn.execute(sql)
        return [row[0] for row in result.fetchall()]


# =============================================================================
# US Daily Ingestor
# =============================================================================


def ingest_us_daily(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest daily OHLCV data for US stocks.

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for tracking sync state
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with row_count, date_range, instrument_count
    """
    dataset = "daily"
    table_name = get_table_name(Market.US, dataset)
    column_map = US_DAILY_MAP

    # Get instrument list
    all_instruments = get_us_instrument_list(engine)

    if full_sync:
        # Full sync - pull all data without date filter
        instruments_with_dates: dict[str, str | None] = dict.fromkeys(all_instruments)
    else:
        # Incremental sync - use manifest
        from trendspec.ingest.incremental import get_instruments_to_sync
        instruments_with_dates = get_instruments_to_sync(manifest, dataset, all_instruments)

    # Sync data
    df = sync_batch_incremental(engine, table_name, column_map, instruments_with_dates)

    if df.is_empty():
        return {"row_count": 0, "date_range": ("", ""), "instrument_count": 0}

    # Write to Parquet
    write_parquet(df, Market.US, dataset, root)

    # Update manifest
    update_manifest_after_sync(manifest, dataset, df)

    # Get summary
    date_range = get_full_date_range(df)
    instrument_count = df.select("instrument_id").n_unique()
    row_count = len(df)

    # Update dataset state in manifest
    manifest.update_dataset_state(dataset, row_count, date_range, instrument_count)

    return {
        "row_count": row_count,
        "date_range": date_range,
        "instrument_count": instrument_count,
    }


# =============================================================================
# US Components Ingestor
# =============================================================================


def ingest_us_components(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest historical component changes for US stocks.

    Events: IPO, DELIST, SP500_ADD, SP500_DEL, R1000_ADD, R1000_DEL, etc.
    Output: (date, instrument_id, event) long table.

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for tracking sync state
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with row_count, date_range, instrument_count
    """
    dataset = "components"
    table_name = get_table_name(Market.US, dataset)
    column_map = US_COMPONENTS_MAP

    # Get date range for sync
    if full_sync:
        last_date = None
    else:
        state = manifest.get_dataset_state(dataset)
        if state and "date_range" in state:
            last_date = state["date_range"]["end"]
        else:
            last_date = None

    # Build SQL query with parameterized date filter (prevents SQL injection)
    sql_columns = list(column_map.values())
    date_column = column_map.get("date", "event_date")

    if last_date:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"WHERE {date_column} > :last_date ORDER BY {date_column}"
        )
        params = {"last_date": last_date}
    else:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"ORDER BY {date_column}"
        )
        params = {}

    # Execute query
    with engine.connect() as conn:
        result = conn.execute(sql, params)
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
    write_parquet(df, Market.US, dataset, root)

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
# US Sectors Ingestor
# =============================================================================


def ingest_us_sectors(
    engine: Engine,
    manifest: Manifest,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest historical sector assignments for US stocks.

    Classification: GICS Sector (8 sectors for backtesting, 11 full)
    Output: (date, instrument_id, sector) long table.

    Args:
        engine: SQLAlchemy engine
        manifest: Manifest for tracking sync state
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with row_count, date_range, instrument_count
    """

    dataset = "sectors"
    table_name = get_table_name(Market.US, dataset)
    column_map = US_SECTORS_MAP

    # Get date range for sync
    if full_sync:
        last_date = None
    else:
        state = manifest.get_dataset_state(dataset)
        if state and "date_range" in state:
            last_date = state["date_range"]["end"]
        else:
            last_date = None

    # Build SQL query with parameterized date filter (prevents SQL injection)
    sql_columns = list(column_map.values())
    date_column = column_map.get("date", "assign_date")

    if last_date:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"WHERE {date_column} > :last_date ORDER BY {date_column}"
        )
        params = {"last_date": last_date}
    else:
        sql = text(
            f"SELECT {', '.join(sql_columns)} FROM {table_name} "
            f"ORDER BY {date_column}"
        )
        params = {}

    # Execute query
    with engine.connect() as conn:
        result = conn.execute(sql, params)
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
    write_parquet(df, Market.US, dataset, root)

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
# Full US Ingest
# =============================================================================


def ingest_us_full(
    engine: Engine,
    root: str,
    full_sync: bool = False,
) -> dict:
    """
    Ingest all US datasets.

    Args:
        engine: SQLAlchemy engine
        root: Root directory for data_lake
        full_sync: If True, ignore manifest and pull all data

    Returns:
        Summary dict with results for each dataset
    """
    manifest = Manifest(Market.US, root)

    results = {
        "daily": ingest_us_daily(engine, manifest, root, full_sync),
        "components": ingest_us_components(engine, manifest, root, full_sync),
        "sectors": ingest_us_sectors(engine, manifest, root, full_sync),
    }

    return results
