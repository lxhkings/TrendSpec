"""
Schema mapping for SQL tables to Polars columns.

Maps SQL column names to the standardized TrendSpec schema.
Different markets may have different column naming conventions.

Standard columns (from schema.py):
- instrument_id: Immutable internal ID (primary key)
- date: Trading date (primary key)
- ticker: Display symbol (mutable)
- open, high, low, close: OHLC prices
- volume: Trading volume
- adj_factor: Adjustment factor for corporate actions
"""

from typing import Final

from trendspec.data.markets import Market

# =============================================================================
# Column Mapping Structure
# =============================================================================

# Each mapping is a dict: {polars_column: sql_column}
# The ingestor reads from SQL using sql_column, then renames to polars_column


# =============================================================================
# CN_A (China A-shares) Schema Mapping
# =============================================================================

CN_A_DAILY_MAP: Final[dict[str, str]] = {
    "instrument_id": "instrument_id",  # SH/SZ prefix + 6-digit code
    "date": "trade_date",
    "ticker": "ticker",  # Chinese name or 6-digit code
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "close": "close_price",
    "volume": "volume",
    "adj_factor": "adj_factor",
}

CN_A_COMPONENTS_MAP: Final[dict[str, str]] = {
    "date": "event_date",
    "instrument_id": "instrument_id",
    "event": "event_type",  # IPO, DELIST, HALT, RESUME
    "details": "event_details",  # Additional info (nullable)
}

CN_A_SECTORS_MAP: Final[dict[str, str]] = {
    "date": "assign_date",
    "instrument_id": "instrument_id",
    "sector": "sector_code",  # Shenwan Level 1 sector code
    "sector_name": "sector_name",  # Chinese sector name
}


# =============================================================================
# US (US stocks) Schema Mapping
# =============================================================================

US_DAILY_MAP: Final[dict[str, str]] = {
    "instrument_id": "instrument_id",  # Ticker-based ID (AAPL, MSFT, etc.)
    "date": "trade_date",
    "ticker": "ticker",
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "close": "close_price",
    "volume": "volume",
    "adj_factor": "adj_factor",
}

US_COMPONENTS_MAP: Final[dict[str, str]] = {
    "date": "event_date",
    "instrument_id": "instrument_id",
    "event": "event_type",  # IPO, DELIST, HALT, RESUME, SP500_ADD, SP500_DEL, etc.
    "details": "event_details",  # Additional info (nullable)
}

US_SECTORS_MAP: Final[dict[str, str]] = {
    "date": "assign_date",
    "instrument_id": "instrument_id",
    "sector": "sector_code",  # GICS sector code
    "sector_name": "sector_name",  # Sector name
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_column_map(market: Market, dataset: str) -> dict[str, str]:
    """
    Get the column mapping for a specific market and dataset.

    Args:
        market: Market enum (CN_A, US)
        dataset: Dataset type ("daily", "components", "sectors")

    Returns:
        Dictionary mapping Polars column names to SQL column names

    Raises:
        ValueError: If market/dataset combination is not supported
    """
    mapping_key = f"{market.value}_{dataset.upper()}_MAP"

    mappings = {
        "CN_A_DAILY_MAP": CN_A_DAILY_MAP,
        "CN_A_COMPONENTS_MAP": CN_A_COMPONENTS_MAP,
        "CN_A_SECTORS_MAP": CN_A_SECTORS_MAP,
        "US_DAILY_MAP": US_DAILY_MAP,
        "US_COMPONENTS_MAP": US_COMPONENTS_MAP,
        "US_SECTORS_MAP": US_SECTORS_MAP,
    }

    if mapping_key not in mappings:
        raise ValueError(
            f"No schema mapping for {market.value}/{dataset}. "
            f"Available: {list(mappings.keys())}"
        )

    return mappings[mapping_key]


def get_sql_columns(market: Market, dataset: str) -> list[str]:
    """
    Get the list of SQL column names to select for a dataset.

    Args:
        market: Market enum
        dataset: Dataset type ("daily", "components", "sectors")

    Returns:
        List of SQL column names to query
    """
    column_map = get_column_map(market, dataset)
    return list(column_map.values())


def get_table_name(market: Market, dataset: str) -> str:
    """
    Get the SQL table name for a market and dataset.

    Args:
        market: Market enum
        dataset: Dataset type ("daily", "components", "sectors")

    Returns:
        SQL table name
    """
    # Convention: {market_lower}_{dataset}
    return f"{market.path}_{dataset}"


# =============================================================================
# Derived Columns (computed during ingest)
# =============================================================================

# These columns are derived from SQL data during ingest

DERIVED_COLUMNS: Final[dict[str, str]] = {
    # year: Extracted from date for partitioning
    "year": "year",
}


def derive_instrument_id_cn(code: str, exchange: str) -> str:
    """
    Derive instrument_id for CN_A stocks.

    Args:
        code: 6-digit stock code (e.g., "600000")
        exchange: Exchange code ("SH" for Shanghai, "SZ" for Shenzhen)

    Returns:
        instrument_id (e.g., "SH600000")
    """
    # Standardize exchange prefix
    exchange_prefix = exchange.upper()
    if exchange_prefix not in ("SH", "SZ"):
        raise ValueError(f"Invalid exchange: {exchange}. Must be 'SH' or 'SZ'.")

    return f"{exchange_prefix}{code}"


def derive_instrument_id_us(ticker: str) -> str:
    """
    Derive instrument_id for US stocks.

    Args:
        ticker: Stock ticker (e.g., "AAPL")

    Returns:
        instrument_id (same as ticker for US)
    """
    # US instrument_id is the ticker itself (uppercase)
    return ticker.upper()
