"""
Standard data schema definitions for TrendSpec.

Defines the canonical data structure for market data across all markets.
The primary key is (instrument_id, date) - instrument_id is immutable while
ticker can change due to renames or delistings.

Key design decision:
- instrument_id: Internal immutable identifier (primary key)
- ticker: Display symbol, mutable (can change on ticker rename)
- This ensures accurate historical tracking regardless of symbol changes
"""

from typing import Final

import polars as pl

# =============================================================================
# Primary Key Definition (One-Way Door Decision)
# =============================================================================

# Primary key is (instrument_id, date), NOT (ticker, date)
# Rationale:
# 1. Tickers can change (company rename, ticker symbol change)
# 2. Tickers can be reused after delisting (different company, same ticker)
# 3. instrument_id is immutable and uniquely identifies a security

PRIMARY_KEY: Final[tuple[str, str]] = ("instrument_id", "date")

# =============================================================================
# Required Columns
# =============================================================================

# Core identification columns
_INSTRUMENT_COLUMNS: Final[frozenset[str]] = frozenset({
    "instrument_id",  # Immutable internal ID (primary key)
    "date",           # Trading date (primary key)
    "ticker",         # Display ticker/symbol (mutable)
})

# Price columns (OHLC)
_PRICE_COLUMNS: Final[frozenset[str]] = frozenset({
    "open",
    "high",
    "low",
    "close",
})

# Volume and adjustment
_VOLUME_COLUMNS: Final[frozenset[str]] = frozenset({
    "volume",      # Trading volume (shares)
    "adj_factor",  # Adjustment factor for corporate actions
})

# All required columns
REQUIRED_COLUMNS: Final[frozenset[str]] = (
    _INSTRUMENT_COLUMNS | _PRICE_COLUMNS | _VOLUME_COLUMNS
)

# OHLC columns for easy access
OHLC_COLUMNS: Final[frozenset[str]] = _PRICE_COLUMNS

# =============================================================================
# Column Types (Polars dtypes)
# =============================================================================

COLUMN_TYPES: Final[dict[str, pl.DataType]] = {
    # Identification
    "instrument_id": pl.String,  # String for flexibility (e.g., "SH600000", "AAPL")
    "date": pl.Date,
    "ticker": pl.String,
    # Prices (Float64 for precision in financial calculations)
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    # Volume and adjustment
    "volume": pl.Int64,      # Can be large, use Int64
    "adj_factor": pl.Float64,  # Typically 1.0 initially, changes with corporate actions
}

# =============================================================================
# Optional Columns (may be present depending on data source)
# =============================================================================

OPTIONAL_COLUMNS: Final[frozenset[str]] = frozenset({
    "amount",        # Trading amount (price * volume)
    "turnover",      # Turnover rate
    "vwap",          # Volume-weighted average price
    "num_trades",    # Number of trades
    "bid_high",      # Bid high
    "bid_low",       # Bid low
    "ask_high",      # Ask high
    "ask_low",       # Ask low
    "limit_up",      # Limit up price (CN_A specific)
    "limit_down",    # Limit down price (CN_A specific)
    "suspended",     # Trading suspended flag
})

# =============================================================================
# Adjustment Modes
# =============================================================================

# Price adjustment modes for handling corporate actions
class AdjustmentMode:
    """Price adjustment modes for handling corporate actions (dividends, splits)."""

    RAW = "raw"            # Unadjusted prices
    FORWARD = "forward"    # Forward-adjusted (dividends reflected in historical prices)
    BACKWARD = "backward" # Backward-adjusted (dividends reflected in current price)

    ALL_MODES: Final[frozenset[str]] = frozenset({RAW, FORWARD, BACKWARD})


# =============================================================================
# Schema Validation
# =============================================================================

def validate_dataframe_schema(
    df: pl.DataFrame,
    require_all: bool = True,
) -> list[str]:
    """
    Validate that a DataFrame has the required schema.

    Args:
        df: Polars DataFrame to validate
        require_all: If True, all required columns must be present.
                    If False, only check that present columns have correct types.

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "instrument_id": ["SH600000"],
        ...     "date": [date(2024, 1, 1)],
        ...     "ticker": ["浦发银行"],
        ...     "open": [10.0],
        ...     "high": [10.5],
        ...     "low": [9.8],
        ...     "close": [10.2],
        ...     "volume": [1000000],
        ...     "adj_factor": [1.0],
        ... })
        >>> errors = validate_dataframe_schema(df)
        >>> len(errors) == 0
        True
    """
    errors: list[str] = []
    df_columns = set(df.columns)

    # Check for missing required columns
    if require_all:
        missing = REQUIRED_COLUMNS - df_columns
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

    # Check for extra unknown columns (warning level, not error)
    # We allow extra columns, so this is informational only

    # Check column types for columns that are present
    for col_name, expected_type in COLUMN_TYPES.items():
        if col_name not in df_columns:
            continue  # Column not present, handled above if require_all

        actual_type = df.schema[col_name]
        if actual_type != expected_type and not _is_compatible_type(
            actual_type, expected_type
        ):
            errors.append(
                f"Column '{col_name}' has wrong type: expected {expected_type}, "
                f"got {actual_type}"
            )

    return errors


def _is_compatible_type(actual: pl.DataType, expected: pl.DataType) -> bool:
    """
    Check if actual type is compatible with expected type.

    Some type differences are acceptable:
    - Int32/Int64 are interchangeable for volume
    - Float32/Float64 are interchangeable for prices
    - String/Categorical are interchangeable for identifiers
    """
    # Exact match
    if actual == expected:
        return True

    # Int types are interchangeable
    int_types = {pl.Int32, pl.Int64, pl.UInt32, pl.UInt64}
    if actual in int_types and expected in int_types:
        return True

    # Float types are interchangeable
    float_types = {pl.Float32, pl.Float64}
    if actual in float_types and expected in float_types:
        return True

    # String and Categorical are interchangeable for identifiers
    string_types = {pl.String, pl.Categorical}
    return actual in string_types and expected in string_types


def get_schema(columns: frozenset[str] | None = None) -> dict[str, pl.DataType]:
    """
    Get the Polars schema for market data.

    Args:
        columns: Optional subset of columns. If None, returns all required columns.

    Returns:
        Dictionary mapping column names to Polars dtypes
    """
    if columns is None:
        columns = REQUIRED_COLUMNS

    return {col: COLUMN_TYPES[col] for col in columns if col in COLUMN_TYPES}


def get_primary_key_schema() -> dict[str, pl.DataType]:
    """Get the Polars schema for primary key columns only."""
    return {col: COLUMN_TYPES[col] for col in PRIMARY_KEY}
