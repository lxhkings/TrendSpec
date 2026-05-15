"""
TrendSpec data module.

Provides market abstraction, schema definitions, and data access utilities.
"""

from trendspec.data.markets import Market
from trendspec.data.schema import (
    COLUMN_TYPES,
    OHLC_COLUMNS,
    PRIMARY_KEY,
    REQUIRED_COLUMNS,
    AdjustmentMode,
    validate_dataframe_schema,
)

__all__ = [
    "Market",
    "REQUIRED_COLUMNS",
    "COLUMN_TYPES",
    "PRIMARY_KEY",
    "OHLC_COLUMNS",
    "AdjustmentMode",
    "validate_dataframe_schema",
]
