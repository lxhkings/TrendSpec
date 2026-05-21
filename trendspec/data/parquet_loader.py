"""
Lazy Parquet data loading for TrendSpec.

Provides efficient data loading from data_lake Parquet files.
Key features:
- Lazy loading with scan_parquet
- Column pruning for memory efficiency
- Date filtering for time-range queries
- Price adjustment modes (forward, raw, backward)

Primary key is (instrument_id, date) - ticker can change.
"""

from datetime import date
from pathlib import Path
from typing import Final, Literal

import polars as pl

from trendspec.config.settings import get_settings
from trendspec.data.markets import Market

# =============================================================================
# Adjustment Modes
# =============================================================================

AdjustmentMode = Literal["raw", "forward", "backward"]

ADJUSTMENT_MODES: Final[frozenset[AdjustmentMode]] = frozenset({
    "raw",
    "forward",
    "backward",
})


# =============================================================================
# Helper Functions
# =============================================================================


def _has_parquet_files(path: Path) -> bool:
    """Check if directory contains Parquet files."""
    if not path.exists():
        return False

    for item in path.iterdir():
        if item.is_file() and item.suffix == ".parquet":
            return True
        if item.is_dir():
            # Check subdirectories for Hive partitions
            for sub_item in item.iterdir():
                if sub_item.is_file() and sub_item.suffix == ".parquet":
                    return True
                # Check deeper for nested partitions
                if sub_item.is_dir():
                    for nested in sub_item.iterdir():
                        if nested.is_file() and nested.suffix == ".parquet":
                            return True
    return False


def _lazyframe_is_empty(lf: pl.LazyFrame) -> bool:
    """Check if LazyFrame is empty by checking its schema."""
    return len(lf.schema) == 0


# =============================================================================
# Lazy Parquet Loading
# =============================================================================


def scan_parquet(
    root: str | None = None,
    market: Market | None = None,
    dataset: str = "daily",
) -> pl.LazyFrame:
    """
    Lazily scan Parquet files from data_lake.

    Args:
        root: Root directory for data_lake.
        market: Market enum. If None, scans all markets.
        dataset: Dataset name (daily, components, sectors).

    Returns:
        Polars LazyFrame for deferred computation
    """
    if root is None:
        root = get_settings().data_lake.data_lake_root

    if market is not None:
        dataset_path = Path(root) / market.path / dataset
    else:
        dataset_path = Path(root) / "*" / dataset

    if not _has_parquet_files(dataset_path):
        return pl.LazyFrame()

    try:
        lf = pl.scan_parquet(
            str(dataset_path / "**" / "*.parquet"),
            hive_partitioning=True,
        )
        return lf
    except (pl.ComputeError, FileNotFoundError):
        return pl.LazyFrame()


def scan_parquet_glob(
    glob_pattern: str,
    root: str | None = None,
) -> pl.LazyFrame:
    """
    Lazily scan Parquet files using glob pattern.

    Args:
        glob_pattern: Glob pattern for Parquet files
        root: Root directory for data_lake

    Returns:
        Polars LazyFrame
    """
    if root is None:
        root = get_settings().data_lake.data_lake_root

    full_pattern = Path(root) / glob_pattern

    # Check if pattern would match any files
    # For glob patterns, we need to check the parent directory
    parent = full_pattern.parent
    while not parent.exists() and str(parent) != str(Path(root)):
        parent = parent.parent

    if not parent.exists():
        return pl.LazyFrame()

    # Check for parquet files in the pattern
    if not _has_parquet_files(full_pattern.parent):
        return pl.LazyFrame()

    try:
        lf = pl.scan_parquet(
            str(full_pattern),
            hive_partitioning=True,
        )
        return lf
    except (pl.ComputeError, FileNotFoundError):
        return pl.LazyFrame()


# =============================================================================
# OHLCV Data Access
# =============================================================================


def bars(
    market: Market,
    start_date: date | None = None,
    end_date: date | None = None,
    instrument_ids: list[str] | None = None,
    columns: list[str] | None = None,
    adjustment_mode: AdjustmentMode = "forward",
    root: str | None = None,
    frequency: Literal["daily", "weekly"] = "daily",
) -> pl.DataFrame:
    """
    Get OHLCV bars for a market with optional date range and adjustment.

    Args:
        market: Market enum (CN_A, US, HK)
        start_date: Optional start date filter (inclusive)
        end_date: Optional end date filter (inclusive)
        instrument_ids: Optional list of instrument_ids to filter
        columns: Optional list of columns to select
        adjustment_mode: Price adjustment mode
        root: Root directory for data_lake
        frequency: 'daily' (default) or 'weekly'

    Returns:
        Polars DataFrame with OHLCV data

    Raises:
        ValueError: If invalid adjustment mode
    """
    if adjustment_mode not in ADJUSTMENT_MODES:
        raise ValueError(
            f"Invalid adjustment mode: {adjustment_mode}. "
            f"Must be one of: {ADJUSTMENT_MODES}"
        )

    lf = scan_parquet(root, market, frequency)

    if _lazyframe_is_empty(lf):
        return pl.DataFrame()

    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)

    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)

    if instrument_ids is not None:
        lf = lf.filter(pl.col("instrument_id").is_in(instrument_ids))

    if columns is not None:
        required_cols = ["instrument_id", "date", "adj_factor"]
        all_columns = list(set(columns) | set(required_cols))
        lf = lf.select(all_columns)

    df = lf.collect()

    if df.is_empty():
        return df

    df = _apply_adjustment(df, adjustment_mode)
    return df


def bars_for_instrument(
    market: Market,
    instrument_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    adjustment_mode: AdjustmentMode = "forward",
    root: str | None = None,
) -> pl.DataFrame:
    """
    Get OHLCV bars for a single instrument.

    Args:
        market: Market enum
        instrument_id: Instrument ID
        start_date: Optional start date
        end_date: Optional end date
        adjustment_mode: Price adjustment mode
        root: Root directory for data_lake

    Returns:
        Polars DataFrame with bars for the instrument
    """
    if root is None:
        root = get_settings().data_lake.data_lake_root

    glob_pattern = f"{market.path}/daily/instrument_id={instrument_id}/*.parquet"
    lf = scan_parquet_glob(glob_pattern, root)

    if _lazyframe_is_empty(lf):
        return pl.DataFrame()

    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)

    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)

    df = lf.collect()

    if df.is_empty():
        return df

    df = _apply_adjustment(df, adjustment_mode)
    return df.sort("date")


# =============================================================================
# Price Adjustment Implementation
# =============================================================================


def _apply_adjustment(df: pl.DataFrame, mode: AdjustmentMode) -> pl.DataFrame:
    """
    Apply price adjustment to OHLCV data.

    Args:
        df: DataFrame with OHLCV data and adj_factor column
        mode: Adjustment mode

    Returns:
        DataFrame with adjusted OHLC prices
    """
    if mode == "raw" or "adj_factor" not in df.columns:
        return df

    ohlc_cols = ["open", "high", "low", "close"]
    has_ohlc = all(col in df.columns for col in ohlc_cols)

    if not has_ohlc:
        return df

    df = df.sort("date", descending=False)

    if mode == "forward":
        # Forward adjustment (前复权)
        # adjusted = raw * (adj_factor / adj_factor_latest)
        latest_factors = df.group_by("instrument_id").agg(
            pl.col("adj_factor").last().alias("latest_adj_factor")
        )
        df = df.join(latest_factors, on="instrument_id")
        adjustment_ratio = pl.col("adj_factor") / pl.col("latest_adj_factor")

        for col in ohlc_cols:
            df = df.with_columns((pl.col(col) * adjustment_ratio).alias(col))

        df = df.drop("latest_adj_factor")

    elif mode == "backward":
        # Backward adjustment (后复权)
        # adjusted = raw * (adj_factor / adj_factor_earliest)
        earliest_factors = df.group_by("instrument_id").agg(
            pl.col("adj_factor").first().alias("earliest_adj_factor")
        )
        df = df.join(earliest_factors, on="instrument_id")
        adjustment_ratio = pl.col("adj_factor") / pl.col("earliest_adj_factor")

        for col in ohlc_cols:
            df = df.with_columns((pl.col(col) * adjustment_ratio).alias(col))

        df = df.drop("earliest_adj_factor")

    return df


# =============================================================================
# Data Access Helpers
# =============================================================================


def get_instrument_ids(market: Market, root: str | None = None) -> list[str]:
    """
    Get all unique instrument_ids for a market.

    Args:
        market: Market enum
        root: Root directory for data_lake

    Returns:
        List of unique instrument_ids
    """
    if root is None:
        root = get_settings().data_lake.data_lake_root

    dataset_path = Path(root) / market.path / "daily"

    if not dataset_path.exists():
        return []

    instrument_ids: list[str] = []
    for partition_dir in dataset_path.iterdir():
        if partition_dir.is_dir() and partition_dir.name.startswith("instrument_id="):
            instrument_id = partition_dir.name.replace("instrument_id=", "")
            instrument_ids.append(instrument_id)

    return sorted(instrument_ids)


def get_date_range(
    market: Market,
    instrument_id: str | None = None,
    root: str | None = None,
) -> tuple[date | None, date | None]:
    """
    Get the date range for a market or specific instrument.

    Args:
        market: Market enum
        instrument_id: Optional instrument_id filter
        root: Root directory for data_lake

    Returns:
        Tuple of (min_date, max_date) or (None, None) if no data
    """
    if instrument_id is not None:
        lf = scan_parquet_glob(
            f"{market.path}/daily/instrument_id={instrument_id}/*.parquet",
            root,
        )
    else:
        lf = scan_parquet(root, market, "daily")

    if _lazyframe_is_empty(lf):
        return (None, None)

    date_stats = lf.select(
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ).collect()

    if date_stats.is_empty():
        return (None, None)

    min_date = date_stats["min_date"].item()
    max_date = date_stats["max_date"].item()

    return (min_date, max_date)


def read_components(
    market: Market,
    root: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """
    Read component events (IPO, delist, halt) from Parquet.

    Args:
        market: Market enum
        root: Root directory for data_lake
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with component events
    """
    lf = scan_parquet(root, market, "components")

    if _lazyframe_is_empty(lf):
        return pl.DataFrame()

    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)

    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)

    return lf.collect().sort("date")


def read_sectors(
    market: Market,
    root: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """
    Read sector assignments from Parquet.

    Args:
        market: Market enum
        root: Root directory for data_lake
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with sector assignments
    """
    lf = scan_parquet(root, market, "sectors")

    if _lazyframe_is_empty(lf):
        return pl.DataFrame()

    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)

    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)

    return lf.collect().sort("date")


def read_indices(
    market: Market,
    root: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    instrument_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Load index close price series from data_lake/{market}/indices/."""
    lf = scan_parquet(root, market, "indices")

    if _lazyframe_is_empty(lf):
        return pl.DataFrame()

    if start_date is not None:
        lf = lf.filter(pl.col("date") >= start_date)

    if end_date is not None:
        lf = lf.filter(pl.col("date") <= end_date)

    if instrument_ids is not None:
        lf = lf.filter(pl.col("instrument_id").is_in(instrument_ids))

    return lf.collect().sort(["instrument_id", "date"])