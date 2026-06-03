"""
Parquet writer for TrendSpec data lake.

Writes Polars DataFrame to partitioned Parquet files.
Partition scheme: data_lake/<market>/<dataset>/instrument_id=<id>/<year>.parquet

Key features:
- zstd compression (good balance of speed and size)
- Same (instrument_id, year) overwrites existing file
- Partitioned by instrument_id and year for efficient querying
"""

import os
from datetime import UTC
from typing import Final

import polars as pl

from trendspec.data.markets import Market

# Compression settings
COMPRESSION: Final[str] = "zstd"
COMPRESSION_LEVEL: Final[int] = 3  # Moderate compression for balance


def write_parquet(
    df: pl.DataFrame,
    market: Market,
    dataset: str,
    root: str,
    overwrite: bool = True,
    show_progress: bool = False,
) -> None:
    """
    Write DataFrame to partitioned Parquet files.

    Partition scheme: <root>/<market>/<dataset>/instrument_id=<id>/<year>.parquet

    Args:
        df: Polars DataFrame to write
        market: Market enum (CN_A, US, HK)
        dataset: Dataset name (daily, components, sectors)
        root: Root directory for data_lake
        overwrite: If True, overwrite existing files for same (instrument_id, year)
        show_progress: If True, render a Rich progress bar over partition writes

    Raises:
        ValueError: If DataFrame doesn't have required columns (instrument_id, date, year)
    """
    # Validate required columns
    if "instrument_id" not in df.columns:
        raise ValueError("DataFrame must have 'instrument_id' column")
    if "date" not in df.columns:
        raise ValueError("DataFrame must have 'date' column")

    # Extract year from date for partitioning
    if "year" not in df.columns:
        df = df.with_columns(
            pl.col("date").dt.year().alias("year")
        )

    # Construct base path
    base_path = os.path.join(root, market.path, dataset)

    # Create base directory if needed
    os.makedirs(base_path, exist_ok=True)

    # Get unique instrument_ids and years
    partitions = df.select(["instrument_id", "year"]).unique()

    partition_rows = partitions.iter_rows(named=True)
    if show_progress:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            TextColumn("[cyan]写入 {task.fields[dataset]}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task("write", total=partitions.height, dataset=dataset)
            partition_rows = progress.track(partition_rows, task_id=task)
            _write_partitions(df, partition_rows, base_path, overwrite)
        return

    _write_partitions(df, partition_rows, base_path, overwrite)


def _write_partitions(df, partition_rows, base_path, overwrite) -> None:
    for row in partition_rows:
        instrument_id = row["instrument_id"]
        year = row["year"]

        # Filter data for this partition
        partition_df = df.filter(
            (pl.col("instrument_id") == instrument_id) & (pl.col("year") == year)
        )

        # Drop year column before writing (it's redundant with partition)
        partition_df = partition_df.drop("year")

        # Construct partition path
        partition_dir = os.path.join(base_path, f"instrument_id={instrument_id}")
        file_path = os.path.join(partition_dir, f"{year}.parquet")

        # Create partition directory
        os.makedirs(partition_dir, exist_ok=True)

        # Write or append based on overwrite flag
        if overwrite or not os.path.exists(file_path):
            partition_df.write_parquet(
                file_path,
                compression=COMPRESSION,
                compression_level=COMPRESSION_LEVEL,
            )
        else:
            # Read existing, combine with new, write merged
            existing_df = pl.read_parquet(file_path)
            # Combine and deduplicate (keep latest by date)
            combined_df = pl.concat([existing_df, partition_df]).unique(
                subset=["instrument_id", "date"],
                keep="last",
                maintain_order=False,
            )
            combined_df.write_parquet(
                file_path,
                compression=COMPRESSION,
                compression_level=COMPRESSION_LEVEL,
            )


def write_dataset_manifest(
    market: Market,
    dataset: str,
    root: str,
    row_count: int,
    date_range: tuple[str, str],
    instrument_count: int,
) -> dict:
    """
    Create manifest entry for a dataset write.

    Args:
        market: Market enum
        dataset: Dataset name
        root: Root directory
        row_count: Number of rows written
        date_range: (start_date, end_date) tuple
        instrument_count: Number of unique instrument_ids

    Returns:
        Manifest entry dictionary
    """
    from datetime import datetime

    return {
        "market": market.value,
        "dataset": dataset,
        "path": f"{root}/{market.path}/{dataset}",
        "last_sync_time": datetime.now(UTC).isoformat(),
        "date_range": {
            "start": date_range[0],
            "end": date_range[1],
        },
        "row_count": row_count,
        "instrument_count": instrument_count,
    }


def get_partition_path(
    root: str,
    market: Market,
    dataset: str,
    instrument_id: str,
    year: int,
) -> str:
    """
    Get the path for a specific partition Parquet file.

    Args:
        root: Root directory for data_lake
        market: Market enum
        dataset: Dataset name
        instrument_id: Instrument ID
        year: Year

    Returns:
        Full path to the Parquet file
    """
    return os.path.join(
        root,
        market.path,
        dataset,
        f"instrument_id={instrument_id}",
        f"{year}.parquet",
    )


def read_partition(
    root: str,
    market: Market,
    dataset: str,
    instrument_id: str,
    year: int | None = None,
) -> pl.DataFrame:
    """
    Read Parquet partition for an instrument.

    Args:
        root: Root directory for data_lake
        market: Market enum
        dataset: Dataset name
        instrument_id: Instrument ID
        year: Optional year filter. If None, reads all years.

    Returns:
        Polars DataFrame with partition data
    """
    partition_dir = os.path.join(
        root,
        market.path,
        dataset,
        f"instrument_id={instrument_id}",
    )

    if year is not None:
        file_path = os.path.join(partition_dir, f"{year}.parquet")
        if os.path.exists(file_path):
            return pl.read_parquet(file_path)
        return pl.DataFrame()

    # Read all years for this instrument
    if not os.path.exists(partition_dir):
        return pl.DataFrame()

    parquet_files = [
        os.path.join(partition_dir, f)
        for f in os.listdir(partition_dir)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        return pl.DataFrame()

    return pl.read_parquet(parquet_files)
