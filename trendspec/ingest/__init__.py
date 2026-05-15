"""
TrendSpec ingest module.

ETL pipeline for downloading stock data from MariaDB on Synology NAS
to local Parquet cache.

Key components:
- mariadb_client: SQLAlchemy engine for MariaDB connection
- schema_map: SQL column to Polars column mapping per market
- writer: Parquet writer with partitioning
- manifest: Sync state tracking
- ingestors: Market-specific data ingestors (CN_A, US)

Design principles:
- instrument_id is immutable (primary key)
- Include delisted stocks (critical for PIT universe)
- Incremental sync (track last date, pull only new data)
- Three datasets per market: daily, components, sectors
"""

from trendspec.ingest.manifest import Manifest, read_manifest, write_manifest
from trendspec.ingest.mariadb_client import get_engine
from trendspec.ingest.schema_map import (
    CN_COMPONENTS_MAP,
    CN_DAILY_MAP,
    CN_SECTORS_MAP,
    US_COMPONENTS_MAP,
    US_DAILY_MAP,
    US_SECTORS_MAP,
    get_column_map,
)
from trendspec.ingest.writer import write_parquet

__all__ = [
    "get_engine",
    "Manifest",
    "read_manifest",
    "write_manifest",
    "CN_DAILY_MAP",
    "CN_COMPONENTS_MAP",
    "CN_SECTORS_MAP",
    "US_DAILY_MAP",
    "US_COMPONENTS_MAP",
    "US_SECTORS_MAP",
    "get_column_map",
    "write_parquet",
]
