"""
MariaDB/MySQL client for TrendSpec ingest pipeline.

Provides SQLAlchemy 2.x engine for connecting to MariaDB on Synology NAS.
This client is ONLY used during ingest, not during backtest/screening.

Security:
- Credentials loaded from Settings (no hardcoded values)
- Read-only user required (Settings validates this)
"""

from typing import Final

from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import QueuePool

from trendspec.config.settings import Settings

# Connection pool settings for batch ingest
POOL_SIZE: Final[int] = 5
MAX_OVERFLOW: Final[int] = 10
POOL_TIMEOUT: Final[int] = 30
POOL_RECYCLE: Final[int] = 3600  # Recycle connections after 1 hour


def get_engine(settings: Settings) -> Engine:
    """
    Create SQLAlchemy engine from settings.

    Args:
        settings: TrendSpec settings containing database configuration

    Returns:
        SQLAlchemy Engine for MariaDB connection

    Note:
        Engine is created with connection pooling optimized for batch ingest.
        Pool size is limited to avoid overwhelming the database.
    """
    return create_engine(
        settings.db.connection_url,
        poolclass=QueuePool,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
        echo=False,  # Set to True for SQL debugging
    )


def get_readonly_engine(settings: Settings) -> Engine:
    """
    Create SQLAlchemy engine specifically for read-only operations.

    This is a convenience wrapper that emphasizes the read-only nature
    of the connection. The user validation is already done in Settings.

    Args:
        settings: TrendSpec settings containing database configuration

    Returns:
        SQLAlchemy Engine for MariaDB connection (read-only)
    """
    return get_engine(settings)


def create_engine_from_settings(db_settings) -> Engine:
    """Create SQLAlchemy engine from DatabaseSettings."""
    return create_engine(db_settings.connection_url)
