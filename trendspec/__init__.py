"""
TrendSpec - Quantitative backtesting and stock screening system.

A local quantitative backtesting system that:
- Reads stock data from MariaDB on a Synology NAS
- Caches data locally as Parquet files
- Supports dual-mode: historical backtesting AND daily stock screening
- Targets China A-shares and US stocks (SP500 + Russell 1000)
- Uses PIT (point-in-time) universe to avoid survivorship bias
"""

__version__ = "0.1.0"
__author__ = "TrendSpec Team"

from trendspec.config import Settings

__all__ = ["Settings", "__version__"]
