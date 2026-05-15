"""
TrendSpec CLI module.

Command-line interface for running backtests and screening.

Commands:
- trendspec ingest: Import market data
- trendspec backtest: Run backtest with strategy
- trendspec screen: Run screening for latest date
"""

from trendspec.cli.main import app

__all__ = ["app"]