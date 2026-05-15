"""
TrendSpec analyzer module.

Performance analysis and reporting for backtest results.

Key components:
- PerformanceMetrics: Calculate trading performance metrics
- EquityCurve: Drawdown and returns analysis
- TradeLog: Trade statistics and summary
- BacktestReport: Chinese output formatting with rich.Table
"""

from trendspec.analyzer.metrics import PerformanceMetrics, calculate_metrics
from trendspec.analyzer.equity_curve import EquityCurve
from trendspec.analyzer.trade_log import TradeLogAnalyzer
from trendspec.analyzer.report import BacktestReport

__all__ = [
    "PerformanceMetrics",
    "calculate_metrics",
    "EquityCurve",
    "TradeLogAnalyzer",
    "BacktestReport",
]