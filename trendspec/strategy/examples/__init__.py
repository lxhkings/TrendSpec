"""
TrendSpec example strategies module.

Provides ready-to-use example strategies demonstrating key features:
- MACrossStrategy: Dual MA crossover (indicator usage)
- RSIReversalStrategy: RSI oversold/overbought reversal
- SectorMomentumStrategy: Sector-relative momentum ranking

These strategies demonstrate:
- Indicator computation via ctx.precompute_indicator()
- Signal generation via ctx.signal()
- PIT universe/sector access via ctx.pit_universe(), ctx.sector()
- Cross-sectional operations via factors
- Risk rule configuration

Example usage:
    >>> from trendspec.strategy.examples import MACrossStrategy
    >>> strategy = MACrossStrategy(params={"short_period": 10, "long_period": 30})
    >>> # Run via backtest or screening engine
"""

from trendspec.strategy.examples.clenow_momentum import ClenowMomentumStrategy
from trendspec.strategy.examples.ma_cross import MACrossStrategy
from trendspec.strategy.examples.minervini_trend_template import MinerviniTrendTemplate
from trendspec.strategy.examples.rsi_reversal import RSIReversalStrategy
from trendspec.strategy.examples.sector_momentum import SectorMomentumStrategy

__all__ = [
    "ClenowMomentumStrategy",
    "MACrossStrategy",
    "MinerviniTrendTemplate",
    "RSIReversalStrategy",
    "SectorMomentumStrategy",
]