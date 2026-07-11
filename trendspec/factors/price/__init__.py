"""
Price-based factors module for TrendSpec.

Exports:
- MomentumFactor: N-day returns
- MomentumRankFactor: Cross-sectional momentum rank
- VolatilityFactor: N-day return std
- VolatilityRankFactor: Cross-sectional volatility rank
- ClenowMomentumFactor: Annualized regression-slope x R^2 momentum
"""

from trendspec.factors.price.clenow_momentum import ClenowMomentumFactor
from trendspec.factors.price.momentum import MomentumFactor, MomentumRankFactor
from trendspec.factors.price.volatility import VolatilityFactor, VolatilityRankFactor

__all__ = [
    "MomentumFactor",
    "MomentumRankFactor",
    "VolatilityFactor",
    "VolatilityRankFactor",
    "ClenowMomentumFactor",
]