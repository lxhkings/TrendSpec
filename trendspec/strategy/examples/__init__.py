"""
TrendSpec example strategies module.
"""

from trendspec.strategy.examples.clenow_momentum import ClenowMomentumStrategy
from trendspec.strategy.examples.ema_cluster_pullback import EMAClusterPullback

__all__ = [
    "ClenowMomentumStrategy",
    "EMAClusterPullback",
]
