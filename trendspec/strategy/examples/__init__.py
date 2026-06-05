"""
TrendSpec example strategies module.
"""

from trendspec.strategy.examples.clenow_momentum import ClenowMomentumStrategy
from trendspec.strategy.examples.ema_cluster_pullback import EMAClusterPullback
from trendspec.strategy.examples.episodic_pivot import EpisodicPivot
from trendspec.strategy.examples.relative_strength_ema import RelativeStrengthEMACross
from trendspec.strategy.examples.rumi import RumiStrategy

__all__ = [
    "ClenowMomentumStrategy",
    "EMAClusterPullback",
    "EpisodicPivot",
    "RelativeStrengthEMACross",
    "RumiStrategy",
]
