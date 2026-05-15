"""
Cross-sectional factors module for TrendSpec.

Exports:
- RankWithinSectorFactor: Rank stocks within sector by factor value
- DemeanBySectorFactor: Subtract sector mean/median from each stock

These factors depend on data/sectors.py for PIT sector lookup.
"""

from trendspec.factors.cross_sectional.rank_within_sector import RankWithinSectorFactor
from trendspec.factors.cross_sectional.demean_by_sector import DemeanBySectorFactor

__all__ = [
    "RankWithinSectorFactor",
    "DemeanBySectorFactor",
]