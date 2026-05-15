"""
Sector-based factors module for TrendSpec.

Exports:
- SectorMomentumFactor: Sector overall N-day returns
- SectorRelativeStrengthFactor: Stock returns - Sector returns

These factors depend on data/sectors.py for PIT sector lookup.
"""

from trendspec.factors.sector.sector_momentum import SectorMomentumFactor
from trendspec.factors.sector.sector_relative_strength import SectorRelativeStrengthFactor

__all__ = [
    "SectorMomentumFactor",
    "SectorRelativeStrengthFactor",
]