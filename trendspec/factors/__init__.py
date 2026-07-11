"""
TrendSpec factors module.

Provides factor framework for computing derived values from market data.
Key components:
- Factor: Abstract base class for factors
- FactorResult: Result of factor computation
- Registry: Central registry for factor lookup

Design principles:
- Factors are named and registered
- Computed via Polars expressions (vectorized)
- Two modes: cross-sectional and time-series

Example:
    >>> from trendspec.factors import Factor, register
    ... import polars as pl
    ...
    >>> @register("my_momentum")
    ... class MyMomentum(Factor):
    ...     name = "my_momentum"
    ...     description = "My custom momentum factor"
    ...     category = "momentum"
    ...
    ...     def compute(self, df):
    ...         return (pl.col("close") - pl.col("close").shift(10)).over("instrument_id")
"""

from trendspec.factors.base import (
    FACTOR_CATEGORIES,
    Factor,
    FactorResult,
    MomentumFactor,
    VolatilityFactor,
    VolumeFactor,
)
from trendspec.factors.registry import (
    clear_registry,
    factor_info,
    get_factor,
    get_factor_class,
    list_factors,
    register,
)

# Price factors
from trendspec.factors.price import (
    ClenowMomentumFactor,
    MomentumFactor as PriceMomentumFactor,
    MomentumRankFactor,
    VolatilityFactor as PriceVolatilityFactor,
    VolatilityRankFactor,
)

# Volume factors
from trendspec.factors.volume import (
    TurnoverFactor,
    VolumeRatioFactor,
)

# Technical factors
from trendspec.factors.technical import (
    MABiasFactor,
)

# Sector factors
from trendspec.factors.sector import (
    SectorMomentumFactor,
    SectorRelativeStrengthFactor,
)

# Cross-sectional factors
from trendspec.factors.cross_sectional import (
    RankWithinSectorFactor,
    DemeanBySectorFactor,
)

# Fundamental factors
from trendspec.factors import fundamental  # noqa: F401

__all__ = [
    # Base classes
    "Factor",
    "FactorResult",
    "FACTOR_CATEGORIES",
    "MomentumFactor",
    "VolatilityFactor",
    "VolumeFactor",
    # Registry
    "register",
    "get_factor",
    "get_factor_class",
    "list_factors",
    "factor_info",
    "clear_registry",
    # Price factors
    "PriceMomentumFactor",
    "MomentumRankFactor",
    "PriceVolatilityFactor",
    "VolatilityRankFactor",
    "ClenowMomentumFactor",
    # Volume factors
    "TurnoverFactor",
    "VolumeRatioFactor",
    # Technical factors
    "MABiasFactor",
    # Sector factors
    "SectorMomentumFactor",
    "SectorRelativeStrengthFactor",
    # Cross-sectional factors
    "RankWithinSectorFactor",
    "DemeanBySectorFactor",
]