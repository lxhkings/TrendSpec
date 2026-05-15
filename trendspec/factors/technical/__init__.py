"""
Technical indicator factors module for TrendSpec.

Exports:
- MABiasFactor: Price / MA - 1 (moving average bias)
"""

from trendspec.factors.technical.ma_bias import MABiasFactor

__all__ = [
    "MABiasFactor",
]