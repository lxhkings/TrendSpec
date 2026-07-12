"""
Technical indicator factors module for TrendSpec.

Exports:
- MABiasFactor: Price / MA - 1 (moving average bias)
- EMAAlignmentFactor: bullish EMA alignment strength (fast/mid/slow)
"""

from trendspec.factors.technical.ma_bias import MABiasFactor
from trendspec.factors.technical.ema_alignment import EMAAlignmentFactor

__all__ = [
    "MABiasFactor",
    "EMAAlignmentFactor",
]