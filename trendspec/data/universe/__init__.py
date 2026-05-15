"""
PIT Universe module for TrendSpec.

Provides point-in-time universe tracking for survivorship bias prevention.
Key design rule: EVERY API MUST ACCEPT DATE PARAMETER.

Supported markets:
- CN_A: China A-shares (Shanghai/Shenzhen) - survivorship-free
- US: US stocks (SP500 + Russell 1000 historical components)
- HK: Placeholder (not implemented)
"""

from trendspec.data.universe.base import Universe
from trendspec.data.universe.cn_a import CNAUniverse
from trendspec.data.universe.hk import HKUniverse
from trendspec.data.universe.us import USUniverse

__all__ = [
    "Universe",
    "CNAUniverse",
    "USUniverse",
    "HKUniverse",
    "get_universe",
]


def get_universe(market: str, root: str | None = None) -> Universe:
    """
    Get universe instance for a market.

    Factory function to create the appropriate universe class.

    Args:
        market: Market code ("CN_A", "US", "HK")
        root: Root directory for data_lake

    Returns:
        Universe instance for the market

    Raises:
        ValueError: If market is not recognized
    """
    if market == "CN_A":
        return CNAUniverse(root)
    elif market == "US":
        return USUniverse(root)
    elif market == "HK":
        return HKUniverse(root)
    else:
        raise ValueError(f"Unknown market: {market}")