"""
Factor registry for TrendSpec.

Provides a central registry for factors:
- @register("factor_name") decorator for registration
- get_factor(name) for lookup
- list_factors() for listing all registered factors

Factors are registered at import time and accessed via the registry.
"""

import inspect
from collections.abc import Callable
from typing import Any

from trendspec.data.markets import Market
from trendspec.factors.base import Factor

# =============================================================================
# Factor Registry
# =============================================================================

_FACTOR_REGISTRY: dict[str, type[Factor]] = {}
_FACTOR_INSTANCES: dict[str, Factor] = {}


def register(name: str) -> Callable:
    """
    Decorator to register a factor class.

    Args:
        name: Factor name for registry lookup

    Returns:
        Decorator function

    Example:
        >>> @register("momentum_10")
        ... class Momentum10Factor(MomentumFactor):
        ...     name = "momentum_10"
        ...
        ...     def compute(self, df):
        ...         return (pl.col("close") - pl.col("close").shift(10)).over("instrument_id")
    """
    def decorator(cls: type[Factor]) -> type[Factor]:
        _FACTOR_REGISTRY[name] = cls
        # Set the name attribute on the class
        cls.name = name
        return cls
    return decorator


def get_factor(name: str, params: dict[str, Any] | None = None) -> Factor | None:
    """
    Get a factor instance by name.

    Creates a new instance with the given params, or returns a cached instance
    if params match.

    Args:
        name: Factor name
        params: Factor parameters (optional)

    Returns:
        Factor instance or None if not found

    Example:
        >>> factor = get_factor("momentum_10", {"period": 10})
        >>> result = factor.compute_full(df)
    """
    cls = _FACTOR_REGISTRY.get(name)
    if cls is None:
        return None

    # Create instance with params
    if params:
        return cls(**params)
    return cls()


def get_factor_with_market(
    name: str, params: dict[str, Any], market: str
) -> Factor | None:
    """
    Like get_factor, but auto-resolves a `market` param for factors that need one.

    If the target factor's constructor accepts `market` and params either omits
    it or supplies it as a plain string, inject Market(market.upper()) — the
    caller's market string uppercased. An explicit Market enum in params is
    left untouched.

    Args:
        name: Factor name
        params: Factor parameters (as would be passed to get_factor)
        market: Market string, e.g. "us" (from FactorSpec.market)

    Returns:
        Factor instance or None if not found
    """
    cls = _FACTOR_REGISTRY.get(name)
    if cls is None:
        return None

    resolved = dict(params or {})
    if "market" in inspect.signature(cls.__init__).parameters:
        given = resolved.get("market")
        if not isinstance(given, Market):
            resolved["market"] = Market(market.upper())

    return get_factor(name, resolved)


def get_factor_class(name: str) -> type[Factor] | None:
    """
    Get a factor class by name (without instantiation).

    Args:
        name: Factor name

    Returns:
        Factor class or None if not found
    """
    return _FACTOR_REGISTRY.get(name)


def list_factors() -> list[str]:
    """Get list of registered factor names."""
    return sorted(_FACTOR_REGISTRY.keys())


def factor_info(name: str) -> dict[str, Any] | None:
    """
    Get information about a registered factor.

    Args:
        name: Factor name

    Returns:
        Dict with factor info or None if not found
    """
    cls = _FACTOR_REGISTRY.get(name)
    if cls is None:
        return None

    return {
        "name": name,
        "class_name": cls.__name__,
        "description": getattr(cls, "description", ""),
        "category": getattr(cls, "category", "general"),
        "params": getattr(getattr(cls, "__init__", None), "__code__", None) and
                  cls.__init__.__code__.co_varnames[1:] or (),  # Skip 'self'
    }


def clear_registry() -> None:
    """Clear the factor registry. Useful for testing."""
    _FACTOR_REGISTRY.clear()
    _FACTOR_INSTANCES.clear()


# =============================================================================
# Built-in Factors (Pre-registered)
# =============================================================================

import polars as pl


@register("momentum")
class Momentum(Factor):
    """Price momentum (return over N days)."""

    description = "Price momentum - percentage change over period"
    category = "momentum"

    def __init__(self, period: int = 10) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        period = self.params.get("period", 10)
        return (
            (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1) * 100
        )


@register("returns")
class Returns(Factor):
    """Daily returns."""

    description = "Daily percentage returns"
    category = "momentum"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        return (
            (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1) * 100
        )


@register("volatility")
class Volatility(Factor):
    """Rolling volatility (annualized std of returns)."""

    description = "Rolling volatility - annualized std of returns"
    category = "volatility"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        period = self.params.get("period", 20)

        # Calculate returns first
        returns_expr = (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1)

        # Rolling std, annualized (252 trading days)
        return (
            returns_expr
            .rolling_std(window_size=period)
            .over("instrument_id")
            * (252 ** 0.5)
        )


@register("volume_ratio")
class VolumeRatio(Factor):
    """Volume relative to average volume."""

    description = "Volume / Average volume over period"
    category = "volume"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        period = self.params.get("period", 20)
        avg_volume = pl.col("volume").rolling_mean(window_size=period).over("instrument_id")
        return pl.col("volume") / avg_volume


@register("price_range")
class PriceRange(Factor):
    """Price range (high - low) relative to close."""

    description = "Price range - (high - low) / close"
    category = "volatility"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        return (pl.col("high") - pl.col("low")) / pl.col("close")


@register("gap")
class Gap(Factor):
    """Opening gap (open vs previous close)."""

    description = "Opening gap - (open - prev_close) / prev_close"
    category = "technical"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        prev_close = pl.col("close").shift(1).over("instrument_id")
        return (pl.col("open") - prev_close) / prev_close


@register("intraday_return")
class IntradayReturn(Factor):
    """Intraday return (close vs open)."""

    description = "Intraday return - (close - open) / open"
    category = "momentum"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        return (pl.col("close") - pl.col("open")) / pl.col("open")


@register("overnight_return")
class OvernightReturn(Factor):
    """Overnight return (open vs previous close)."""

    description = "Overnight return - same as gap but as percentage"
    category = "momentum"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        prev_close = pl.col("close").shift(1).over("instrument_id")
        return (pl.col("open") - prev_close) / prev_close * 100


@register("turnover")
class Turnover(Factor):
    """Turnover rate (requires turnover column in data)."""

    description = "Turnover rate - volume / shares outstanding"
    category = "volume"

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        # Assumes turnover column exists in data
        # If not, this will need to be computed differently
        if "turnover" in df.columns:
            return pl.col("turnover")
        # Placeholder - would need shares outstanding data
        return pl.lit(0.0)


@register("relative_strength")
class RelativeStrength(Factor):
    """Relative strength vs market/benchmark."""

    description = "Relative strength - return vs benchmark"
    category = "sector"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        # This would need benchmark data
        # For now, compute cumulative return over period
        period = self.params.get("period", 20)
        return (
            (pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1) * 100
        )


@register("max_drawdown")
class MaxDrawdown(Factor):
    """Maximum drawdown over period."""

    description = "Maximum drawdown over period"
    category = "risk"

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        period = self.params.get("period", 20)
        # Rolling max drawdown
        rolling_max = pl.col("close").rolling_max(window_size=period).over("instrument_id")
        return (pl.col("close") - rolling_max) / rolling_max


@register("sharpe_proxy")
class SharpeProxy(Factor):
    """Sharpe ratio proxy (return/volatility)."""

    description = "Sharpe proxy - momentum / volatility"
    category = "risk"

    def __init__(self, return_period: int = 10, vol_period: int = 20) -> None:
        self.params = {"return_period": return_period, "vol_period": vol_period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        ret_period = self.params.get("return_period", 10)
        vol_period = self.params.get("vol_period", 20)

        # Return over period
        momentum = (
            pl.col("close") / pl.col("close").shift(ret_period).over("instrument_id") - 1
        )
        # Volatility
        returns = pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1
        vol = returns.rolling_std(window_size=vol_period).over("instrument_id")

        return momentum / vol
