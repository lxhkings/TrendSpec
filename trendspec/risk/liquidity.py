"""
Liquidity risk rule for TrendSpec.

Rule:
- MinLiquidityRule: Skip stocks with trading volume < threshold

Ensures positions are in sufficiently liquid stocks for execution.
"""

from typing import Any, ClassVar

from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


class MinLiquidityRule(RiskRule):
    """
    Minimum liquidity rule.

    Filters stocks with insufficient trading volume.
    Ensures positions can be executed without significant slippage.

    This rule checks ADV (Average Daily Volume) or current volume.

    Attributes:
        name: Rule name
        priority: Rule priority (50)

    Parameters:
        min_volume: Minimum daily volume (default: 100000 shares)
        use_adv: Use average daily volume instead of current volume (default: True)
        adv_period: Period for ADV calculation (default: 20)
        min_turnover: Minimum daily turnover value (default: None, in currency)

    Example:
        >>> rule = MinLiquidityRule(min_volume=500000, use_adv=True)
        >>> # Rejects stocks with ADV < 500,000 shares
    """

    name: ClassVar[str] = "min_liquidity"
    priority: ClassVar[int] = 50

    def __init__(
        self,
        min_volume: int = 100000,
        use_adv: bool = True,
        adv_period: int = 20,
        min_turnover: float | None = None,
    ) -> None:
        """
        Initialize minimum liquidity rule.

        Args:
            min_volume: Minimum daily volume in shares
            use_adv: Use average daily volume instead of current volume
            adv_period: Period for ADV calculation (days)
            min_turnover: Minimum daily turnover value (e.g., 1000000 for $1M)
        """
        self.params = {
            "min_volume": min_volume,
            "use_adv": use_adv,
            "adv_period": adv_period,
            "min_turnover": min_turnover,
        }

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if stock meets minimum liquidity requirements.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            Allow or Reject result
        """
        min_volume = self.params.get("min_volume", 100000)
        use_adv = self.params.get("use_adv", True)
        adv_period = self.params.get("adv_period", 20)
        min_turnover = self.params.get("min_turnover")

        # Get volume from context
        try:
            current_volume = ctx.volume
        except RuntimeError:
            # No volume data available - allow by default
            # Could also reject here depending on strategy preference
            return Allow(self.name)

        # Check liquidity
        volume_to_check = current_volume

        if use_adv:
            # Try to get ADV from precomputed indicators (only if already computed)
            # Don't try to compute on demand as that can cause issues
            try:
                cache_key = f"ADV_{adv_period}"
                if cache_key in ctx._indicator_cache:
                    adv_df = ctx._indicator_cache[cache_key]
                    filtered = adv_df.filter(
                        (pl.col("instrument_id") == signal.instrument_id)
                        & (pl.col("date") == ctx.date)
                    )
                    if not filtered.is_empty():
                        adv_value = filtered["ADV"].item() if "ADV" in filtered.columns else None
                        if adv_value is not None:
                            volume_to_check = int(adv_value)
            except Exception:
                # Fall back to current volume if ADV not available
                volume_to_check = current_volume

        # Volume check
        if volume_to_check < min_volume:
            return Reject(
                self.name,
                f"Volume {volume_to_check} below minimum {min_volume}",
                {
                    "volume": volume_to_check,
                    "min_volume": min_volume,
                    "use_adv": use_adv,
                    "instrument_id": signal.instrument_id,
                },
            )

        # Turnover check (if specified)
        if min_turnover is not None:
            try:
                # Turnover = volume * price
                current_turnover = volume_to_check * signal.price
                if current_turnover < min_turnover:
                    return Reject(
                        self.name,
                        f"Turnover {current_turnover:.0f} below minimum {min_turnover:.0f}",
                        {
                            "turnover": current_turnover,
                            "min_turnover": min_turnover,
                            "volume": volume_to_check,
                            "price": signal.price,
                            "instrument_id": signal.instrument_id,
                        },
                    )
            except Exception:
                # If turnover calculation fails, skip this check
                pass

        return Allow(self.name)


# Register rule in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["min_liquidity"] = MinLiquidityRule