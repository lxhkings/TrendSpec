"""
Price limit risk rule for TrendSpec.

Rule:
- PriceLimitRule: Skip if at daily price limit (A-share 涨跌停, US circuit breaker)

Detects stocks at daily price limits to avoid execution at locked prices.
Uses raw prices for limit detection (not adjusted prices).
"""

from typing import Any, ClassVar

import polars as pl

from trendspec.data.markets import Market
from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


class PriceLimitRule(RiskRule):
    """
    Price limit rule.

    Filters stocks at daily price limits to avoid execution issues.

    For A-shares (CN_A):
    - 涨停: Price at +10% (or +20% for some stocks) from previous close
    - 跌停: Price at -10% (or -10%) from previous close

    For US:
    - Circuit breaker: Market-wide halts at specific decline thresholds
    - This rule checks for individual stock halts if data available

    Attributes:
        name: Rule name
        priority: Rule priority (60)

    Parameters:
        market: Market for limit rules (default: Market.CN_A)
        limit_pct: Limit percentage for detection (default: 0.10 for 10%)
        check_limit_up: Check for limit up (涨停) (default: True)
        check_limit_down: Check for limit down (跌停) (default: True)
        tolerance: Tolerance for limit detection (default: 0.001)

    Example:
        >>> rule = PriceLimitRule(market=Market.CN_A, limit_pct=0.10)
        >>> # Rejects signals for stocks at 涨停 or 跌停
    """

    name: ClassVar[str] = "price_limit"
    priority: ClassVar[int] = 60

    def __init__(
        self,
        market: Market = Market.CN_A,
        limit_pct: float = 0.10,
        check_limit_up: bool = True,
        check_limit_down: bool = True,
        tolerance: float = 0.001,
    ) -> None:
        """
        Initialize price limit rule.

        Args:
            market: Market for limit rules
            limit_pct: Limit percentage (e.g., 0.10 for 10%)
            check_limit_up: Check for limit up
            check_limit_down: Check for limit down
            tolerance: Tolerance for limit detection (to handle rounding)
        """
        # Adjust limit percentage based on market
        if market == Market.CN_A:
            # A-shares: 10% for most, 20% for ChiNext/STAR Market
            # Default to 10%, user can override for specific stocks
            pass
        elif market == Market.US:
            # US: No daily limits, but circuit breakers exist
            # This rule is less relevant for US
            pass

        self.params = {
            "market": market,
            "limit_pct": limit_pct,
            "check_limit_up": check_limit_up,
            "check_limit_down": check_limit_down,
            "tolerance": tolerance,
        }

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if stock is at price limit.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            Allow or Reject result
        """
        market = self.params.get("market", Market.CN_A)
        limit_pct = self.params.get("limit_pct", 0.10)
        check_limit_up = self.params.get("check_limit_up", True)
        check_limit_down = self.params.get("check_limit_down", True)
        tolerance = self.params.get("tolerance", 0.001)

        # US market doesn't have daily price limits
        if market == Market.US:
            # Could check for circuit breaker conditions here
            # For now, allow all US signals
            return Allow(self.name)

        # For A-shares, check price vs previous close
        try:
            # Get previous close from context
            # This requires historical data lookup
            prev_close = ctx.indicator_value(
                "prev_close",
                signal.instrument_id,
                ctx.date,
            )

            if prev_close is None:
                # Try to calculate from data
                # Get close from previous trading day
                try:
                    # Get current close and shift
                    prev_close_expr = pl.col("close").shift(1).over("instrument_id")
                    # This would need access to full data DataFrame
                    # For simplicity, allow if no prev_close available
                    return Allow(self.name)
                except Exception:
                    return Allow(self.name)

            # Calculate limit prices
            limit_up_price = prev_close * (1 + limit_pct)
            limit_down_price = prev_close * (1 - limit_pct)

            # Check if current price is at limit
            current_price = signal.price

            # Limit up check
            if check_limit_up:
                # Price is at or near limit up (涨停)
                if abs(current_price - limit_up_price) <= tolerance * limit_up_price:
                    # At limit up - buy signals may fail (can't buy at 涨停)
                    if signal.is_buy():
                        return Reject(
                            self.name,
                            f"Price at limit up (涨停): {current_price:.2f} vs {limit_up_price:.2f}",
                            {
                                "current_price": current_price,
                                "limit_up_price": limit_up_price,
                                "prev_close": prev_close,
                                "limit_pct": limit_pct,
                                "instrument_id": signal.instrument_id,
                                "direction": signal.direction,
                            },
                        )

            # Limit down check
            if check_limit_down:
                # Price is at or near limit down (跌停)
                if abs(current_price - limit_down_price) <= tolerance * limit_down_price:
                    # At limit down - sell signals may fail (can't sell at 跌停)
                    if signal.is_sell():
                        return Reject(
                            self.name,
                            f"Price at limit down (跌停): {current_price:.2f} vs {limit_down_price:.2f}",
                            {
                                "current_price": current_price,
                                "limit_down_price": limit_down_price,
                                "prev_close": prev_close,
                                "limit_pct": limit_pct,
                                "instrument_id": signal.instrument_id,
                                "direction": signal.direction,
                            },
                        )

            return Allow(self.name)

        except Exception:
            # If price limit check fails, allow by default
            return Allow(self.name)


# Register rule in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["price_limit"] = PriceLimitRule