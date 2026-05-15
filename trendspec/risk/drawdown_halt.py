"""
Drawdown halt risk rule for TrendSpec.

Rule:
- DrawdownHaltRule: Stop opening new positions when drawdown > threshold

Prevents adding new risk during drawdown periods, resuming when recovered.
"""

from typing import Any, ClassVar
from dataclasses import dataclass, field

from trendspec.risk.base import RiskRule, Allow, Reject, RiskResult, Portfolio
from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal


@dataclass
class DrawdownState:
    """
    State tracking for drawdown halt rule.

    Tracks the peak equity and current drawdown level.

    Attributes:
        peak_equity: Maximum equity observed
        current_drawdown: Current drawdown as percentage
        halted: Whether currently halted
        halt_date: Date when halted
        recovery_date: Date when recovered (if halted)
    """

    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    halted: bool = False
    halt_date: str | None = None
    recovery_date: str | None = None


class DrawdownHaltRule(RiskRule):
    """
    Drawdown halt rule.

    Stops opening new positions when portfolio drawdown exceeds threshold.
    Resumes when drawdown recovers below threshold.

    This is a protective rule that prevents adding new risk during
    unfavorable market conditions.

    Attributes:
        name: Rule name
        priority: Rule priority (5, very early)

    Parameters:
        halt_threshold: Drawdown % to trigger halt (default: 0.10 = 10%)
        recovery_threshold: Drawdown % to resume (default: 0.05 = 5%)
        halt_buy_only: Only halt buy signals (default: True)

    Example:
        >>> rule = DrawdownHaltRule(halt_threshold=0.15, recovery_threshold=0.08)
        >>> # Halts new positions when drawdown > 15%
        >>> # Resumes when drawdown < 8%
    """

    name: ClassVar[str] = "drawdown_halt"
    priority: ClassVar[int] = 5  # Run very early

    def __init__(
        self,
        halt_threshold: float = 0.10,
        recovery_threshold: float = 0.05,
        halt_buy_only: bool = True,
    ) -> None:
        """
        Initialize drawdown halt rule.

        Args:
            halt_threshold: Drawdown % to trigger halt
            recovery_threshold: Drawdown % to resume (must be lower than halt)
            halt_buy_only: Only halt buy signals (allow sells during halt)
        """
        if recovery_threshold >= halt_threshold:
            raise ValueError(
                f"recovery_threshold ({recovery_threshold}) must be less than "
                f"halt_threshold ({halt_threshold})"
            )

        self.params = {
            "halt_threshold": halt_threshold,
            "recovery_threshold": recovery_threshold,
            "halt_buy_only": halt_buy_only,
        }

        # Track drawdown state
        self._state: DrawdownState = DrawdownState()

    def _update_drawdown(self, portfolio: Portfolio) -> float:
        """
        Update drawdown state from portfolio.

        Args:
            portfolio: Current portfolio state

        Returns:
            Current drawdown percentage
        """
        if portfolio.equity <= 0:
            return 0.0

        # Update peak equity
        if portfolio.equity > self._state.peak_equity:
            self._state.peak_equity = portfolio.equity

        # Calculate current drawdown
        if self._state.peak_equity > 0:
            drawdown = 1 - portfolio.equity / self._state.peak_equity
            self._state.current_drawdown = drawdown
            return drawdown

        return 0.0

    def check(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> RiskResult:
        """
        Check if drawdown exceeds halt threshold.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            Allow or Reject result
        """
        halt_threshold = self.params.get("halt_threshold", 0.10)
        recovery_threshold = self.params.get("recovery_threshold", 0.05)
        halt_buy_only = self.params.get("halt_buy_only", True)

        # Update drawdown state
        current_drawdown = self._update_drawdown(portfolio)

        # Check halt/recovery logic
        if self._state.halted:
            # Currently halted - check if recovered
            if current_drawdown <= recovery_threshold:
                # Recovered - resume trading
                self._state.halted = False
                self._state.recovery_date = str(ctx.date)
                return Allow(self.name)
            else:
                # Still halted
                if halt_buy_only and signal.is_sell():
                    # Allow sell signals during halt
                    return Allow(self.name)

                return Reject(
                    self.name,
                    f"Halted due to drawdown {current_drawdown:.1%} > {halt_threshold:.1%}",
                    {
                        "current_drawdown": current_drawdown,
                        "halt_threshold": halt_threshold,
                        "peak_equity": self._state.peak_equity,
                        "current_equity": portfolio.equity,
                        "halt_date": self._state.halt_date,
                    },
                )

        else:
            # Not halted - check if should halt
            if current_drawdown > halt_threshold:
                # Trigger halt
                self._state.halted = True
                self._state.halt_date = str(ctx.date)

                if halt_buy_only and signal.is_sell():
                    # Allow sell signals even when halting
                    return Allow(self.name)

                return Reject(
                    self.name,
                    f"Halting new positions: drawdown {current_drawdown:.1%} > {halt_threshold:.1%}",
                    {
                        "current_drawdown": current_drawdown,
                        "halt_threshold": halt_threshold,
                        "peak_equity": self._state.peak_equity,
                        "current_equity": portfolio.equity,
                        "halt_date": self._state.halt_date,
                    },
                )

            return Allow(self.name)

    def is_halted(self) -> bool:
        """
        Check if currently halted.

        Returns:
            True if halted, False otherwise
        """
        return self._state.halted

    def get_state(self) -> DrawdownState:
        """
        Get current drawdown state.

        Returns:
            Current DrawdownState
        """
        return self._state

    def reset(self) -> None:
        """
        Reset drawdown state.

        Useful for testing or restarting strategy.
        """
        self._state = DrawdownState()


# Register rule in the registry
from trendspec.risk.base import _RULE_REGISTRY

_RULE_REGISTRY["drawdown_halt"] = DrawdownHaltRule