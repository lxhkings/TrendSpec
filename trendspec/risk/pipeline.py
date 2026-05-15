"""
Risk pipeline for TrendSpec.

RiskPipeline runs a series of risk rules in priority order.
Any rejection drops the signal and logs the reason.

Key design:
- Serial execution: Rules run in order
- First rejection wins: Signal dropped at first rejection
- Logging: All checks logged for analysis
- Modifiable signals: Rules can modify signals (e.g., add stop-loss)

Usage:
    >>> pipeline = RiskPipeline([
    ...     UniverseMembership(),
    ...     MaxPositions(10),
    ...     MinCapital(1000),
    ... ])
    ...
    >>> result = pipeline.run(signal, portfolio, ctx)
    >>> if result.is_allowed():
    ...     broker.submit(result.modified_signal or signal)
"""

from dataclasses import dataclass, field
from typing import Any

from trendspec.strategy.context import StrategyContext
from trendspec.strategy.signal import Signal
from trendspec.risk.base import (
    Allow,
    Reject,
    RiskResult,
    RiskRule,
    Portfolio,
)


@dataclass
class PipelineResult:
    """
    Result of running signal through pipeline.

    Contains final decision and all rule results for logging.

    Attributes:
        final_result: Final Allow or Reject
        rule_results: Dict of rule_name -> RiskResult
        signal: Original signal (or modified)
        rejection_reason: Reason if rejected (None if allowed)
    """

    final_result: RiskResult
    signal: Signal
    rule_results: dict[str, RiskResult] = field(default_factory=dict)
    rejection_reason: str | None = None

    def is_allowed(self) -> bool:
        """Check if signal passed pipeline."""
        return self.final_result.is_allowed()

    def is_rejected(self) -> bool:
        """Check if signal was rejected."""
        return self.final_result.is_rejected()

    def get_modified_signal(self) -> Signal | None:
        """Get modified signal if any rule modified it."""
        return getattr(self.final_result, "modified_signal", None)

    def get_rejection_details(self) -> dict[str, Any]:
        """Get rejection details if rejected."""
        if self.is_rejected():
            return getattr(self.final_result, "details", {})
        return {}

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_allowed():
            passed_rules = [r for r, res in self.rule_results.items() if res.is_allowed()]
            return f"Signal passed {len(passed_rules)} rules: {', '.join(passed_rules)}"
        else:
            return f"Signal rejected by {self.final_result.rule_name}: {self.rejection_reason}"


@dataclass
class PipelineStats:
    """
    Statistics for risk pipeline.

    Tracks rule performance and rejection rates.

    Attributes:
        total_signals: Total signals processed
        allowed_count: Signals allowed through
        rejected_count: Signals rejected
        rejection_by_rule: Dict of rule_name -> rejection count
        rejection_by_reason: Dict of reason -> count
    """

    total_signals: int = 0
    allowed_count: int = 0
    rejected_count: int = 0
    rejection_by_rule: dict[str, int] = field(default_factory=dict)
    rejection_by_reason: dict[str, int] = field(default_factory=dict)

    def record_result(self, result: PipelineResult) -> None:
        """Record a pipeline result."""
        self.total_signals += 1
        if result.is_allowed():
            self.allowed_count += 1
        else:
            self.rejected_count += 1
            rule_name = result.final_result.rule_name
            reason = result.rejection_reason or "unknown"
            self.rejection_by_rule[rule_name] = self.rejection_by_rule.get(rule_name, 0) + 1
            self.rejection_by_reason[reason] = self.rejection_by_reason.get(reason, 0) + 1

    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_signals == 0:
            return 0.0
        return self.rejected_count / self.total_signals

    def most_common_rejection(self) -> tuple[str | None, int]:
        """Get most common rejection rule."""
        if not self.rejection_by_rule:
            return (None, 0)
        return max(self.rejection_by_rule.items(), key=lambda x: x[1])

    def summary(self) -> str:
        """Get human-readable stats summary."""
        lines = [
            f"Risk Pipeline Statistics:",
            f"  Total signals: {self.total_signals}",
            f"  Allowed: {self.allowed_count} ({1 - self.rejection_rate():.1%})",
            f"  Rejected: {self.rejected_count} ({self.rejection_rate():.1%})",
        ]
        if self.rejection_by_rule:
            lines.append("  Rejection by rule:")
            for rule, count in sorted(self.rejection_by_rule.items(), key=lambda x: -x[1]):
                lines.append(f"    {rule}: {count}")
        return "\n".join(lines)


class RiskPipeline:
    """
    Risk pipeline that runs multiple rules in order.

    Rules are executed in priority order (lower priority runs first).
    Any rejection drops the signal immediately.

    Example:
        >>> pipeline = RiskPipeline([
        ...     UniverseMembership(),       # priority 1
        ...     DuplicatePosition(),        # priority 5
        ...     MaxPositionSize(0.10),      # priority 10
        ...     MaxPositions(10),           # priority 20
        ...     MinCapital(1000),           # priority 30
        ... ])
        ...
        >>> result = pipeline.run(signal, portfolio, ctx)
        >>> if result.is_allowed():
        ...     broker.submit(result.signal)
    """

    def __init__(self, rules: list[RiskRule] | None = None) -> None:
        """
        Initialize pipeline with rules.

        Args:
            rules: List of risk rules (sorted by priority)
        """
        self._rules: list[RiskRule] = []
        self._stats = PipelineStats()

        if rules:
            # Sort by priority
            self._rules = sorted(rules, key=lambda r: r.get_priority())

    def add_rule(self, rule: RiskRule) -> None:
        """
        Add a rule to the pipeline.

        Rules are inserted in priority order.

        Args:
            rule: Risk rule to add
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.get_priority())

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule by name.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was found and removed
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                self._rules.pop(i)
                return True
        return False

    def get_rules(self) -> list[RiskRule]:
        """Get list of rules in execution order."""
        return list(self._rules)

    def run(
        self,
        signal: Signal,
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> PipelineResult:
        """
        Run signal through all rules.

        Args:
            signal: Signal to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            PipelineResult with final decision and all rule results
        """
        rule_results: dict[str, RiskResult] = {}
        current_signal = signal

        for rule in self._rules:
            if not rule.is_enabled():
                continue

            # Run the rule
            result = rule.check(current_signal, portfolio, ctx)
            rule_results[rule.name] = result

            # Handle rejection
            if result.is_rejected():
                self._stats.record_result(PipelineResult(
                    final_result=result,
                    rule_results=rule_results,
                    signal=signal,
                    rejection_reason=result.reason,
                ))
                return PipelineResult(
                    final_result=result,
                    rule_results=rule_results,
                    signal=signal,
                    rejection_reason=result.reason,
                )

            # Handle signal modification
            if isinstance(result, Allow) and result.modified_signal:
                current_signal = result.modified_signal

        # All rules passed
        self._stats.record_result(PipelineResult(
            final_result=Allow("pipeline", modified_signal=current_signal),
            rule_results=rule_results,
            signal=current_signal,
        ))
        return PipelineResult(
            final_result=Allow("pipeline", modified_signal=current_signal),
            rule_results=rule_results,
            signal=current_signal,
        )

    def run_batch(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> list[PipelineResult]:
        """
        Run multiple signals through pipeline.

        Args:
            signals: List of signals to check
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            List of PipelineResults
        """
        results = []
        for signal in signals:
            result = self.run(signal, portfolio, ctx)
            results.append(result)
        return results

    def filter_allowed(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        ctx: StrategyContext,
    ) -> list[Signal]:
        """
        Filter signals, returning only allowed ones.

        Args:
            signals: List of signals to filter
            portfolio: Current portfolio state
            ctx: Strategy context

        Returns:
            List of allowed signals (possibly modified)
        """
        allowed = []
        for signal in signals:
            result = self.run(signal, portfolio, ctx)
            if result.is_allowed():
                modified = result.get_modified_signal()
                allowed.append(modified if modified else signal)
        return allowed

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._stats = PipelineStats()

    def summary(self) -> str:
        """Get human-readable pipeline summary."""
        rule_names = [r.name for r in self._rules]
        lines = [
            f"RiskPipeline with {len(self._rules)} rules:",
            f"  Execution order: {', '.join(rule_names)}",
            self._stats.summary(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RiskPipeline(rules={len(self._rules)})"


# =============================================================================
# Default Pipeline Factory
# =============================================================================


def default_pipeline(
    max_positions: int = 10,
    max_position_pct: float = 0.10,
    min_capital: float = 1000.0,
    max_sector_pct: float = 0.30,
    min_volume: int = 100000,
) -> RiskPipeline:
    """
    Create default risk pipeline with common rules.

    Args:
        max_positions: Maximum number of positions
        max_position_pct: Maximum position size as % of equity
        min_capital: Minimum remaining capital after trade
        max_sector_pct: Maximum sector concentration
        min_volume: Minimum volume for liquidity filter

    Returns:
        RiskPipeline with default rules
    """
    from trendspec.risk.base import (
        UniverseMembership,
        DuplicatePosition,
        MaxPositionSize,
        MaxPositions,
        MinCapital,
        SectorConcentration,
        LiquidityFilter,
    )

    return RiskPipeline([
        UniverseMembership(),
        DuplicatePosition(),
        MaxPositionSize(max_position_pct),
        MaxPositions(max_positions),
        MinCapital(min_capital),
        SectorConcentration(max_sector_pct),
        LiquidityFilter(min_volume),
    ])