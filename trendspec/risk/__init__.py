"""
TrendSpec risk module.

Provides risk management framework for filtering signals before execution.
Key components:
- RiskRule: Abstract base class for risk rules
- Allow/Reject: Result types for rule checks
- Portfolio: Portfolio state for risk decisions
- RiskPipeline: Serial pipeline for running rules

Design principles:
- Serial pipeline: Rules run in priority order
- First rejection wins: Signal dropped at first rejection
- Logging: All checks logged for analysis
- Modifiable signals: Rules can modify signals

Example:
    >>> from trendspec.risk import RiskPipeline, MaxPositions, MinCapital
    ...
    >>> pipeline = RiskPipeline([
    ...     MaxPositions(10),
    ...     MinCapital(1000),
    ... ])
    ...
    >>> result = pipeline.run(signal, portfolio, ctx)
    >>> if result.is_allowed():
    ...     broker.submit(result.signal)
"""

from trendspec.risk.base import (
    Allow,
    DuplicatePosition,
    LiquidityFilter,
    MaxPositionSize,
    MaxPositions,
    MinCapital,
    Portfolio,
    Reject,
    RiskResult,
    RiskRule,
    SectorConcentration,
    UniverseMembership,
    get_rule,
    list_rules,
    register_rule,
)
from trendspec.risk.pipeline import (
    PipelineResult,
    PipelineStats,
    RiskPipeline,
    default_pipeline,
)

__all__ = [
    # Result types
    "Allow",
    "Reject",
    "RiskResult",
    # Portfolio
    "Portfolio",
    # Base classes
    "RiskRule",
    # Built-in rules
    "MaxPositionSize",
    "MaxPositions",
    "MinCapital",
    "SectorConcentration",
    "LiquidityFilter",
    "DuplicatePosition",
    "UniverseMembership",
    # Registry
    "register_rule",
    "get_rule",
    "list_rules",
    # Pipeline
    "RiskPipeline",
    "PipelineResult",
    "PipelineStats",
    "default_pipeline",
]