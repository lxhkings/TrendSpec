"""
Qullamaggie Momentum Breakout Strategy.

Based on Kristjan Kullamägi's ("Qullamaggie") episodic-pivot / momentum-breakout
playbook: trade high-ADR stocks already in a strong move, wait for tight
consolidation, buy the volume-confirmed breakout, partial sell after a few bars,
trail the rest with a short MA.

See `docs/superpowers/specs/2026-05-17-qullamaggie-momentum-spec.md` for full
design.
"""

from datetime import date as DateType
from typing import Any

import polars as pl

from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("qullamaggie_momentum")
class QullamaggieMomentumStrategy(BaseStrategy):
    """Qullamaggie-style momentum breakout with partial sell + MA trailing exit."""

    name = "qullamaggie_momentum"
    version = "1.0.0"
    params: dict[str, Any] = {
        "ma_short_period": 10,
        "ma_mid_period": 20,
        "ma_long_period": 50,
        "roc_period": 60,
        "prior_move_threshold": 0.30,
        "adr_period": 20,
        "adr_pct_min": 0.04,
        "dollar_volume_min": 5_000_000,
        "consolidation_days": 5,
        "consolidation_tightness": 1.5,
        "volume_mult": 1.5,
        "partial_sell_after_days": 4,
        "partial_sell_fraction": 0.5,
        "trail_ma_period": 10,
        "risk_pct": 0.005,
    }

    def _validate_dict_params(self) -> None:
        risk_pct = self.get_param("risk_pct", 0.005)
        psf = self.get_param("partial_sell_fraction", 0.5)
        adr_min = self.get_param("adr_pct_min", 0.04)
        cons_days = self.get_param("consolidation_days", 5)

        if not (0 < risk_pct < 1):
            raise ValueError(f"risk_pct ({risk_pct}) must be in (0, 1)")
        if not (0 <= psf <= 1):
            raise ValueError(f"partial_sell_fraction ({psf}) must be in [0, 1]")
        if adr_min < 0:
            raise ValueError(f"adr_pct_min ({adr_min}) must be >= 0")
        if cons_days < 2:
            raise ValueError(f"consolidation_days ({cons_days}) must be >= 2")

    def init(self, ctx: StrategyContext) -> None:  # pragma: no cover - filled in Task 3
        raise NotImplementedError("Task 3 implements init()")

    def next(self, ctx: StrategyContext) -> None:  # pragma: no cover - filled later
        raise NotImplementedError("Tasks 4-7 implement next()")
