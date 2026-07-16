"""兼容 re-export：实现见 trendspec.analyzer.ema_cross_winrate。"""

from trendspec.analyzer.ema_cross_winrate import (
    aggregate,
    compute_adv20_daily,
    compute_ema_cross,
    current_screen,
    monte_carlo,
    pair_trades,
    per_ticker,
    recent_golden_cross,
    run_novice_simulations,
    run_winrate,
    simulate_novice,
)

__all__ = [
    "aggregate",
    "compute_adv20_daily",
    "compute_ema_cross",
    "current_screen",
    "monte_carlo",
    "pair_trades",
    "per_ticker",
    "recent_golden_cross",
    "run_novice_simulations",
    "run_winrate",
    "simulate_novice",
]
