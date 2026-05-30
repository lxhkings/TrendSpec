"""Walk-forward 滚动样本外评估（固定 spec，不逐窗重拟合）。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as DateType
from typing import Any

from trendspec.data.calendar import trading_days_between
from trendspec.data.markets import Market

BacktestFn = Callable[[dict, str, DateType, DateType, float], dict[str, Any]]


@dataclass
class WindowResult:
    start: DateType
    end: DateType
    metrics: dict[str, Any]


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    window_sharpes: list[float] = field(default_factory=list)
    oos_sharpe: float = 0.0
    worst_window_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_total_return: float = 0.0


def _default_backtest_fn(
    spec_dict: dict, market: str, start: DateType, end: DateType, capital: float
) -> dict[str, Any]:
    from trendspec.engine.backtest_engine import BacktestEngine
    from trendspec.engine.base_engine import EngineConfig
    from trendspec.strategy.factor_strategy import FactorStrategy

    config = EngineConfig(
        market=Market(market.upper()),
        start_date=start,
        end_date=end,
        initial_capital=capital,
    )
    result = BacktestEngine(config).run(FactorStrategy, params={"spec": spec_dict})
    return result.metrics


def _split_windows(market: str, start: DateType, end: DateType, n: int) -> list[tuple]:
    days = trading_days_between(Market(market.upper()), start, end)
    if len(days) < n:
        n = max(1, len(days))
    size = len(days) // n
    windows = []
    for i in range(n):
        lo = i * size
        hi = (i + 1) * size - 1 if i < n - 1 else len(days) - 1
        windows.append((days[lo], days[hi]))
    return windows


def run_walkforward(
    spec_dict: dict,
    market: str,
    start: DateType,
    end: DateType,
    n_windows: int,
    capital: float = 100000.0,
    backtest_fn: BacktestFn | None = None,
) -> WalkForwardResult:
    fn = backtest_fn or _default_backtest_fn
    windows = _split_windows(market, start, end, n_windows)

    out = WalkForwardResult()
    for w_start, w_end in windows:
        metrics = fn(spec_dict, market, w_start, w_end, capital)
        out.windows.append(WindowResult(w_start, w_end, metrics))
        out.window_sharpes.append(float(metrics.get("sharpe_ratio", 0.0)))

    if out.window_sharpes:
        out.oos_sharpe = sum(out.window_sharpes) / len(out.window_sharpes)
        out.worst_window_sharpe = min(out.window_sharpes)
    out.oos_max_drawdown = max(
        (float(w.metrics.get("max_drawdown", 0.0)) for w in out.windows), default=0.0
    )
    out.oos_total_return = sum(float(w.metrics.get("total_return", 0.0)) for w in out.windows)
    return out
