"""研究专用快评估器：去重因子组合 + 注入引擎。数值等价于逐候选全回测。"""

import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date as DateType
from pathlib import Path
from typing import Any, Callable

from trendspec.data.markets import Market
from trendspec.data.calendar import trading_days_between
from trendspec.data.universe import get_universe
from trendspec.engine.backtest_engine import BacktestEngine
from trendspec.engine.base_engine import EngineConfig
from trendspec.research.factor_cache import build_combo_score
from trendspec.research.market_panel import MarketPanel
from trendspec.research.panel_io import read_ipc_mmap, write_ipc
from trendspec.strategy.factor_strategy import FactorStrategy


def _combo_key(spec: dict) -> str:
    """factor_combo 唯一键：只含 factors（含 params/direction/weight），不含 top_k/rebalance。"""
    norm = [
        {"name": f["name"], "params": dict(sorted((f.get("params") or {}).items())),
         "direction": f["direction"], "weight": f.get("weight", 1.0)}
        for f in spec["factors"]
    ]
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


def _split_windows(market: str, start: DateType, end: DateType, n: int) -> list[tuple]:
    days = trading_days_between(Market(market.upper()), start, end)
    if len(days) < n:
        n = max(1, len(days))
    size = len(days) // n
    out = []
    for i in range(n):
        lo = i * size
        hi = (i + 1) * size - 1 if i < n - 1 else len(days) - 1
        out.append((days[lo], days[hi]))
    return out


def _worker_eval_candidate(args: tuple) -> tuple[int, dict]:
    """子进程：mmap读panel，独立跑完整walk-forward。返回(idx, result)。

    数值等价：每个窗口独立计算combo_score（不复用其他窗口的scores）。
    """
    (idx, spec, market, start, end, n_windows, capital, root,
     panel_path, polars_threads) = args
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)

    # 子进程内import
    import trendspec.factors  # noqa: F401
    import trendspec.strategy.factor_strategy  # noqa: F401

    panel_df = read_ipc_mmap(panel_path)
    panel = MarketPanel(data=panel_df)
    panel.universe = get_universe(Market(market.upper()), root)

    windows = _split_windows(market, start, end, n_windows)
    sharpes, dds, rets = [], [], []

    for w_start, w_end in windows:
        win_df = panel.slice(w_start, w_end)
        # 每个窗口独立计算combo_score（不复用其他窗口）
        scores = build_combo_score(win_df, spec["factors"])

        cfg = EngineConfig(market=Market(market.upper()),
                           start_date=w_start, end_date=w_end,
                           initial_capital=capital, root=root)
        eng = BacktestEngine(cfg)
        eng.inject(data=win_df, universe=panel.universe)
        res = eng.run(FactorStrategy,
                      params={"spec": spec, "precomputed_scores": scores})
        m = res.metrics
        sharpes.append(float(m.get("sharpe_ratio", 0.0)))
        dds.append(float(m.get("max_drawdown", 0.0)))
        rets.append(float(m.get("total_return", 0.0)))

    oos_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    return idx, {
        "spec": spec,
        "oos_sharpe": oos_sharpe,
        "oos_max_drawdown": max(dds, default=0.0),
        "worst_window_sharpe": min(sharpes, default=-1.0),
        "window_sharpes": sharpes,
        "oos_total_return": sum(rets),
    }


class ResearchEvaluator:
    def __init__(self, market: str, start: DateType, end: DateType,
                 n_windows: int = 4, capital: float = 100000.0,
                 parallel: bool = True, root: str | None = None) -> None:
        self.market = market
        self.start = start
        self.end = end
        self.n_windows = n_windows
        self.capital = capital
        self.parallel = parallel
        self.root = root
        self._panel: MarketPanel | None = None

    def _ensure_panel(self) -> MarketPanel:
        if self._panel is None:
            self._panel = MarketPanel.load(self.market, self.start, self.end, self.root)
        return self._panel

    def evaluate_batch(
        self, candidates: list[dict],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """批量评估候选。

        为保持与 default_evaluate_fn 数值等价，每个窗口独立计算 combo_score。
        并行模式：panel mmap共享，候选间并行（worker独立跑walk-forward）。
        """
        panel = self._ensure_panel()
        windows = _split_windows(self.market, self.start, self.end, self.n_windows)

        # 串行分支
        if not self.parallel or len(candidates) <= 1:
            window_combo_scores: dict[tuple, dict[str, Any]] = {}
            for w_start, w_end in windows:
                win_df = panel.slice(w_start, w_end)
                combo_scores: dict[str, Any] = {}
                for spec in candidates:
                    k = _combo_key(spec)
                    if k not in combo_scores:
                        combo_scores[k] = build_combo_score(win_df, spec["factors"])
                window_combo_scores[(w_start, w_end)] = combo_scores

            results: list[dict] = []
            total = len(candidates)
            for i, spec in enumerate(candidates, start=1):
                sharpes, dds, rets = [], [], []
                for w_start, w_end in windows:
                    scores = window_combo_scores[(w_start, w_end)][_combo_key(spec)]
                    cfg = EngineConfig(market=Market(self.market.upper()),
                                       start_date=w_start, end_date=w_end,
                                       initial_capital=self.capital, root=self.root)
                    eng = BacktestEngine(cfg)
                    eng.inject(data=panel.slice(w_start, w_end), universe=panel.universe)
                    res = eng.run(FactorStrategy,
                                  params={"spec": spec, "precomputed_scores": scores})
                    m = res.metrics
                    sharpes.append(float(m.get("sharpe_ratio", 0.0)))
                    dds.append(float(m.get("max_drawdown", 0.0)))
                    rets.append(float(m.get("total_return", 0.0)))
                oos_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
                results.append({
                    "spec": spec,
                    "oos_sharpe": oos_sharpe,
                    "oos_max_drawdown": max(dds, default=0.0),
                    "worst_window_sharpe": min(sharpes, default=-1.0),
                    "window_sharpes": sharpes,
                    "oos_total_return": sum(rets),
                })
                if progress_cb:
                    progress_cb(i, total)
            return results

        # === 并行分支：panel mmap共享，候选间并行 ===
        cores = os.cpu_count() or 4
        n_workers = max(1, min(cores, len(candidates)))
        # 自适应：候选多 → 多进程/每worker单线程；候选少 → 少进程/多线程
        polars_threads = max(1, cores // n_workers) if len(candidates) < cores else 1

        tmpdir = Path(tempfile.mkdtemp(prefix="trendspec_panel_"))
        panel_path = tmpdir / "panel.arrow"
        try:
            write_ipc(panel.data, panel_path)

            tasks = [
                (idx, spec, self.market, self.start, self.end, self.n_windows,
                 self.capital, self.root, str(panel_path), polars_threads)
                for idx, spec in enumerate(candidates)
            ]

            results_by_idx: dict[int, dict] = {}
            done = 0
            total = len(tasks)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for idx, res in pool.map(_worker_eval_candidate, tasks):
                    results_by_idx[idx] = res
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)

            # 确定性回填（按候选顺序）
            return [results_by_idx[i] for i in range(len(candidates))]
        finally:
            shutil.rmtree(tmpdir)