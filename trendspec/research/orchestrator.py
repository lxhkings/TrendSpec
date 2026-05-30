"""研究闭环主编排。"""

from collections.abc import Callable
from datetime import date as DateType
from pathlib import Path
from typing import Any

from trendspec.research.agent import HypothesisParseError
from trendspec.research.ledger import append_ledger, read_ledger, write_state
from trendspec.research.report import write_advice
from trendspec.research.search import expand_grid

EvaluateFn = Callable[[dict], dict[str, Any]]

THRESHOLD_SHARPE = 1.0
THRESHOLD_MAX_DD = 0.20


def passes_threshold(result: dict[str, Any]) -> bool:
    return (
        result.get("oos_sharpe", 0.0) >= THRESHOLD_SHARPE
        and result.get("oos_max_drawdown", 1.0) <= THRESHOLD_MAX_DD
        and result.get("worst_window_sharpe", -1.0) > 0
    )


def default_evaluate_fn(
    market: str, start: DateType, end: DateType, n_windows: int, capital: float
) -> EvaluateFn:
    """生产用 evaluate_fn：对单个 spec 跑 walk-forward。"""
    from trendspec.research.walkforward import run_walkforward

    def _fn(spec_dict: dict) -> dict[str, Any]:
        wf = run_walkforward(spec_dict, market, start, end, n_windows, capital)
        return {
            "spec": spec_dict,
            "oos_sharpe": wf.oos_sharpe,
            "oos_max_drawdown": wf.oos_max_drawdown,
            "worst_window_sharpe": wf.worst_window_sharpe,
            "window_sharpes": wf.window_sharpes,
            "oos_total_return": wf.oos_total_return,
        }

    return _fn


class ResearchOrchestrator:
    def __init__(
        self,
        agent,
        evaluate_fn=None,
        out_dir: str = "./research_out",
        max_rounds: int = 10,
        max_candidates: int = 200,
        top_n: int = 5,
        stop_on_first: bool = True,
        batch_evaluator=None,
    ) -> None:
        self._agent = agent
        self._evaluate = evaluate_fn
        self._batch = batch_evaluator
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._ledger_path = self._out / "ledger.jsonl"
        self._state_path = self._out / "state.json"
        self._max_rounds = max_rounds
        self._max_candidates = max_candidates
        self._top_n = top_n
        self._stop_on_first = stop_on_first

    def run(self) -> None:
        winners_total = 0
        for rnd in range(1, self._max_rounds + 1):
            ledger_rows = read_ledger(self._ledger_path)
            write_state(
                self._state_path,
                {
                    "phase": "running",
                    "round": rnd,
                    "max_rounds": self._max_rounds,
                    "winners": winners_total,
                },
            )

            try:
                hypo = self._agent.propose(ledger_rows)
            except HypothesisParseError:
                append_ledger(self._ledger_path, {"round": rnd, "error": "hypothesis_parse_failed"})
                continue

            candidates = expand_grid(hypo, max_candidates=self._max_candidates)

            def _progress(done: int, total: int, _hypo=hypo, _rnd=rnd) -> None:
                write_state(self._state_path, {
                    "phase": "running", "round": _rnd,
                    "sweep_done": done, "sweep_total": total,
                    "hypothesis": _hypo, "winners": winners_total})

            if self._batch is not None:
                results = self._batch.evaluate_batch(candidates, progress_cb=_progress)
            else:
                results = []
                for i, spec_dict in enumerate(candidates, start=1):
                    results.append(self._evaluate(spec_dict))
                    _progress(i, len(candidates))

            results.sort(key=lambda r: r.get("oos_sharpe", 0.0), reverse=True)
            top = results[: self._top_n]
            round_winners = [r for r in results if passes_threshold(r)]

            for w in round_winners:
                write_advice(self._out, w, round_no=rnd)
            winners_total += len(round_winners)

            append_ledger(
                self._ledger_path,
                {
                    "round": rnd,
                    "hypothesis": hypo,
                    "top_candidates": top,
                    "winners": len(round_winners),
                },
            )

            if round_winners and self._stop_on_first:
                break

        write_state(self._state_path, {"phase": "done", "winners": winners_total})
