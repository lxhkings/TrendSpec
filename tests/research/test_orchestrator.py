import json
from pathlib import Path

from trendspec.research.orchestrator import ResearchOrchestrator, passes_threshold


class _FakeAgent:
    def __init__(self, hypos):
        self._hypos = hypos
        self._i = 0

    def propose(self, ledger_rows):
        h = self._hypos[self._i]
        self._i += 1
        return h


def _hypo(period):
    return {
        "market": "us",
        "factors": [{"name": "momentum", "direction": "high", "weight": 1.0,
                     "param_grid": {"period": [period]}}],
        "top_k_grid": [20], "rebalance_grid": [5], "rationale": f"p{period}",
    }


def test_passes_threshold():
    assert passes_threshold({"oos_sharpe": 1.2, "oos_max_drawdown": 0.1,
                             "worst_window_sharpe": 0.3})
    assert not passes_threshold({"oos_sharpe": 0.5, "oos_max_drawdown": 0.1,
                                 "worst_window_sharpe": 0.3})
    assert not passes_threshold({"oos_sharpe": 1.2, "oos_max_drawdown": 0.5,
                                 "worst_window_sharpe": 0.3})
    assert not passes_threshold({"oos_sharpe": 1.2, "oos_max_drawdown": 0.1,
                                 "worst_window_sharpe": -0.1})


def test_run_writes_state_ledger_and_winner(tmp_path: Path):
    # evaluate_fn: 给一个高分结果 → 必达标
    def evaluate_fn(spec_dict):
        return {"spec": spec_dict, "oos_sharpe": 1.4, "oos_max_drawdown": 0.10,
                "worst_window_sharpe": 0.5, "window_sharpes": [1.3, 1.5],
                "oos_total_return": 0.3}

    orch = ResearchOrchestrator(
        agent=_FakeAgent([_hypo(60)]),
        evaluate_fn=evaluate_fn,
        out_dir=str(tmp_path),
        max_rounds=1, max_candidates=10, stop_on_first=True,
    )
    orch.run()

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["phase"] in ("done", "running")
    assert (tmp_path / "ledger.jsonl").exists()
    winners = list(tmp_path.glob("strategy-*.md"))
    assert len(winners) == 1


def test_run_no_winner_when_below_threshold(tmp_path: Path):
    def evaluate_fn(spec_dict):
        return {"spec": spec_dict, "oos_sharpe": 0.2, "oos_max_drawdown": 0.30,
                "worst_window_sharpe": -0.5, "window_sharpes": [0.2],
                "oos_total_return": 0.0}

    orch = ResearchOrchestrator(
        agent=_FakeAgent([_hypo(60)]),
        evaluate_fn=evaluate_fn,
        out_dir=str(tmp_path),
        max_rounds=1, max_candidates=10, stop_on_first=True,
    )
    orch.run()
    assert list(tmp_path.glob("strategy-*.md")) == []
    assert (tmp_path / "ledger.jsonl").exists()
