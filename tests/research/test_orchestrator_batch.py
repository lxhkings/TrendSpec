import json
from trendspec.research.orchestrator import ResearchOrchestrator


class _StubAgent:
    def __init__(self): self.n = 0
    def propose(self, ledger_rows):
        self.n += 1
        if self.n > 1:
            from trendspec.research.agent import HypothesisParseError
            raise HypothesisParseError("stop")
        return {"market": "us",
                "factors": [{"name": "momentum", "direction": "high", "weight": 1.0,
                             "param_grid": {"period": [10, 20]}}],
                "top_k_grid": [5], "rebalance_grid": [5], "rationale": "t"}


class _StubBatch:
    def __init__(self): self.calls = 0
    def evaluate_batch(self, cands, progress_cb=None):
        self.calls += 1
        out = []
        for i, c in enumerate(cands):
            if progress_cb: progress_cb(i + 1, len(cands))
            out.append({"spec": c, "oos_sharpe": 0.1, "oos_max_drawdown": 0.05,
                        "worst_window_sharpe": 0.1, "oos_total_return": 0.2})
        return out


def test_orchestrator_uses_batch_evaluator(tmp_path):
    batch = _StubBatch()
    orch = ResearchOrchestrator(agent=_StubAgent(), evaluate_fn=None,
                                out_dir=str(tmp_path), max_rounds=1,
                                batch_evaluator=batch)
    orch.run()
    assert batch.calls == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["phase"] == "done"