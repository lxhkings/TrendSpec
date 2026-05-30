from pathlib import Path

from typer.testing import CliRunner

from trendspec.cli.research_cmd import app

runner = CliRunner()


def test_run_with_mock_llm_writes_winner(tmp_path: Path, monkeypatch):
    import trendspec.research.fast_eval as fe_mod

    def fake_evaluate_batch(self, candidates, progress_cb=None):
        return [
            {"spec": c, "oos_sharpe": 1.5, "oos_max_drawdown": 0.1,
             "worst_window_sharpe": 0.6, "window_sharpes": [1.4, 1.6],
             "oos_total_return": 0.3}
            for c in candidates
        ]

    monkeypatch.setattr(fe_mod.ResearchEvaluator, "evaluate_batch", fake_evaluate_batch)

    good = ('{"market":"us","factors":[{"name":"momentum","direction":"high",'
            '"weight":1.0,"param_grid":{"period":[60]}}],'
            '"top_k_grid":[20],"rebalance_grid":[5],"rationale":"动量"}')

    result = runner.invoke(app, [
        "run", "--market", "us", "--start", "2018-01-01", "--end", "2023-12-31",
        "--rounds", "1", "--out", str(tmp_path),
        "--mock-llm", good,
    ])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("strategy-*.md"))
