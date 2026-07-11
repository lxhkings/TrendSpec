import json
from pathlib import Path
from unittest.mock import patch
import datetime as dt

import polars as pl
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


def test_run_with_theme_flag_completes(tmp_path: Path, monkeypatch):
    import trendspec.research.fast_eval as fe_mod

    def fake_evaluate_batch(self, candidates, progress_cb=None):
        return [
            {"spec": c, "oos_sharpe": 1.5, "oos_max_drawdown": 0.1,
             "worst_window_sharpe": 0.6, "window_sharpes": [1.4, 1.6],
             "oos_total_return": 0.3}
            for c in candidates
        ]

    monkeypatch.setattr(fe_mod.ResearchEvaluator, "evaluate_batch", fake_evaluate_batch)

    good = ('{"market":"us","factors":[{"name":"ma_bias","direction":"low",'
            '"weight":1.0,"param_grid":{"period":[20]}}],'
            '"top_k_grid":[20],"rebalance_grid":[5],"rationale":"均值回归"}')

    result = runner.invoke(app, [
        "run", "--market", "us", "--start", "2018-01-01", "--end", "2023-12-31",
        "--rounds", "1", "--out", str(tmp_path),
        "--theme", "均值回归",
        "--mock-llm", good,
    ])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("strategy-*.md"))


def _mock_panel():
    from trendspec.research.market_panel import MarketPanel
    rows = []
    for iid, base in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
        for i in range(40):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": base + i})
    return MarketPanel(data=pl.DataFrame(rows))


def _mock_panel_single_ic():
    """Create panel with exactly 26 days so horizon=20 yields exactly 1 IC date."""
    from trendspec.research.market_panel import MarketPanel
    rows = []
    for iid, (base, multiplier) in [("A", (10.0, 1.0)), ("B", (20.0, 0.9)), ("C", (30.0, 1.1))]:
        for i in range(26):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            # Use different patterns to avoid rank correlation becoming null
            close = base + i * multiplier
            rows.append({"instrument_id": iid, "date": d, "close": close})
    return MarketPanel(data=pl.DataFrame(rows))


def test_research_ic_command_prints_summary(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}],
    }))

    with patch("trendspec.research.market_panel.MarketPanel.load", return_value=_mock_panel()):
        result = runner.invoke(app, [
            "ic", "--spec-file", str(spec_path), "--market", "cn",
            "--start", "2020-01-01", "--end", "2020-02-10", "--horizon", "5",
        ])

    assert result.exit_code == 0, result.output
    assert "IC均值" in result.output


def test_research_ic_command_missing_spec_file_exits_with_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    result = runner.invoke(app, [
        "ic", "--spec-file", str(missing), "--market", "cn", "--start", "2020-01-01",
    ])
    assert result.exit_code == 1
    assert "不存在" in result.output


def test_research_ic_command_handles_single_ic_date_no_crash(tmp_path):
    """Test that single IC date (ic_std=None) is handled gracefully without TypeError."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}],
    }))

    with patch("trendspec.research.market_panel.MarketPanel.load", return_value=_mock_panel_single_ic()):
        result = runner.invoke(app, [
            "ic", "--spec-file", str(spec_path), "--market", "cn",
            "--start", "2020-01-01", "--end", "2020-01-21", "--horizon", "20",
        ])

    assert result.exit_code == 0, result.output
    assert "IC均值" in result.output
    assert "N/A" in result.output
