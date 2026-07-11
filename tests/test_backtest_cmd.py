"""Tests for trendspec/cli/backtest_cmd.py --spec-file handling."""

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from trendspec.cli.backtest_cmd import app
from trendspec.engine.base_engine import EngineResult

runner = CliRunner()


def _empty_result() -> EngineResult:
    return EngineResult(signals=[], trades=[], equity_curve=[], metrics={})


def test_spec_file_passed_through_to_engine_run(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "momentum", "params": {"period": 60},
                     "direction": "high", "weight": 1.0}],
        "top_k": 3, "rebalance": 21,
        "group_by": {"金融": ["银行"]},
    }))

    with patch("trendspec.engine.backtest_engine.BacktestEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.run.return_value = _empty_result()
        instance._strategy = None
        result = runner.invoke(app, [
            "run", "--strategy", "factor_combo", "--market", "cn",
            "--spec-file", str(spec_path),
        ])

    assert result.exit_code == 0, result.output
    instance.run.assert_called_once()
    _, kwargs = instance.run.call_args
    assert kwargs["params"]["spec"]["group_by"] == {"金融": ["银行"]}
    assert kwargs["params"]["spec"]["top_k"] == 3


def test_spec_file_missing_exits_with_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    result = runner.invoke(app, [
        "run", "--strategy", "factor_combo", "--market", "cn",
        "--spec-file", str(missing_path),
    ])
    assert result.exit_code == 1
    assert "不存在" in result.output


def test_spec_file_invalid_json_exits_with_error(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{not valid json")
    result = runner.invoke(app, [
        "run", "--strategy", "factor_combo", "--market", "cn",
        "--spec-file", str(spec_path),
    ])
    assert result.exit_code == 1
    assert "不是合法 JSON" in result.output


def test_compare_spec_file_sweeps_multiple_specs_with_param_override(tmp_path):
    """--spec-file (repeatable) on `compare` runs factor_combo once per spec,
    applying the same --param override (top_pct) to each, instead of
    sweeping the registered-strategy list."""
    spec_a = tmp_path / "pe_only.json"
    spec_a.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "fund_pe_ttm", "direction": "low", "weight": 1.0}],
        "top_k": 5, "rebalance": 21,
    }))
    spec_b = tmp_path / "roe_only.json"
    spec_b.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "fund_roe", "direction": "high", "weight": 1.0}],
        "top_k": 5, "rebalance": 21,
    }))

    with patch("trendspec.engine.backtest_engine.BacktestEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.run.return_value = _empty_result()
        result = runner.invoke(app, [
            "compare", "--market", "cn",
            "--spec-file", str(spec_a), "--spec-file", str(spec_b),
            "--param", "top_pct=0.05",
        ])

    assert result.exit_code == 0, result.output
    assert instance.run.call_count == 2
    for call in instance.run.call_args_list:
        _, kwargs = call
        spec = kwargs["params"]["spec"]
        assert spec["top_pct"] == 0.05
        assert "top_k" not in spec  # top_pct 覆盖时二选一，原 top_k 被清掉
    assert "pe_only" in result.output
    assert "roe_only" in result.output


def test_compare_spec_file_missing_file_records_error_row(tmp_path):
    """A missing --spec-file becomes an ERROR row, not a hard CLI exit —
    matches the sweep's per-item tolerance so one bad file doesn't kill
    the rest of the comparison."""
    missing = tmp_path / "does_not_exist.json"
    with patch("trendspec.engine.backtest_engine.BacktestEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.run.return_value = _empty_result()
        result = runner.invoke(app, [
            "compare", "--market", "cn", "--spec-file", str(missing),
        ])

    assert result.exit_code == 0, result.output
    instance.run.assert_not_called()
    assert "ERROR" in result.output
