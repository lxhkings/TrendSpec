"""Tests for trendspec/cli/screen_cmd.py --spec-file handling."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from trendspec.cli.screen_cmd import app
from trendspec.engine.screening_engine import ScreeningResult

runner = CliRunner()


def _empty_result() -> ScreeningResult:
    return ScreeningResult(signals=[], trades=[], equity_curve=[], metrics={}, universe_size=0)


def test_spec_file_passed_through_to_engine_run(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "market": "cn",
        "factors": [{"name": "momentum", "params": {"period": 60},
                     "direction": "high", "weight": 1.0}],
        "top_k": 3, "rebalance": 21,
        "group_by": {"金融": ["银行"]},
    }))

    with patch("trendspec.engine.screening_engine.ScreeningEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.run.return_value = _empty_result()
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
