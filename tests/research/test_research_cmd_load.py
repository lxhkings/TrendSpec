from pathlib import Path

import pytest
import typer

import trendspec.factors  # noqa: F401
from trendspec.cli.research_cmd import _load_factor_spec_json


def test_load_factor_spec_json_valid(tmp_path: Path):
    p = tmp_path / "ok.json"
    p.write_text(
        '{"factors":[{"name":"momentum","params":{"period":5},"direction":"high"}]}',
        encoding="utf-8",
    )
    out = _load_factor_spec_json(p)
    assert out["factors"][0]["name"] == "momentum"
    assert out["filters"] == []


def test_load_factor_spec_json_bad_filter_op_exits(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"factors":[{"name":"momentum","direction":"high"}],'
        '"filters":[{"name":"momentum","op":"!=","value":0}]}',
        encoding="utf-8",
    )
    with pytest.raises(typer.Exit) as ei:
        _load_factor_spec_json(p)
    assert ei.value.exit_code == 1
