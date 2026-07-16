import pytest
from pydantic import ValidationError

import trendspec.factors  # noqa: F401 — 触发因子注册，name 校验才过
from trendspec.research.spec import (
    FactorSpec,
    FactorTerm,
    FilterTerm,
    parse_research_eval_spec,
)


def _valid_spec_dict():
    return {
        "market": "us",
        "factors": [
            {"name": "momentum", "params": {"period": 60}, "direction": "high", "weight": 1.0},
            {"name": "volatility", "params": {"period": 20}, "direction": "low", "weight": 0.5},
        ],
        "top_k": 20,
        "rebalance": 5,
        "rationale": "动量叠加低波动",
    }


def test_valid_spec_parses():
    spec = FactorSpec(**_valid_spec_dict())
    assert spec.market == "us"
    assert len(spec.factors) == 2
    assert spec.factors[0].direction == "high"
    assert spec.top_k == 20


def test_unknown_factor_name_rejected():
    d = _valid_spec_dict()
    d["factors"][0]["name"] = "no_such_factor"
    with pytest.raises(ValueError, match="未注册因子"):
        FactorSpec(**d)


def test_bad_direction_rejected():
    d = _valid_spec_dict()
    d["factors"][0]["direction"] = "sideways"
    with pytest.raises(ValueError):
        FactorSpec(**d)


def test_top_k_must_be_positive():
    d = _valid_spec_dict()
    d["top_k"] = 0
    with pytest.raises(ValueError):
        FactorSpec(**d)


def test_round_trip_dict():
    spec = FactorSpec(**_valid_spec_dict())
    again = FactorSpec(**spec.model_dump())
    assert again.top_k == spec.top_k


def test_group_by_defaults_to_none():
    spec = FactorSpec(**_valid_spec_dict())
    assert spec.group_by is None


def test_winsorize_pct_defaults_to_one_percent():
    spec = FactorSpec(**_valid_spec_dict())
    assert spec.winsorize_pct == 0.01


def test_group_by_accepts_mapping():
    d = _valid_spec_dict()
    d["group_by"] = {"金融": ["银行", "证券"], "能源": ["煤炭开采"]}
    spec = FactorSpec(**d)
    assert spec.group_by == {"金融": ["银行", "证券"], "能源": ["煤炭开采"]}


def test_group_by_round_trips_through_model_dump():
    d = _valid_spec_dict()
    d["group_by"] = {"金融": ["银行"]}
    spec = FactorSpec(**d)
    again = FactorSpec(**spec.model_dump())
    assert again.group_by == {"金融": ["银行"]}


def test_filters_default_empty():
    spec = FactorSpec(**_valid_spec_dict())
    assert spec.filters == []


def test_filters_parse_and_round_trip():
    d = _valid_spec_dict()
    d["filters"] = [
        {"name": "momentum", "op": ">", "value": 0.0},
        {"name": "volatility", "op": ">=", "value": 1e9},
    ]
    spec = FactorSpec(**d)
    assert len(spec.filters) == 2
    assert spec.filters[0].op == ">"
    again = FactorSpec(**spec.model_dump())
    assert again.filters[1].value == 1e9


def test_filter_unknown_factor_rejected():
    d = _valid_spec_dict()
    d["filters"] = [{"name": "no_such_factor", "op": ">", "value": 0.0}]
    with pytest.raises(ValueError, match="未注册因子"):
        FactorSpec(**d)


def test_filter_bad_op_rejected():
    d = _valid_spec_dict()
    d["filters"] = [{"name": "momentum", "op": "!=", "value": 0.0}]
    with pytest.raises(ValueError):
        FactorSpec(**d)


def test_framework_v1_example_parses():
    import json
    from pathlib import Path

    p = Path(__file__).parents[2] / "examples" / "factor_combo_cn_framework_v1.json"
    spec = FactorSpec(**json.loads(p.read_text()))
    assert spec.market == "cn"
    assert len(spec.filters) == 3
    assert {t.name for t in spec.filters} == {
        "fund_op_margin", "fund_q_ocf_to_sales", "fund_total_revenue"
    }
    assert "fund_revenue_yoy_band" in {t.name for t in spec.factors}
    assert spec.group_by is not None


def test_parse_research_eval_spec_accepts_factors_and_filters():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}
        ],
        "filters": [
            {"name": "momentum", "params": {"period": 5}, "op": ">", "value": 0.0}
        ],
        "winsorize_pct": 0.02,
    }
    out = parse_research_eval_spec(raw)
    assert len(out["factors"]) == 1
    assert out["factors"][0]["name"] == "momentum"
    assert out["factors"][0]["direction"] == "high"
    assert len(out["filters"]) == 1
    assert out["filters"][0]["op"] == ">"
    assert out["winsorize_pct"] == 0.02


def test_parse_research_eval_spec_defaults_filters_empty_and_winsorize():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
    }
    out = parse_research_eval_spec(raw)
    assert out["filters"] == []
    assert out["winsorize_pct"] == 0.01


def test_parse_research_eval_spec_rejects_bad_filter_op():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "filters": [{"name": "momentum", "op": "!=", "value": 0.0}],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_unknown_factor_name():
    raw = {
        "factors": [
            {"name": "no_such_factor_xyz", "direction": "high"}
        ],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_missing_direction():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}}
        ],
    }
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_rejects_empty_factors():
    raw = {"factors": []}
    with pytest.raises(ValidationError):
        parse_research_eval_spec(raw)


def test_parse_research_eval_spec_preserves_group_by():
    raw = {
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "group_by": {"金融": ["银行"]},
    }
    out = parse_research_eval_spec(raw)
    assert out["group_by"] == {"金融": ["银行"]}


def test_parse_research_eval_spec_ignores_full_spec_extra_fields():
    raw = {
        "market": "cn",
        "top_k": 10,
        "rebalance": 5,
        "rationale": "x",
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "filters": [],
    }
    out = parse_research_eval_spec(raw)
    assert "market" not in out
    assert "top_k" not in out
    assert out["filters"] == []
    assert out["winsorize_pct"] == 0.01
    assert out["factors"][0]["name"] == "momentum"


def test_filter_ops_keys_match_spec_constant():
    from trendspec.combo.scores import _FILTER_OPS
    from trendspec.research.spec import FILTER_OP_NAMES

    assert set(_FILTER_OPS.keys()) == set(FILTER_OP_NAMES)
