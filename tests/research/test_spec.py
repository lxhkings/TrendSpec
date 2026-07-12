import pytest

from trendspec.research.spec import FactorSpec, FactorTerm, FilterTerm


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
