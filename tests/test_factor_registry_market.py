import trendspec.factors  # noqa: F401 触发因子注册
from trendspec.data.markets import Market
from trendspec.factors.registry import get_factor_with_market


def test_injects_market_when_factor_needs_it_and_none_given():
    factor = get_factor_with_market("rank_within_sector", {"factor_name": "returns"}, "us")
    assert factor.params["market"] == Market.US


def test_normalizes_lowercase_market_string():
    factor = get_factor_with_market(
        "rank_within_sector", {"factor_name": "returns", "market": "us"}, "us"
    )
    assert factor.params["market"] == Market.US


def test_does_not_override_explicit_market_enum():
    factor = get_factor_with_market(
        "rank_within_sector", {"factor_name": "returns", "market": Market.CN}, "us"
    )
    assert factor.params["market"] == Market.CN


def test_factor_without_market_param_is_unaffected():
    factor = get_factor_with_market("momentum", {"period": 10}, "us")
    assert "market" not in factor.params
    assert factor.params["period"] == 10


def test_unknown_factor_name_returns_none():
    assert get_factor_with_market("no_such_factor", {}, "us") is None
