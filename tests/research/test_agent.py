import json

import pytest

from trendspec.research.agent import HypothesisAgent, HypothesisParseError
from trendspec.research.llm_client import MockLLMClient


def _good_hypo_json():
    return json.dumps({
        "market": "us",
        "factors": [{"name": "momentum", "direction": "high", "weight": 1.0,
                     "param_grid": {"period": [20, 60]}}],
        "top_k_grid": [10, 20],
        "rebalance_grid": [5],
        "rationale": "动量",
    })


def test_propose_parses_plain_json():
    agent = HypothesisAgent(MockLLMClient([_good_hypo_json()]))
    hypo = agent.propose(ledger_rows=[])
    assert hypo["market"] == "us"
    assert hypo["factors"][0]["name"] == "momentum"


def test_propose_strips_code_fence():
    fenced = "```json\n" + _good_hypo_json() + "\n```"
    agent = HypothesisAgent(MockLLMClient([fenced]))
    hypo = agent.propose(ledger_rows=[])
    assert hypo["top_k_grid"] == [10, 20]


def test_propose_retries_once_then_succeeds():
    agent = HypothesisAgent(MockLLMClient(["not json at all", _good_hypo_json()]))
    hypo = agent.propose(ledger_rows=[])
    assert hypo["factors"][0]["name"] == "momentum"


def test_propose_raises_after_retry_fails():
    agent = HypothesisAgent(MockLLMClient(["garbage", "still garbage"]))
    with pytest.raises(HypothesisParseError):
        agent.propose(ledger_rows=[])


def test_propose_rejects_unknown_factor():
    bad = json.dumps({
        "market": "us",
        "factors": [{"name": "no_such", "direction": "high", "weight": 1.0,
                     "param_grid": {}}],
        "top_k_grid": [10], "rebalance_grid": [5], "rationale": "x",
    })
    agent = HypothesisAgent(MockLLMClient([bad, bad]))
    with pytest.raises(HypothesisParseError):
        agent.propose(ledger_rows=[])


class _SpyLLMClient:
    """测试用：记录最近一次 complete() 收到的 system prompt。"""

    def __init__(self, response: str) -> None:
        self._response = response
        self.last_system: str | None = None

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        self.last_system = system
        return self._response


def test_propose_without_theme_has_no_constraint_text():
    spy = _SpyLLMClient(_good_hypo_json())
    agent = HypothesisAgent(spy)
    agent.propose(ledger_rows=[])
    assert "本轮主题限定" not in spy.last_system


def test_propose_with_theme_injects_mean_reversion_constraint():
    spy = _SpyLLMClient(_good_hypo_json())
    agent = HypothesisAgent(spy, theme="均值回归")
    agent.propose(ledger_rows=[])
    assert "本轮主题限定: 均值回归" in spy.last_system
    assert "rank_within_sector" in spy.last_system
    assert "ma_bias" in spy.last_system
