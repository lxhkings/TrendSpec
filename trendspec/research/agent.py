"""假设生成 agent：纯假设生成器，无 tool-calling。"""

import json
import re
from typing import Any

from trendspec.factors.registry import factor_info, list_factors
from trendspec.research.llm_client import LLMClient


class HypothesisParseError(Exception):
    pass


_SYSTEM_TMPL = """你是量化因子研究助手。只能使用下列已注册因子，禁止杜撰：
{catalog}

输出严格 JSON（不要多余文字），结构：
{{
  "market": "us",
  "factors": [
    {{"name": <因子名>, "direction": "high"|"low", "weight": <float>,
      "param_grid": {{<参数名>: [<候选值>...]}}}}
  ],
  "top_k_grid": [<int>...],
  "rebalance_grid": [<int>...],
  "rationale": "<市场逻辑一句话>"
}}
direction=high 表示因子值越大越优选，low 反之。"""


def _factor_catalog() -> str:
    lines = []
    for name in list_factors():
        info = factor_info(name) or {}
        lines.append(f"- {name} ({info.get('category', '')}): {info.get('description', '')}")
    return "\n".join(lines)


def _ledger_summary(ledger_rows: list[dict]) -> str:
    if not ledger_rows:
        return "（无历史，给出第一个假设）"
    parts = []
    for r in ledger_rows[-10:]:
        best = (r.get("top_candidates") or [{}])[0]
        parts.append(
            f"轮{r.get('round')}: {r.get('hypothesis', {}).get('rationale', '')} "
            f"→ 最佳 OOS Sharpe={best.get('oos_sharpe', 'NA')}"
        )
    return "\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise HypothesisParseError(f"无 JSON: {text[:120]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise HypothesisParseError(str(e)) from e


def _validate(hypo: dict) -> None:
    factors = hypo.get("factors")
    if not factors:
        raise HypothesisParseError("factors 为空")
    registered = set(list_factors())
    for f in factors:
        if f.get("name") not in registered:
            raise HypothesisParseError(f"未注册因子: {f.get('name')}")
    if not hypo.get("top_k_grid") or not hypo.get("rebalance_grid"):
        raise HypothesisParseError("缺 top_k_grid / rebalance_grid")


class HypothesisAgent:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def propose(self, ledger_rows: list[dict]) -> dict[str, Any]:
        system = _SYSTEM_TMPL.format(catalog=_factor_catalog())
        user = "历史研究记录：\n" + _ledger_summary(ledger_rows) + "\n\n给出下一个未试过的假设。"

        last_err: Exception | None = None
        for _ in range(2):
            text = self._client.complete(system, user)
            try:
                hypo = _extract_json(text)
                _validate(hypo)
                return hypo
            except HypothesisParseError as e:
                last_err = e
                user = user + f"\n\n上次输出无法解析（{e}）。请只输出合法 JSON。"
        raise HypothesisParseError(f"重试后仍失败: {last_err}")
