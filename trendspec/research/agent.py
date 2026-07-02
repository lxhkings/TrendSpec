"""假设生成 agent：纯假设生成器，无 tool-calling。"""

import inspect
import json
import re
from typing import Any

from trendspec.factors.registry import factor_info, get_factor_class, list_factors
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

_MEAN_REVERSION_CONSTRAINT = """

本轮主题限定: {theme}
只允许提出以下三类均值回归假设之一（禁止动量追涨假设）：
1. 短期反转: factors 用 momentum 或 returns，period 取 5-20 交易日，direction=low
2. 均线偏离回归: factors 用 ma_bias，period 任意，direction=low
3. 行业中性反转: factors 用 rank_within_sector 或 demean_by_sector，
   param_grid 只允许给 factor_name（候选值从 momentum/returns 里选），direction=low，
   不要给 period/market/ascending/root 等其它参数——这两个因子本身没有 period，
   包的是哪个因子就用哪个因子的默认参数（market 会按当前市场自动注入）"""


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
        name = f.get("name")
        if name not in registered:
            raise HypothesisParseError(f"未注册因子: {name}")
        cls = get_factor_class(name)
        valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        given = set((f.get("param_grid") or {}).keys())
        bad = given - valid_params
        if bad:
            raise HypothesisParseError(
                f"因子 {name} 不支持参数 {sorted(bad)}（支持: {sorted(valid_params)}）"
            )
    if not hypo.get("top_k_grid") or not hypo.get("rebalance_grid"):
        raise HypothesisParseError("缺 top_k_grid / rebalance_grid")


class HypothesisAgent:
    def __init__(self, client: LLMClient, theme: str | None = None) -> None:
        self._client = client
        self._theme = theme

    def propose(self, ledger_rows: list[dict]) -> dict[str, Any]:
        system = _SYSTEM_TMPL.format(catalog=_factor_catalog())
        if self._theme:
            system += _MEAN_REVERSION_CONSTRAINT.format(theme=self._theme)
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
