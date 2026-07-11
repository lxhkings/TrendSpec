"""声明式因子组合 spec。AI 每轮输出此结构；FactorStrategy 解释执行。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from trendspec.factors.registry import list_factors


class FactorTerm(BaseModel):
    """单个因子项。"""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    direction: Literal["high", "low"]
    weight: float = 1.0

    @field_validator("name")
    @classmethod
    def _name_registered(cls, v: str) -> str:
        if v not in list_factors():
            raise ValueError(f"未注册因子: {v}（可用: {', '.join(list_factors())}）")
        return v


class FactorSpec(BaseModel):
    """声明式因子组合策略 spec。"""

    market: str
    factors: list[FactorTerm] = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    top_pct: float | None = Field(default=None, gt=0, le=1)
    """可选：按每组（或全局，group_by 未设时）当日 PIT 候选数量的百分比取
    top（如 0.05 = 前 5%），随 universe 每日浮动动态计算，而非固定数量。
    与 top_k 二选一。"""
    rebalance: int = Field(gt=0)
    rationale: str = ""
    sector_filter: list[str] | None = None
    """可选：限定排名/持仓的行业代码列表（申万一级/GICS，market 对应体系）。
    None = 不过滤，全市场参与排名。"""
    group_by: dict[str, list[str]] | None = None
    """可选：{组名: [细分行业名...]}。设置后 FactorStrategy 按组分别选
    top_k（每组前 top_k 支），而不是全局排名取 top_k。None = 保持原有
    全局模式，向后兼容。"""
    winsorize_pct: float = 0.01
    """入模前每个因子的双侧截断分位数（默认 1%/99%），组内计算。"""

    @model_validator(mode="after")
    def _exactly_one_of_top_k_top_pct(self) -> "FactorSpec":
        if (self.top_k is None) == (self.top_pct is None):
            raise ValueError("top_k 和 top_pct 必须二选一（恰好设置一个）")
        return self
