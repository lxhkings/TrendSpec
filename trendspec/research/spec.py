"""声明式因子组合 spec。AI 每轮输出此结构；FactorStrategy 解释执行。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
    top_k: int = Field(gt=0)
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
