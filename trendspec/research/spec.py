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
