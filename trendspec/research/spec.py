"""兼容 re-export：实现见 trendspec.combo.spec。"""

from trendspec.combo.spec import (
    FILTER_OP_NAMES,
    FactorSpec,
    FactorTerm,
    FilterTerm,
    parse_research_eval_spec,
)

__all__ = [
    "FILTER_OP_NAMES",
    "FactorSpec",
    "FactorTerm",
    "FilterTerm",
    "parse_research_eval_spec",
]
