"""因子组合运行时：声明式 spec + 截面组合打分。"""

from trendspec.combo.scores import compute_combo_scores
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
    "compute_combo_scores",
    "parse_research_eval_spec",
]
