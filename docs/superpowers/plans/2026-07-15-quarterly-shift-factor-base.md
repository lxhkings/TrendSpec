# trend `_QuarterlyShiftFactor` 压扁 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用参数化基类压扁 `trend.py` 中六个跨季 Factor 的重复 `compute()`，注册名与数值语义完全不变。

**Architecture:** 在 `trendspec/factors/fundamental/trend.py` 增加模块内 `_QuarterlyShiftFactor(Factor)`，用 ClassVar 描述 `value_col` / `n` / gaps / `mode` / `cagr_years` / `anchor_shift`。唯一一份 `compute()` 调用既有 `_quarterly_shift_compute` + `_asof_join_quarterly_result`。六个 `@register` 类变为只含 description + classvar 的薄壳。

**Tech Stack:** Python, Polars, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-15-research-pipeline-structure-cleanup-design.md` §B

## Global Constraints

- 行为冻结：六个注册名与数值不变；不改 `_quarterly_*` 算法。
- 不改公开 API / 因子注册表键名。
- 不碰 ingest、CN/US 列名统一。
- 现有 `tests/test_trend_factors.py` 全绿即主验收。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/factors/fundamental/trend.py` | 基类 + 六个薄壳 |
| `tests/test_trend_factors.py` | 回归（已有）；可选补「子类无独立 compute 大段」结构测 |

### 注册名 → ClassVar 映射（实现时必须一致）

| 注册名 | value_col | n | gap_min | gap_max | mode | cagr_years | anchor_shift |
|--------|-----------|---|---------|---------|------|------------|--------------|
| `fund_revenue_qoq` | total_revenue | 1 | 2 | 4 | ratio | — | 0 |
| `fund_revenue_qoq_prev` | total_revenue | 1 | 2 | 4 | ratio | — | 1 |
| `fund_net_income_qoq` | net_income | 1 | 2 | 4 | ratio | — | 0 |
| `fund_net_income_qoq_prev` | net_income | 1 | 2 | 4 | ratio | — | 1 |
| `fund_revenue_cagr_3y` | total_revenue | 12 | 34 | 38 | cagr | 3.0 | 0 |
| `fund_roe_trend_4q` | roe | 4 | 10 | 14 | diff | — | 0 |

---

### Task 1: 引入 `_QuarterlyShiftFactor` 并改写六个类

**Files:**
- Modify: `trendspec/factors/fundamental/trend.py`
- Test: `tests/test_trend_factors.py` (existing)

**Interfaces:**
- Produces: `_QuarterlyShiftFactor` with ClassVars + single `compute`
- Consumes: `_quarterly_shift_compute`, `_asof_join_quarterly_result`

- [ ] **Step 1: Run baseline tests (must be green before edit)**

Run: `uv run pytest tests/test_trend_factors.py -q`

Expected: PASS（全部绿——这是重构前基线）

- [ ] **Step 2: Write a structural failing test (optional but recommended)**

Append to `tests/test_trend_factors.py`:

```python
import inspect

from trendspec.factors.fundamental import trend as trend_mod
from trendspec.factors.registry import get_factor


def test_quarterly_factors_share_base_compute():
    """六个跨季因子应共用基类 compute，而不是各自拷贝方法体。"""
    names = [
        "fund_revenue_qoq",
        "fund_revenue_qoq_prev",
        "fund_net_income_qoq",
        "fund_net_income_qoq_prev",
        "fund_revenue_cagr_3y",
        "fund_roe_trend_4q",
    ]
    base = trend_mod._QuarterlyShiftFactor
    methods = []
    for n in names:
        cls = type(get_factor(n))
        assert issubclass(cls, base), f"{n} should subclass _QuarterlyShiftFactor"
        # 子类不应再 override compute（MRO 上 compute 来自 base）
        assert "compute" not in cls.__dict__, f"{n} should not define its own compute"
        methods.append(inspect.getattr_static(cls, "compute"))
    assert len(set(methods)) == 1
```

- [ ] **Step 3: Run structural test to verify it fails**

Run: `uv run pytest tests/test_trend_factors.py::test_quarterly_factors_share_base_compute -xvs`

Expected: FAIL（`_QuarterlyShiftFactor` 不存在或子类仍有自己的 `compute`）

- [ ] **Step 4: Implement base class and thin shells**

Replace the six Factor class blocks in `trendspec/factors/fundamental/trend.py`（从第一个 `@register("fund_revenue_qoq")` 到文件末尾）为：

```python
class _QuarterlyShiftFactor(Factor):
    """跨季 shift 因子共用壳：ClassVar 描述列与 gap，compute 只写一次。"""

    category: ClassVar[str] = "fundamental"
    value_col: ClassVar[str] = ""
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    cagr_years: ClassVar[float | None] = None
    anchor_shift: ClassVar[int] = 0

    def compute(self, df: pl.DataFrame) -> pl.Expr | pl.Series:
        if "end_date" not in df.columns or self.value_col not in df.columns:
            return pl.lit(None, dtype=pl.Float64)
        result = _quarterly_shift_compute(
            df,
            self.value_col,
            n=self.n,
            gap_min_months=self.gap_min_months,
            gap_max_months=self.gap_max_months,
            mode=self.mode,
            cagr_years=self.cagr_years,
            anchor_shift=self.anchor_shift,
        )
        if result.is_empty():
            return pl.lit(None, dtype=pl.Float64)
        return _asof_join_quarterly_result(df, result).alias(self.name)


@register("fund_revenue_qoq")
class FundRevenueQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Revenue QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "total_revenue"
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    anchor_shift: ClassVar[int] = 0


@register("fund_revenue_qoq_prev")
class FundRevenueQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Revenue QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "total_revenue"
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    anchor_shift: ClassVar[int] = 1


@register("fund_net_income_qoq")
class FundNetIncomeQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Net income QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "net_income"
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    anchor_shift: ClassVar[int] = 0


@register("fund_net_income_qoq_prev")
class FundNetIncomeQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Net income QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "net_income"
    n: ClassVar[int] = 1
    gap_min_months: ClassVar[int] = 2
    gap_max_months: ClassVar[int] = 4
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
    anchor_shift: ClassVar[int] = 1


@register("fund_revenue_cagr_3y")
class FundRevenueCagr3Y(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Revenue 3-year CAGR (12 quarters back, PIT)"
    value_col: ClassVar[str] = "total_revenue"
    n: ClassVar[int] = 12
    gap_min_months: ClassVar[int] = 34
    gap_max_months: ClassVar[int] = 38
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "cagr"
    cagr_years: ClassVar[float | None] = 3.0
    anchor_shift: ClassVar[int] = 0


@register("fund_roe_trend_4q")
class FundRoeTrend4Q(_QuarterlyShiftFactor):
    description: ClassVar[str] = "ROE change vs 4 quarters ago, absolute points not ratio (PIT)"
    value_col: ClassVar[str] = "roe"
    n: ClassVar[int] = 4
    gap_min_months: ClassVar[int] = 10
    gap_max_months: ClassVar[int] = 14
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "diff"
    anchor_shift: ClassVar[int] = 0
```

Keep the module docstring; update the line that says helpers are shared by all Factors to also mention `_QuarterlyShiftFactor`.

Do **not** change `_quarterly_series` / `_quarterly_shift_compute` / `_asof_join_quarterly_result`.

- [ ] **Step 5: Run full trend tests**

Run: `uv run pytest tests/test_trend_factors.py -xvs`

Expected: PASS（含结构性测 + 原有数值测）

- [ ] **Step 6: Commit**

```bash
git add trendspec/factors/fundamental/trend.py tests/test_trend_factors.py
git commit -m "refactor(factors): collapse quarterly shift factors into base class"
```

---

## Plan complete criteria

- [ ] `_QuarterlyShiftFactor.compute` 是唯一实现
- [ ] 六个注册名仍可用且 `tests/test_trend_factors.py` 全绿
- [ ] 子类 `__dict__` 无独立 `compute`
