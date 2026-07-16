# Quarterly ClassVar 去复述（Plan 5）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `_QuarterlyShiftFactor` 子类只声明相对基类默认值的差异 ClassVar；注册名与数值语义不变。enrich 脆测若 Plan 1 已删则本 plan 无额外动作。

**Architecture:** 仅改 `trendspec/factors/fundamental/trend.py` 六个薄壳类体；基类默认值保持：`n=1`, `gap_min_months=2`, `gap_max_months=4`, `mode="ratio"`, `cagr_years=None`, `anchor_shift=0`。

**Tech Stack:** Python, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-16-combo-boundary-cleanup-design.md` §Plan 5

## Global Constraints

- 行为冻结：注册名、因子数值、gap/mode/anchor 语义不变。
- 不改 `_quarterly_series` / `_quarterly_shift_compute` / `_asof_join_quarterly_result` / 基类 `compute`。
- 不改 enrich 生产代码。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/factors/fundamental/trend.py` | 六个 Factor 类 ClassVar 精简 |
| `tests/test_trend_factors.py` | 回归（含 `test_quarterly_factors_share_base_compute`） |

---

### Task 1: 精简六个子类 ClassVar

**Files:**
- Modify: `trendspec/factors/fundamental/trend.py`
- Test: `tests/test_trend_factors.py`

**Interfaces:**
- 基类保持：
  ```python
  class _QuarterlyShiftFactor(Factor):
      category: ClassVar[str] = "fundamental"
      value_col: ClassVar[str] = ""
      n: ClassVar[int] = 1
      gap_min_months: ClassVar[int] = 2
      gap_max_months: ClassVar[int] = 4
      mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "ratio"
      cagr_years: ClassVar[float | None] = None
      anchor_shift: ClassVar[int] = 0
  ```

- [ ] **Step 1: 将六个注册类改为仅差异字段**

替换 `@register("fund_revenue_qoq")` 起至文件末尾为：

```python
@register("fund_revenue_qoq")
class FundRevenueQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Revenue QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "total_revenue"


@register("fund_revenue_qoq_prev")
class FundRevenueQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Revenue QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "total_revenue"
    anchor_shift: ClassVar[int] = 1


@register("fund_net_income_qoq")
class FundNetIncomeQoQ(_QuarterlyShiftFactor):
    description: ClassVar[str] = "Net income QoQ growth (quarter vs immediately prior quarter, PIT)"
    value_col: ClassVar[str] = "net_income"


@register("fund_net_income_qoq_prev")
class FundNetIncomeQoQPrev(_QuarterlyShiftFactor):
    description: ClassVar[str] = (
        "Net income QoQ growth for the quarter prior to the latest one (t-1 vs t-2, PIT)"
    )
    value_col: ClassVar[str] = "net_income"
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


@register("fund_roe_trend_4q")
class FundRoeTrend4Q(_QuarterlyShiftFactor):
    description: ClassVar[str] = "ROE change vs 4 quarters ago, absolute points not ratio (PIT)"
    value_col: ClassVar[str] = "roe"
    n: ClassVar[int] = 4
    gap_min_months: ClassVar[int] = 10
    gap_max_months: ClassVar[int] = 14
    mode: ClassVar[Literal["ratio", "cagr", "diff"]] = "diff"
```

注意：`FundRevenueCagr3Y` / `FundRoeTrend4Q` 不再写 `anchor_shift=0`（继承默认）。

- [ ] **Step 2: 可选加固测 — ClassVar 解析正确**

若尚无覆盖 anchor_shift 继承的测，可在 `tests/test_trend_factors.py` 追加：

```python
def test_qoq_prev_anchor_shift_is_one():
    from trendspec.factors.registry import get_factor
    assert type(get_factor("fund_revenue_qoq_prev")).anchor_shift == 1
    assert type(get_factor("fund_revenue_qoq")).anchor_shift == 0
    assert type(get_factor("fund_revenue_cagr_3y")).cagr_years == 3.0
    assert type(get_factor("fund_roe_trend_4q")).mode == "diff"
```

- [ ] **Step 3: 跑测**

Run:
```bash
uv run pytest tests/test_trend_factors.py -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add trendspec/factors/fundamental/trend.py tests/test_trend_factors.py
git commit -m "$(cat <<'EOF'
refactor(factors): drop redundant ClassVars on quarterly shift factors

Subclasses only override fields that differ from _QuarterlyShiftFactor defaults.
EOF
)"
```

---

### Task 2: enrich 脆测确认

**Files:**
- Read: `tests/test_fundamentals_merge.py`

- [ ] **Step 1:** 确认 `test_call_sites_use_enrich_daily_panel_not_inline_merges` 已不存在（Plan 1 Task 4）。若仍在，按 Plan 1 Task 4 删除并单独 commit。

- [ ] **Step 2:** 确认 `test_enrich_daily_panel_empty_returns_unchanged` 仍在。

Run:
```bash
uv run pytest tests/test_fundamentals_merge.py -q
```
Expected: PASS

---

### Plan 5 完成检查

```bash
uv run pytest tests/test_trend_factors.py tests/test_fundamentals_merge.py -q
```

- [ ] 六个因子仍 subclass `_QuarterlyShiftFactor` 且不 override `compute`
- [ ] ClassVar 无与基类相同的冗余复述（目视）
- [ ] 无源码扫描 enrich 测
