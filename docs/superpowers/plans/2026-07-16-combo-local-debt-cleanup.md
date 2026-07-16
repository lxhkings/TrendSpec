# Combo 本地去债（Plan 1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在搬家前收口本地债：单一 memo helper、删除 `FactorCache`、filter op 与 `FilterTerm` 同源、`parse_research_eval_spec` 用 `extra="ignore"`、去掉 enrich 源码扫描脆测。

**Architecture:** 仍在 `research/spec.py` 与 `research/factor_cache.py` 内完成；不新建 `combo/`。为 Plan 2 搬家留下干净实现。

**Tech Stack:** Python 3.13, Pydantic v2, Polars, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-16-combo-boundary-cleanup-design.md` §Plan 1

## Global Constraints

- 严格行为冻结：合法路径分数/filter/IC 语义不变；enrich 仍 `except Exception: pass`。
- **禁止** filter 与 score 跨阶段共用 cache。
- 删除 `FactorCache`；生产仅 `_compute_full_cached` + 阶段内 dict。
- 不新建顶级模块；不改 `compute_combo_scores` 公开签名。
- 本 plan 不做 import 搬家。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/research/spec.py` | `FILTER_OP_NAMES` 常量；`FilterTerm.op`；`extra="ignore"` parse |
| `trendspec/research/factor_cache.py` | `_compute_full_cached`；同源 `_FILTER_OPS`；删 `FactorCache` |
| `tests/research/test_factor_cache.py` | 去 `FactorCache` 测；保留同阶段 memo 测 |
| `tests/research/test_spec.py` | parse 忽略多余字段 |
| `tests/test_fundamentals_merge.py` | 删源码扫描测 |

---

### Task 1: 删除 FactorCache + 统一 `_compute_full_cached`

**Files:**
- Modify: `trendspec/research/factor_cache.py`
- Modify: `tests/research/test_factor_cache.py`

**Interfaces:**
- Produces:
  ```python
  def _compute_full_cached(
      cache: dict[tuple, Any],
      name: str,
      params: dict[str, Any],
      market: str,
      df: pl.DataFrame,
  ) -> Any:  # FactorResult
      """Look up (name,params) in cache; on miss get_factor_with_market + compute_full."""
  ```
- Consumes: existing `_key`, `get_factor_with_market`, `compute_combo_scores` 签名不变

- [ ] **Step 1: 更新测试 — 去掉 FactorCache，保留 memo 行为测**

在 `tests/research/test_factor_cache.py`：

1. 将 import 改为：
```python
import trendspec.research.factor_cache as factor_cache_module
from trendspec.research.factor_cache import compute_combo_scores
```
（删除 `FactorCache` 导入）

2. **删除**整个 `test_factor_cache_memoizes_by_name_params` 函数。

3. 确认仍存在（若无则补回，内容与现仓库一致）：
   - `test_apply_filters_same_factor_twice_computes_once`
   - `test_score_stage_duplicate_factors_compute_once`
   - `test_filter_and_score_same_factor_may_compute_twice`

- [ ] **Step 2: 运行确认 FactorCache 相关已无引用**

Run:
```bash
uv run pytest tests/research/test_factor_cache.py -q
rg -n "FactorCache" trendspec tests --glob '*.py'
```
Expected: `FactorCache` 仅仍出现在 `factor_cache.py` 类定义（实现前）；测可能因仍 import 类而 fail，或删测后其余绿。

- [ ] **Step 3: 实现 `_compute_full_cached` 并替换双处拷贝；删除 `FactorCache` 与无用 `get_factor` import**

在 `trendspec/research/factor_cache.py`：

1. 删除 `from trendspec.factors.registry import get_factor, get_factor_with_market` 中的 `get_factor`，仅保留 `get_factor_with_market`。

2. 在 `_key` 之后加入：
```python
def _compute_full_cached(
    cache: dict[tuple, Any],
    name: str,
    params: dict[str, Any],
    market: str,
    df: pl.DataFrame,
) -> Any:
    """同阶段 memo：key=(name, sorted params)；miss 时 compute_full 并写入 cache。"""
    k = _key(name, params or {})
    hit = cache.get(k)
    if hit is not None:
        return hit
    factor = get_factor_with_market(name, params or {}, market)
    result = factor.compute_full(df)
    cache[k] = result
    return result
```

3. `_apply_filters` 循环体改为：
```python
    for term in filters:
        result = _compute_full_cached(
            cache,
            term["name"],
            term.get("params") or {},
            market,
            df,
        )
        cond = _FILTER_OPS[term["op"]](pl.col(result.name), term["value"])
        passed = result.values.filter(cond).select(["instrument_id", "date"])
        df = df.join(passed, on=["instrument_id", "date"], how="semi")
```

4. `compute_combo_scores` 的 factors 循环改为：
```python
    score_cache: dict[tuple, Any] = {}
    weight_cols: list[pl.Expr] = []
    missing_any = pl.lit(False)
    for i, term in enumerate(factors):
        result = _compute_full_cached(
            score_cache,
            term["name"],
            term.get("params") or {},
            market,
            df,
        )
        col = result.name
        # ... 其余 winsorize/z-score 逻辑不变
```

5. **删除**整个 `class FactorCache: ...`（文件末尾）。

6. 更新模块 docstring 首段：去掉对「FactorCache」作为主 API 的暗示；说明 memo 为 `compute_combo_scores` 内同阶段缓存。

- [ ] **Step 4: 跑测**

Run:
```bash
uv run pytest tests/research/test_factor_cache.py tests/research/test_factor_strategy_inject.py tests/research/test_fast_eval.py -q
rg -n "FactorCache" trendspec tests --glob '*.py'
```
Expected: 全绿；`FactorCache` 零命中。

- [ ] **Step 5: Commit**

```bash
git add trendspec/research/factor_cache.py tests/research/test_factor_cache.py
git commit -m "$(cat <<'EOF'
refactor(research): collapse combo memo into _compute_full_cached

Remove unused FactorCache; filter and score stages each use a private
cache dict via one helper. Same-stage dedupe preserved; no cross-stage share.
EOF
)"
```

---

### Task 2: Filter op 单一真相源

**Files:**
- Modify: `trendspec/research/spec.py`
- Modify: `trendspec/research/factor_cache.py`
- Modify: `tests/research/test_spec.py`（可选加固）

**Interfaces:**
- Produces:
  ```python
  # spec.py
  FILTER_OP_NAMES: tuple[str, ...] = (">", ">=", "<", "<=")
  # FilterTerm.op uses Literal[">", ">=", "<", "<="] matching FILTER_OP_NAMES
  ```
- Consumes: `_FILTER_OPS` 执行 map 只从 `FILTER_OP_NAMES` 生成

- [ ] **Step 1: 在 spec 定义常量并让 FilterTerm 对齐**

在 `trendspec/research/spec.py` 顶部（imports 后、`FactorTerm` 前）加入：

```python
FILTER_OP_NAMES: tuple[str, ...] = (">", ">=", "<", "<=")
```

将 `FilterTerm.op` 改为（保持四算子，与常量一致；Pydantic Literal 需字面量，可写）：

```python
    op: Literal[">", ">=", "<", "<="]
```

并在 `FilterTerm` docstring 或模块注释中写明：执行层 `_FILTER_OPS` 必须由 `FILTER_OP_NAMES` 构建。

可选单测（追加 `tests/research/test_spec.py`）：

```python
from trendspec.research.spec import FILTER_OP_NAMES
from trendspec.research.factor_cache import _FILTER_OPS

def test_filter_ops_keys_match_spec_constant():
    assert set(_FILTER_OPS.keys()) == set(FILTER_OP_NAMES)
```

- [ ] **Step 2: factor_cache 从常量建 map**

在 `trendspec/research/factor_cache.py`：

```python
from trendspec.research.spec import FILTER_OP_NAMES

def _build_filter_ops() -> dict[str, Any]:
    ops: dict[str, Any] = {
        ">": lambda col, v: col > v,
        ">=": lambda col, v: col >= v,
        "<": lambda col, v: col < v,
        "<=": lambda col, v: col <= v,
    }
    # 确保与 spec 常量同步；多/少 key 立即失败
    if set(ops) != set(FILTER_OP_NAMES):
        raise RuntimeError(
            f"_FILTER_OPS keys {set(ops)!r} != FILTER_OP_NAMES {set(FILTER_OP_NAMES)!r}"
        )
    return ops

_FILTER_OPS = _build_filter_ops()
```

注意：避免循环 import——`spec.py` **不得** import `factor_cache`。当前 `spec` 只依赖 `factors.registry`，安全。

- [ ] **Step 3: 跑测**

Run:
```bash
uv run pytest tests/research/test_spec.py tests/research/test_factor_cache.py -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add trendspec/research/spec.py trendspec/research/factor_cache.py tests/research/test_spec.py
git commit -m "$(cat <<'EOF'
refactor(research): single source for filter op names

FILTER_OP_NAMES in spec; factor_cache builds _FILTER_OPS from it and
asserts key set match.
EOF
)"
```

---

### Task 3: parse_research_eval_spec 用 extra="ignore"

**Files:**
- Modify: `trendspec/research/spec.py`
- Modify: `tests/research/test_spec.py`

**Interfaces:**
- Produces: `parse_research_eval_spec(raw: dict[str, Any]) -> dict[str, Any]` 返回形状不变
- `_ResearchEvalSpec` 增加 `model_config = ConfigDict(extra="ignore")`

- [ ] **Step 1: 写/补失败测 — 完整 FactorSpec 字段不炸 + 形状**

在 `tests/research/test_spec.py` 追加：

```python
def test_parse_research_eval_spec_ignores_full_spec_extra_fields():
    raw = {
        "market": "cn",
        "top_k": 10,
        "rebalance": 5,
        "rationale": "x",
        "factors": [
            {"name": "momentum", "params": {"period": 5}, "direction": "high"}
        ],
        "filters": [],
    }
    out = parse_research_eval_spec(raw)
    assert "market" not in out
    assert "top_k" not in out
    assert out["filters"] == []
    assert out["winsorize_pct"] == 0.01
    assert out["factors"][0]["name"] == "momentum"
```

- [ ] **Step 2: 跑测应已绿或仍绿（手搓 payload 已 ignore 多余键）**

Run:
```bash
uv run pytest tests/research/test_spec.py::test_parse_research_eval_spec_ignores_full_spec_extra_fields -q
```
Expected: PASS（当前手搓 payload 已忽略 extra；本任务简化实现后仍须 PASS）

- [ ] **Step 3: 简化实现**

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class _ResearchEvalSpec(BaseModel):
    """ic/quantile 子集：不要求 market/top_k/rebalance。"""

    model_config = ConfigDict(extra="ignore")

    factors: list[FactorTerm] = Field(min_length=1)
    filters: list[FilterTerm] = Field(default_factory=list)
    group_by: dict[str, list[str]] | None = None
    winsorize_pct: float = 0.01


def parse_research_eval_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the ic/quantile subset of a factor-spec JSON object.

    Requires non-empty ``factors`` (each a FactorTerm). Optional ``filters``
    (FilterTerm list, default []), optional ``group_by``, optional
    ``winsorize_pct`` (default 0.01). Ignores market/top_k/rebalance if present.

    Returns a plain dict with dumped factors/filters for compute_* callers.
    Raises pydantic.ValidationError on invalid input.
    """
    parsed = _ResearchEvalSpec.model_validate(raw)
    out: dict[str, Any] = {
        "factors": [t.model_dump() for t in parsed.factors],
        "filters": [t.model_dump() for t in parsed.filters],
        "winsorize_pct": parsed.winsorize_pct,
    }
    if parsed.group_by is not None:
        out["group_by"] = parsed.group_by
    return out
```

- [ ] **Step 4: 全 spec 测**

Run:
```bash
uv run pytest tests/research/test_spec.py tests/research/test_research_cmd_load.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trendspec/research/spec.py tests/research/test_spec.py
git commit -m "$(cat <<'EOF'
refactor(research): parse_research_eval_spec via extra=ignore

Drop hand-built payload; validate raw with _ResearchEvalSpec.
EOF
)"
```

---

### Task 4: 删除 enrich 源码扫描脆测

**Files:**
- Modify: `tests/test_fundamentals_merge.py`

**Interfaces:**
- 不改生产代码
- 保留 `test_enrich_daily_panel_empty_returns_unchanged` 行为测

- [ ] **Step 1: 删除脆测函数**

删除 `test_call_sites_use_enrich_daily_panel_not_inline_merges` 整函数。

若文件顶部仅因该测而 `from pathlib import Path`，且无其它引用，删除 `Path` import。

- [ ] **Step 2: 跑测**

Run:
```bash
uv run pytest tests/test_fundamentals_merge.py -q
```
Expected: PASS（剩余 merge PIT + enrich 空表）

- [ ] **Step 3: Commit**

```bash
git add tests/test_fundamentals_merge.py
git commit -m "$(cat <<'EOF'
test(data): drop source-scan architectural test for enrich

Keep behavior tests only; string asserts on call sites were brittle.
EOF
)"
```

---

### Plan 1 完成检查

```bash
uv run pytest tests/research/test_factor_cache.py tests/research/test_spec.py tests/research/test_research_cmd_load.py tests/test_fundamentals_merge.py tests/research/test_factor_strategy_inject.py -q
rg -n "FactorCache" trendspec tests --glob '*.py'   # expect no matches
rg -n "FILTER_OP_NAMES" trendspec/research
```

- [ ] 无 `FactorCache`
- [ ] `_compute_full_cached` 为唯一 get-or-compute
- [ ] filter op 同源
- [ ] parse 无手搓 payload
- [ ] 无源码扫描 enrich 测
