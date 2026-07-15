# compute_combo_scores 同阶段 memoize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `compute_combo_scores` 的 **filter 阶段** 与 **score 阶段** 各自对 `(name, params)` memoize `compute_full`，消除同阶段重复计算；**不**跨阶段复用 cache（行为冻结）。

**Architecture:** `_apply_filters` 增加可选/内部 memo dict，或在函数内自建 cache。`compute_combo_scores` 在 filter 后、factors 循环另建 cache。Key 复用模块已有 `_key(name, params)`。公开签名不变。

**Tech Stack:** Python, Polars, pytest, monkeypatch, uv。

**Spec:** `docs/superpowers/specs/2026-07-15-research-pipeline-structure-cleanup-design.md` §D

## Global Constraints

- 行为冻结：filter 语义、z-score survivor 语义、公开签名不变。
- **禁止** filter 阶段与 score 阶段共用同一 cache 条目。
- 同一因子同时出现在 filters 与 factors 时，允许最多 2 次 `compute_full`（每阶段一次）。
- 同阶段内：两 filter 同 key → 1 次；两 factor 同 key → 1 次。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/research/factor_cache.py` | `_apply_filters` + `compute_combo_scores` memo |
| `tests/research/test_factor_cache.py` | 调用次数测 + 既有语义回归 |

---

### Task 1: 同阶段 memo 单测（先红）

**Files:**
- Modify: `tests/research/test_factor_cache.py`

**Interfaces:**
- Consumes: `compute_combo_scores`, `get_factor_with_market` (monkeypatch spy)
- Produces: 规格锁定的调用次数断言

- [ ] **Step 1: Write the failing tests**

Append to `tests/research/test_factor_cache.py`:

```python
import trendspec.research.factor_cache as factor_cache_module
from trendspec.factors.base import FactorResult


def test_apply_filters_same_factor_twice_computes_once(monkeypatch):
    """两个 filter 使用同一 name+params 时，filter 阶段只 compute_full 一次。"""
    df = _panel_with_margin()
    calls: list[tuple] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append((name, tuple(sorted((params or {}).items())), id(frame)))
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    filters = [
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0},
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": -1.0},  # 同 key，更松
    ]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    assert score.height > 0
    filter_calls = [c for c in calls if c[0] == "fund_op_margin"]
    assert len(filter_calls) == 1, f"expected 1 filter-stage compute, got {filter_calls}"


def test_score_stage_duplicate_factors_compute_once(monkeypatch):
    """factors 列表两个相同 name+params 时，score 阶段只 compute_full 一次。"""
    df = _panel()
    calls: list[str] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append(name)
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 0.5},
        {"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 0.5},
    ]
    score = compute_combo_scores(df, factors, market="us")
    assert score.height > 0
    assert calls.count("momentum") == 1


def test_filter_and_score_same_factor_may_compute_twice(monkeypatch):
    """同因子既在 filters 又在 factors：允许每阶段各一次（共 2），禁止跨阶段强制 1 次。"""
    df = _panel_with_margin()
    calls: list[str] = []
    real_gfm = factor_cache_module.get_factor_with_market

    def spy(name, params, market):
        factor = real_gfm(name, params, market)
        real_full = factor.compute_full

        def tracked_full(frame):
            calls.append(name)
            return real_full(frame)

        factor.compute_full = tracked_full  # type: ignore[method-assign]
        return factor

    monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)

    factors = [
        {"name": "fund_op_margin", "params": {}, "direction": "high", "weight": 1.0},
    ]
    filters = [
        {"name": "fund_op_margin", "params": {}, "op": ">", "value": 0.0},
    ]
    score = compute_combo_scores(df, factors, market="cn", filters=filters)
    assert score.height > 0
    # 行为冻结下允许 2；实现 memo 后仍应为 2，不能误改成 1 若那样会改截面语义
    assert calls.count("fund_op_margin") == 2
```

依赖：文件中已有 `_panel_with_margin`（filters 测引入）。若当前分支没有该 helper，先确认 `tests/research/test_factor_cache.py` 含 `_panel_with_margin` 与 filters 测；没有则从同文件既有 filters 测复制，或使用：

```python
def _panel_with_margin():
    df = _panel()
    extra_rows = []
    for i in range(40):
        d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
        extra_rows.append({
            "instrument_id": "D", "date": d,
            "open": 40.0 + i, "high": 40.0 + i + 1,
            "low": 40.0 + i - 1, "close": 40.0 + i, "volume": 1000 + i,
            "ticker": "D",
        })
    df = pl.concat([df, pl.DataFrame(extra_rows)])
    margin = {"A": 10.0, "B": -5.0, "C": None, "D": 8.0}
    return df.with_columns(
        pl.col("instrument_id").replace_strict(margin, default=None).alias("op_margin")
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/research/test_factor_cache.py::test_apply_filters_same_factor_twice_computes_once \
  tests/research/test_factor_cache.py::test_score_stage_duplicate_factors_compute_once \
  tests/research/test_factor_cache.py::test_filter_and_score_same_factor_may_compute_twice -xvs
```

Expected:
- `same_factor_twice` FAIL：filter 调用 2 次而非 1
- `duplicate_factors` FAIL：momentum 调用 2 次而非 1
- `filter_and_score` 可能已 PASS（当前无 memo 时已是 2）

- [ ] **Step 3: Implement same-stage memoize**

Edit `trendspec/research/factor_cache.py`.

1. Change `_apply_filters` to accept and use an optional cache, or always build one internally:

```python
def _apply_filters(
    df: pl.DataFrame,
    filters: list[dict[str, Any]],
    market: str,
    cache: dict[tuple, Any] | None = None,
) -> pl.DataFrame:
    """按 filters 逐条 semi-join 剔除不合格 (instrument_id, date) 行。

    Polars 比较遇 null 结果为 null，被 filter 丢弃——缺失值自然落入
    "剔除"分支，与 FilterTerm 的语义一致。

    cache: 可选 memo，key 与模块 _key 相同，value 为 FactorResult。
    仅用于本阶段（同一 df 快照）内去重。
    """
    if cache is None:
        cache = {}
    for term in filters:
        k = _key(term["name"], term.get("params") or {})
        result = cache.get(k)
        if result is None:
            factor = get_factor_with_market(
                term["name"], term.get("params") or {}, market
            )
            result = factor.compute_full(df)
            cache[k] = result
        cond = _FILTER_OPS[term["op"]](pl.col(result.name), term["value"])
        passed = result.values.filter(cond).select(["instrument_id", "date"])
        df = df.join(passed, on=["instrument_id", "date"], how="semi")
    return df
```

Note: `_key` is currently defined **below** `compute_combo_scores`. Move `_key` **above** `_apply_filters` (or keep order and ensure `_key` is defined before use — in Python function body resolves at call time so defining `_key` later in module is OK).

2. In `compute_combo_scores`, replace the factors loop compute with memo, and pass a **fresh** filter cache:

```python
    if filters:
        filter_cache: dict[tuple, Any] = {}
        df = _apply_filters(df, filters, market, cache=filter_cache)

    # ... group_by block unchanged ...

    score_cache: dict[tuple, Any] = {}
    weight_cols: list[pl.Expr] = []
    missing_any = pl.lit(False)
    for i, term in enumerate(factors):
        k = _key(term["name"], term.get("params") or {})
        result = score_cache.get(k)
        if result is None:
            factor = get_factor_with_market(
                term["name"], term.get("params") or {}, market
            )
            result = factor.compute_full(df)
            score_cache[k] = result
        col = result.name
        # ... rest of z-score loop unchanged, still using result ...
```

Do **not** pass `filter_cache` into score stage.

- [ ] **Step 4: Run memo + semantic tests**

Run:

```bash
uv run pytest tests/research/test_factor_cache.py -xvs
uv run pytest tests/research/test_factor_strategy.py -q --tb=line
```

Expected: PASS（含 filters 语义测、调用次数测）

- [ ] **Step 5: Commit**

```bash
git add trendspec/research/factor_cache.py tests/research/test_factor_cache.py
git commit -m "perf(research): same-stage memoize compute_full in combo scores"
```

---

## Plan complete criteria

- [ ] 同阶段重复 `(name, params)` → 单次 `compute_full`
- [ ] filter+score 同名 → 恰好 2 次
- [ ] 既有 filter survivor / null / z 对称测仍绿

---

## 四 plan 执行顺序（提醒）

| 顺序 | Plan 文件 |
|------|-----------|
| 1 | `2026-07-15-enrich-daily-panel.md` |
| 2 | `2026-07-15-quarterly-shift-factor-base.md` |
| 3 | `2026-07-15-research-cmd-filter-validation.md` |
| 4 | `2026-07-15-combo-scores-same-stage-memoize.md` |

1∥2 可并行；3、4 彼此独立，建议 filter 语义稳定后再做 4。
