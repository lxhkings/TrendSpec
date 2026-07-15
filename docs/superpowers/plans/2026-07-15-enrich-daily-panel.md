# enrich_daily_panel 抽取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `BaseEngine.load_data` 与 `MarketPanel.load` 中重复的 fundamentals/valuation best-effort merge 抽成 `enrich_daily_panel`，行为完全不变。

**Architecture:** 在 `trendspec/data/fundamentals.py` 新增编排函数 `enrich_daily_panel(daily, market, root)`：空表直接返回；否则依次 try/merge fundamentals 与 valuation，`except Exception: pass`。两处调用方各改为一行调用。不包含 `bars()`，不碰 weekly 路径。

**Tech Stack:** Python, Polars, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-15-research-pipeline-structure-cleanup-design.md` §A

## Global Constraints

- 行为冻结：可观测结果不变；保留 `except Exception: pass`；不新增日志。
- 不把 `bars()` 并入 enrich；不改 weekly best-effort 块。
- 不新建顶级模块；函数落在 `data/fundamentals.py`。
- `/docs` 在 `.gitignore` 中；若改 plan 文件需 `git add -f`。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/data/fundamentals.py` | 新增 `enrich_daily_panel` |
| `trendspec/engine/base_engine.py` | `load_data` 改用 enrich |
| `trendspec/research/market_panel.py` | `load` 改用 enrich |
| `tests/test_fundamentals_merge.py` | 新增 enrich 单测 |

---

### Task 1: `enrich_daily_panel` + 空表单测

**Files:**
- Modify: `trendspec/data/fundamentals.py`
- Modify: `tests/test_fundamentals_merge.py`

**Interfaces:**
- Produces: `enrich_daily_panel(daily: pl.DataFrame, market: Market, root: str | None) -> pl.DataFrame`
- Consumes: existing `merge_fundamentals`, `merge_valuation`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fundamentals_merge.py`:

```python
from trendspec.data.fundamentals import enrich_daily_panel, merge_fundamentals_frame
from trendspec.data.markets import Market


def test_enrich_daily_panel_empty_returns_unchanged():
    empty = pl.DataFrame(
        schema={
            "instrument_id": pl.Utf8,
            "date": pl.Date,
            "close": pl.Float64,
        }
    )
    out = enrich_daily_panel(empty, Market.CN, root=None)
    assert out.is_empty()
    assert out.equals(empty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fundamentals_merge.py::test_enrich_daily_panel_empty_returns_unchanged -xvs`

Expected: FAIL with `ImportError` or `cannot import name 'enrich_daily_panel'`

- [ ] **Step 3: Write minimal implementation**

Append to `trendspec/data/fundamentals.py` (after `merge_valuation`):

```python
def enrich_daily_panel(
    daily: pl.DataFrame, market: Market, root: str | None
) -> pl.DataFrame:
    """Best-effort fundamentals + valuation PIT merge onto a daily OHLCV frame.

    Mirrors the historical BaseEngine.load_data / MarketPanel.load inline blocks:
    empty input is returned as-is; each merge is wrapped in bare
    ``except Exception: pass`` so a missing dataset or merge error never fails load.
    Does not call ``bars()`` — callers load OHLCV first, then enrich.
    """
    if daily.is_empty():
        return daily
    try:
        daily = merge_fundamentals(daily, market, root)
    except Exception:
        pass
    try:
        daily = merge_valuation(daily, market, root)
    except Exception:
        pass
    return daily
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fundamentals_merge.py -xvs`

Expected: PASS (全部，含原有 merge 测 + 新测)

- [ ] **Step 5: Commit**

```bash
git add trendspec/data/fundamentals.py tests/test_fundamentals_merge.py
git commit -m "feat(data): add enrich_daily_panel for shared fund/val merge"
```

---

### Task 2: 接线 BaseEngine + MarketPanel

**Files:**
- Modify: `trendspec/engine/base_engine.py` (load_data 内 merge 块，约 L187–202；import 约 L30)
- Modify: `trendspec/research/market_panel.py` (L7 import；L30–41 merge 块)

**Interfaces:**
- Consumes: `enrich_daily_panel(daily, market, root) -> pl.DataFrame`
- Produces: 两处 load 行为与抽取前一致

- [ ] **Step 1: Write the failing regression check (call-site shape)**

本任务是接线重构。先加一条轻量测试，断言 `MarketPanel` / engine 源码不再直接调用 `merge_fundamentals`（避免只改一半）。追加到 `tests/test_fundamentals_merge.py`：

```python
from pathlib import Path


def test_call_sites_use_enrich_daily_panel_not_inline_merges():
    root = Path(__file__).resolve().parents[1]
    engine = (root / "trendspec/engine/base_engine.py").read_text()
    panel = (root / "trendspec/research/market_panel.py").read_text()
    assert "enrich_daily_panel" in engine
    assert "enrich_daily_panel" in panel
    # load 路径不应再内联调用这两个名字（import 行除外：允许 from ... import enrich only）
    assert "merge_fundamentals(" not in engine
    assert "merge_valuation(" not in engine
    assert "merge_fundamentals(" not in panel
    assert "merge_valuation(" not in panel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fundamentals_merge.py::test_call_sites_use_enrich_daily_panel_not_inline_merges -xvs`

Expected: FAIL（源码仍是 inline merge）

- [ ] **Step 3: Wire BaseEngine**

In `trendspec/engine/base_engine.py`:

1. Change import from:
   `from trendspec.data.fundamentals import merge_fundamentals, merge_valuation`
   to:
   `from trendspec.data.fundamentals import enrich_daily_panel`

2. Replace the two try/merge blocks after `bars(...)` assignment to `self._data` with:

```python
            self._data = enrich_daily_panel(
                self._data, self.config.market, self.root
            )
```

Leave the weekly `bars(... frequency="weekly")` try/except block unchanged.

- [ ] **Step 4: Wire MarketPanel**

In `trendspec/research/market_panel.py`:

1. Change import to:
   `from trendspec.data.fundamentals import enrich_daily_panel`

2. Replace the try/merge block after `bars(...)` with:

```python
        df = enrich_daily_panel(df, m, root)
```

Full `load` body should look like:

```python
        m = Market(market.upper())
        df = bars(market=m, start_date=start, end_date=end,
                  adjustment_mode=adjustment_mode, root=root)
        df = enrich_daily_panel(df, m, root)
        uni = get_universe(m, root)
        return cls(data=df, universe=uni)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_fundamentals_merge.py -xvs
uv run pytest tests/research/ tests/test_fundamental_factors.py -q --tb=line
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add trendspec/engine/base_engine.py trendspec/research/market_panel.py tests/test_fundamentals_merge.py
git commit -m "refactor: BaseEngine and MarketPanel use enrich_daily_panel"
```

---

## Plan complete criteria

- [ ] `enrich_daily_panel` 存在且空表原样返回
- [ ] `base_engine.py` / `market_panel.py` 无 `merge_fundamentals(` / `merge_valuation(` 调用
- [ ] 相关 pytest 绿
