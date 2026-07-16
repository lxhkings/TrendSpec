# 新建 trendspec/combo 并迁入（Plan 2+3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `FactorSpec` 族与 `compute_combo_scores` 迁入中性包 `trendspec/combo/`，断 `strategy → research`；research 仅薄 re-export；更新 ARCHITECTURE；加边界静态测。

**Architecture:** `combo/spec.py` + `combo/scores.py` 持有实现；`research/spec.py` / `research/factor_cache.py` 显式 re-export；仓内 import 改 `trendspec.combo`。`combo` 不得依赖 strategy/research/engine/cli。

**Tech Stack:** Python, Pydantic, Polars, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-16-combo-boundary-cleanup-design.md` §Plan 2–3

**Depends on:** Plan 1 完成（memo 收口、`FactorCache` 已删、`FILTER_OP_NAMES`、parse `extra="ignore"`）。若未做 Plan 1，必须先在迁入内容中完成同等收口，禁止把双轨 `FactorCache` 搬进 combo。

## Global Constraints

- 严格行为冻结。
- `combo` 可依赖 `data` / `factors` only。
- shim 显式名单，禁止 `import *` 泄漏私有符号（`_compute_full_cached` 等可不导出）。
- 仓内新代码走 `trendspec.combo`；shim 仅兼容。
- Plan 3 边界测与本 plan 同合入。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/combo/__init__.py` | 公共 API 导出 |
| `trendspec/combo/spec.py` | 自 research/spec 迁入 |
| `trendspec/combo/scores.py` | 自 research/factor_cache 迁入（可改名） |
| `trendspec/research/spec.py` | re-export |
| `trendspec/research/factor_cache.py` | re-export `compute_combo_scores` |
| `trendspec/strategy/factor_strategy.py` | import combo |
| `trendspec/research/fast_eval.py`, `factor_eval.py` | import combo |
| `trendspec/cli/*.py`, `scripts/consumer_rank.py` | import combo |
| `tests/**` | import combo（可保留少数 shim 兼容测） |
| `ARCHITECTURE.md` | Topology + Key Class + 依赖 |
| `tests/research/test_combo_boundaries.py` | 或 `tests/test_combo_boundaries.py` 静态边界 |

---

### Task 1: 创建 combo 包（实现文件）

**Files:**
- Create: `trendspec/combo/__init__.py`
- Create: `trendspec/combo/spec.py`
- Create: `trendspec/combo/scores.py`

**Interfaces:**
- Public:
  ```python
  # trendspec.combo
  FactorTerm, FilterTerm, FactorSpec, FILTER_OP_NAMES
  parse_research_eval_spec
  compute_combo_scores
  ```

- [ ] **Step 1: 复制并调整 spec**

将 **Plan 1 完成后** 的 `trendspec/research/spec.py` 全文复制到 `trendspec/combo/spec.py`。

模块 docstring 改为：

```python
"""声明式因子组合 spec。FactorStrategy 与 research 评估共用。

实现位于 trendspec.combo；research.spec 仅为兼容 re-export。
"""
```

imports 保持 `from trendspec.factors.registry import list_factors`（合法）。

- [ ] **Step 2: 复制并调整 scores**

将 Plan 1 完成后的 `trendspec/research/factor_cache.py` 复制为 `trendspec/combo/scores.py`。

1. 模块 docstring 改为说明实现位于 `combo.scores`，为组合打分唯一实现。
2. 将：
```python
from trendspec.research.spec import FILTER_OP_NAMES
```
改为：
```python
from trendspec.combo.spec import FILTER_OP_NAMES
```
3. 删除任何对 `research` 的引用。
4. **不要** export `_compute_full_cached` / `_apply_filters` 为包公共 API（模块级私有可保留）。

- [ ] **Step 3: `__init__.py`**

```python
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
```

- [ ] **Step 4: 冒烟 import**

Run:
```bash
uv run python -c "from trendspec.combo import FactorSpec, compute_combo_scores, parse_research_eval_spec; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add trendspec/combo/
git commit -m "$(cat <<'EOF'
feat(combo): add trendspec.combo package with spec and scores

Neutral home for FactorSpec and compute_combo_scores (copied from
research after local-debt cleanup). research re-exports come next.
EOF
)"
```

---

### Task 2: research 改为薄 re-export

**Files:**
- Modify: `trendspec/research/spec.py`（整文件替换为 shim）
- Modify: `trendspec/research/factor_cache.py`（整文件替换为 shim）

**Interfaces:**
- `from trendspec.research.spec import FactorSpec` 仍可用
- `from trendspec.research.factor_cache import compute_combo_scores` 仍可用
- `FactorCache` **不可**再 import（Plan 1 已删）

- [ ] **Step 1: 替换 research/spec.py**

```python
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
```

- [ ] **Step 2: 替换 research/factor_cache.py**

```python
"""兼容 re-export：实现见 trendspec.combo.scores。"""

from trendspec.combo.scores import compute_combo_scores

__all__ = ["compute_combo_scores"]
```

- [ ] **Step 3: 跑依赖旧路径的测（应仍绿）**

Run:
```bash
uv run pytest tests/research/test_spec.py tests/research/test_factor_cache.py tests/research/test_factor_strategy_inject.py -q
```
Expected: PASS（若测试 monkeypatch `trendspec.research.factor_cache.get_factor_with_market`，**会失败**——实现已不在该模块）。

**若 monkeypatch 失败：** 将 `tests/research/test_factor_cache.py` 与 `tests/research/test_factor_strategy.py` 中的：

```python
import trendspec.research.factor_cache as factor_cache_module
...
monkeypatch.setattr(factor_cache_module, "get_factor_with_market", spy)
```

改为：

```python
import trendspec.combo.scores as scores_module
...
monkeypatch.setattr(scores_module, "get_factor_with_market", spy)
```

并对 `compute_combo_scores` 的 import 改为 `from trendspec.combo.scores import compute_combo_scores`（或 `from trendspec.combo import compute_combo_scores`）。

- [ ] **Step 4: Commit**

```bash
git add trendspec/research/spec.py trendspec/research/factor_cache.py tests/research/
git commit -m "$(cat <<'EOF'
refactor(research): re-export FactorSpec and scores from combo

Implementation lives in trendspec.combo; research keeps thin shims.
EOF
)"
```

---

### Task 3: 仓内调用方改 import 到 combo

**Files:**
- Modify: `trendspec/strategy/factor_strategy.py`
- Modify: `trendspec/research/fast_eval.py`
- Modify: `trendspec/research/factor_eval.py`
- Modify: `trendspec/cli/research_cmd.py`
- Modify: `trendspec/cli/backtest_cmd.py`
- Modify: `trendspec/cli/screen_cmd.py`
- Modify: `scripts/consumer_rank.py`
- Modify: 所有 tests 中 `from trendspec.research.spec` / `factor_cache` 的生产意图 import
- 保留：可选 1 个 shim 兼容测（`from trendspec.research.spec import FactorSpec` 仍工作）

**替换对照表：**

| 旧 | 新 |
|----|-----|
| `from trendspec.research.spec import FactorSpec` | `from trendspec.combo import FactorSpec` |
| `from trendspec.research.spec import FactorTerm, FilterTerm, ...` | `from trendspec.combo import ...` |
| `from trendspec.research.spec import parse_research_eval_spec` | `from trendspec.combo import parse_research_eval_spec` |
| `from trendspec.research.factor_cache import compute_combo_scores` | `from trendspec.combo import compute_combo_scores` |
| `import trendspec.research.factor_cache as X`（spy） | `import trendspec.combo.scores as X` |

- [ ] **Step 1: 改 strategy（关键：断反向依赖）**

`trendspec/strategy/factor_strategy.py`：

```python
from trendspec.combo import FactorSpec, compute_combo_scores
```

删除对 `trendspec.research` 的任何 import。

- [ ] **Step 2: 改 research 消费方与 CLI / scripts / tests**

批量替换上表。`research/agent.py` / `search.py` / `orchestrator.py` 若 import FactorSpec，同样改 combo。

Run 查找残留（**strategy 必须为零**）：
```bash
rg -n "trendspec\.research\.(spec|factor_cache)" trendspec/strategy
rg -n "trendspec\.research\.(spec|factor_cache)" trendspec tests scripts --glob '*.py'
```

- [ ] **Step 3: 跑相关测**

Run:
```bash
uv run pytest tests/research/ tests/test_fundamental_factors.py tests/test_backtest_cmd.py tests/test_screen_cmd.py -q --tb=line
```
Expected: PASS（按项目已有测文件名微调）

- [ ] **Step 4: Commit**

```bash
git add trendspec/strategy trendspec/research trendspec/cli scripts tests
git commit -m "$(cat <<'EOF'
refactor: point strategy/cli/research consumers at trendspec.combo

Break strategy→research dependency for FactorSpec and combo scores.
EOF
)"
```

---

### Task 4: ARCHITECTURE.md + 边界静态测（Plan 3）

**Files:**
- Modify: `ARCHITECTURE.md`
- Create: `tests/test_combo_boundaries.py`

- [ ] **Step 1: 更新 Directory Topology**

在 `factors/` 与 `research/` 之间插入：

```markdown
| `combo/` | 因子组合运行时（声明式 spec + 截面打分） | `spec.py` (FactorSpec/FactorTerm/FilterTerm), `scores.py` (compute_combo_scores) |
```

更新 `research/` 行：去掉「spec.py (FactorSpec...)」作为实现描述，改为「编排/评估；FactorSpec 见 combo（research.spec 为 re-export）」。

**Key Class Index：**

```markdown
| `FactorSpec` / `FactorTerm` / `FilterTerm` | `combo/spec.py` | Pydantic 因子组合规范（含硬过滤层 filters） |
| `compute_combo_scores` | `combo/scores.py` | winsorize + z-score 组合打分唯一实现 |
```

**Key Design Principles** 或 Data Flow 旁增加一句：

```markdown
8. **组合契约中性** — FactorSpec 与 compute_combo_scores 位于 combo/；strategy 与 research 均可依赖 combo，strategy 不得依赖 research
```

Research 管道图可注明步骤使用 combo 打分。

- [ ] **Step 2: 边界测试**

创建 `tests/test_combo_boundaries.py`：

```python
"""Architectural boundaries for trendspec.combo."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STRAT = ROOT / "trendspec" / "strategy"
COMBO = ROOT / "trendspec" / "combo"

_RESEARCH_IMPORT = re.compile(
    r"^\s*(from\s+trendspec\.research|import\s+trendspec\.research)",
    re.M,
)
_FORBIDDEN_COMBO = re.compile(
    r"^\s*(from|import)\s+trendspec\.(strategy|research|engine|cli)\b",
    re.M,
)


def _py_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*.py") if p.is_file())


def test_strategy_does_not_import_research():
    offenders: list[str] = []
    for p in _py_files(STRAT):
        text = p.read_text(encoding="utf-8")
        if _RESEARCH_IMPORT.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"strategy must not import research: {offenders}"


def test_combo_does_not_import_upper_layers():
    offenders: list[str] = []
    for p in _py_files(COMBO):
        text = p.read_text(encoding="utf-8")
        if _FORBIDDEN_COMBO.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"combo must not import strategy/research/engine/cli: {offenders}"
```

- [ ] **Step 3: 跑边界测 + 冒烟**

Run:
```bash
uv run pytest tests/test_combo_boundaries.py tests/research/test_factor_cache.py tests/research/test_spec.py -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md tests/test_combo_boundaries.py
git commit -m "$(cat <<'EOF'
docs(arch): document combo package and enforce import boundaries

Add topology/key classes; tests lock strategy↛research and combo isolation.
EOF
)"
```

---

### Plan 2+3 完成检查

```bash
uv run pytest tests/research/ tests/test_combo_boundaries.py -q
rg -n "trendspec\.research" trendspec/strategy --glob '*.py'   # empty
rg -n "trendspec\.(strategy|research|engine|cli)" trendspec/combo --glob '*.py'  # empty
```

- [ ] `strategy` 零 research import
- [ ] `combo` 零上层 import
- [ ] shim 仍可 `from trendspec.research.spec import FactorSpec`
- [ ] ARCHITECTURE 已更新
