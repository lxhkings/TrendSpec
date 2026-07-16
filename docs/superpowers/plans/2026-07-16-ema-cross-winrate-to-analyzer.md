# ema_cross_winrate 迁入 analyzer（Plan 4）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ema_cross_winrate` 主实现从 `research/` 迁到 `analyzer/`，research 仅 re-export；CLI 与 tests 改新路径。纯搬家，不改函数逻辑。

**Architecture:** `trendspec/analyzer/ema_cross_winrate.py` 持有实现；`research/ema_cross_winrate.py` 显式 re-export 公开函数。不进 `combo/`（与因子组合契约无关）。

**Tech Stack:** Python, Polars, pytest, uv。

**Spec:** `docs/superpowers/specs/2026-07-16-combo-boundary-cleanup-design.md` §Plan 4

## Global Constraints

- 行为冻结：函数签名与返回值不变。
- **禁止**本 plan 内修改算法、默认参数、列名。
- diff 纪律：git mv（或等价 copy+shim）+ import 改写 only。
- 不强制把符号加入 `analyzer/__init__.py` 的 `__all__`（避免扩大公共面；CLI 直接 import 子模块即可）。

---

## File map

| 文件 | 职责 |
|------|------|
| `trendspec/analyzer/ema_cross_winrate.py` | 主实现（自 research 迁入） |
| `trendspec/research/ema_cross_winrate.py` | 薄 re-export |
| `trendspec/cli/winrate_cmd.py` | import analyzer |
| `tests/test_ema_cross_winrate.py` | import analyzer |
| `tests/test_montecarlo.py` | import analyzer |
| `tests/test_novice_sim.py` | import analyzer |
| `ARCHITECTURE.md` | analyzer 行可提及（可选一句） |

**公开函数（须 re-export）：** 以源文件为准，至少包括：

- `compute_adv20_daily`
- `compute_ema_cross`
- `pair_trades`
- `aggregate`
- `monte_carlo`
- `simulate_novice`
- `run_novice_simulations`
- `per_ticker`
- `current_screen`
- `recent_golden_cross`
- `run_winrate`

（实现迁入后用 `rg -n "^def " trendspec/analyzer/ema_cross_winrate.py` 核对 shim `__all__`。）

---

### Task 1: 迁移文件 + research shim

**Files:**
- Create: `trendspec/analyzer/ema_cross_winrate.py`
- Modify: `trendspec/research/ema_cross_winrate.py` → shim

- [ ] **Step 1: 移动实现**

```bash
git mv trendspec/research/ema_cross_winrate.py trendspec/analyzer/ema_cross_winrate.py
```

若 `git mv` 后需立刻写 shim，可：

```bash
cp trendspec/research/ema_cross_winrate.py trendspec/analyzer/ema_cross_winrate.py
# 再把 research 文件改成 shim（勿丢 analyzer 实现）
```

优先 `git mv` 以保留 blame。

- [ ] **Step 2: 检查 analyzer 文件内相对 import**

打开 `trendspec/analyzer/ema_cross_winrate.py`，确认仅依赖 `trendspec.data.*` 等合法层；**不得** import `trendspec.research`。

若有 `from trendspec.research...`，改为正确层或删除（当前文件应只有 data）。

- [ ] **Step 3: 写 research shim**

`trendspec/research/ema_cross_winrate.py`：

```python
"""兼容 re-export：实现见 trendspec.analyzer.ema_cross_winrate。"""

from trendspec.analyzer.ema_cross_winrate import (
    aggregate,
    compute_adv20_daily,
    compute_ema_cross,
    current_screen,
    monte_carlo,
    pair_trades,
    per_ticker,
    recent_golden_cross,
    run_novice_simulations,
    run_winrate,
    simulate_novice,
)

__all__ = [
    "aggregate",
    "compute_adv20_daily",
    "compute_ema_cross",
    "current_screen",
    "monte_carlo",
    "pair_trades",
    "per_ticker",
    "recent_golden_cross",
    "run_novice_simulations",
    "run_winrate",
    "simulate_novice",
]
```

用源文件实际 `def` 列表校对；多/少导出名以源为准。

- [ ] **Step 4: 冒烟**

Run:
```bash
uv run python -c "from trendspec.analyzer.ema_cross_winrate import run_winrate; from trendspec.research.ema_cross_winrate import run_winrate as r2; print(run_winrate is r2)"
```
Expected: `True`

- [ ] **Step 5: Commit**

```bash
git add trendspec/analyzer/ema_cross_winrate.py trendspec/research/ema_cross_winrate.py
git commit -m "$(cat <<'EOF'
refactor(analyzer): move ema_cross_winrate out of research

Main implementation under analyzer; research keeps compatibility re-export.
EOF
)"
```

---

### Task 2: CLI 与 tests 改 import

**Files:**
- Modify: `trendspec/cli/winrate_cmd.py`
- Modify: `tests/test_ema_cross_winrate.py`
- Modify: `tests/test_montecarlo.py`
- Modify: `tests/test_novice_sim.py`

- [ ] **Step 1: 替换 import**

全部：

```python
from trendspec.research.ema_cross_winrate import ...
```

改为：

```python
from trendspec.analyzer.ema_cross_winrate import ...
```

`winrate_cmd.py` 内三处 lazy import 一并改。

- [ ] **Step 2: 确认无生产代码仍依赖 research 路径（shim 除外）**

Run:
```bash
rg -n "research\.ema_cross_winrate" trendspec/cli tests --glob '*.py'
```
Expected: 无命中（或仅注释）

- [ ] **Step 3: 跑测**

Run:
```bash
uv run pytest tests/test_ema_cross_winrate.py tests/test_montecarlo.py tests/test_novice_sim.py -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add trendspec/cli/winrate_cmd.py tests/test_ema_cross_winrate.py tests/test_montecarlo.py tests/test_novice_sim.py
git commit -m "$(cat <<'EOF'
refactor(cli,tests): import ema_cross_winrate from analyzer
EOF
)"
```

---

### Task 3: ARCHITECTURE 一句（可选但建议）

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1:** 在 `analyzer/` Topology 行补充关键文件提及 `ema_cross_winrate.py`（胜率/蒙特卡洛工具）。

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs(arch): note ema_cross_winrate under analyzer"
```

---

### Plan 4 完成检查

```bash
test -f trendspec/analyzer/ema_cross_winrate.py
wc -l trendspec/research/ema_cross_winrate.py   # shim 应很短（~30 行级）
uv run pytest tests/test_ema_cross_winrate.py tests/test_montecarlo.py tests/test_novice_sim.py -q
```

- [ ] 主实现在 analyzer
- [ ] research 仅为 re-export
- [ ] 行为测全绿
