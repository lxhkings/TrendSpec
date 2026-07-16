# Combo 边界收口与结构去债 — 设计文档

## 背景

2026-07-15 research 管道结构清理（enrich 抽取、quarterly 基类、CLI eval-spec 校验、同阶段 memoize）消除了**重复实现**，但 code-review 与后续边界审计指出两类仍存债务：

**局部债**

1. `FactorCache` 与 `compute_combo_scores` 内 bare dict memo 双轨并存；生产路径只用后者，前者仅测试引用。
2. `_QuarterlyShiftFactor` 子类复述与基类相同的 ClassVar 默认值。
3. `parse_research_eval_spec` 手搓 payload 再 `model_validate`，可用 `extra="ignore"` 简化。
4. `test_call_sites_use_enrich_daily_panel_not_inline_merges` 以源码字符串锁架构，脆且不测行为。
5. filter 允许 op 双真相源：`FilterTerm.op` Literal 与 `factor_cache._FILTER_OPS` dict keys。

**边界债**

6. `FactorSpec` + `compute_combo_scores` 挂在 `research/`，导致 **`strategy → research` 反向依赖**；架构图中 research 本应独立于引擎环。
7. `research/` 夹带与 LLM 研究编排无关的 `ema_cross_winrate`（被 winrate CLI / analyzer 类测试使用）。
8. `enrich_daily_panel` 的 `except Exception: pass` 为存量策略；本设计**不改语义**（严格行为冻结）。

本设计在**严格行为冻结**前提下全量收口，拆成多个可独立合入的 implementation plan。落地路线为「先收口再搬家」（方案 1）。

## 已确认决策

| 项 | 选择 |
|----|------|
| 范围 | 全量收口，拆多个 plan |
| 行为边界 | 严格行为冻结：可观测打分/filter/IC/回测结果不变；enrich 仍吞异常 |
| 运行时契约归属 | 新中性包 `trendspec/combo/` |
| 旧 import | 薄 re-export shim + 仓内改新路径；shim 非新代码入口 |
| 落地顺序 | Plan1 本地去债 → Plan2 建 combo 搬家 → Plan3 边界验收 → Plan4 ema 迁出 → Plan5 噪音清理 |

## 目标

| ID | 目标 |
|----|------|
| G1 | 组合运行时契约归 `trendspec/combo/`（spec + 打分 + filter 契约） |
| G2 | 断 `strategy → research`；`research` 只做编排/评估/LLM，消费 `combo` |
| G3 | 同阶段 memo 单一实现；删除生产无用的 `FactorCache` 类 |
| G4 | filter 允许 op 单一真相源（与 `FilterTerm` 同源） |
| G5 | `ema_cross_winrate` 迁出 `research/`；ClassVar 复述与脆测清掉 |
| G6 | 旧路径薄 re-export；仓内 import 改 `combo` / `analyzer` |

## 非目标

- 改 IC / 分层 / 回测 / filter 可观测数值语义
- 改 `enrich_daily_panel` 的 `except Exception: pass` 可见性或改为 raise/日志必选
- filter↔score **跨阶段**共享同一次 `compute_full`
- 全管道强制 Pydantic 类型贯穿（`compute_combo_scores` 仍可接受 `list[dict]`）
- 本轮删除全部 re-export shim（可后续单独 plan）
- 新建 `load_daily_panel = bars + enrich` 大一统 loader
- 性能基准、并行化重做
- 将 `ema_cross_winrate` 拆文件重构（本轮只搬家）
- 统一 CN/US `end_date` vs `period_end`

## 约束

- **严格行为冻结**：因子数值、注册名、filter AND/null 剔除、z-score survivor 语义、best-effort enrich 吞异常，均不变。
- **依赖方向**：`combo` 可依赖 `data` / `factors`；**不得**依赖 `engine` / `strategy` / `research` / `cli`。
- 新顶级模块 `combo/` 须同步更新 `ARCHITECTURE.md`（与代码同一次相关 commit）。
- 拆多个独立 plan，可分别合入；不受 `RESEARCH_RULES.md` 研究循环写权限白名单约束（工程清理）。
- 本工作是工程清理，不改变研究阈值或因子经济学含义。

## 成功标准

| 标准 | 判定 |
|------|------|
| 行为冻结 | 相关 pytest 全绿；无数值/注册名/filter 语义类断言变更 |
| 反向依赖消除 | `strategy/` 无任何 `from trendspec.research...` / `import trendspec.research` |
| 单一打分入口 | 生产实现位于 `combo.scores`；research 仅 re-export |
| 无双轨 memo | 生产代码无 `FactorCache` 类 |
| filter op 同源 | 执行表与 `FilterTerm` 允许 op 同一常量源 |
| 外围迁出 | `ema_cross_winrate` 主实现不在 `research/` |
| ARCHITECTURE | Directory Topology 与依赖说明含 `combo/` 及正确方向 |

---

## 架构与包结构

### 目标依赖拓扑

```
ingest → data_lake → data/ → factors/
                         ↘
                          combo/  ← spec + scores（中性运行时）
                         ↗    ↘
              strategy/ ─      → research/（编排、IC、LLM、fast_eval）
              engine/  ────────→ strategy/
              cli/     ────────→ 各层（含 combo、research、analyzer）
              analyzer/  ← winrate 工具（自 research 迁入）
```

**禁止：**

- `strategy → research`
- `combo → strategy | research | engine | cli`

### `trendspec/combo/` 文件

| 文件 | 职责 | 来源 |
|------|------|------|
| `__init__.py` | 导出公共 API | 新建 |
| `spec.py` | `FactorTerm` / `FilterTerm` / `FactorSpec` / `_ResearchEvalSpec` / `parse_research_eval_spec` | 自 `research/spec.py` 搬迁 |
| `scores.py` | `compute_combo_scores`、`_apply_filters`、同阶段 memo helper；**无** `FactorCache` | 自 `research/factor_cache.py` 搬迁并收口 |

**公共 API（仅下列符号作为正式出口）：**

- `FactorTerm`, `FilterTerm`, `FactorSpec`
- `parse_research_eval_spec`
- `compute_combo_scores`

不导出：`_key`、`_apply_filters`、内部 memo helper、`_ResearchEvalSpec`（可模块内私有）。

### 旧路径 shim

| 旧模块 | 行为 |
|--------|------|
| `research/spec.py` | **显式** re-export 公共 spec 符号（禁止 `import *` 若会泄漏私有名则改名单） |
| `research/factor_cache.py` | re-export `compute_combo_scores` only；不再提供 `FactorCache` |
| `research/ema_cross_winrate.py` | re-export analyzer 中同名实现 |

仓内调用方（`strategy` / `research` / `cli` / `tests` / `scripts`）一律改为新路径。shim 仅兜底外部或漏网引用，**新代码不得依赖 shim 作为入口**。

### 消费关系（完成后）

| 消费者 | 依赖 |
|--------|------|
| `strategy/factor_strategy.py` | `combo.spec` + `combo.scores` |
| `research/fast_eval.py`, `factor_eval.py` | `combo.scores` |
| `research/orchestrator`, `agent`, `search` | `combo.spec` |
| `cli/research_cmd`, `backtest_cmd`, `screen_cmd` | `combo.spec`（及各自 engine/strategy） |
| `cli/winrate_cmd` | `analyzer.ema_cross_winrate` |

### `ema_cross_winrate` 归属

| 项 | 决定 |
|----|------|
| 新位置 | `trendspec/analyzer/ema_cross_winrate.py` |
| 理由 | 与绩效/诊断工具同类；与因子组合契约无关，**不进** `combo` |
| 行为 | 函数签名与返回值不变；本 plan 禁止逻辑改动 |

### `ARCHITECTURE.md` 同步

- Directory Topology 增加 `combo/` 行：职责为因子组合声明式规范与截面打分。
- Research 管道说明改为：打分与 FactorSpec 来自 `combo`；research 负责假设/搜索/评估编排。
- 依赖原则补充：strategy 不得依赖 research。

---

## 分 Plan 详细设计

### Plan 1 — 本地去债（仍在 `research/`，不搬家）

**1a. Memo 收口**

- 抽取 `_compute_full_cached(cache, name, params, market, df) -> FactorResult`。
- `_apply_filters` 与 score 循环各持**独立** `cache: dict = {}`。
- **禁止**跨阶段共用 cache（与现测一致：同因子既在 filters 又在 factors 时允许 2 次 `compute_full`）。
- **删除** `FactorCache` 类及仅服务该类的测试。
- 保留同阶段调用次数测（两 filter 同 key → 1；两 factor 同 key → 1；跨阶段允许 2）。

**1b. Filter op 单一真相源**

- 在 `spec` 模块定义允许 op 的常量源（例如 `_FILTER_OPS_TUPLE = (">", ">=", "<", "<=")`），`FilterTerm.op` 的 `Literal` 与之对齐。
- `factor_cache` / 日后 `scores` 的执行 map **只从该常量生成**，禁止手写第二份 key 列表。

**1c. parse 简化**

- `_ResearchEvalSpec` 设置 `model_config = ConfigDict(extra="ignore")`。
- `parse_research_eval_spec`：`model_validate(raw)`，去掉手搓 `payload` 字典。
- 返回 dict 形状冻结：
  - 总有 `factors` / `filters`（默认 `[]`）/ `winsorize_pct`（默认 `0.01`）
  - `group_by` 仅在解析结果非 `None` 时出现在 out 中

**1d. 测试**

- 删除源码字符串扫描式 `test_call_sites_use_enrich_daily_panel_not_inline_merges`（或等价测）。
- enrich：保留/补强行为测（空表原样返回）；不锁源码子串。

**行为冻结：** 合法路径输出不变；非法 op / 未注册名仍 `ValidationError`；CLI 仍 Exit(1)。

---

### Plan 2 — 新建 `trendspec/combo/` 并迁入

| 步骤 | 动作 |
|------|------|
| 2.1 | 新建 `combo/spec.py`、`combo/scores.py`（内容 = Plan1 收口后的最终形态） |
| 2.2 | `research/spec.py`、`research/factor_cache.py` → 显式薄 re-export |
| 2.3 | 仓内 import 批量改为 `trendspec.combo...` |
| 2.4 | 更新 `ARCHITECTURE.md` |
| 2.5 | `uv run pytest` 相关子集 + 约定全量 |

**不在此 plan：** 改算法、改 `FactorSpec` 字段集、删除 shim、改 enrich 语义。

**命名：** 实现文件为 `scores.py`；shim 文件名可保留 `research/factor_cache.py` 以兼容旧 import 路径字符串。

---

### Plan 3 — 消费方边界验收

可与 Plan 2 合并；若拆开则仅含：

- 静态检查：`strategy/**/*.py` 不得出现 `trendspec.research`
- 静态检查：`combo/**` 不得 import `strategy|research|engine|cli`
- 文档与真实 import 一致

建议实现：测试或脚本内 `rg`/AST 断言，防止回潮。

---

### Plan 4 — `ema_cross_winrate` 迁出

| 项 | 规定 |
|----|------|
| 目标 | `trendspec/analyzer/ema_cross_winrate.py` |
| shim | `research/ema_cross_winrate.py` re-export |
| 调用方 | `cli/winrate_cmd`、相关 tests → `analyzer` |
| diff 纪律 | 纯搬迁 + import；禁止同 PR 改逻辑 |

---

### Plan 5 — 噪音清理

**5a. quarterly ClassVar**

子类只覆盖相对 `_QuarterlyShiftFactor` 默认值的差异字段：

| 类 | 须声明的差异 |
|----|----------------|
| `FundRevenueQoQ` | `value_col` + description |
| `FundRevenueQoQPrev` | `value_col`, `anchor_shift=1` |
| `FundNetIncomeQoQ` | `value_col` |
| `FundNetIncomeQoQPrev` | `value_col`, `anchor_shift=1` |
| `FundRevenueCagr3Y` | `value_col`, `n`, gaps, `mode`, `cagr_years` |
| `FundRoeTrend4Q` | `value_col`, `n`, gaps, `mode` |

注册名与数值语义不变；`tests/test_trend_factors.py` 全绿。

**5b. enrich**

- **不改** `except Exception: pass`
- 测试只锁行为，不锁源码字符串（与 Plan1 一致；若 1 已删脆测则 5b 可为空）

---

## 错误处理（全 plan 统一）

| 场景 | 行为（冻结） |
|------|----------------|
| 未注册因子名（经 parse / FactorSpec） | `ValidationError` |
| 非法 filter op | `ValidationError` / CLI `Exit(1)` |
| 直接调用 `compute_combo_scores` 传入坏 dict | 与今一致（可能 `KeyError`）；本轮不扩大运行时校验面 |
| enrich 数据集缺失或 merge 异常 | 静默保持 daily 不变 |
| 删除 `FactorCache` 后旧 `from ... import FactorCache` | **允许 ImportError**（仅测试曾用；生产无调用方） |
| re-export 公开符号 | 与迁出前公开 API 对齐（`compute_combo_scores` / FactorSpec 族） |

---

## 测试策略

| Plan | 测试 |
|------|------|
| 1 | 现有 `test_factor_cache` / `test_spec` + memo 次数；删 FactorCache 测；parse 忽略多余字段；去脆测 |
| 2 | import 改写后全量/相关 pytest；可选公共导出面冒烟 |
| 3 | 边界 grep/静态测 |
| 4 | winrate / montecarlo / novice 相关测改 import 后全绿 |
| 5 | `test_trend_factors`；enrich 行为测 |

合入前：`uv run pytest`（至少 research / strategy / factors / analyzer / cli 相关 + 约定全量）。

---

## 工作包与落地顺序

```
Plan1（本地去债）
  → Plan2（combo 搬家 + ARCHITECTURE，含 Plan3 checklist）
  → Plan4（ema 迁 analyzer）
  → Plan5（ClassVar / 测清理；可与 Plan4 并行）
```

| Plan | 标题 | 依赖 | 可独立合入 |
|------|------|------|------------|
| 1 | memo/FilterCache/op/parse/脆测 | 无 | 是 |
| 2 | `trendspec/combo` 迁入 + shim + ARCHITECTURE | Plan1（建议） | 是（若跳过 1 则搬家后仍须在 combo 内做 1 的收口） |
| 3 | 边界静态验收 | Plan2 | 可并入 2 |
| 4 | ema_cross_winrate → analyzer | 无硬依赖 | 是 |
| 5 | quarterly ClassVar + enrich 测 | 无硬依赖 | 是 |

**建议：** 不要跳过 Plan1 直接搬家——避免把双轨 memo 原样搬进 `combo/`。

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 搬家漏改 import | 仓内 `rg`；shim 兜底；Plan3 静态禁 strategy→research |
| 删 `FactorCache` 有隐藏调用方 | Plan1 先删 + CI；已审计仅 tests |
| re-export 循环 import | `combo` 禁止 import research；shim 单向指向 combo/analyzer |
| 大文件搬家难审 | Plan4 纯 mv + import，禁止夹带逻辑 |
| 误做跨阶段 memo | 规格禁止；测锁定「允许 2 次」 |
| ARCHITECTURE 漂移 | Plan2 与拓扑改动同 commit |

## 回滚

| Plan | 回滚 |
|------|------|
| 1 | 单 commit revert |
| 2 | revert 搬家；或临时让调用方再指 shim |
| 4 | 文件迁回 research + 改 import |
| 5 | revert，无行为风险 |

## 后续可开（本设计明确不做）

- 删除全部 research shim
- enrich 改为可观测日志或失败可见
- `compute_combo_scores(..., filters: list[FilterTerm])` 类型强制
- `ema_cross_winrate` 拆分模块
- 跨阶段 memo（会改变截面因子语义）

## 整体完成定义（DoD）

1. `strategy/` 零 `trendspec.research` 引用
2. `combo/**` 零 `strategy|research|engine|cli` 引用
3. 生产路径无 `FactorCache`；同阶段 memo 单测绿
4. filter op 与 `FilterTerm` 同源
5. `ema_cross_winrate` 主实现在 `analyzer/`
6. `ARCHITECTURE.md` 含 `combo/` 与正确依赖方向
7. `uv run pytest` 全绿（或项目约定的等价全量）
8. 可观测打分 / filter / IC 语义无故意变更

## 决策记录

| 决策 | 选择 |
|------|------|
| 范围 | 全量收口，多 plan |
| 行为 | 严格冻结 |
| 契约归属 | `trendspec/combo/` |
| 兼容 | 薄 re-export + 仓内改路径 |
| 落地路线 | 先收口（Plan1）再搬家（Plan2） |
| ema 归属 | `analyzer/`，不进 combo |
| FactorCache | 删除，不迁入 combo |
| enrich 吞异常 | 不改 |
