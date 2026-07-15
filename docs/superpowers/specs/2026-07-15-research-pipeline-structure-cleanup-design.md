# Research 管道结构清理 — 设计文档

## 背景

`factor-research` 分支相对 `origin/main` 引入了硬过滤层（`FilterTerm` / `filters`）、`MarketPanel` 基本面 merge、跨季 `*_qoq_prev` 因子，以及 `MaxPositionSize` 等修复。code-review 认为**行为方向正确**，但存在四处结构债：

1. `MarketPanel.load` 与 `BaseEngine.load_data` 各自内联同一套 fundamentals/valuation best-effort merge（含 `except Exception: pass`），后续易再次分叉。
2. `trend.py` 在已有 `anchor_shift` 参数化计算后，仍用整类拷贝展开 `*_qoq_prev`，六个 Factor 的 `compute()` 模板重复。
3. `research ic/quantile` 的 `_load_factor_spec_json` docstring 写「validate」，实际只有 `json.loads`；非法 `filters.op` 在 `_FILTER_OPS[...]` 处变成含糊 `KeyError`，绕开了 `FilterTerm` 契约。
4. `compute_combo_scores` 中 filter 与 score 各自 `compute_full`；同阶段内重复 `(name, params)` 会双算。`FactorCache` 存在但未接入该路径。

本设计在**行为冻结**前提下消除上述债务，并拆成四个可独立合入的 implementation plan。

## 目标

| ID | 目标 |
|----|------|
| A | Panel enrich 单一实现：`BaseEngine` 与 `MarketPanel` 共用入口 |
| B | QoQ/跨季 Factor 去模板复制：参数化基类，注册名与数值不变 |
| C | CLI 对 ic/quantile 的 factors/filters 做与 `FilterTerm`/`FactorTerm` 一致的轻量校验 |
| D | `compute_combo_scores` 同阶段内对 `(name, params)` memoize，避免重复 `compute_full` |

## 约束（已确认）

- **行为冻结**：可观测结果不变（因子数值、注册名、filter AND/null 剔除、best-effort 吞异常、z-score 在 survivor 上计算等）。
- **允许的错误路径清晰化**：非法 JSON / 非法 op / 未注册因子名 → 明确失败信息 + CLI `Exit(1)`，取代更深处的 `KeyError`/`ValidationError` 泄漏。
- **一份 umbrella design**，后续 **四个 plan**（按 A/B/C/D 各一）。
- **方案 1（最小抽取）**：不新建顶级模块；不把 `bars()` 并入 enrich；不重做 Engine 全量 load；不强制 `compute_combo_scores` 签名改为 `list[FilterTerm]`。
- 不改 research 阈值、walkforward、IC 公式、因子经济学含义。
- 不统一 CN/US `end_date` vs `period_end`。
- 本工作是工程清理，在常规开发流程中完成，不受 `RESEARCH_RULES.md` 研究循环写权限白名单约束。

## 非目标

- 新建 `data/panel_load.py` 或统一 `load_daily_panel = bars + enrich`（范围膨胀，曾作为方案 2 否决）。
- 类型贯穿全管道（签名强制 Pydantic 模型；曾作为方案 3 否决）。
- Filter↔score **跨阶段**共用同一次 `compute_full`（截面因子在 filter 前后成员集合不同，复用会改变语义；见 D）。
- 性能基准、并行化、大规模优化。
- 修改 `BaseEngine` 中 weekly 等其它 best-effort 块。
- 修改 `FactorStrategy` 的 spec 加载（已走完整 `FactorSpec`）。

## 成功标准

| 标准 | 判定 |
|------|------|
| 行为冻结 | 现有相关 pytest 全绿；无数值/注册名/filter 语义类断言变更 |
| A 去重 | 两处 load 仅调用共享 `enrich_daily_panel`；无重复 try/merge 块 |
| B 去重 | 六个跨季 Factor 共享一份 `compute` 实现；`*_prev` 无独立大段拷贝 |
| C 契约 | 非法 `op` / 未注册名 → CLI Exit(1) + 可读错误；合法 spec 行为与今一致 |
| D memo | 同阶段内相同 `(name, params)` 只 `compute_full` 一次（单测可测调用次数） |
| 可拆合入 | A/B/C/D 可独立 PR/commit，无硬编码交叉依赖 |

## 架构与落位

依赖方向保持：

```
data/fundamentals  ←  engine/base_engine, research/market_panel
factors/fundamental/trend  ←  registry / consumers（只消费）
research/spec (FilterTerm, FactorTerm)  ←  cli/research_cmd, research/factor_cache
research/factor_cache  ←  strategy, fast_eval, factor_eval, CLI
```

| 工作包 | 主要文件 | 概念 API |
|--------|----------|----------|
| A | `trendspec/data/fundamentals.py`；调用方 `base_engine.py`、`market_panel.py` | `enrich_daily_panel(df, market, root) -> df` |
| B | `trendspec/factors/fundamental/trend.py` | 模块内 `_QuarterlyShiftFactor`（名可调整） |
| C | `trendspec/cli/research_cmd.py`；（可选）`research/spec.py` 抽纯函数 | `_load_factor_spec_json` 真校验 |
| D | `trendspec/research/factor_cache.py` | `compute_combo_scores` 内部同阶段 memo |

不新增 ARCHITECTURE.md 顶级模块；若 CLI 行为说明有文档句，可在实现 plan 中顺手改一句 help 文案（已部分提到 filters）。

## 详细设计

### A — `enrich_daily_panel`

落点：`trendspec/data/fundamentals.py`。

语义（与当前 `BaseEngine` / `MarketPanel` 对齐）：

1. 若 `daily` 为空，直接返回。
2. `try: daily = merge_fundamentals(daily, market, root)` / `except Exception: pass`。
3. `try: daily = merge_valuation(daily, market, root)` / `except Exception: pass`。
4. 返回 `daily`。

约定：

- 不新增日志；不收窄异常类型（行为冻结）。
- `merge_*` 内部「数据集缺失则原样返回」不变。
- **不**包含 `bars()`；weekly 路径不动。
- `BaseEngine.load_data` 与 `MarketPanel.load` 在 `bars` 之后各调用一次，删除内联重复块。

### B — 跨季 Factor 参数化基类

落点：`trendspec/factors/fundamental/trend.py`。

- 保留 `_quarterly_series` / `_quarterly_shift_compute` / `_asof_join_quarterly_result` 算法与 `anchor_shift` 语义。
- 新增模块内基类，用 ClassVar 描述：`value_col`、`n`、`gap_min_months`、`gap_max_months`、`mode`（`ratio`|`cagr`|`diff`）、`cagr_years`（可选）、`anchor_shift`（默认 0）。
- 唯一一份 `compute()`：缺列 → null expr；调用 `_quarterly_shift_compute`；empty → null；asof join 并 alias 为 `self.name`。
- 六个已注册类变为薄壳（仅 description + classvar + `@register`）。

注册名与参数意图（不得改名、不得改数值语义）：

| 注册名 | 意图 |
|--------|------|
| `fund_revenue_qoq` | total_revenue, n=1, ratio, anchor_shift=0 |
| `fund_revenue_qoq_prev` | 同上, anchor_shift=1 |
| `fund_net_income_qoq` | net_income, n=1, ratio, anchor_shift=0 |
| `fund_net_income_qoq_prev` | 同上, anchor_shift=1 |
| `fund_revenue_cagr_3y` | total_revenue, n=12, cagr, gaps 34–38, cagr_years=3 |
| `fund_roe_trend_4q` | roe, n=4, diff, gaps 10–14 |

### C — CLI 轻量校验

落点：`_load_factor_spec_json`（必要时在 `spec.py` 抽出可单测的纯函数，例如 `parse_research_eval_spec(raw: dict) -> dict`，避免 CLI 内堆逻辑）。

校验范围（ic/quantile 子集，**不是**完整 `FactorSpec`）：

| 字段 | 规则 |
|------|------|
| `factors` | 必填、非空；每项按 `FactorTerm` 校验（含已注册 `name`、`direction` 等） |
| `filters` | 可选，默认 `[]`；每项按 `FilterTerm` 校验 |
| `group_by` | 可选 |
| `winsorize_pct` | 可选，默认与今相同（0.01） |
| `top_k` / `top_pct` / `rebalance` / `market` | **不要求** |

失败：console 红字说明原因 + `typer.Exit(1)`。  
成功：返回与今相同形状的 `dict`（filters/factors 可用 `model_dump()`），下游 `spec.get("filters")` 等无需改。

说明：缺 `direction` 的 factors 在今日路径上会在更深处失败；本设计将失败前移并文案化，视为允许的错误路径清晰化，不是成功路径行为变更。

`FactorStrategy` 继续走完整 `FactorSpec`，本包不改。

### D — 同阶段 memoize

落点：`compute_combo_scores` / `_apply_filters`（`factor_cache.py`）。

- Key：`(name, tuple(sorted((params or {}).items())))`，可与模块内现有 `_key` 对齐。
- **Filter 阶段**：在 pre-filter `df` 上，多个 filter 共享一个 memo；同 key 只 `compute_full` 一次。
- **Score 阶段**：在 **全部 filter 应用之后** 的 `df` 上，另建 memo；factors 循环同 key 只算一次。
- **禁止** filter 阶段与 score 阶段共用同一 cache 条目作为默认行为。原因：filter 在全量（或当前阶段）成员上算原始值；score 在 survivor 集合上算。对依赖截面成员集合的因子，跨阶段复用会改变 z-score 输入语义。列直通/时序类因子虽然数值常一致，但为行为冻结与简单性，统一不跨阶段复用。
- 因此：同一因子既出现在 filters 又出现在 factors 时，**允许**最多两次 `compute_full`（每阶段一次）。D 的收益是：多 filter 同因子、factors 列表重复 params 时去重。
- 不强制改 `FactorCache` 对外 API；内部可复用相同 key 函数。
- 公开签名保持 `filters: list[dict] | None` 等现有形式。

## 测试策略

| 工作包 | 测试 |
|--------|------|
| A | 现有 fundamentals/engine 相关测回归；可选：`enrich_daily_panel` 空表原样返回 |
| B | `tests/test_trend_factors.py` 全绿 |
| C | 新测：合法 filters 通过；`op: "!="` 失败；未注册因子名失败；缺 direction 失败且信息可读 |
| D | 新测：两 filter 同 name → `compute_full` 1 次；两 factor 同 name+params → 1 次；filter+factor 同名允许 2 次；现有 filter 语义测（survivor、null、z 对称）仍绿 |
| 汇总 | 实现末期 `uv run pytest` 相关子集，合入前全量 |

## 工作包与 plan 切分

实现阶段按下列四个 plan 拆分（`writing-plans` 各写一份，或一份总 plan 下四个 phase——推荐**四个独立 plan 文件**，与已确认选项一致）：

| Plan | 标题 | 依赖 | 可独立合入 |
|------|------|------|------------|
| 1 | enrich_daily_panel 抽取 | 无 | 是 |
| 2 | trend `_QuarterlyShiftFactor` 压扁 | 无 | 是 |
| 3 | research_cmd factors/filters 校验 | 无（逻辑上可与 A/B 并行） | 是 |
| 4 | compute_combo_scores 同阶段 memoize | 无硬依赖；建议在 filter 语义测稳定后做 | 是 |

建议落地顺序：**1 → 2 → 3 → 4**（1∥2 亦可），非合并阻塞关系。

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 吞异常复制时「修掉」了静默行为 | A 明确保留 `except Exception: pass`，单测不断言日志 |
| 基类 classvar 与 `self.name` 注册不一致 | B 不改 `@register` 名；全量 trend 测 |
| FactorTerm 校验过严导致旧 JSON 挂 | 仅 ic/quantile 入口；错误信息指明缺字段；成功路径 JSON 本就带 direction |
| Memo 跨阶段误用导致截面因子漂移 | D 规格禁止跨阶段 cache；测调用次数而非强行 1 次跨阶段 |

## 决策记录

| 决策 | 选择 |
|------|------|
| Spec 粒度 | 一份 umbrella design |
| 行为边界 | 纯重构 / 行为冻结 |
| Plan 切分 | 按 review 四项 A/B/C/D |
| 实现路线 | 方案 1 最小抽取 |
| D 跨阶段复用 | 不做 |
| C factors 校验 | 按 `FactorTerm`（含 direction） |
