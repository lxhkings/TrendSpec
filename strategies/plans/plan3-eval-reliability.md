# plan3 — 研究循环可靠性修复(设计,Claude 执行)

> 背景:factor-research 连续 3 轮 0 winners。诊断结论(2026-07-16):
> ① 评估管线 NaN/inf 传播 bug 使 2/3 假设从未被真正测量(round 20260716 H1/H3);
> ② 基本面因子全部裸测、无行业中性化,框架已有的 `group_by`/`filters` 武器 runbook 未开放;
> ③ 评估故障与真负结论混记 ledger,判重规则永久锁死未测因子;plan1/plan2 日期自相矛盾。
> 本设计分 A(代码修复)/ B(runbook 开放武器库)/ C(流程修补)三部分,全部执行。
> 复制位置:TrendSpec 的 `strategies/plans/plan3-eval-reliability.md`。

## 根因(读代码实锤)

- `trendspec/combo/scores.py`:z-score = `(w - mean) / std`,某日某组截面因子值全相同时
  `std = 0.0`,除法产出 NaN/inf。剔除逻辑只检查 `is_null()`,而 **polars 中 NaN ≠ null**,
  带 NaN/inf 的 combo_score 漏网进入下游。
- `trendspec/research/factor_eval.py`:`pl.corr` 在退化截面上产出 NaN rank_ic,
  `drop_nulls` 拦不住 NaN,一个 NaN 毒化整个 `mean()` → 报告出现 `IC均值=nan`。
- `compute_quantile_returns` 的 `qcut`:NaN/常数截面触发 polars 内部
  `PanicException: called Option::unwrap() on a None value`。
- 实际受害:round 20260716 的 `fund_revenue_cagr_3y`(H1)、`ema_alignment`(H3)
  被记「负结论」,但从未被测量;按 plan1 判重规则它们将永久不许再提。

## A. 评估管线修复(代码,main 分支,TDD)

**A1 根因修复 `trendspec/combo/scores.py`**
z-score 表达式计算后,非有限值(NaN/inf)置 null:

```python
pl.when(z.is_finite()).then(z).otherwise(None)
```

零 std 分组的 z 变 null → 现有 `missing_any`(已含 `zcol.is_null()`)自动整行剔除。
语义:该日该组因子无区分度,该行不参与排名。

**A2 防御加固 `trendspec/research/factor_eval.py`**
- `compute_rank_ic`:`drop_nulls("rank_ic")` 后追加 `.filter(pl.col("rank_ic").is_finite())`
  (即便 A1 修复,退化日截面 corr 仍可能出 NaN)。
- `summarize_ic`:入口先过滤非有限 rank_ic;过滤后为空则返回全 None 组。

**A3 qcut 护栏**
`compute_quantile_returns` 改用 `qcut(..., allow_duplicates=True)`:
常数/重复分位边界截面日落入同一桶,不再 panic。

**A4 新 CLI 子命令 `trendspec research coverage`**

```bash
uv run trendspec research coverage --spec-file <spec.json> --market cn --start 2010-01-01
```

输出三个数:总行数、combo_score 非空率、有效截面日占比(当日 ≥30 只标的且 std>0
的日期数 / 总日期数)。复用 `compute_combo_scores`,放 `trendspec/cli/research_cmd.py`
+ `trendspec/research/factor_eval.py`。ARCHITECTURE.md 的 CLI 命令表同步加一行,
与代码同 commit。

**A5 测试(先写测试复现,再修)**
- 常数截面因子:IC 均值/IR 有限(退化日被剔除),quantile 不 panic。
- 零 std 分组:该组行被整行剔除,其余组正常计分。
- `summarize_ic`:注入 NaN rank_ic 仍返回有限汇总。
- coverage 命令:对已知稀疏因子输出正确占比。

**A6 数据修正(修复合入后执行)**
- `research_out/ledger.jsonl` 中 2026-07-16 的 `fund_revenue_cagr_3y`、`ema_alignment`
  两条记录改标 `stage_failed: "eval_error"`,conclusion 注明「框架 NaN bug,未被测量」。
- 修复后按 plan1 Phase 3 重跑这两个 spec 的 ic + quantile(`--start 2010-01-01`,
  记 `--end`),真实数字追加 ledger 新条目(note 标注 retest after eval fix)。

## B. plan1 开放武器库(文档)

- **2.2 spec 模板**补两个可选字段:
  - `group_by`:行业中性化分组,格式 `{组名: [行业代码,...]}`;GICS 11 组现成映射
    在 `examples/factor_combo_cn_gics.json`,可整段复制。
  - `filters`:打分前硬过滤,格式 `[{"name","params","op","value"}]`
    (op ∈ `>`/`>=`/`<`/`<=`),用于市值/流动性下限。
- **Phase 3 新规则(中性化变体)**:原始因子初筛近失(IC/IR 单项达标,或未达标项
  距门槛 <20%)→ 允许同一假设跑一次 GICS 行业中性化变体(spec 加 group_by),
  不占假设预算;原始与中性化两组数字都进报告。
- **Phase 2 新指引(家族平衡)**:每轮 ≥1 个价格/量类假设(反转、换手、波动率等);
  基本面类假设默认建议带行业中性化。
- **RESEARCH_RULES 第 5 节**同步一行:同一假设的行业中性化变体不算新假设、不占预算。

## C. 流程修补(文档)

- **日期全统一 2010-01-01**(用户决定,写死):
  - plan1 头部「本轮参数」、Phase 3.1/3.2 命令(2018→2010)、Phase 4.1 walkforward
    (2015→2010);
  - plan2 C2 复现命令(2018→2010)。
- **记 `--end`**:plan1 Phase 3 命令显式加 `--end <当日日期>`,报告粘贴完整命令;
  plan2 C2 改为逐字重跑报告所记命令,期望数字逐位一致,删除「第 3 位小数内漂移」
  容忍;不一致 → 先查 fundamentals 是否 restate(对比 parquet 该区间行数),
  仍不明 → 判不可信。
- **plan1 新增 Phase 3.0 覆盖率预检**:每个假设跑 IC 前先跑 `research coverage`
  (A4);有效截面日占比 <50% → 按 `data_insufficient` 记 ledger、停该假设,
  不进入 IC。占比阈值 50% 写入 plan1,可由用户调整。
- **判重豁免**(plan1 1.2 + 5.2 模板):`stage_failed ∈ {eval_error, data_insufficient}`
  的 ledger 条目不参与判重,允许将来重提。
- **RESEARCH_RULES 第 4 节**补一条:ic/quantile 输出含 nan → 按 `eval_error` 记录
  并停该假设(评估故障,非负结论);同样禁止绕过。

## 不做什么

- **门槛不动**:IC ≥0.02 / IR ≥0.3 / oos_sharpe ≥1.0 等全部保留,
  等中性化变体跑 1-2 轮后用数据重新评估。
- 不改 walkforward 切窗、不加新因子、不动 ingest/StockPull 流程。

## 执行顺序

1. 用户先 merge `factor-research` → `main`(20260716 验收已通过),避免两头改。
2. A 代码在 main 上做:A5 测试先行 → A1-A4 → 全测试绿 → 提交(代码与
   ARCHITECTURE.md 同 commit)。
3. A6 ledger 修正 + H1/H3 重测,追加记录。
4. B + C:plan1 / plan2 / RESEARCH_RULES.md 三份文档同一提交改完。

## 验收标准

- `uv run pytest` 全绿;新增测试覆盖常数截面、零 std 分组、NaN 汇总、coverage 命令。
- 修复后重跑 `research_out/specs/revenue_cagr_3y.json`、`ema_alignment.json` 的
  ic/quantile:无 nan、无 panic,产出有限数字。
- plan1/plan2/RESEARCH_RULES 中不再出现 `2018-01-01`/`2015-01-01` 起点;
  spec 模板含 group_by/filters;判重豁免与 Phase 3.0 预检落文。
- ledger 中不存在「nan 指标 + 负结论」组合的条目。
