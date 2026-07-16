# plan2 — 因子研究验收清单(Claude 执行)

> 触发时机:factor-research 分支累积 2–3 轮研究后,用户手动发起。
> 目的:确认 DeepSeek 产出可信,决定是否建议用户并入 main。
> 复制位置:TrendSpec 的 `strategies/plans/plan2-accept.md`。

**工作目录:** `/Users/xiaohong/Project/TrendSpec`,在 `factor-research` 分支上执行,全程只读(验收不改代码)。

---

## A. 越界检查(一票否决)

- [ ] **A1 改动范围**

```bash
git diff main...factor-research --stat
```

所有改动必须落在:`trendspec/factors/**`、`research_out/**`、`strategies/specs/**`。
出现任何其他路径(engine/research/risk/analyzer/data/ingest/config/cli/strategy/tests/pyproject.toml)→ **验收失败**,单列违规文件,终止后续步骤。

- [ ] **A2 阈值未被篡改**

```bash
git diff main...factor-research -- trendspec/research/ trendspec/engine/
```

预期:空输出。非空 → 验收失败。

## B. 未来函数审计(逐个新因子)

- [ ] **B1 机械扫描**

```bash
grep -rn "shift(-" trendspec/factors/ && echo "VIOLATION" || echo "clean"
grep -rn "requests\|urllib\|random" trendspec/factors/ --include="*.py" | grep -v "^Binary" || echo "clean"
```

预期两项均 clean。

- [ ] **B2 人工审计**:读每个新增因子文件,逐条核对 RESEARCH_RULES.md 第 2 节:
  - t 日值是否只依赖 ≤t 数据(重点:rolling 窗口方向、join 是否引入未来行)
  - 横截面运算是否限定在 `over("date")` 同日截面内
  - 基本面是否走现有 PIT 加载路径(不许在因子内自读数据库/文件)

## C. 数字复现(抽查)

- [ ] **C1 选样**:每轮报告抽 1 个「通过」+ 1 个「负结论」假设(如有)。

- [ ] **C2 重跑报告中记录的原命令**(spec json 在 `research_out/specs/`):

从报告第 3 节复制该假设记录的**完整原命令**(含 `--start 2010-01-01 --end <报告所记日期>`)逐字重跑,不得自行改参数。

判定:IC均值/IC标准差/IR/IC胜率/分层各组收益/top-bottom 价差与报告**逐位一致**。`--end` 已锚定区间,不存在数据增量漂移;任何数字不一致 = 复现失败。

- [ ] **C3 复现失败处理**:先核对 fundamentals 是否发生 restate(对比该区间 parquet 行数与报告第 1 节 ingest status);无 restate 证据仍不一致 → 该因子判「不可信」,记入验收结论,建议 revert 对应 commit。

## D. 账实一致

- [ ] **D1 ledger vs 报告**:`research_out/ledger.jsonl` 中每条 `manual_research`(含负结论、`eval_error`、`data_insufficient`)与各轮报告第 3 节一一对应,无报告里有、ledger 里没有(或反之)的假设;`eval_error`/`data_insufficient` 条目的 metrics 不得被报告当作负结论引用。

- [ ] **D2 commit vs 报告**:每个「入库」结论都有对应 commit(hash 在报告里),且该 commit 只含该因子相关文件。

- [ ] **D3 死代码检查**:`git status` 干净;负结论因子在 `trendspec/factors/` 下无残留文件、`__init__.py` 无悬空 import。

## E. 验收结论

- [ ] **E1 写 `research_out/accept-<YYYYMMDD>.md`**:

```markdown
# 验收报告 <YYYY-MM-DD>

覆盖轮次: report-<...> ~ report-<...>
A 越界: 通过|失败(<违规文件>)
B 未来函数: 通过|失败(<因子名+问题>)
C 复现抽查: <假设名>: 一致|不一致(<差异>)
D 账实: 通过|失败(<不一致项>)

结论: 建议并入 main | 部分 revert 后并入(<commit 列表>)| 整体退回
```

- [ ] **E2 告知用户**:验收结论 + 并入命令(用户手动执行):

```bash
git checkout main && git merge factor-research
```
