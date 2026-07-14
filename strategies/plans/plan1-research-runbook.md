# plan1 — 因子研究循环 Runbook(DeepSeek 执行)

> **执行者须知:** 你是研究员,不是框架开发者。每轮开始前先完整读一遍仓库根目录
> `RESEARCH_RULES.md`(铁律,冲突时以它为准)。本 runbook 每一步都有明确命令和
> 通过/失败标准,照做;做不到就停,如实记录。
> 复制位置:TrendSpec 的 `strategies/plans/plan1-research-runbook.md`。

**目标:** 跑一轮完整因子研究:数据准备 → 提假设(可写新因子) → 初筛 → 严格验证 → 入库/记负结论 → 报告。

**工作目录:** TrendSpec = `/Users/xiaohong/Project/TrendSpec`,StockPull = `/Users/xiaohong/Project/StockPull`。所有 `trendspec` 命令在 TrendSpec 目录下 `uv run` 执行。

**本轮参数(开跑前确认):** 市场 `us` 或 `cn`(用户指定,默认 us);评估区间:初筛 `--start 2010-01-01`,严格验证 `--start 2010-01-01`,`--end` 均不传(默认到今日)。

---

## Phase 0:准备与数据

- [ ] **0.1 切分支**

```bash
cd /Users/xiaohong/Project/TrendSpec
git checkout factor-research 2>/dev/null || git checkout -b factor-research
git status   # 必须 clean;不 clean 先停,报告后终止
```

- [ ] **0.2 数据新鲜度检查 → 按需增量**(不再无条件每轮跑 StockPull;同步耗时可达1-2小时,数据已新鲜时跳过)

```bash
cd /Users/xiaohong/Project/TrendSpec
uv run trendspec ingest status --market us   # 或 cn，本轮先看一眼当前状态
```

daily/valuation 最新日期落后最近交易日 **≥1 个交易日** → 触发下方 StockPull 增量;
已覆盖最近交易日(落后 0 个交易日) → **跳过 StockPull**,报告里记一句「数据已新鲜,跳过同步」,直接进 0.3。

```bash
cd /Users/xiaohong/Project/StockPull
uv run main.py prices daily --market us     # 本轮跑 cn 就 --market cn
uv run main.py futu sync                    # 美股基本面(本轮 us 时)
uv run main.py tushare sync --market cn     # A股基本面+估值(本轮 cn 时)
uv run main.py status
```

任一命令报错 → **停**,跳到 Phase 6 写故障报告,本轮终止。

- [ ] **0.3 TrendSpec 摄入**

```bash
cd /Users/xiaohong/Project/TrendSpec
uv run trendspec ingest daily --market us          # 或 cn
uv run trendspec ingest fundamentals --market us   # 或 cn
uv run trendspec ingest valuation --market cn      # 仅 cn 需要
uv run trendspec ingest status --market us         # 或 cn
```

`ingest status` 输出里最新日期落后最近交易日超过 3 个交易日,或有报错 → **停**,故障报告,终止。

- [ ] **0.4 数据新鲜度证据**:把 `ingest status` 原始输出完整粘贴进本轮报告草稿(Phase 6 模板第 1 节)。

## Phase 1:读历史,建「已试清单」

- [ ] **1.1 读 ledger**

```bash
cd /Users/xiaohong/Project/TrendSpec
uv run python -c "
from trendspec.research.ledger import read_ledger
for r in read_ledger('research_out/ledger.jsonl'):
    h = r.get('hypothesis', {})
    fs = [(f.get('name'), f.get('direction'), f.get('param_grid') or f.get('params')) for f in h.get('factors', [])]
    print(r.get('round', r.get('type','?')), '|', fs, '|', 'winners' if r.get('top_candidates') else r.get('conclusion',''))
"
```

- [ ] **1.2 已试清单写进报告草稿**。判重规则:因子名相同 + 方向相同 + 参数网格有重叠 = 重复,禁止再提。

- [ ] **1.3 可用因子清单**(提组合假设前必看,因子名必须来自这里):

```bash
uv run python -c "
import trendspec.factors
from trendspec.factors.registry import list_factors
print('\n'.join(sorted(list_factors())))
"
```

## Phase 2:提假设(≤3 个,新因子 ≤2 个)

- [ ] **2.1 每个假设写三行**(进报告草稿):

```
假设 H1: <一句话描述>
经济学逻辑: <为什么该有效——行为偏差/风险补偿/结构性原因,一句话>
类型: 组合(已有因子) | 新因子
```

讲不出经济学逻辑的假设作废。与已试清单重复的作废。

- [ ] **2.2 组合类假设 → 写 spec json**,存 `research_out/specs/<假设名>.json`:

```json
{
  "factors": [
    {"name": "price_momentum", "params": {"period": 20}, "direction": "low", "weight": 1.0}
  ],
  "winsorize_pct": 0.01
}
```

`name` 必须在 Phase 1.3 清单里;`direction`:`high`=值大者好,`low`=值小者好。

- [ ] **2.3 新因子类假设 → 写 Factor 子类**。文件放 `trendspec/factors/<类目>/<因子名>.py`,类目从现有目录选:`price` / `technical` / `volume` / `cross_sectional` / `sector` / `fundamental`。模板(完整可跑,仿 `factors/price/momentum.py`):

```python
"""<一句话:因子含义与经济学逻辑>"""

from typing import ClassVar

import polars as pl

from trendspec.factors.base import Factor, FactorResult
from trendspec.factors.registry import register


@register("my_factor_name")
class MyFactor(Factor):
    """<因子说明:计算口径、参数含义>"""

    name: ClassVar[str] = "my_factor_name"
    description: ClassVar[str] = "<一句话描述>"
    category: ClassVar[str] = "momentum"  # 或 volatility/value/quality/...

    def __init__(self, period: int = 20) -> None:
        self.params = {"period": period}

    def compute(self, df: pl.DataFrame) -> pl.Expr:
        period = self.params.get("period", 20)
        # 只许用 <=t 的数据:shift(正数) OK,shift(负数) 禁止
        return (
            pl.col("close") / pl.col("close").shift(period).over("instrument_id") - 1
        ) * 100

    def compute_full(self, df: pl.DataFrame) -> FactorResult:
        df_sorted = df.sort("date")
        period = self.params.get("period", 20)
        col_name = f"my_factor_name_{period}"
        df_result = df_sorted.with_columns(self.compute(df_sorted).alias(col_name))
        return FactorResult(
            values=df_result.select(["instrument_id", "date", col_name]),
            name=col_name,
            metadata={
                "description": self.description,
                "category": self.category,
                "params": self.params,
            },
        )
```

然后在 `trendspec/factors/<类目>/__init__.py` 加一行 import(仿照该文件里现有行):

```python
from trendspec.factors.<类目>.<因子名> import MyFactor  # noqa: F401
```

- [ ] **2.4 新因子逐条过 RESEARCH_RULES.md 第 2 节防未来函数清单**,在报告草稿里逐条打勾。

- [ ] **2.5 新因子 smoke test**(确认注册成功、能算出值):

```bash
uv run python -c "
import trendspec.factors
from trendspec.factors.registry import list_factors
assert 'my_factor_name' in list_factors(), '注册失败'
print('registered OK')
"
```

失败 → 修(最多 2 次)→ 仍失败 → 删文件、还原 `__init__.py`,记负结论。

- [ ] **2.6 新因子同样写一份 spec json**(2.2 格式,name 用新因子注册名),后续初筛统一走 spec。

## Phase 3:初筛(每个假设依次跑)

- [ ] **3.1 RankIC**

```bash
uv run trendspec research ic --spec-file research_out/specs/<假设名>.json \
  --market us --start 2018-01-01 --horizon 20
```

预期输出形如:`IC均值=0.0xxx  IC标准差=0.0xxx  IR=0.xxxx  IC胜率=xx.xx%`。原样粘贴进报告。

**通过标准:|IC均值| ≥ 0.02 且 |IR| ≥ 0.3,且符号与假设方向一致。**

- [ ] **3.2 分层回测**

```bash
uv run trendspec research quantile --spec-file research_out/specs/<假设名>.json \
  --market us --start 2018-01-01 --horizon 20 --n-quantiles 5
```

预期输出:5 组平均前瞻收益 + `top-bottom 价差均值`。原样粘贴进报告。

**通过标准:组间收益基本单调(允许 1 处小倒挂),top-bottom 价差方向与假设一致。**

- [ ] **3.3 未过初筛的假设**:新因子代码立即清理(`git checkout -- trendspec/factors/` 或删除新文件+还原 `__init__.py`),spec json 保留,跳 Phase 5 记负结论。输出「没有可用样本」按失败处理并停该假设,禁止缩日期区间重试。

## Phase 4:严格验证(仅初筛通过者)

- [ ] **4.1 walk-forward**(借 `--mock-llm` 注入单假设,自动扫参+切窗+过 `passes_threshold`):

```bash
uv run trendspec research run --market us --start 2015-01-01 --rounds 1 \
  --out ./research_out \
  --mock-llm '{"market":"us","factors":[{"name":"<因子名>","direction":"<high|low>","weight":1.0,"param_grid":{"period":[10,20,60]}}],"top_k_grid":[50,100],"rebalance_grid":[5,10,20],"rationale":"<经济学逻辑>"}'
```

结果自动追加进 `research_out/ledger.jsonl`;达标策略生成 `research_out/strategy-r*-*.md`。

- [ ] **4.2 判定**:以命令输出与 ledger 里 `oos_sharpe / oos_max_drawdown / worst_window_sharpe` 为准(门槛见 RESEARCH_RULES.md 第 7 节)。关键数字原样粘贴进报告。

## Phase 5:结果处理

- [ ] **5.1 通过者 → 提交**(仅新因子有代码可交;一因子一提交):

```bash
git add trendspec/factors/<类目>/<因子名>.py trendspec/factors/<类目>/__init__.py research_out/specs/<假设名>.json
git commit -m "feat(factors): add <因子名> — <一句话逻辑>

IC均值=<粘贴> IR=<粘贴> oos_sharpe=<粘贴> max_dd=<粘贴> worst_window=<粘贴>
evidence: research_out/report-<YYYYMMDD>.md"
```

- [ ] **5.2 失败者 → 清理代码 + 记负结论**(初筛失败的假设 walkforward 没跑、ledger 里没有记录,手动补一条):

```bash
uv run python -c "
from trendspec.research.ledger import append_ledger
append_ledger('research_out/ledger.jsonl', {
    'type': 'manual_research',
    'date': '<YYYY-MM-DD>',
    'hypothesis': {'market': 'us', 'factors': [{'name': '<因子名>', 'direction': '<high|low>', 'param_grid': {'period': [20]}}], 'rationale': '<经济学逻辑>'},
    'stage_failed': 'ic|quantile|walkforward',
    'metrics': {'ic_mean': <粘贴>, 'ir': <粘贴>},
    'conclusion': '<一句话负结论>',
})
print('ledger appended')
"
```

- [ ] **5.3 确认工作区干净**:`git status` 只剩已提交内容与 research_out 产物,无残留半成品代码。

## Phase 6:报告

- [ ] **6.1 写 `research_out/report-<YYYYMMDD>.md`**,固定结构:

```markdown
# 因子研究报告 <YYYY-MM-DD> — market=<us|cn>

## 1. 数据新鲜度
<ingest status 原始输出粘贴>

## 2. 已试清单(来自 ledger)
<Phase 1 输出>

## 3. 假设与结果
### H1: <描述>
- 经济学逻辑: ...
- 类型: 组合|新因子(防未来函数清单: 逐条打勾)
- IC: <原始输出粘贴>
- 分层: <原始输出粘贴>
- walk-forward: <关键数字粘贴 | 未进入>
- 结论: 入库(commit <hash>)| 负结论(ledger 已记)
(H2/H3 同构)

## 4. 故障(如有)
<报错原文粘贴 + 停在哪一步>

## 5. 下轮方向
<超预算想法、想试的变体>
```

- [ ] **6.2 提交报告**:

```bash
git add research_out/
git commit -m "research: round <YYYYMMDD> report (<N> hypotheses, <M> winners)"
```

- [ ] **6.3 收尾自查**(全部要"是"才算完整跑完一轮):
  - 改动文件全部在白名单内?(`git diff main...factor-research --stat` 自查)
  - 报告里每个数字都有对应命令输出?
  - 失败因子代码已清理?
  - ledger 与报告结论一一对应?
