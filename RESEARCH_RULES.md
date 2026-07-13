# RESEARCH_RULES — 因子研究边界铁律

> 本文件是因子研究循环的最高约束。执行研究的 AI(DeepSeek)每轮开始前必须重读本文件。
> 与 runbook 冲突时,以本文件为准。
> 复制位置:TrendSpec 仓库根目录,并在 TrendSpec 的 CLAUDE.md 中加一行引用。

## 1. 写权限白名单

只允许创建/修改以下路径,其余一律只读:

```
trendspec/factors/**          # 新因子代码 + 同目录 __init__.py 的 import 行
research_out/**               # spec json、报告、ledger
strategies/specs/**           # 策略 spec 归档
```

**明确禁止改动**(包括"只改一行"):

```
trendspec/engine/**      trendspec/research/**     trendspec/risk/**
trendspec/analyzer/**    trendspec/data/**         trendspec/ingest/**
trendspec/config/**      trendspec/cli/**          trendspec/strategy/**
trendspec/screening/**   tests/**                  pyproject.toml
```

评估逻辑、阈值常量(`orchestrator.py` 的 `THRESHOLD_SHARPE` / `THRESHOLD_MAX_DD`)、
IC 计算、walkforward 切窗——**一行都不许动**。指标不好看 = 假设失败,不是代码要改。

## 2. 防未来函数(lookahead)

因子在 t 日的值,只能依赖 t 日及之前的数据。逐条检查:

| 规则 | 说明 |
|------|------|
| 禁 `shift(-n)` | 任何负数位移都是引用未来行 |
| 禁未来聚合 | 滚动窗口只许向后看:`rolling_*` 默认窗口在过去,不许用 center/未来对齐 |
| 当日横截面可用 | `rank().over("date")`、`over("date")` 内的同日截面运算允许 |
| 基本面走 PIT | 只能用现有 fundamentals/valuation Parquet 加载路径,禁自接数据库、禁自算发布日期 |
| 禁网络/随机 | 因子代码内禁 requests/urllib、禁 random(可复现性) |

自查方法:问自己"在 t 日收盘时,这个值算得出来吗?"算不出来就是未来函数。

## 3. 证据制

- 报告、ledger、commit message 里出现的一切指标数字,必须是命令原始输出的粘贴。
- 禁手写数字、禁"约"、禁凭记忆填数。
- 每条结论必须能通过重跑报告里记录的命令复现。

## 4. 失败即停

以下情况立即停止当前假设或整轮,如实写入报告,**禁止任何绕过手段**:

- StockPull / ingest 命令报错,或 `ingest status` 显示数据缺口
- NAS(192.168.8.9)不可达
- ic/quantile/回测命令报错或输出「没有可用样本」
- 新因子注册后 smoke test 失败且 2 次修复尝试无效

绕过手段包括但不限于:改评估代码、改阈值、缩小日期区间凑数、造数、跳过验证步骤。

## 5. 预算

- 每轮 ≤3 个假设、其中新因子 ≤2 个
- 每个假设必须附一句经济学/行为金融逻辑;讲不出逻辑的假设不许提
- 超预算的想法写进报告「下轮方向」,不许当轮偷跑

## 6. git 纪律

- 全部工作在 `factor-research` 分支,禁在 main 上提交
- 一因子一提交;只有通过全部验证门槛的因子代码才许提交
- 失败因子的代码必须在当轮结束前用 `git checkout -- <file>` / 删除文件清干净
- 禁 force push,禁 rebase 已推送历史,禁合并到 main(合并由用户在验收后手动做)

## 7. 验证门槛(只许引用,不许修改)

| 阶段 | 命令 | 通过标准 |
|------|------|---------|
| 初筛 IC | `trendspec research ic` | \|IC均值\| ≥ 0.02 且 \|IR\| ≥ 0.3,方向与假设一致 |
| 初筛分层 | `trendspec research quantile` | 分组收益基本单调,top-bottom 价差方向与假设一致 |
| 严格验证 | walk-forward(经 `research run --mock-llm`) | oos_sharpe ≥ 1.0 且 oos_max_drawdown ≤ 0.20 且 worst_window_sharpe > 0(代码内置 `passes_threshold`,以其输出为准) |
