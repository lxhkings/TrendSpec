# TrendSpec

量化回测与选股系统，支持 A 股（CSI800）和美股（SP500 + Russell1000）。

## 功能

- 双模式：历史回测 + 每日选股，同一策略代码通用
- PIT（Point-in-Time）宇宙，避免生存者偏差
- 本地 Parquet 数据湖，选股无需实时连接数据库
- 行业中文显示（GICS 标准分类）
- 信号历史：策略历史信号命中率、均值收益、胜率统计

## 环境要求

- Python >= 3.11
- 群辉 NAS MariaDB（仅入库时需要）

## 安装

```bash
uv sync
```

## 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```
DB_HOST=192.168.8.9
DB_PORT=3306
DB_USER=root
DB_PASSWORD=<密码>
DB_NAME=stocks
DATA_LAKE_ROOT=./data_lake
ALLOW_ROOT_DB_USER=true
```

> `ALLOW_ROOT_DB_USER=true` 必须写在 `.env` 文件（shell 环境变量无效）。

## CLI 命令一览

| 命令 | 说明 |
|------|------|
| `trendspec ingest` | 数据摄入 |
| `trendspec backtest` | 回测 |
| `trendspec screen` | 选股 |
| `trendspec winrate` | 胜率研究 |
| `trendspec signal-history` | 信号历史 |
| `trendspec research` | AI 因子研究 |

---

## 数据摄入

### 首次初始化

```bash
# 美股日线 + 周线
uv run trendspec ingest daily --market us --full
uv run trendspec ingest weekly --market us --full
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us

# A 股日线 + 周线
uv run trendspec ingest daily --market cn --full
uv run trendspec ingest weekly --market cn --full
uv run trendspec ingest components --market cn
uv run trendspec ingest sectors --market cn

# 1h intraday（胜率研究前置）
uv run trendspec ingest intraday --market us --full
```

### 日常增量

```bash
uv run trendspec ingest daily --market us
uv run trendspec ingest daily --market cn
uv run trendspec ingest intraday --market us   # 1h 增量
```

### 查看状态

```bash
uv run trendspec ingest status --market us
```

---

## 回测

```bash
# 查看可用策略
uv run trendspec backtest list

# 运行回测
uv run trendspec backtest run --strategy clenow_momentum --market us --start 2020-01-01 --end 2024-12-31

# 指定初始资金
uv run trendspec backtest run --strategy rs_ema_cross --market us --start 2020-01-01 --capital 1000000

# 策略对比
uv run trendspec backtest compare --market us --start 2022-01-01 --end 2024-12-31 --sort sharpe
```

---

## 选股

```bash
# Clenow 动量选股
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-14

# EMA 密集回踩
uv run trendspec screen run --strategy ema_cluster_pullback --market us --date 2026-05-14

# 传参数
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-14 --param max_per_sector=1
```

CSV 输出：`results/screening/signals_<strategy>_<date>.csv`

---

## 胜率研究

基于 1h intraday 数据，计算 EMA 金叉/死叉信号胜率：

```bash
# 计算胜率 + 当前金叉态选股
uv run trendspec winrate ema-cross --market us --csv ./winrate_out

# 自定义 EMA 周期
uv run trendspec winrate ema-cross --market us --ema-short 60 --ema-long 120 --csv ./winrate_out
```

输出：
- 终端汇总表（总交易数、胜率、平均盈利/亏损、盈亏比、平均持有 1h 根数）
- 当前金叉态选股表（浮动收益降序，前 20）
- CSV：`<csv>_trades.csv`、`<csv>_summary.csv`、`<csv>_screen.csv`

> **前置**：需先摄入 intraday 数据：
> ```bash
> uv run trendspec ingest intraday --market us --full
> ```

---

## 信号历史

回放策略历史信号，计算远期收益率（T+1/3/5/10/20）：

```bash
# 首次构建
uv run trendspec signal-history build --strategy clenow_momentum --market us

# 增量更新
uv run trendspec signal-history build --strategy clenow_momentum --market us

# 查看状态
uv run trendspec signal-history status --strategy clenow_momentum --market us
```

选股报告自动附带历史统计列（样本数、均值收益、胜率、置信度）。

---

## AI 因子研究闭环

自动化策略研究：LLM 提假设 → 扫参 → walk-forward 验证 → 建议书。

### 配置 LLM

`.env` 追加：

```
RESEARCH_LLM_BASE_URL=https://api.deepseek.com/v1
RESEARCH_LLM_API_KEY=sk-...
RESEARCH_LLM_MODEL=deepseek-chat
RESEARCH_OUT_DIR=./research_out
```

### 运行研究

```bash
uv run trendspec research run --market us --start 2015-01-01 --end 2023-12-31 --rounds 10 --out ./research_out

# 测试模式（不连 LLM）
uv run trendspec research run --market us --start 2015-01-01 --end 2023-12-31 --rounds 1 --mock-llm '{"market":"us","factors":[{"name":"momentum","direction":"high","weight":1.0}],"top_k_grid":[20]}'
```

### 监控面板

```bash
# 前台运行（Ctrl+C 中止）
uv run trendspec research serve --out ./research_out --port 8800

# 后台运行
nohup uv run trendspec research serve --out ./research_out --port 8800 > serve.log 2>&1 &
```

浏览器打开 `http://127.0.0.1:8800`。

---

## 可用策略

| 策略名 | 类型 | 说明 |
|--------|------|------|
| `clenow_momentum` | 量化动量 | Clenow《Stocks on the Move》：指数回归斜率×R² 排名 |
| `ema_cluster_pullback` | EMA 密集回踩 | 日 EMA 密集 + 周线回踩 + 多头趋势 |
| `episodic_pivot` | 突破回踩 | 缺口 + 放量 + 底部压缩突破 |
| `rs_ema_cross` | 相对强度 | Top-N 周度轮动，相对基准走强 |
| `ma_cross` | 趋势跟踪 | 双均线交叉 |
| `minervini_trend` | 动量筛选 | Minervini 趋势模板 6 项过滤 |
| `rsi_reversal` | 均值回归 | RSI 超卖买入 |
| `sector_momentum` | 行业动量 | 行业内相对动量排名 |

策略参数详见 CLAUDE.md 或各策略源码。

---

## 股票池

| 市场 | 来源 | 只数 |
|------|------|------|
| 美股 | SP500 + Russell1000 | ~1017 |
| A 股 | CSI800 | ~800 |

---

## 开发

```bash
uv run pytest          # 测试
uv run ruff check .    # lint
uv run ruff format .   # 格式化
```

---

## 数据源

群辉 NAS `stocks` 数据库：

| 表名 | 说明 |
|------|------|
| `prices` | 日线 OHLCV |
| `prices_weekly` | 周线 OHLCV |
| `prices_intraday` | 1h OHLCV |
| `stocks` | 基本信息 + GICS 行业 |
| `index_constituents` | 指数成分快照 |
| `constituent_changes` | 成分变动历史 |
| `index_prices` | 指数日线 |

---

## 编写自定义策略

继承 `BaseStrategy`，实现 `init()` + `next()`：

```python
from trendspec.strategy import BaseStrategy, register_strategy, StrategyContext

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    params = {"period": 20}

    def init(self, ctx: StrategyContext) -> None:
        ctx.precompute_indicator("MA", period=self.params["period"])

    def next(self, ctx: StrategyContext) -> None:
        ma = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=self.params["period"])
        if ma and ctx.close > ma and not ctx.has_position():
            ctx.signal("BUY", ctx.instrument_id, ctx.close)
```

参考 `trendspec/strategy/examples/`。