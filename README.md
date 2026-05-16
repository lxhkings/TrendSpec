# TrendSpec

量化回测与选股系统，支持 A 股和美股。

## 功能

- 双模式：历史回测 + 每日选股，同一策略代码通用
- PIT（Point-in-Time）宇宙，避免生存者偏差
- 本地 Parquet 缓存，快速数据访问
- 支持 A 股（沪深）和美股（NYSE/Nasdaq）

## 环境要求

- Python >= 3.11
- 群辉 NAS 或其他 MariaDB/MySQL 数据源

## 安装

```bash
uv sync
```

## 配置

复制 `.env.example` 到 `.env` 并填写：

```bash
cp .env.example .env
```

主要配置项：

```
DB_HOST=192.168.8.9        # 群辉 NAS IP
DB_PORT=3306
DB_USER=root               # 建议使用只读账户
DB_PASSWORD=...
DB_NAME=stocks
DATA_LAKE_ROOT=./data_lake
ALLOW_ROOT_DB_USER=true    # 使用 root 账户时需要设置
```

## 数据摄入

### 首次初始化（只需一次）

```bash
# 美股：拉取全量历史 + 成分 + 行业
uv run trendspec ingest daily --market us --full
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us

# A 股：同上，换成 cn
uv run trendspec ingest daily --market cn --full
uv run trendspec ingest components --market cn
uv run trendspec ingest sectors --market cn
```

> `components` 和 `sectors` 是低频数据（成分每年变几次，行业基本不变），首次运行后无需重复。

### 日常增量更新

```bash
uv run trendspec ingest daily --market us   # 只拉新数据，已有数据不重复
uv run trendspec ingest daily --market cn
```

### 查看同步状态

```bash
uv run trendspec ingest status --market us
```

## 可用策略

| 策略名 | 类型 | 说明 |
|--------|------|------|
| `clenow_momentum` | 量化动量 | Clenow《Stocks on the Move》：指数回归斜率×R² 排名，ATR 仓位，每周调仓 |
| `ma_cross` | 趋势跟踪 | 双均线交叉（短期 MA 上穿长期 MA 买入） |
| `rsi_reversal` | 均值回归 | RSI 超卖买入、超买卖出 |
| `sector_momentum` | 行业动量 | 行业内相对动量排名，买入前 10% |

### clenow_momentum 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `score_period` | 90 | 指数回归回望窗口（交易日） |
| `sma_period` | 200 | 趋势过滤均线（价格须在此均线上方） |
| `atr_period` | 20 | ATR 周期，用于仓位计算 |
| `risk_factor` | 0.001 | 每单位 ATR 分配的权益比例 |
| `rebalance_weekday` | 2 | 调仓日（0=周一…4=周五，默认周三） |
| `top_pct` | 0.8 | 持有排名前多少比例（默认前 80%） |
| `max_gap` | -0.15 | 90 日内单日最大跌幅过滤（-15%） |

## 回测

```bash
# 查看可用策略
uv run trendspec backtest list

# Clenow 动量策略回测
uv run trendspec backtest run --strategy clenow_momentum --market us --start 2020-01-01 --end 2024-12-31

# MA 交叉策略回测
uv run trendspec backtest run --strategy ma_cross --market us --start 2020-01-01 --end 2024-12-31

# 指定初始资金
uv run trendspec backtest run --strategy clenow_momentum --market us --start 2023-01-01 --capital 1000000
```

## 选股

```bash
# Clenow 动量选股
uv run trendspec screen --strategy clenow_momentum --market us --date 2024-05-15

# MA 交叉选股
uv run trendspec screen --strategy ma_cross --market us --date 2024-05-15
```

## 查看所有命令

```bash
uv run trendspec --help
```

## 数据源说明

系统读取群辉 NAS 上的 `stocks` 数据库，表结构：

| 表名 | 说明 |
|------|------|
| `prices` | 日线 OHLCV（美股为 Yahoo 复权价，A 股为 Tushare 后复权价）|
| `stocks` | 股票基本信息，含 GICS 行业分类 |
| `constituent_changes` | 指数成分变动（CSI800 / SP500 / HSI）|

## 编写自定义策略

继承 `BaseStrategy`，实现 `init()` 和 `next()`：

```python
from trendspec.strategy import BaseStrategy, register_strategy, StrategyContext

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    params = {"period": 20}

    def init(self, ctx: StrategyContext) -> None:
        ctx.precompute_indicator("MA", period=self.get_param("period", 20))

    def next(self, ctx: StrategyContext) -> None:
        ma = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=self.get_param("period", 20))
        if ma and ctx.close > ma and not ctx.has_position():
            ctx.signal("BUY", ctx.instrument_id, ctx.close)
```

参考 `trendspec/strategy/examples/` 下的四个示例策略。

## 开发

```bash
# 运行测试
uv run pytest

# 代码检查
uv run ruff check .
uv run ruff format .
```
