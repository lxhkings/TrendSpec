# TrendSpec

量化回测与选股系统，支持 A 股（CSI800）和美股（SP500 + Russell1000）。

## 功能

- 双模式：历史回测 + 每日选股，同一策略代码通用
- PIT（Point-in-Time）宇宙，避免生存者偏差
- 本地 Parquet 数据湖，选股无需实时连接数据库
- 行业中文显示（GICS 标准分类）
- 选股结果输出终端 10 列决策表 + CSV 导出

## 环境要求

- Python >= 3.11
- 群辉 NAS MariaDB（仅入库时需要）

## 安装

```bash
uv sync
```

## 配置

复制 `.env.example` 到 `.env`：

```
DB_HOST=192.168.8.9
DB_PORT=3306
DB_USER=root
DB_PASSWORD=...
DB_NAME=stocks
DATA_LAKE_ROOT=./data_lake
ALLOW_ROOT_DB_USER=true
```

## 数据摄入

### 首次初始化（只需一次）

```bash
# 美股：SP500 + Russell1000，全量历史
uv run trendspec ingest daily --market us --full
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us

# A 股：CSI800，全量历史
uv run trendspec ingest daily --market cn --full
uv run trendspec ingest components --market cn
uv run trendspec ingest sectors --market cn
```

> `components` 和 `sectors` 是低频数据（成分每年变几次，行业基本不变），首次运行后无需重复。

### 日常增量更新

```bash
uv run trendspec ingest daily --market us   # 合并新数据，已有历史不覆盖
uv run trendspec ingest daily --market cn
```

### 查看同步状态

```bash
uv run trendspec ingest status --market us
```

## 股票池范围

| 市场 | 来源 | 只数 |
|------|------|------|
| 美股 | SP500 + Russell1000（`index_constituents`） | ~1017 |
| A 股 | CSI800（`index_constituents`） | ~800 |

## 可用策略

| 策略名 | 类型 | 说明 |
|--------|------|------|
| `clenow_momentum` | 量化动量 | Clenow《Stocks on the Move》：指数回归斜率×R² 排名，ATR 仓位，每周调仓 |
| `ma_cross` | 趋势跟踪 | 双均线交叉（短期 MA 上穿长期 MA 买入） |
| `minervini_trend` | 动量筛选 | Minervini 趋势模板：6 项纯技术指标过滤，2 日确认 |
| `rsi_reversal` | 均值回归 | RSI 超卖买入、超买卖出 |
| `sector_momentum` | 行业动量 | 行业内相对动量排名，买入前 10% |

### clenow_momentum 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `score_period` | 90 | 指数回归回望窗口（交易日） |
| `sma_period` | 200 | 趋势过滤均线（价格须在此均线上方） |
| `atr_period` | 20 | ATR 周期，用于仓位计算 |
| `risk_factor` | 0.001 | 每单位 ATR 分配的权益比例 |
| `rebalance_weekday` | 2 | 调仓日（0=周一…4=周五，默认周三）；选股模式自动跳过此限制 |
| `top_pct` | 0.8 | 持有排名前多少比例（默认前 80%） |
| `max_gap` | -0.15 | 90 日内单日最大跌幅过滤（-15%） |
| `atr_stop_k` | 3.0 | 初始止损 = 收盘价 − k × ATR |
| `drawdown_period` | 63 | 回撤基准窗口（日） |
| `volume_avg_period` | 50 | 成交量均量窗口（日） |
| `warn_deviation_max` | 40.0 | 乖离率预警阈值（超过则标注"均线乖离过大"） |
| `warn_vol_mult_low` | 1.0 | 放量倍数下限（低于则"量能萎缩"） |
| `warn_vol_mult_high` | 3.0 | 放量倍数上限（高于则"放量过快"） |
| `warn_drawdown_max` | -15.0 | 回撤预警阈值（低于则"回撤过深"） |

## 选股

```bash
# Clenow 动量选股（任意日期，自动用当天数据）
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-14
uv run trendspec screen run --strategy clenow_momentum --market cn --date 2026-05-14

# 其他策略
uv run trendspec screen run --strategy ma_cross --market us --date 2026-05-14
```

选股输出包含：行业、选股排名、建议买入价、初始止损线、趋势质量（R²）、乖离率、回撤、放量倍数、预警信息。

CSV 文件保存为 `results/screening/signals_<strategy>_<date>.csv`。

## 回测

```bash
uv run trendspec backtest list

uv run trendspec backtest run --strategy clenow_momentum --market us --start 2020-01-01 --end 2024-12-31

uv run trendspec backtest run --strategy clenow_momentum --market us --start 2023-01-01 --capital 1000000

uv run trendspec backtest compare --market us --start 2022-01-01 --end 2024-12-31 --sort sharpe --export csv
```

## 数据源

数据来自群辉 NAS `stocks` 数据库（入库后即可离线使用）：

| 表名 | 说明 |
|------|------|
| `prices` | 日线 OHLCV（美股 Yahoo 复权价，A 股 Tushare 后复权价） |
| `stocks` | 基本信息，含 GICS 行业分类 |
| `index_constituents` | 指数成分快照（SP500 / Russell1000 / CSI800 / HSI） |
| `constituent_changes` | 指数成分变动历史 |
| `index_prices` | 指数日线价格 |

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

参考 `trendspec/strategy/examples/` 下的示例策略。

## 开发

```bash
uv run pytest          # 运行测试
uv run ruff check .    # 代码检查
uv run ruff format .   # 代码格式化
```
