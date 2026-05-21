# TrendSpec

量化回测与选股系统，支持 A 股（CSI800）和美股（SP500 + Russell1000）。

## 功能

- 双模式：历史回测 + 每日选股，同一策略代码通用
- PIT（Point-in-Time）宇宙，避免生存者偏差
- 本地 Parquet 数据湖，选股无需实时连接数据库
- 行业中文显示（GICS 标准分类）
- 选股结果输出终端决策表 + CSV 导出
- **信号历史**：策略历史信号命中率、均值收益、胜率统计，附于每次选股报告

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

按实际情况修改 `.env`：

```
DB_HOST=192.168.8.9
DB_PORT=3306
DB_USER=root
DB_PASSWORD=<你的密码>
DB_NAME=stocks
DATA_LAKE_ROOT=./data_lake
ALLOW_ROOT_DB_USER=true   # 开发环境使用 root 账号时必须加
```

> **注意：** `ALLOW_ROOT_DB_USER=true` 必须写在 `.env` 文件中，写在 shell 环境变量里无效（pydantic-settings 从 `.env` 读取，不读 `os.environ`）。

## 数据摄入

### 首次初始化（只需一次）

```bash
# 美股：SP500 + Russell1000，全量历史
uv run trendspec ingest daily --market us --full
uv run trendspec ingest weekly --market us --full   # 周线数据
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us

# A 股：CSI800，全量历史
uv run trendspec ingest daily --market cn --full
uv run trendspec ingest weekly --market cn --full   # 周线数据
uv run trendspec ingest components --market cn
uv run trendspec ingest sectors --market cn
```

> `components` 和 `sectors` 是低频数据（成分每年变几次，行业基本不变），首次运行后无需重复。周线数据可选，仅部分策略需要。

### 日常增量更新

```bash
uv run trendspec ingest daily --market us   # 合并新数据，已有历史不覆盖
uv run trendspec ingest daily --market cn
uv run trendspec ingest weekly --market us  # 周线增量（可选）
uv run trendspec ingest weekly --market cn
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
| `ema_cluster_pullback` | EMA 密集回踩 | 日 EMA20/60/120 密集缠绕 + 周线回踩 EMA20 + 多头趋势确认，连续 2 日触发 |
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
| `max_per_sector` | 0 | 每个行业最多选几只（0 = 不限；1 = 每行业只选 score 最高那只） |
| `atr_stop_k` | 3.0 | 初始止损 = 收盘价 − k × ATR |
| `drawdown_period` | 63 | 回撤基准窗口（日） |
| `volume_avg_period` | 50 | 成交量均量窗口（日） |
| `warn_deviation_max` | 40.0 | 乖离率预警阈值（超过则标注"均线乖离过大"） |
| `warn_vol_mult_low` | 1.0 | 放量倍数下限（低于则"量能萎缩"） |
| `warn_vol_mult_high` | 3.0 | 放量倍数上限（高于则"放量过快"） |
| `warn_drawdown_max` | -15.0 | 回撤预警阈值（低于则"回撤过深"） |

### ema_cluster_pullback 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ema_short` | 20 | 短期 EMA 周期（日线） |
| `ema_mid` | 60 | 中期 EMA 周期（日线） |
| `ema_long` | 120 | 长期 EMA 周期（日线） |
| `daily_cluster_threshold` | 0.04 | 日线 EMA 密集阈值：(max−min)/min < 4% |
| `weekly_proximity_threshold` | 0.025 | 周线 proximity 阈值：|close−weekly_EMA20| / weekly_EMA20 < 2.5% |
| `weekly_ema_period` | 20 | 周线 EMA 周期 |
| `ema_long_slope_lookback` | 20 | EMA120 斜率回望（交易日） |
| `adv_threshold_us` | 5_000_000 | 美股 ADV20 阈值（美元） |
| `adv_threshold_cn` | 50_000_000 | A 股 ADV20 阈值（人民币） |
| `market_filter_enabled` | True | 指数过滤：指数收盘 > 指数 EMA200 |
| `confirmation_days` | 2 | 连续满足条件天数 |
| `stop_loss_pct` | 0.08 | 硬止损：收盘 ≤ entry × (1−8%) |
| `sell_ma_period` | 60 | SELL 条件 EMA 周期（跌破 EMA60） |

> EMA 密集回踩策略需要周线数据。运行前请先执行 `uv run trendspec ingest weekly --market us`。

## 选股

```bash
# Clenow 动量选股（任意日期，自动用当天数据）
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-14
uv run trendspec screen run --strategy clenow_momentum --market cn --date 2026-05-14

# 每行业只选最高分一只
uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-14 --param max_per_sector=1

# 其他策略
uv run trendspec screen run --strategy ma_cross --market us --date 2026-05-14
```

选股输出包含：行业、选股排名、建议买入价、初始止损线、趋势质量（R²）、乖离率、回撤、放量倍数、预警信息，以及历史信号统计（如已构建）。

CSV 文件保存为 `results/screening/signals_<strategy>_<date>.csv`。

## 信号历史

回放策略历史信号，计算每个标的的远期收益率（T+1/3/5/10/20 交易日），聚合为命中率和均值收益，缓存为 Parquet 供选股报告实时查询。

### 首次构建（慢，约 10 年历史）

```bash
uv run trendspec signal-history build --strategy clenow_momentum --market us
uv run trendspec signal-history build --strategy clenow_momentum --market cn
```

### 日常增量更新

```bash
# 不加 --rebuild 自动增量，只补最后缓存日之后的新数据
uv run trendspec signal-history build --strategy clenow_momentum --market us
```

### 查看缓存状态

```bash
uv run trendspec signal-history status --strategy clenow_momentum --market us
```

### 选股报告中的历史统计列

| 列名 | 说明 |
|------|------|
| `历史样本数` | 该标的历史买入信号总次数 |
| `历史 1d 均值收益 %` | 信号后 1 交易日平均收益 |
| `历史 5d 均值收益 %` | 信号后 5 交易日平均收益 |
| `历史 20d 均值收益 %` | 信号后 20 交易日平均收益 |
| `历史 5d 胜率 %` | 信号后 5 交易日收益为正的概率 |
| `信号置信度` | ★ < 5 次，★★ 5–9 次，★★★ ≥ 10 次 |

> 未运行 `signal-history build` 前，上述列显示 `-`，选股报告仍正常工作。

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
| `weekly_prices` | 周线 OHLCV（周收盘日聚合，用于 EMA 密集回踩等策略） |
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
