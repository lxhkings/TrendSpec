# EMA Cluster Pullback 策略设计

**日期：** 2026-05-21
**策略名：** `ema_cluster_pullback`
**版本：** 1.0.0
**作者：** lxhkings (with Claude)

## 1. 目标与动机

将「日线均线密集缠绕 + 周线股价回踩 20 周线 + 趋势仍向上」的视觉形态量化为可回测、可筛选的策略。捕捉中长期上升趋势中的健康回撤入场点。

视觉特征参考用户提供的 K 线截图：日线 EMA20/60/120 在 E 点处极度贴合，周线收盘回到 20 周 EMA 附近，整体仍处多头排列。

## 2. 决策摘要（来自 brainstorming）

| 项 | 决策 |
|---|---|
| 均线类型 | EMA（不是 SMA） |
| 周线数据来源 | 群辉 MariaDB `weekly_prices` 表 → ingest 到 `data_lake/<market>/weekly/` |
| 周线接入 | 扩展 `StrategyContext`，新增 `precompute_weekly_indicator` / `weekly_indicator_value` |
| 日 EMA120 斜率 | 与 20 日前比较（与 Minervini 范式一致） |
| 周 EMA20 斜率 | 与上一已完成周比较 |
| 信号方向 | BUY + SELL（完整回测可用） |
| SELL 条件 | 收盘 < EMA60 连续 2 日 **或** 回撤 ≥ 8%（硬止损） |
| 确认天数 | 2 日 |
| 阈值（默认，可 params 覆盖） | 日 4%、周 2.5% |
| 大盘过滤 | 启用：指数 > EMA200 才发 BUY；US=SP500，CN=CSI800 |
| Universe | 活跃股 + 20 日均成交额 ≥ 阈值（US 5M USD，CN 50M CNY） |

## 3. 架构总览

```
群辉 MariaDB
  ├─ prices (日线) ────────► data_lake/<market>/daily/  (现有)
  └─ weekly_prices ───────► data_lake/<market>/weekly/  (新增)

BacktestEngine / ScreeningEngine
  ├─ load daily bars (现有)
  └─ load weekly bars (新增) ─► StrategyContext
                                  ├─ precompute_indicator       (现有, 日线)
                                  └─ precompute_weekly_indicator (新增)

EMACluster Strategy
  init():
    EMA20/60/120 (日)、ADV20 (日)、EMA20 (周)、大盘 EMA200 缓存
  next():
    BUY  = 日密集 ∧ 周近 ∧ 日趋势↑ ∧ 周趋势↑ ∧ 大盘 ∧ 流动性  (连 2 日)
    SELL = 收 < EMA60 (连 2 日) ∨ 回撤 ≥ 8%
```

### 关键 boundary

- `weekly_indicator_value(iid, as_of_date=t)` 返回**最近完成周** bar 的指标值；**绝不读未完成本周**，防止 lookahead 泄漏。
- 周 & 日 indicator 注册表共用：`compute_indicator` 不关心输入 DataFrame 的频率。
- 周 ingestor 与日 ingestor 共用 `stocks_db_ingestor.py`，新增函数对应新 SQL，不动现有函数。

## 4. 组件清单

### 4.1 `trendspec/ingest/stocks_db_ingestor.py`（修改）

新增函数：

```python
def ingest_us_weekly(client, root, full=False) -> int: ...
def ingest_cn_weekly(client, root, full=False) -> int: ...
```

- SQL：`SELECT ticker, date, open, high, low, close, volume FROM weekly_prices p JOIN stocks s ON p.ticker=s.ticker WHERE s.exchange IN (...)`
- `date` 对应每周收盘日（约定按群辉表自身定义，通常为周五或周末最后交易日）
- 输出 schema 与日线完全一致：`(instrument_id, date, open, high, low, close, volume, adj_factor=1.0)`
- 写入路径：`data_lake/<market>/weekly/instrument_id=<id>/<year>.parquet`

### 4.2 `trendspec/cli/ingest_commands.py`（修改）

新增子命令：

```bash
uv run trendspec ingest weekly --market us [--full]
uv run trendspec ingest weekly --market cn [--full]
```

### 4.3 `trendspec/data/parquet_loader.py`（修改）

`bars()` 增加 `frequency` 参数：

```python
def bars(
    market: Market,
    ...,
    frequency: Literal["daily", "weekly"] = "daily",
) -> pl.DataFrame:
    lf = scan_parquet(root, market, frequency)   # 替换硬编码 "daily"
    ...
```

默认 `daily`，向后兼容现有所有调用。

### 4.4 `trendspec/engine/base_engine.py`（修改）

`_load_data()` 加载周线（可选）：

```python
self._data = bars(market=..., frequency="daily", ...)
try:
    self._weekly_data = bars(market=..., frequency="weekly", ...)
except (FileNotFoundError, NoParquetFilesError):
    self._weekly_data = None
```

注入 `StrategyContext`：

```python
StrategyContext(..., weekly_data=self._weekly_data)
```

### 4.5 `trendspec/strategy/context.py`（修改）

新字段：

```python
self._weekly_data: pl.DataFrame | None = weekly_data
self._weekly_indicator_cache: dict[str, pl.DataFrame] = {}
self._weekly_indicator_fast: dict[str, dict[tuple, float]] = {}
self._weekly_dates_by_iid: dict[str, list[DateType]] = {}  # 用于二分查找
```

新方法：

```python
def precompute_weekly_indicator(self, name: str, **params) -> pl.DataFrame:
    """对 self._weekly_data 调用 compute_indicator, 与日线指标缓存隔离."""

def weekly_indicator_value(
    self, name: str, instrument_id: str | None = None,
    as_of_date: DateType | None = None, **params,
) -> float | None:
    """
    定位 ≤ as_of_date 的最近已完成周 bar, 返回该周 bar 的指标值.
    不会返回未完成的本周 bar (防 lookahead).
    """

def weekly_close(self, instrument_id: str | None = None,
                 as_of_date: DateType | None = None) -> float | None:
    """便捷方法: 最近完成周的收盘价."""

def _resolve_week_end(self, iid: str, as_of_date: DateType) -> DateType | None:
    """二分查找该 iid 在 weekly_data 中 ≤ as_of_date 的最大 date."""
```

### 4.6 `trendspec/strategy/examples/ema_cluster_pullback.py`（新建）

```python
@register_strategy("ema_cluster_pullback")
class EMAClusterPullback(BaseStrategy):
    name = "ema_cluster_pullback"
    version = "1.0.0"
    params = {
        "ema_short": 20,
        "ema_mid": 60,
        "ema_long": 120,
        "daily_cluster_threshold": 0.04,        # 4%
        "weekly_proximity_threshold": 0.025,    # 2.5%
        "ema_long_slope_lookback": 20,          # EMA120 vs 20 日前
        "weekly_ema_period": 20,
        "adv_lookback": 20,
        "adv_threshold_us": 5_000_000,          # USD
        "adv_threshold_cn": 50_000_000,         # CNY
        "market_index_id_us": "SP500",
        "market_index_id_cn": "CSI800",
        "market_ema_period": 200,
        "market_filter_enabled": True,
        "confirmation_days": 2,
        "stop_loss_pct": 0.08,
        "sell_ma_period": 60,                   # 跌破 EMA60 触发卖
    }
```

### 4.7 测试文件（新建）

- `tests/test_weekly_ingestor.py` — `ingest_us_weekly` / `ingest_cn_weekly` 用 SQLite 内存库
- `tests/test_weekly_loader.py` — `bars(frequency="weekly")` 加载
- `tests/test_weekly_context.py` — `weekly_indicator_value` 日→周映射、未完成周不泄漏
- `tests/strategy/test_ema_cluster_pullback.py` — 完整信号判定（手工构造样本）

## 5. 信号计算细节

### 5.1 `init()` 预计算

```python
def init(self, ctx):
    ctx.precompute_indicator("EMA", period=20)
    ctx.precompute_indicator("EMA", period=60)
    ctx.precompute_indicator("EMA", period=120)
    ctx.precompute_weekly_indicator("EMA", period=20)

    # ADV20 = rolling 20 日均成交额 (close * volume)
    # 现有指标库无 ADV, 在 init() 里向量化算一次后存入 self._adv20_fast: dict[(iid, date), float]
    self._adv20_fast = self._compute_adv20(ctx._data, lookback=20)

    self._market_ema_cache: dict[tuple, float | None] = {}
    self._entry_price: dict[str, float] = {}
    self._buy_pass_history: dict[str, deque[bool]] = {}
    self._sell_break_history: dict[str, deque[bool]] = {}
```

### 5.2 `next()` BUY 判定（六合一，连 2 日确认）

```
ema20  = indicator_value("EMA", iid, t, period=20)
ema60  = indicator_value("EMA", iid, t, period=60)
ema120 = indicator_value("EMA", iid, t, period=120)
ema120_prev = indicator_value("EMA", iid, t_minus_20_trading_days, period=120)
                                    # 交易日索引 today_idx - 20, 与 Minervini Rule 2 一致

weekly_ema20      = weekly_indicator_value("EMA", iid, t,                            period=20)
weekly_ema20_prev = weekly_indicator_value("EMA", iid, t_minus_1_completed_week,     period=20)
                                    # 取再上一已完成周 (week_end_prev = week_end - 7 日, 再用 _resolve_week_end 二分定位)

adv20 = self._adv20_fast.get((iid, t))   # 已在 init 预算 (close * volume) 滚动均值
index_ema200 = market_ema(index_id, t, 200)

C1_daily_cluster = (max(e20, e60, e120) - min(e20, e60, e120)) / min(...) < 0.04
C2_weekly_near   = abs(close - weekly_ema20) / weekly_ema20 < 0.025
C3_daily_trend   = ema120 > ema120_prev
C4_weekly_trend  = weekly_ema20 > weekly_ema20_prev
C5_market_ok     = (not market_filter_enabled) or (index_close(t) > index_ema200)
C6_liquid        = adv20 >= adv_threshold

buy_signal_today = C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6
push to buy_pass_history[iid] (maxlen=2)
if all(history) and len==2 and not has_position(iid):
    signal("BUY", iid, close, note="EMA cluster + weekly pullback")
    entry_price[iid] = close
```

### 5.3 `next()` SELL 判定

```python
# 硬止损 — 单日触发
if has_position(iid) and close <= entry_price[iid] * (1 - 0.08):
    signal("SELL", iid, close, note="stop_loss_8pct")
    cleanup(iid)
    return

# 跌破 EMA60 — 连 2 日确认
break_ema60 = close < ema60
push to sell_break_history[iid] (maxlen=2)
if all(history) and len==2 and has_position(iid):
    signal("SELL", iid, close, note="break_ema60_2d")
    cleanup(iid)
```

`cleanup(iid)`：清空 `entry_price[iid]`、`sell_break_history[iid]`、`buy_pass_history[iid]`。

## 6. 防 lookahead 泄漏证明

| 数据点 | 保证 |
|---|---|
| 日线 EMA20/60/120 | `compute_indicator` rolling 不向后看 |
| `weekly_indicator_value(iid, t)` | `_resolve_week_end` 二分查找 `date <= t` 的最大周收盘日，只读已完成周 |
| `weekly_ema20_prev` | 同上，取再上一周 |
| `ema120_prev = (t-20)` | 历史值 |
| `adv20` | 滚动 20 日窗口 |
| `index_ema200` | 基于 `index_close` 历史，缓存键含 `t` |

特殊场景：

- **新股**周线 < 20 周：`weekly_ema20=None` → BUY 不发
- **节假日**周末日期（如 SHFE 春节连休）：周线表本身按交易日聚合，不影响
- **t = 周一开盘**：最近完成周是上周五（或更早）
- **t = 本周三**：最近完成周仍是上周五，不会偷看本周 bar

## 7. 错误处理与容错

| 场景 | 行为 |
|---|---|
| `weekly_data is None`（周线未 ingest） | `init()` 不预算周指标；`weekly_indicator_value` 返回 `None`；BUY 永不触发；日志警告 |
| 单只股票指标 `None`（数据不足） | 信号视为 `False`，写入 history（保持时序对齐），不发信号 |
| `entry_price[iid]` 缺失但持仓中 | 跳过止损判定，仅依赖 EMA60 跌破出场。日志警告。 |
| `market_filter_enabled=True` 但大盘指数缺失 | 视为通过（与 minervini 一致），日志警告 |
| 同一只股票同一天满足 BUY 与 SELL | SELL 优先（先平仓再考虑新仓） |

## 8. 测试矩阵

### 8.1 Ingestor (`test_weekly_ingestor.py`)

- US weekly：mock MariaDB 含 `weekly_prices` + `stocks`，验证写入正确分区
- CN weekly：验证 `instrument_id` 推导（`SH{ticker}` / `SZ{ticker}`）
- 缺失字段：`volume=NULL` 时降级到 0
- 增量：`full=False` 模式只追加新数据

### 8.2 Loader (`test_weekly_loader.py`)

- `bars(market=US, frequency="weekly")` 读出周 bar
- `bars(frequency="daily")` 仍正常工作
- 周线目录不存在时返回空 DataFrame

### 8.3 Context (`test_weekly_context.py`)

- `weekly_indicator_value(iid, as_of_date=周一)` 返回上周五值
- `weekly_indicator_value(iid, as_of_date=周三)` 返回上周五值，不偷看本周
- `weekly_indicator_value(iid, as_of_date=已完成周五收盘后)` 返回当天周值
- 周线数据缺失返回 `None`
- 二分查找性能（大数据量）

### 8.4 Strategy (`test_ema_cluster_pullback.py`)

- 构造满足所有条件的样本 → 2 日确认后发 BUY
- 第二天指标失败 → 不发 BUY（确认中断）
- 持仓后收盘 < EMA60 连 2 日 → 发 SELL
- 持仓后收盘 ≤ entry * 0.92 → 单日发 SELL（硬止损）
- 大盘指数 < EMA200 → BUY 不发
- ADV20 < 阈值 → BUY 不发
- 周线数据缺失 → BUY 不发
- 同日 BUY 与 SELL 冲突 → SELL 优先

## 9. CLI 用法（实施完成后）

```bash
# 一次性: 入库周线全量
uv run trendspec ingest weekly --market us --full
uv run trendspec ingest weekly --market cn --full

# 日常增量
uv run trendspec ingest daily   --market us
uv run trendspec ingest weekly  --market us

# 选股
uv run trendspec screen run --strategy ema_cluster_pullback --market us --date 2026-05-21

# 回测
uv run trendspec backtest run \
  --strategy ema_cluster_pullback \
  --market us \
  --start 2020-01-01 --end 2026-05-21

# 调参（举例：放宽到 5% 日密集 + 3% 周近）
uv run trendspec screen run --strategy ema_cluster_pullback --market us \
  --param daily_cluster_threshold=0.05 \
  --param weekly_proximity_threshold=0.03
```

## 10. 开放问题（实施时再定）

- ADV20 实施细节：默认用 `mean(close * volume, 20)` 自算（兼容现有 schema）；若群辉 `prices` 表有 `amount` 字段，可改为更精确的 `mean(amount, 20)`
- 同时多只股票发 BUY 时排序：按 `C1` 密集度（越小越好）排序，前 N 个进 universe
- A 股周线日期定义：用群辉表自身的 `date`（不强制周五），与日线无缝对齐

## 11. 工作量估算

| 层 | 行数 |
|---|---|
| Ingestor (`ingest_us_weekly` + `ingest_cn_weekly`) | ~50 |
| CLI (`ingest weekly` 子命令) | ~20 |
| Loader (`bars` 加 frequency 参数) | ~10 |
| Engine (加载 weekly 注入 context) | ~15 |
| Context (`precompute_weekly_indicator` + `weekly_indicator_value` + 辅助) | ~80 |
| Strategy (`ema_cluster_pullback.py`) | ~200 |
| 测试 (四个测试文件) | ~250 |
| **合计** | **~625** |

约半天到一天工作量（在群辉已有 `weekly_prices` 表的前提下）。
