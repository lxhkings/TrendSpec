# Clenow Momentum 选股输出增强 — 设计规范

**日期：** 2026-05-18
**范围：** 仅 `clenow_momentum` 策略的 BUY 信号输出（终端 + CSV）
**通用副作用：** `ScreeningReport` 导出文件名规则变更（所有策略）

---

## 1. 目标

在 `trendspec screen run --strategy clenow_momentum` 输出中，将当前 6 列信号表扩展为 10 列决策表，便于人工复核每只买入候选的：

- 板块归属
- 排名位次
- 风控止损位
- 趋势质量（R²）
- 价格相对趋势的位置（MA200 乖离率）
- 短期回撤
- 量能配合
- 自动预警标签

非目标：

- 不动其他策略输出
- 不引入分时数据 / 财务因子
- 不参与策略下单逻辑（止损线、预警仅为可视化字段，策略退出规则仍按现有 trend filter + rank-out 逻辑）

## 2. 输出示例（终端表）

| 股票代码 | 行业 | 选股排名 | 建议买入价 | 初始止损线 | 趋势质量 (R²) | 乖离率 (距 MA200) | 回撤 (距 63 日高点) | 放量倍数 | 备注/预警 |
|---------|------|---------|-----------|-----------|---------------|-------------------|---------------------|---------|----------|
| LITE | Technology | #1 | $1001.81 | $769.31 | 0.85 (极平稳) | +32.5% | -2.1% | 1.5x | 正常 |
| CIEN | Technology | #2 | $591.57 | $493.38 | 0.79 (优秀) | +48.2% | -1.5% | 0.8x | [警报] 均线乖离过大，量能萎缩 |
| DELL | Technology | #3 | $247.89 | $213.15 | 0.72 (良好) | +15.4% | -8.5% | 2.1x | 正常 |
| VRT | Industrials | #4 | $376.23 | $328.05 | 0.68 (一般) | +22.1% | -5.0% | 1.2x | 正常 |

注：行业列使用 GICS 粗 sector（US 共 11 个英文分类，CN 使用申万 L1 28 个中文分类）。

## 3. 架构与数据流

```
universe ─┬─→ ClenowMomentumStrategy.next()
          │       ├─ 查 indicator: MA200, ATR20, CLENOW_SCORE/R2, HH63, SMA_VOL_50
          │       ├─ 查 sectors.sector(iid, date) PIT
          │       ├─ 排序后赋 rank
          │       └─ 创建 Signal(direction=BUY, ..., extras={...})
          │
          ↓
        Signal.extras = {
            sector, rank, r2, deviation_pct,
            drawdown_pct, vol_mult, stop_loss, alerts: [str]
        }
          ↓
        ScreeningReport.output()
            ├─ 默认 6 列布局（其他策略走原路径）
            └─ strategy=='clenow_momentum' → 10 列布局
        ScreeningReport.export()
            └─ CSV 文件名: signals_<strategy>_<YYYYMMDD>.csv  ← 通用修复
```

模块改动清单：

| 文件 | 改动类型 | 改动 |
|------|---------|------|
| `trendspec/strategy/signal.py` | 扩展 | `Signal` 增 `extras: dict = field(default_factory=dict, repr=False)` |
| `trendspec/strategy/indicators.py` | 新增 | 注册 `HH`、`SMA_VOLUME`、`CLENOW_R2` 三个 indicator |
| `trendspec/strategy/examples/clenow_momentum.py` | 扩展 | `init()` 多 precompute 三个指标；`next()` BUY 时填 extras；引入 sector PIT 查询；扩参数表 |
| `trendspec/screening/report.py` | 扩展 | 按 strategy_name 切渲染分支；CSV 文件名加 `<strategy>` 段；CSV schema 按策略切 |
| `tests/test_indicators.py` | 新增用例 | HH / SMA_VOLUME / CLENOW_R2 |
| `tests/test_strategies.py` | 新增用例 | clenow extras、阈值、缺失数据、参数校验 |
| `tests/test_screening_report.py` | 新建或扩展 | 14 列渲染、CSV 文件名、CSV 行长度 |

不动：engine、context、CLI、risk pipeline、其他策略文件。

## 4. 字段计算公式

| 字段 | 公式 | 数据来源 |
|------|------|---------|
| 股票代码 | `signal.ticker` | 已有 |
| 行业 | `sectors.sector(iid, current_date)` | `trendspec/data/sectors.py` PIT |
| 选股排名 | `rank = ranked.index(iid) + 1`，仅 BUY top 集合内（1-based） | 策略 next() 内排序后写入 |
| 建议买入价 | `signal.price` = 当日 close | 已有 |
| 初始止损线 | `stop_loss = close - atr_stop_k * ATR(20)` | ATR 已 precompute；`atr_stop_k` 新参数 |
| 趋势质量 R² | `r2 = CLENOW_R2_{score_period}` 列值 | 新注册 indicator |
| 乖离率 (距 MA200) | `deviation_pct = (close - MA200) / MA200 * 100` | MA200 已 precompute |
| 回撤 (距 63 日高点) | `drawdown_pct = (close - HH63) / HH63 * 100` | 新增 `HH(63)` indicator |
| 放量倍数 | `vol_mult = volume / SMA(volume, 50)` | 新增 `SMA_VOLUME(50)` indicator |
| 备注/预警 | 见 §6 预警规则 | 在策略层组装 |

### 4.1 R² indicator 实现选择

新注册 `CLENOW_R2` indicator，函数体内独立调用 `scipy.stats.linregress(x, ln(close)).rvalue ** 2`，**不**复用 `CLENOW_SCORE` 已算的 `CLENOW_R2_{period}` 列。

理由：
- 避免跨 indicator 依赖（注册系统每个 indicator 自包含）
- 保持单输出语义（一个 indicator 一个值）
- 重算开销可接受：单次 linregress ~20µs × universe size

### 4.2 R² 质量分档（仅渲染层）

终端 R² 列显示为 `f"{r2:.2f} ({label})"`：

| R² 范围 | label |
|---------|-------|
| ≥ 0.85 | 极平稳 |
| ≥ 0.75 | 优秀 |
| ≥ 0.65 | 良好 |
| 其他 | 一般 |

CSV 仅存原始 r2 数值（4 位小数）；分档标签仅用于终端可读性。

### 4.3 sector 数据源

- 调用 `trendspec.data.sectors.sector(iid, current_date)`（PIT 查询，O(1) lru_cache）
- 返回 `sector` 字段（GICS 粗板块：Energy、Materials、Industrials、Consumer Discretionary、Consumer Staples、Health Care、Financials、Technology、Communication Services、Utilities、Real Estate）
- CN 市场返回申万 L1 中文（28 个）
- 查询失败（None）→ extras["sector"] = None，渲染列显示 "-"

## 5. 数据结构变更

### 5.1 Signal 扩展

```python
# trendspec/strategy/signal.py
from typing import Any

@dataclass
class Signal:
    direction: Literal["BUY", "SELL"]
    ticker: str
    instrument_id: str
    price: float
    trigger_value: float | None = None
    note: str | None = None
    shares: float | None = field(default=None, repr=False)
    timestamp: float | None = field(default=None, repr=False)
    extras: dict[str, Any] = field(default_factory=dict, repr=False)  # NEW
```

向后兼容：其他策略不填 extras 即空 dict；现有路径不读 extras。

### 5.2 clenow BUY signal extras 结构（固定 8 键）

```python
sig.extras = {
    "sector": str | None,        # PIT sector, None if missing
    "rank": int,                  # 1-based, within top_pct cut
    "r2": float,                  # 0-1
    "deviation_pct": float,       # (close - MA200)/MA200 * 100
    "drawdown_pct": float,        # (close - HH63)/HH63 * 100
    "vol_mult": float,            # close_vol / SMA(vol, 50)
    "stop_loss": float,           # close - k * ATR(20)
    "alerts": list[str],          # 命中的预警标签
}
```

SELL signal 不填 extras。sell_reason 保留在 `note` 字段（与现有逻辑一致）。

## 6. 策略参数与预警规则

### 6.1 新增参数（追加到 `params` dict）

| 参数 | 类型 | 默认 | 用途 |
|------|------|-----|------|
| `atr_stop_k` | float | 3.0 | 止损 ATR 乘数 |
| `drawdown_period` | int | 63 | 回撤窗口（HH 周期） |
| `volume_avg_period` | int | 50 | 放量分母窗口 |
| `warn_deviation_max` | float | 40.0 | 乖离预警阈值 (%) |
| `warn_vol_mult_low` | float | 1.0 | 量能萎缩阈值 |
| `warn_vol_mult_high` | float | 3.0 | 放量过快阈值 |
| `warn_drawdown_max` | float | -15.0 | 回撤过深阈值 (%) |

### 6.2 参数校验（`_validate_dict_params`）

- `atr_stop_k > 0`
- `drawdown_period >= 2`
- `volume_avg_period >= 2`
- `warn_vol_mult_low < warn_vol_mult_high`
- `warn_drawdown_max < 0`
- `warn_deviation_max > 0`

违法值 raise ValueError（沿用现有 validation 风格）。

### 6.3 预警规则

按顺序判断，命中即追加到 `alerts: list[str]`：

| 触发条件 | alert 文本 |
|---------|-----------|
| `deviation_pct > warn_deviation_max` | "均线乖离过大" |
| `vol_mult < warn_vol_mult_low` | "量能萎缩" |
| `vol_mult > warn_vol_mult_high` | "放量过快" |
| `drawdown_pct < warn_drawdown_max` | "回撤过深" |

备注列拼接：

- 命中任意 alert：`"[警报] " + "，".join(alerts)`
- 无 alert：`"正常"`

## 7. Report 渲染与导出

### 7.1 渲染分流

`ScreeningReport._create_signals_table()` 入口：

```python
if title == "买入信号" and self.strategy_name == "clenow_momentum":
    return self._create_clenow_buy_table(signals)
return self._create_default_signals_table(signals, title)
```

SELL 表沿用原渲染。其他策略 BUY 表沿用原渲染。

### 7.2 10 列布局（终端 rich.Table）

| # | 列名 | 来源 | 格式化 |
|---|------|------|--------|
| 1 | 股票代码 | `signal.ticker` | str |
| 2 | 行业 | `extras["sector"]` | str，None → "-" |
| 3 | 选股排名 | `extras["rank"]` | `f"#{n}"` |
| 4 | 建议买入价 | `signal.price` | `f"${p:.2f}"` |
| 5 | 初始止损线 | `extras["stop_loss"]` | `f"${p:.2f}"` |
| 6 | 趋势质量 (R²) | `extras["r2"]` | `f"{r2:.2f} ({label})"` |
| 7 | 乖离率 (距 MA200) | `extras["deviation_pct"]` | `f"{p:+.1f}%"` |
| 8 | 回撤 (距 63 日高点) | `extras["drawdown_pct"]` | `f"{p:+.1f}%"` |
| 9 | 放量倍数 | `extras["vol_mult"]` | `f"{m:.1f}x"` |
| 10 | 备注/预警 | 拼接 alerts 或 "正常" | str |

预警行 rich style `red`，正常行 `white`。

### 7.3 CSV 导出（通用修复）

文件名变更：`signals_{strategy}_{date_str}.csv`

- 影响所有策略，不仅 clenow
- 已有 `signals_YYYYMMDD.csv` 文件不被覆盖；新旧文件并存直到用户清理
- CHANGELOG 注明文件名规则变更

CSV schema 按 strategy 切分：

- **clenow_momentum：统一 13 列**（BUY/SELL 同表同 schema）：
  - 列：`股票代码, instrument_id, 日期, 方向, 行业, 选股排名, 建议买入价, 初始止损线, 趋势质量 (R²), 乖离率 (距 MA200), 回撤 (距 63 日高点), 放量倍数, 备注/预警`
  - BUY 行：13 列全部填充（备注/预警 = alerts 拼接或 "正常"）
  - SELL 行：`股票代码`、`instrument_id`、`日期`、`方向="SELL"`、`建议买入价` (=price)、`备注/预警` (=sell_reason) 填充；其余 7 列（行业、排名、止损、R²、乖离率、回撤、放量）为空字符串
- **其他策略：维持现 7 列 schema** — `股票代码, instrument_id, 日期, 方向, 价格, 触发指标值, 备注`

R² 在 CSV 中存为 `f"{r2:.4f}"` 数值字符串（无分档标签），方便下游解析。

## 8. 错误处理 / 数据缺失

| 场景 | 处理 |
|------|------|
| `sector(iid, date)` 返回 None | `extras["sector"] = None`，渲染 "-"。不丢信号 |
| `HH63` / `SMA_VOLUME_50` 缺失（历史不足） | 该 iid 不进入 ranked（与现有 SMA/score 缺失同处理：`continue`） |
| `MA200` 缺失 | 同上 |
| `ATR(20)` 缺失 | 同上（现有逻辑） |
| `CLENOW_R2` 缺失 | 同上 |
| `vol_mult` 分母 SMA_VOL 为 0 | skip iid |
| 历史不足 400 日（`screen_cmd` 已 buffer） | HH63 / SMA_VOL_50 << 400，无须扩 |
| 无 BUY 信号 | 不输出 BUY 表（现有逻辑） |
| `Signal.extras` 在其他策略为空 | Report 走默认 6 列路径，不读 extras |

不引入异常路径：字段缺失走 skip + `ctx.strategy.log(...)`，不抛 ValueError。

## 9. 测试矩阵

| 测试文件 | 用例 | 验证 |
|---------|------|------|
| `tests/test_indicators.py` | `test_hh_basic` | HH(3) on [10,12,11,15,14] → [None,None,12,15,15] |
| ↑ | `test_sma_volume_basic` | SMA 函数变体，针对 volume 列 |
| ↑ | `test_clenow_r2_basic` | 注册名查找 + 输出列 `CLENOW_R2_{period}` 值 ∈ [0,1] |
| `tests/test_strategies.py` | `test_clenow_buy_signal_has_full_extras` | BUY signal 含 8 个 extras key，类型正确 |
| ↑ | `test_clenow_rank_monotonic` | top score rank=1；rank ∈ [1, n_keep] |
| ↑ | `test_clenow_stop_loss_formula` | `extras["stop_loss"] == close - 3.0 * ATR20` |
| ↑ | `test_clenow_alerts_deviation_trigger` | close/MA200 比 1.5 → "均线乖离过大" |
| ↑ | `test_clenow_alerts_vol_low_trigger` | vol_mult=0.5 → "量能萎缩" |
| ↑ | `test_clenow_alerts_vol_high_trigger` | vol_mult=4.0 → "放量过快" |
| ↑ | `test_clenow_alerts_drawdown_trigger` | drawdown_pct=-20 → "回撤过深" |
| ↑ | `test_clenow_alerts_normal_when_none_hit` | 都不命中 → extras["alerts"] == [] |
| ↑ | `test_clenow_sector_missing_returns_none` | sector None 时 extras["sector"] is None，信号仍出 |
| ↑ | `test_clenow_param_validation` | atr_stop_k<=0 / warn_drawdown_max>=0 等 → ValueError |
| `tests/test_screening_report.py` | `test_clenow_buy_table_has_10_columns` | rich.Table column count == 10 |
| ↑ | `test_other_strategy_buy_table_keeps_6_columns` | strategy_name='ma_cross' → 原 6 列 |
| ↑ | `test_csv_filename_contains_strategy` | 文件名 `signals_clenow_momentum_20260518.csv` |
| ↑ | `test_csv_clenow_schema_13_columns` | CSV header == 13 列；BUY 行全填充；SELL 行 7 列为空 |
| ↑ | `test_csv_other_strategy_keeps_7_columns` | 非 clenow 策略 CSV 保持现有 7 列 |

## 10. 验证清单（实施验收）

执行后必须：

1. `uv run pytest tests/test_indicators.py tests/test_strategies.py tests/test_screening_report.py` 全绿
2. `uv run trendspec screen run --strategy clenow_momentum --market us --date 2026-05-15` 输出 10 列 BUY 表
3. `uv run trendspec screen run --strategy ma_cross --market cn --date 2026-05-15` 仍输出原 6 列（回归）
4. 生成的 CSV 文件名包含 `clenow_momentum`
5. ruff / type check 无新增告警
