# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
uv sync

# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_stocks_db_ingestor.py

# 运行单个测试函数
uv run pytest tests/test_metrics.py::test_calculate_basic_metrics -v

# 代码检查（lint）
uv run ruff check .

# 代码格式化
uv run ruff format .
```

## 数据摄入（CLI）

数据源：群辉 NAS MariaDB（`stocks` 数据库，192.168.8.9）

**首次初始化（只需一次）：**

```bash
uv run trendspec ingest daily --market us --full      # 全量历史
uv run trendspec ingest components --market us        # SP500 成分变动（低频，一次性）
uv run trendspec ingest sectors --market us           # GICS 行业（静态，一次性）

uv run trendspec ingest daily --market cn --full      # A 股同上
uv run trendspec ingest components --market cn        # CSI800 成分
uv run trendspec ingest sectors --market cn
```

**日常增量（每天）：**

```bash
uv run trendspec ingest daily --market us   # 只拉新数据
uv run trendspec ingest daily --market cn
```

**回测 / 选股：**

```bash
uv run trendspec backtest list
uv run trendspec backtest run --strategy ma_cross --market us --start 2020-01-01 --end 2024-12-31
uv run trendspec screen --strategy ma_cross --market us --date 2024-05-15
uv run trendspec ingest status --market us
```

## 环境配置

复制 `.env.example` 到 `.env`：

```
DB_HOST=192.168.8.9
DB_PORT=3306
DB_USER=root
DB_PASSWORD=...
DB_NAME=stocks
DATA_LAKE_ROOT=./data_lake
ALLOW_ROOT_DB_USER=true   # root 账户开发专用绕过
```

`ALLOW_ROOT_DB_USER=true` 必须在 `.env` 中设置才能使用 root 账户（`os.getenv` 读不到 `.env`，已用 pydantic model_validator 修复）。

## 架构概览

### 核心数据流

```
群辉 MariaDB (stocks DB) → stocks_db_ingestor → data_lake (Parquet) → Engine → Strategy → RiskPipeline → Broker → Portfolio → Analyzer
```

### 数据源 Schema（群辉现有表）

```
prices(ticker, date, open, high, low, close, volume)          -- 1645 万行，2010-2026
stocks(ticker, exchange, gics_sector, gics_industry, is_active)
constituent_changes(index_id, ticker, change_type, change_date) -- CSI800/HSI/SP500
```

exchange 分布：US = NYSE/Nasdaq/CBOE，CN = SSE/SH/SZSE/SZ，HK = HKEX/HK

**价格已调整**：US = Yahoo Finance 复权价，CN = Tushare 后复权价，两者 `adj_factor=1.0`。

### 自定义 Ingestor（`trendspec/ingest/stocks_db_ingestor.py`）

读取群辉现有 schema，转换为 TrendSpec 标准 Parquet：

- `ingest_us_daily` / `ingest_cn_daily`：JOIN prices + stocks，按 exchange 过滤，CN 推导 `SH{ticker}` / `SZ{ticker}` 格式的 `instrument_id`
- `ingest_us_components` / `ingest_cn_components`：SP500/CSI800 成分变动 ADDED→IPO、REMOVED→DELIST；无成分变动记录的 ticker 用 `MIN(date)` 作 IPO 日期
- `ingest_us_sectors` / `ingest_cn_sectors`：GICS 行业静态快照，`assign_date=2000-01-01`（无历史变更）

注：CN 行业使用 GICS（非申万），来自群辉数据库现有字段。

### data_lake 分区结构

```
data_lake/<market>/<dataset>/instrument_id=<id>/<year>.parquet
```

market = `us` 或 `cn`（旧代码中曾用 `cn_a`，已统一改为 `cn`）。

主键 `(instrument_id, date)`，**不是 ticker**（ticker 可变，instrument_id 不可变）。

### 引擎与策略

`BacktestEngine`（历史遍历）和 `ScreeningEngine`（最新日期）共用同一策略接口（dual-mode）。策略继承 `BaseStrategy`，实现 `init(ctx)`（向量化预计算）和 `next(ctx)`（每日信号生成）。

### 风控

`RiskPipeline` 串行执行 `RiskRule` 列表，第一个拒绝即丢弃信号。内置规则：仓位限制、行业限制、回撤熔断、流动性过滤。

### 设置系统

`trendspec/config/settings.py`：pydantic-settings，`.env` 文件加载。`Settings.get()` 返回 `@lru_cache` 单例。`DatabaseSettings` 用 `@model_validator(mode="after")` 校验 root 用户（而非 `@field_validator`，因为 `.env` 值不在 `os.environ` 中）。

### PIT（Point-in-Time）原则

Universe API 必须接受日期参数。`instrument_id` 不可变，跟踪 IPO/退市/停牌事件，确保历史回测包含当时已退市股票，避免生存者偏差。

## 编写新策略

参考 `trendspec/strategy/examples/`：

```python
from trendspec.strategy import BaseStrategy, StrategyContext

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    params = {"period": 20}

    def init(self, ctx: StrategyContext) -> None:
        ctx.precompute_indicator("MA", period=self.params["period"])

    def next(self, ctx: StrategyContext) -> None:
        ma_val = ctx.indicator_value("MA", ctx.instrument_id, period=self.params["period"])
        if ctx.close > ma_val and not ctx.has_position():
            ctx.signal("BUY", ctx.instrument_id, ctx.close)
```

## 测试规范

- 数据库：SQLite 内存库模拟 MariaDB，`conftest.py` 中有基础 fixtures
- `stocks_db_ingestor` 测试：`tests/test_stocks_db_ingestor.py`，含 US + CN 场景共 13 个测试
- PIT 场景：退市股票、行业重分类、除权复权均有覆盖
- 不使用 mock 验证数据模型层行为

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
