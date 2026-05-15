# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
uv sync

# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_metrics.py

# 运行单个测试函数
uv run pytest tests/test_metrics.py::test_calculate_basic_metrics -v

# 代码检查（lint）
uv run ruff check .

# 代码格式化
uv run ruff format .

# CLI 入口
uv run trendspec --help
uv run trendspec ingest --market cn_a --dataset daily
uv run trendspec backtest --strategy ma_cross --market cn_a --start 2020-01-01
uv run trendspec screen --strategy ma_cross --market cn_a --date 2024-05-15
```

## 环境配置

复制 `.env.example` 到 `.env`，配置以下变量：

```
DB_HOST=...
DB_USER=...      # 不能是 root，必须是只读账户
DB_PASSWORD=...
DATA_LAKE_ROOT=./data_lake
```

## 架构概览

### 核心数据流

```
MariaDB → Ingest → data_lake (Parquet) → Engine → Strategy → RiskPipeline → Broker → Portfolio → Analyzer
```

**数据摄入（ingest/）**：从 MariaDB 读取原始数据，写入本地 Parquet 缓存（data_lake）。增量同步由 manifest 管理，状态记录在 `data_lake/_manifest/<market>.json`。

**数据存储（data_lake/）**：Parquet 格式按市场/数据集分区，结构为 `data_lake/<market>/<dataset>/`。主键为 `(instrument_id, date)`，**不是 ticker**，因为 ticker 可变。

**引擎（engine/）**：`BacktestEngine` 运行回测，`ScreeningEngine` 运行选股。两者共用同一个策略接口（dual-mode 设计）。

**策略（strategy/）**：用户继承 `BaseStrategy`，实现 `init(ctx)` 和 `next(ctx)`。`init` 用 Polars 向量化预计算指标，`next` 在每个交易日调用一次。

**风控（risk/）**：`RiskPipeline` 串行执行一组 `RiskRule`，第一个拒绝即丢弃信号。内置规则：仓位限制、行业限制、回撤熔断、流动性过滤。

**因子（factors/）**：按类别分组（price/、volume/、technical/、sector/、cross_sectional/），继承 `BaseFactor`，通过 Polars 表达式向量化计算。

### PIT（Point-in-Time）原则

**所有 Universe API 必须接受日期参数**，避免生存者偏差。Universe 跟踪 IPO/退市/停牌事件，确保历史窗口包含当时已退市股票。`instrument_id` 是不可变主键（如 `SH600000`），ticker 仅作展示用途。

### 关键设计约束

- **instrument_id 不可变**：公司改名、ticker 复用均不影响 `instrument_id`
- **双模式策略**：同一 `next()` 方法在回测（历史遍历）和选股（最新日期）中均可运行
- **向量化优先**：`init()` 中用 Polars 预计算所有指标；`next()` 只做查值，不做计算
- **deferred import in CLI**：CLI 命令函数内部导入重模块，保证 `--help` 响应速度

### 设置系统

`Settings` 类通过 pydantic-settings 从 `.env` 加载，使用 `Settings.get()` 获取单例。下级设置组：`settings.db`、`settings.data_lake`、`settings.backtest`、`settings.risk`。

## 编写新策略

继承 `BaseStrategy`，实现两个方法（参考 `trendspec/strategy/examples/`）：

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

测试使用 SQLite 内存数据库模拟 MariaDB，使用临时目录模拟 data_lake。共享 fixtures 在 `tests/conftest.py`。测试数据包含真实 PIT 场景（退市股票、行业重分类、除权复权），不使用 mock 验证数据模型层行为。
