# Architecture Panorama

## Data Flow

```
群辉 MariaDB (stocks DB)
    ├── prices(ticker, date, ohlcv)
    ├── stocks(ticker, exchange, gics_sector, is_active)
    ├── constituent_changes(index_id, ticker, change_type, change_date)
    │
    ▼
trendspec/ingest/  ───►  data_lake/<market>/<dataset>/ (Parquet 分区)
    │
    ▼
trendspec/engine/  ───►  trendspec/strategy/  ───►  trendspec/risk/
                        (BaseEngine +             (RiskPipeline)
                         BaseStrategy)
    │
    ▼
trendspec/analyzer/   (PerformanceMetrics, EquityCurve, BacktestReport)
```

Research 管道（独立于引擎环路）：

```
ResearchOrchestrator
  1. HypothesisAgent (LLM) → FactorSpec 假设
  2. expand_grid() → 搜索空间
  3. run_walkforward() → 回测验证
  4. passes_threshold() → 过滤通过策略
  5. write_advice() → Markdown 报告
```

## Directory Topology

| 目录 | 职责 | 关键文件/类 |
|------|------|-----------|
| `ingest/` | 从 MariaDB 摄入数据 → Parquet | `stocks_db_ingestor.py`, `us_ingestor.py`, `cn_ingestor.py`, `writer.py`, `Manifest` |
| `data/` | 数据加载层、市场配置、PIT Universe | `parquet_loader.py` (bars()), `markets.py` (Market 枚举), `universe/` (Universe + CN/US/HK), `schema.py` (校验), `sectors.py` (SectorIndex), `calendar.py` (交易日历) |
| `engine/` | 执行编排（回测/选股） | `base_engine.py` (BaseEngine, EngineConfig, EngineResult), `backtest_engine.py` (BacktestEngine), `screening_engine.py` (ScreeningEngine), `broker.py` (Broker), `portfolio.py` (Portfolio), `costs.py` (CostsModel) |
| `strategy/` | 策略框架（扩展点） | `base.py` (BaseStrategy, @register_strategy), `context.py` (StrategyContext), `signal.py` (Signal), `factor_strategy.py` (FactorStrategy), `indicators.py` (MA/EMA/RSI/MACD/ATR/Bollinger) |
| `factors/` | 因子计算引擎 | `base.py` (Factor), `registry.py` (Registry + @register), `price/`, `technical/`, `volume/`, `cross_sectional/`, `sector/`, `fundamental/` |
| `research/` | LLM 驱动策略研究管道 | `orchestrator.py` (ResearchOrchestrator), `agent.py` (HypothesisAgent), `spec.py` (FactorSpec/FactorTerm), `llm_client.py` (LLMClient), `walkforward.py`, `fast_eval.py`, `factor_eval.py` (RankIC/分层回测), `search.py`, `report.py`, `ledger.py` |
| `risk/` | 风控规则链 | `base.py` (RiskRule/Allow/Reject), `pipeline.py` (RiskPipeline), `position_limit.py`, `sector_limit.py`, `drawdown_halt.py`, `liquidity.py`, `price_limit.py`, `sector_neutral.py` |
| `screening/` | 选股输出报告 | `report.py` (ScreeningReport — rich 表格 + CSV) |
| `analyzer/` | 回测绩效分析 | `metrics.py` (PerformanceMetrics), `equity_curve.py` (EquityCurve), `trade_log.py` (TradeLogAnalyzer), `report.py` (BacktestReport), `signal_history.py`, `strategy_comparison.py` |
| `config/` | pydantic-settings 配置 | `settings.py` (Settings → Database/DataLake/Backtest/RiskSettings) |
| `cli/` | Typer CLI 命令组 | `main.py` (app), `ingest_cmd.py`, `backtest_cmd.py`, `screen_cmd.py`, `research_cmd.py`, `signal_history_cmd.py`, `winrate_cmd.py` |

## CLI 命令树

```
trendspec (Typer)
├── ingest
│   ├── daily --market us|cn [--full]
│   ├── weekly --market us|cn
│   ├── components --market us|cn
│   ├── sectors --market us|cn
│   └── status --market us|cn
├── screen run --strategy NAME --market us|cn --date DATE
├── backtest run --strategy NAME --market us|cn --start DATE --end DATE [--params]
├── research
│   ├── run --theme STR [--fast] [--goal STR] [--initial-cap FLOAT]
│   ├── serve (启动 dashboard)
│   ├── ic --spec-file PATH --market us|cn --start DATE [--end DATE] [--horizon N]
│   └── quantile --spec-file PATH --market us|cn --start DATE [--end DATE] [--horizon N] [--n-quantiles N]
├── signal-history
│   ├── build
│   └── status
└── winrate ema-cross
```

## Key Class Index

| 类 | 文件:行 | 用途 |
|-----|---------|------|
| `BaseEngine` | `engine/base_engine.py` | 执行引擎抽象基类 |
| `BacktestEngine` | `engine/backtest_engine.py` | 完整回测 → Risk → Broker → Portfolio → Analyzer |
| `ScreeningEngine` | `engine/screening_engine.py` | 单日选股信号生成 |
| `BaseStrategy` | `strategy/base.py` | 策略基类，需实现 `init(ctx)` + `next(ctx)` |
| `StrategyContext` | `strategy/context.py` | bar 数据 / 指示器缓存 / PIT 查询 |
| `Signal` | `strategy/signal.py` | 买卖信号 dataclass |
| `FactorStrategy` | `strategy/factor_strategy.py` | 声明式因子组合策略 |
| `Factor` | `factors/base.py` | 因子抽象基类 |
| `FactorRegistry` | `factors/registry.py` | `@register` 装饰器注册因子 |
| `RiskPipeline` | `risk/pipeline.py` | 串行 RiskRule 链 |
| `RiskRule` / `Allow` / `Reject` | `risk/base.py` | 风控基底 + 结果类型 |
| `Universe` | `data/universe/base.py` | PIT 成分股查询（所有方法接受日期） |
| `Market` | `data/markets.py` | 市场枚举（CN / US / HK） |
| `PerformanceMetrics` | `analyzer/metrics.py` | Sharpe / 回撤 / 胜率 |
| `BacktestReport` | `analyzer/report.py` | 中文 rich 表格输出 |
| `ResearchOrchestrator` | `research/orchestrator.py` | 研究循环主控 |
| `HypothesisAgent` | `research/agent.py` | LLM 策略假设生成 |
| `FactorSpec` / `FactorTerm` / `FilterTerm` | `research/spec.py` | Pydantic 因子组合规范（含硬过滤层 filters） |
| `compute_rank_ic` / `compute_quantile_returns` | `research/factor_eval.py` | RankIC / 分层回测评估 |
| `Settings` | `config/settings.py` | `get_settings()` 聚合配置 |
| `Manifest` | `ingest/manifest.py` | 摄入同步状态跟踪 |

## Key Design Principles

1. **PIT (Point-in-Time)** — Universe API 所有方法接受日期参数，消除生存者偏差
2. **instrument_id 不可变** — 主键 `(instrument_id, date)`，非 ticker
3. **数据分区** — `data_lake/<market>/<dataset>/instrument_id=<id>/<year>.parquet`
4. **向量化预计算** — `init()` 做 Polars 批量计算，`next()` 引用缓存
5. **风控串行** — RiskPipeline 按序执行，首个 Reject 跳过信号
6. **双模式引擎** — BacktestEngine + ScreeningEngine 共享策略接口
7. **CN 列名映射** — `INGEST_SCHEMA_MAP` 处理 CN 子表差异；US 直接使用标准列名
