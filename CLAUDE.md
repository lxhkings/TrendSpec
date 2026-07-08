# CLAUDE.md

## Role & Rules

You are an expert quantitative developer. 

**CRITICAL RULE: DO NOT explore workspace on startup.** Read ARCHITECTURE.md for module structure. Use prompts to point directly at target files.

## 常用命令

```bash
uv sync                    # 安装依赖
uv run pytest              # 运行所有测试
uv run pytest -xvs         # 单测文件/函数
uv run ruff check .        # lint
uv run ruff format .       # 格式化
```

## 数据摄入

```bash
# 首次全量
uv run trendspec ingest daily --market us --full
uv run trendspec ingest components --market us
uv run trendspec ingest sectors --market us
uv run trendspec ingest daily --market cn --full
uv run trendspec ingest components --market cn
uv run trendspec ingest sectors --market cn

# 日常增量（每天）
uv run trendspec ingest daily --market us
uv run trendspec ingest daily --market cn
```

## 回测 / 选股

```bash
uv run trendspec backtest list
uv run trendspec backtest run --strategy ma_cross --market us --start 2020-01-01 --end 2024-12-31
uv run trendspec screen --strategy ma_cross --market us --date 2024-05-15
uv run trendspec ingest status --market us
```

## 环境配置

复制 `.env.example` 到 `.env`。`ALLOW_ROOT_DB_USER=true` 必须在 `.env` 中设置才能使用 root 账户。

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
- Pitfall PIT 场景覆盖：退市股票、行业重分类、除权复权
- 不使用 mock 验证数据模型层行为
