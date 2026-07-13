# CLAUDE.md

## Role & Rules

You are an expert quantitative developer. 

**CRITICAL RULE: DO NOT explore workspace on startup.** Read ARCHITECTURE.md for module structure. Use prompts to point directly at target files.

因子研究循环必须先读根目录 RESEARCH_RULES.md.

## 模块化设计约束

1. **分层依赖方向**：`ingest → data_lake(Parquet) → data/ → engine/ → strategy/ → risk/ → analyzer/`。`config/`、`cli/` 可被任何层引用，但自身不得反向依赖业务层。跨模块 import 前先确认方向图里有路径；反向或跳层引用（如 `data/` 直接查 MariaDB）先说明理由征得确认，不直接写。

2. **新功能落位判断**：动手前查 ARCHITECTURE.md 的 Directory Topology 表，判断现有模块能否覆盖新功能。能覆盖就加进该模块；不能覆盖先问用户是否新建顶级模块，不擅自创建。功能同时触碰两个模块职责时拆开分别落位，不合并成大杂烩文件。

3. **文件体积阈值**：单文件超 500 行时停下，说明已超阈值并提出拆分方案，征得同意后再继续，不擅自拆分也不放任膨胀。

4. **架构文档同步**：涉及模块结构的改动前先读 ARCHITECTURE.md 相关章节；改动后如新增/删除顶级模块、改变模块间依赖方向、新增 CLI 命令，须同步更新 ARCHITECTURE.md 对应表格，与代码同一次 commit 提交。

5. **CLI 命令间共用逻辑抽函数**：同一个 CLI 文件里多个 command 出现相同的参数解析/文件加载逻辑（如 `--param key=value` 解析、`--spec-file` 加载）时，抽成模块级函数复用，不要每个 command 各自手写一份——曾经出现过同一段 `--spec-file` 解析代码在 `run` 和新加的 `compare` 之间即将被复制第二遍。

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
- **SQLite mock schema 不要每个测试文件各写一份**：新测试需要连表结构时，先查 `conftest.py` 有没有现成 fixture；没有就在 `conftest.py` 新增共用 fixture，不要在测试文件里手写 `CREATE TABLE`。曾经出现两个测试文件各自手写同一张表却打错同一个字（`weekly_prices` vs 生产代码实际用的 `prices_weekly`），两处都静默挂了好几周没人发现。
- **测试用 SQLite fixture 里的表名/列名要跟被测生产代码实际查询的对上**，不要凭印象编——新增/修改 fixture 前先读一遍对应 ingest 函数的 SQL，而不是照抄旁边一个可能已经过期的测试。
- MariaDB 专属语法（如 `COLLATE utf8mb4_unicode_ci`）在 SQLite fixture 里跑不通时，优先在 fixture 侧注册兼容处理（如 `sqlalchemy.event` 注册同名 no-op collation），不要为了让测试通过而改生产 SQL。
