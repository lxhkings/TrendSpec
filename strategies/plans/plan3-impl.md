# plan3 研究循环可靠性修复 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复评估管线 NaN/inf 传播 bug、新增覆盖率预检命令、修正被污染的 ledger 记录,并把行业中性化/过滤武器与统一日期写入研究循环文档。

**Architecture:** 三层修复——(A) `combo/scores.py` 在 z-score 源头把非有限值置 null 借现有剔除逻辑清行,`research/factor_eval.py` 下游防御 + qcut 护栏 + 新 `compute_coverage`;(B/C) plan1/plan2/RESEARCH_RULES 文档同步。spec 见 `strategies/plans/plan3-eval-reliability.md`。

**Tech Stack:** Python 3.12 + polars 1.40 + typer + pytest,包管理 uv。

## Global Constraints

- **前置(用户已确认设计):** 本计划在 `main` 分支执行。开工前先由用户 merge `factor-research` → `main`(20260716 验收已通过);若未合并,先停下提醒用户。
- 所有命令在 `/Users/xiaohong/Project/TrendSpec` 下执行;Python 命令一律 `uv run` 前缀。
- 回测/评估起点全部写死 `2010-01-01`(用户决定);文档中不得残留 `2018-01-01`/`2015-01-01` 起点。
- 验证门槛(IC ≥0.02 / IR ≥0.3 / oos_sharpe ≥1.0 / max_dd ≤0.20 / worst_window >0)一律不改。
- 每个任务提交前:改动文件跑 `uv run ruff check <files>`;commit message 结尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 测试基线:开工前跑一次 `uv run pytest -q` 记录基线;任务只允许让测试变多变绿,不允许把既有测试改红。

---

### Task 1: z-score 非有限值置 null(根因修复)

**Files:**
- Modify: `trendspec/combo/scores.py`(z-score 计算块,约 :185-200)
- Test: `tests/research/test_factor_eval.py`(追加)

**Interfaces:**
- Consumes: `compute_combo_scores(df, factors, market, ...)`(现有)
- Produces: `compute_combo_scores` 输出保证 `combo_score` 全部有限(无 NaN/inf 行);零区分度截面(std=0)整行剔除。Task 2/3/4 依赖此保证。

- [ ] **Step 1: 写失败测试**

在 `tests/research/test_factor_eval.py` 末尾追加(该文件已 import `compute_combo_scores`、`dt`、`pl`、`pytest`):

```python
def _panel_all_identical() -> pl.DataFrame:
    """3支股票 close 序列完全相同 → momentum 截面每天全相等 → 组内 std=0。"""
    rows = []
    for iid in ["A", "B", "C"]:
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 10.0 + i})
    return pl.DataFrame(rows)


def test_combo_scores_zero_std_cross_section_rows_dropped():
    """截面全相等 → std=0 → z-score 非有限,整行剔除;不得漏出 NaN/inf combo_score。

    回归:2026-07-16 round,fund_revenue_cagr_3y/ema_alignment 因此出 IC均值=nan。"""
    df = _panel_all_identical()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    scores = compute_combo_scores(df, factors, "cn")
    assert scores.is_empty()


def test_combo_scores_partial_ties_all_finite():
    """部分并列(4 同 + 1 异)截面 std>0,行保留且 combo_score 全部有限。"""
    rows = []
    slopes = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 3.0}
    for iid, slope in slopes.items():
        for i in range(20):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 100.0 + slope * i})
    df = pl.DataFrame(rows)
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    scores = compute_combo_scores(df, factors, "cn")
    assert scores.height > 0
    assert scores["combo_score"].is_finite().all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/research/test_factor_eval.py::test_combo_scores_zero_std_cross_section_rows_dropped -xvs`
Expected: FAIL——`scores` 非空(NaN combo_score 行漏过 `is_null` 检查被保留)。

- [ ] **Step 3: 最小实现**

`trendspec/combo/scores.py`,z-score 计算块。原代码:

```python
        vals = vals.with_columns([
            winsorized.alias("_w"),
            pl.col(col).is_null().alias(ncol),
        ]).with_columns(
            (
                sign * (pl.col("_w") - pl.col("_w").mean().over(["date", "_group"]))
                / pl.col("_w").std().over(["date", "_group"])
            ).alias(zcol)
        ).select(["instrument_id", "date", zcol, ncol])
```

改为(插入一段 `with_columns`,其余不动):

```python
        vals = vals.with_columns([
            winsorized.alias("_w"),
            pl.col(col).is_null().alias(ncol),
        ]).with_columns(
            (
                sign * (pl.col("_w") - pl.col("_w").mean().over(["date", "_group"]))
                / pl.col("_w").std().over(["date", "_group"])
            ).alias(zcol)
        ).with_columns(
            # 零 std 分组(截面全相等)除出 NaN/inf——polars 中 NaN ≠ null,
            # 必须显式置 null 才能被下方 missing_any 的 is_null 检查整行剔除
            pl.when(pl.col(zcol).is_finite())
            .then(pl.col(zcol))
            .otherwise(None)
            .alias(zcol)
        ).select(["instrument_id", "date", zcol, ncol])
```

同时把该函数 docstring 里「任一因子 z 无定义(原始值缺失 or 单成员分组 std 为 null)的行整条剔除」一句补上「或零 std 分组 z 非有限」。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/research/test_factor_eval.py -q`
Expected: 全部 PASS(含既有测试,确认无回归)。

- [ ] **Step 5: Commit**

```bash
uv run ruff check trendspec/combo/scores.py tests/research/test_factor_eval.py
git add trendspec/combo/scores.py tests/research/test_factor_eval.py
git commit -m "fix(combo): null out non-finite z-scores from zero-std groups

polars NaN is not null: zero-std cross-sections produced NaN combo_scores
that slipped past the is_null row-drop check and poisoned downstream IC
means (round 20260716 H1/H3 were never actually measured).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: compute_rank_ic / summarize_ic 防御 NaN

**Files:**
- Modify: `trendspec/research/factor_eval.py:58-76`(`compute_rank_ic` 尾部、`summarize_ic`)
- Test: `tests/research/test_factor_eval.py`(追加)

**Interfaces:**
- Consumes: Task 1 后的 `compute_combo_scores`(combo_score 全有限)
- Produces: `compute_rank_ic(...)` 返回的 `rank_ic` 列全部有限;`summarize_ic(ic_df)` 对含 NaN 输入仍返回有限汇总(NaN 剔除后计算),全非有限时返回全 None dict。

- [ ] **Step 1: 写失败测试**

追加到 `tests/research/test_factor_eval.py`:

```python
def test_summarize_ic_ignores_non_finite_rank_ic():
    """一颗 NaN 不得毒化整个均值(回归:IC均值=nan 但胜率有值)。"""
    ic_df = pl.DataFrame({
        "date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)],
        "rank_ic": [0.5, float("nan"), 0.3],
    })
    s = summarize_ic(ic_df)
    assert s["ic_mean"] == pytest.approx(0.4)
    assert s["ic_win_rate"] == pytest.approx(1.0)


def test_summarize_ic_all_nan_returns_none():
    ic_df = pl.DataFrame({"date": [dt.date(2020, 1, 1)], "rank_ic": [float("nan")]})
    assert summarize_ic(ic_df) == {
        "ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None,
    }


def _panel_with_flat_tail() -> pl.DataFrame:
    """前 10 天 3 支股票斜率不同(momentum 有区分度),之后全部横盘——
    横盘段前瞻收益全为 0,收益秩零方差,corr 在这些日期产出 NaN。"""
    rows = []
    slopes = {"A": 0.5, "B": 1.0, "C": 2.0}
    for iid, slope in slopes.items():
        price = 100.0
        for i in range(25):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            if i < 10:
                price += slope
            rows.append({"instrument_id": iid, "date": d, "close": price})
    return pl.DataFrame(rows)


def test_compute_rank_ic_excludes_degenerate_dates():
    df = _panel_with_flat_tail()
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    ic_df = compute_rank_ic(df, factors, "cn", horizon=5)
    assert ic_df.height > 0
    assert ic_df["rank_ic"].is_finite().all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/research/test_factor_eval.py -q -k "non_finite or all_nan or degenerate"`
Expected: 3 个新测试 FAIL(NaN 进均值 → nan ≠ 0.4;全 NaN 时返回 nan 而非 None;ic_df 含 NaN 行)。

- [ ] **Step 3: 最小实现**

`trendspec/research/factor_eval.py`。`compute_rank_ic` 尾部,原:

```python
    return (
        ranked.group_by("date")
        .agg(pl.corr("_score_rank", "_ret_rank").alias("rank_ic"))
        .drop_nulls("rank_ic")
        .sort("date")
    )
```

改为:

```python
    return (
        ranked.group_by("date")
        .agg(pl.corr("_score_rank", "_ret_rank").alias("rank_ic"))
        .drop_nulls("rank_ic")
        # 退化截面(如当日收益秩零方差)corr 产出 NaN,drop_nulls 拦不住 NaN
        .filter(pl.col("rank_ic").is_finite())
        .sort("date")
    )
```

`summarize_ic` 开头,原:

```python
    if ic_df.is_empty():
        return {"ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None}
```

改为:

```python
    if not ic_df.is_empty():
        ic_df = ic_df.filter(pl.col("rank_ic").is_finite())
    if ic_df.is_empty():
        return {"ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None}
```

并在 `summarize_ic` docstring 补一句「非有限 rank_ic 先剔除,不参与汇总」。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/research/test_factor_eval.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
uv run ruff check trendspec/research/factor_eval.py tests/research/test_factor_eval.py
git add trendspec/research/factor_eval.py tests/research/test_factor_eval.py
git commit -m "fix(research): drop non-finite rank_ic before IC summary

pl.corr yields NaN on degenerate cross-sections; drop_nulls does not
catch NaN, so one bad date poisoned the whole IC mean.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: qcut 护栏(allow_duplicates)

**Files:**
- Modify: `trendspec/research/factor_eval.py:107`(`compute_quantile_returns` 的 qcut)
- Test: `tests/research/test_factor_eval.py`(追加)

**Interfaces:**
- Consumes: Task 1 后的 `compute_combo_scores`
- Produces: `compute_quantile_returns` 对并列分数截面不再 panic;重复分位边界合并落桶。

- [ ] **Step 1: 写失败测试**

追加到 `tests/research/test_factor_eval.py`:

```python
def test_compute_quantile_returns_handles_tied_scores_without_panic():
    """6支股票 4 支因子值并列:5 分位边界重复,未加 allow_duplicates 时
    polars qcut 直接 PanicException(回归:20260716 H1 分层)。"""
    rows = []
    slopes = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 2.0, "F": 3.0}
    for iid, slope in slopes.items():
        for i in range(25):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 100.0 + slope * i})
    df = pl.DataFrame(rows)
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    qr = compute_quantile_returns(df, factors, "cn", horizon=5, n_quantiles=5)
    assert not qr.is_empty()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/research/test_factor_eval.py::test_compute_quantile_returns_handles_tied_scores_without_panic -xvs`
Expected: FAIL——polars 抛异常(PanicException 或 DuplicateError,视版本;两者都算复现)。

- [ ] **Step 3: 最小实现**

`trendspec/research/factor_eval.py` `compute_quantile_returns`,原:

```python
    bucketed = joined.with_columns(
        pl.col("combo_score").qcut(n_quantiles, labels=labels).over("date").alias("quantile")
    )
```

改为:

```python
    bucketed = joined.with_columns(
        pl.col("combo_score")
        .qcut(n_quantiles, labels=labels, allow_duplicates=True)
        .over("date")
        .alias("quantile")
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/research/test_factor_eval.py -q`
Expected: 全部 PASS(含既有 qcut 回归测试 `test_compute_quantile_returns_per_date_qcut_regression`)。

- [ ] **Step 5: Commit**

```bash
uv run ruff check trendspec/research/factor_eval.py tests/research/test_factor_eval.py
git add trendspec/research/factor_eval.py tests/research/test_factor_eval.py
git commit -m "fix(research): allow duplicate qcut boundaries in quantile eval

Tied combo scores within a date produced duplicate quantile edges and
a polars PanicException (round 20260716 H1 quantile run).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `research coverage` 预检命令

**Files:**
- Modify: `trendspec/research/factor_eval.py`(新增 `compute_coverage`)
- Modify: `trendspec/cli/research_cmd.py`(新增 `coverage` 子命令)
- Modify: `ARCHITECTURE.md:63-67`(CLI 命令树)、`ARCHITECTURE.md:97`(Key Class Index)
- Test: `tests/research/test_factor_eval.py`、`tests/research/test_research_cmd.py`(追加)

**Interfaces:**
- Consumes: `compute_combo_scores`(Task 1 后)、`_load_factor_spec_json`(research_cmd.py 现有)、`MarketPanel.load(market, start, end)`(现有,返回对象带 `.data: pl.DataFrame`)
- Produces:
  ```python
  def compute_coverage(
      panel: pl.DataFrame, factors: list[dict[str, Any]], market: str,
      group_by: dict[str, list[str]] | None = None, winsorize_pct: float = 0.01,
      root: str | None = None, filters: list[dict[str, Any]] | None = None,
      min_stocks: int = 30,
  ) -> dict[str, float | int]
  # keys: panel_rows, scored_rows, score_coverage, n_dates, n_valid_dates, valid_date_ratio
  ```
  CLI: `trendspec research coverage --spec-file PATH --market cn --start DATE [--end DATE] [--min-stocks N]`

- [ ] **Step 1: 写失败测试(函数级)**

追加到 `tests/research/test_factor_eval.py`(import 行加 `compute_coverage`):

```python
def test_compute_coverage_reports_valid_date_ratio():
    df = _panel_with_monotonic_relation()  # 既有 helper:5 支股票 x 30 天
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    cov = compute_coverage(df, factors, "cn", min_stocks=5)
    assert cov["panel_rows"] == df.height
    assert 0 < cov["score_coverage"] <= 1
    assert cov["n_dates"] == 30
    # 前 5 天 momentum 为 null 没有打分行 → 有效日 < 总日数
    assert 0 < cov["n_valid_dates"] < cov["n_dates"]
    assert cov["valid_date_ratio"] == pytest.approx(cov["n_valid_dates"] / cov["n_dates"])


def test_compute_coverage_zero_std_dates_not_valid():
    df = _panel_all_identical()  # Task 1 的 helper:截面全相等
    factors = [{"name": "momentum", "params": {"period": 5}, "direction": "high", "weight": 1.0}]
    cov = compute_coverage(df, factors, "cn", min_stocks=2)
    assert cov["n_valid_dates"] == 0
    assert cov["valid_date_ratio"] == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/research/test_factor_eval.py -q -k coverage`
Expected: FAIL with `ImportError: cannot import name 'compute_coverage'`。

- [ ] **Step 3: 实现 `compute_coverage`**

`trendspec/research/factor_eval.py` 末尾追加:

```python
def compute_coverage(
    panel: pl.DataFrame,
    factors: list[dict[str, Any]],
    market: str,
    group_by: dict[str, list[str]] | None = None,
    winsorize_pct: float = 0.01,
    root: str | None = None,
    filters: list[dict[str, Any]] | None = None,
    min_stocks: int = 30,
) -> dict[str, float | int]:
    """因子覆盖率预检:打分行覆盖率 + 有效截面日占比。

    有效截面日 = 当日打分行数 ≥ min_stocks 且 combo_score 截面 std > 0
    (有区分度)。占比过低说明因子历史覆盖不足,IC/分层结果不可信,
    应按 data_insufficient 停假设而不是烧一次评估。"""
    scores = compute_combo_scores(
        panel, factors, market, group_by, winsorize_pct, root, filters=filters
    )
    n_dates = panel.select(pl.col("date").n_unique()).item()
    per_date = scores.group_by("date").agg(
        pl.len().alias("n"),
        pl.col("combo_score").std().alias("std"),
    )
    n_valid = per_date.filter(
        (pl.col("n") >= min_stocks) & (pl.col("std") > 0)
    ).height
    return {
        "panel_rows": panel.height,
        "scored_rows": scores.height,
        "score_coverage": scores.height / panel.height if panel.height else 0.0,
        "n_dates": n_dates,
        "n_valid_dates": n_valid,
        "valid_date_ratio": n_valid / n_dates if n_dates else 0.0,
    }
```

- [ ] **Step 4: 跑函数级测试确认通过**

Run: `uv run pytest tests/research/test_factor_eval.py -q`
Expected: 全部 PASS。

- [ ] **Step 5: 写失败测试(CLI 级)**

追加到 `tests/research/test_research_cmd.py`(文件已有 `dt/json/pl/CliRunner/app/runner`):

```python
def test_coverage_command_outputs_ratio(tmp_path: Path, monkeypatch):
    import trendspec.research.market_panel as mp_mod

    rows = []
    slopes = {"A": 0.5, "B": 1.0, "C": 2.0}
    for iid, slope in slopes.items():
        for i in range(30):
            d = dt.date(2020, 1, 1) + dt.timedelta(days=i)
            rows.append({"instrument_id": iid, "date": d, "close": 100.0 + slope * i})
    panel = pl.DataFrame(rows)

    class _FakePanel:
        data = panel

    monkeypatch.setattr(
        mp_mod.MarketPanel, "load", classmethod(lambda cls, *a, **k: _FakePanel())
    )

    spec = {"factors": [{"name": "momentum", "params": {"period": 5},
                         "direction": "high", "weight": 1.0}]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    result = runner.invoke(app, [
        "coverage", "--spec-file", str(spec_path), "--market", "cn",
        "--start", "2020-01-01", "--end", "2020-02-01", "--min-stocks", "3",
    ])
    assert result.exit_code == 0, result.output
    assert "有效日占比" in result.output
```

Run: `uv run pytest tests/research/test_research_cmd.py::test_coverage_command_outputs_ratio -xvs`
Expected: FAIL——typer 报 no such command "coverage"(exit_code != 0)。

- [ ] **Step 6: 实现 CLI 子命令**

`trendspec/cli/research_cmd.py` 末尾追加(风格仿 `research_ic`):

```python
@app.command("coverage")
def research_coverage(
    spec_file: Path = typer.Option(
        ..., "--spec-file",
        help="FactorSpec JSON 文件路径（只读 factors/filters/group_by/winsorize_pct 字段）",
    ),
    market: str = typer.Option("cn", "--market", "-m", help="市场"),
    start: str = typer.Option(..., "--start", help="起始 YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="结束 YYYY-MM-DD，默认今日"),
    min_stocks: int = typer.Option(30, "--min-stocks", help="有效截面日最少标的数"),
) -> None:
    """因子覆盖率预检：打分行覆盖率 + 有效截面日占比（≥min_stocks 只且截面有区分度）。
    跑 IC 前先看数据够不够——占比过低时 IC/分层结果不可信。"""
    import trendspec.factors  # noqa: F401 — 触发因子注册
    from trendspec.research.factor_eval import compute_coverage
    from trendspec.research.market_panel import MarketPanel

    spec = _load_factor_spec_json(spec_file)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    panel = MarketPanel.load(market, start_date, end_date)

    cov = compute_coverage(
        panel.data, spec["factors"], market,
        group_by=spec.get("group_by"), winsorize_pct=spec.get("winsorize_pct", 0.01),
        filters=spec.get("filters"), min_stocks=min_stocks,
    )
    console.print(f"[cyan]覆盖率预检[/cyan] (min_stocks={min_stocks})")
    console.print(
        f"panel 行数={cov['panel_rows']:,}  打分行数={cov['scored_rows']:,}  "
        f"行覆盖率={cov['score_coverage']:.2%}"
    )
    console.print(
        f"总日数={cov['n_dates']}  有效截面日={cov['n_valid_dates']}  "
        f"有效日占比={cov['valid_date_ratio']:.2%}"
    )
```

- [ ] **Step 7: 跑 CLI 测试确认通过**

Run: `uv run pytest tests/research/test_research_cmd.py -q`
Expected: 全部 PASS。

- [ ] **Step 8: 更新 ARCHITECTURE.md**

CLI 命令树(:63-67)research 分支,`ic` 行之前插一行:

```
│   ├── coverage --spec-file PATH --market us|cn --start DATE [--end DATE] [--min-stocks N]
```

Key Class Index(:97)行:

```
| `compute_rank_ic` / `compute_quantile_returns` | `research/factor_eval.py` | RankIC / 分层回测评估 |
```

改为:

```
| `compute_rank_ic` / `compute_quantile_returns` / `compute_coverage` | `research/factor_eval.py` | RankIC / 分层回测 / 覆盖率预检 |
```

- [ ] **Step 9: Commit**

```bash
uv run ruff check trendspec/research/factor_eval.py trendspec/cli/research_cmd.py \
  tests/research/test_factor_eval.py tests/research/test_research_cmd.py
git add trendspec/research/factor_eval.py trendspec/cli/research_cmd.py \
  tests/research/test_factor_eval.py tests/research/test_research_cmd.py ARCHITECTURE.md
git commit -m "feat(research): add coverage precheck command

trendspec research coverage reports scored-row coverage and the share of
dates with a usable cross-section (>=min_stocks and std>0), so sparse
factors are flagged as data_insufficient before burning an IC run.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: 全量验证检查点

**Files:** 无新改动;验证 Task 1-4。

- [ ] **Step 1: 全量测试**

Run: `uv run pytest -q`
Expected: 全绿(对照开工前基线,只多不少)。任何红 → 停下修复,不进 Task 6。

- [ ] **Step 2: 全量 lint**

Run: `uv run ruff check .`
Expected: 无新增告警。

---

### Task 6: ledger 修正 + H1/H3 重测(真实数据)

**Files:**
- Modify: `research_out/ledger.jsonl`(第 12、14 行,即 date=2026-07-16 的 `fund_revenue_cagr_3y` 与 `ema_alignment` 两条)

**Interfaces:**
- Consumes: Task 1-4 修复后的 ic/quantile/coverage 命令;`trendspec.research.ledger.append_ledger(path, dict)`(现有);spec 文件 `research_out/specs/revenue_cagr_3y.json`、`research_out/specs/ema_alignment.json`(已存在)
- Produces: ledger 无「nan 指标 + 负结论」条目;两因子有真实测量记录。

注意:本任务跑真实 CN 面板(2010 起,~1300 万行),单条命令可能数分钟;NAS/parquet 读取失败 → 停下报告,不造数。

- [ ] **Step 1: 改标两条 eval_error 记录**

`research_out/ledger.jsonl` 第 12 行(fund_revenue_cagr_3y),把

```json
"stage_failed": "ic", "metrics": {"ic_mean": null, "ic_std": null, "ir": null, "ic_win_rate": 0.4925, "n_periods": 3202}, "conclusion": "IC均值/IR 均为 nan（3202期，胜率49.25%），未达 |IC|>=0.02 且 |IR|>=0.3；quantile 在 qcut 处 PanicException 未完成"
```

改为

```json
"stage_failed": "eval_error", "metrics": {"ic_mean": null, "ic_std": null, "ir": null, "ic_win_rate": 0.4925, "n_periods": 3202}, "conclusion": "评估故障：零std截面NaN传播致IC均值=nan、qcut panic（框架bug，plan3修复）。该因子未被测量，不构成负结论，不参与判重"
```

第 14 行(ema_alignment)同样处理:`"stage_failed": "eval_error"`,conclusion 改为

```json
"conclusion": "评估故障：零std截面NaN传播致IC均值=nan（框架bug，plan3修复）。该因子未被测量，不构成负结论，不参与判重"
```

第 13 行(fund_net_income_qoq,数字健康)不动。

- [ ] **Step 2: 重测前覆盖率预检**

```bash
uv run trendspec research coverage --spec-file research_out/specs/revenue_cagr_3y.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD>
uv run trendspec research coverage --spec-file research_out/specs/ema_alignment.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD>
```

Expected: 输出三行覆盖率数字。记下「有效日占比」:≥50% → 继续 Step 3;<50% → 该因子按 `data_insufficient` 追加 ledger(Step 4 模板,`stage_failed` 换成 `data_insufficient`,metrics 填覆盖率数字),跳过其 ic/quantile。

- [ ] **Step 3: 重跑 ic + quantile**

```bash
uv run trendspec research ic --spec-file research_out/specs/revenue_cagr_3y.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20
uv run trendspec research quantile --spec-file research_out/specs/revenue_cagr_3y.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20 --n-quantiles 5
uv run trendspec research ic --spec-file research_out/specs/ema_alignment.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20
uv run trendspec research quantile --spec-file research_out/specs/ema_alignment.json \
  --market cn --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20 --n-quantiles 5
```

Expected: 四条命令全部正常退出,输出全为有限数字(无 nan、无 PanicException)。任一输出仍含 nan → 修复不完整,停下回 Task 1-3 诊断,禁止继续。

- [ ] **Step 4: 真实数字追加 ledger**

数字从 Step 3 输出原样粘贴(证据制,禁手写):

```bash
uv run python -c "
from trendspec.research.ledger import append_ledger
append_ledger('research_out/ledger.jsonl', {
    'type': 'manual_research',
    'date': '<当日YYYY-MM-DD>',
    'note': 'retest after plan3 eval fix (was eval_error on 2026-07-16)',
    'hypothesis': {'market': 'cn', 'factors': [{'name': 'fund_revenue_cagr_3y', 'direction': 'high', 'params': {}}], 'rationale': '多年营收复合增速捕捉可持续成长，区别于单季yoy噪音'},
    'stage_failed': '<ic|quantile|无则不填此键>',
    'metrics': {'ic_mean': <粘贴>, 'ic_std': <粘贴>, 'ir': <粘贴>, 'ic_win_rate': <粘贴>, 'n_periods': <粘贴>},
    'conclusion': '<按 RESEARCH_RULES 第7节门槛如实判定，一句话>',
})
print('ledger appended')
"
```

`ema_alignment` 同构再跑一条(rationale 用原条目的:'多周期EMA多头排列强度刻画趋势一致性，行为上追逐趋势的资金拥挤与动量延续')。
若某因子重测**通过**初筛(|IC|≥0.02 且 |IR|≥0.3 且分层基本单调):不在本任务继续 walk-forward——记入 conclusion「初筛通过,待下轮研究走 Phase 4」,留给研究循环。

- [ ] **Step 5: 验证 ledger 一致性**

```bash
uv run python -c "
import json
for i, line in enumerate(open('research_out/ledger.jsonl'), 1):
    r = json.loads(line)
    m = r.get('metrics', {})
    bad = any(v is None for k, v in m.items() if k in ('ic_mean', 'ir'))
    if bad and r.get('stage_failed') not in ('eval_error', 'data_insufficient'):
        print('VIOLATION line', i)
        break
else:
    print('ledger clean')
"
```

Expected: `ledger clean`。

- [ ] **Step 6: Commit**

```bash
git add research_out/ledger.jsonl
git commit -m "research: reclassify 20260716 nan entries as eval_error, retest after fix

fund_revenue_cagr_3y and ema_alignment were never actually measured
(NaN propagation bug); retested with fixed pipeline, real metrics appended.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: plan1 / plan2 / RESEARCH_RULES 文档修订(B+C)

**Files:**
- Modify: `strategies/plans/plan1-research-runbook.md`
- Modify: `strategies/plans/plan2-accept.md`
- Modify: `RESEARCH_RULES.md`

**Interfaces:**
- Consumes: Task 4 的 `research coverage` 命令(文档引用它)
- Produces: 下轮 DeepSeek 研究循环按新文档执行。

- [ ] **Step 1: plan1 —— 日期与 --end(头部「本轮参数」)**

原(:12):

```
**本轮参数(开跑前确认):** 市场 `us` 或 `cn`(用户指定,默认 us);评估区间:初筛 `--start 2010-01-01`,严格验证 `--start 2010-01-01`,`--end` 均不传(默认到今日)。
```

改为:

```
**本轮参数(开跑前确认):** 市场 `us` 或 `cn`(用户指定,默认 us);评估区间:初筛与严格验证一律 `--start 2010-01-01`;`--end` 显式传当日日期,并把完整命令(含 --end)原样记入报告——这是验收复现的锚点。
```

- [ ] **Step 2: plan1 —— 判重豁免(Phase 1.2)**

原:

```
- [ ] **1.2 已试清单写进报告草稿**。判重规则:因子名相同 + 方向相同 + 参数网格有重叠 = 重复,禁止再提。
```

改为:

```
- [ ] **1.2 已试清单写进报告草稿**。判重规则:因子名相同 + 方向相同 + 参数网格有重叠 = 重复,禁止再提。例外:`stage_failed` 为 `eval_error` 或 `data_insufficient` 的条目是「没测成」不是「测过失败」,不参与判重,允许重提。
```

- [ ] **Step 3: plan1 —— 家族平衡指引(Phase 2.1 末尾追加)**

在「讲不出经济学逻辑的假设作废。与已试清单重复的作废。」后追加一行:

```
家族平衡:每轮至少 1 个价格/量类假设(反转、换手率、波动率等——历史 winners 全部出自该家族);基本面类假设默认建议加 GICS 行业中性化(见 2.2 的 group_by)。
```

- [ ] **Step 4: plan1 —— spec 模板补 group_by/filters(Phase 2.2)**

原 json 块与其后一行:

````
```json
{
  "factors": [
    {"name": "price_momentum", "params": {"period": 20}, "direction": "low", "weight": 1.0}
  ],
  "winsorize_pct": 0.01
}
```

`name` 必须在 Phase 1.3 清单里;`direction`:`high`=值大者好,`low`=值小者好。
````

改为:

````
```json
{
  "factors": [
    {"name": "price_momentum", "params": {"period": 20}, "direction": "low", "weight": 1.0}
  ],
  "winsorize_pct": 0.01,
  "group_by": {"能源": ["煤炭开采", "石油加工"], "...": ["..."]},
  "filters": [
    {"name": "fund_total_mv", "op": ">=", "value": 200000}
  ]
}
```

`name` 必须在 Phase 1.3 清单里;`direction`:`high`=值大者好,`low`=值小者好。
可选字段:
- `group_by`(行业中性化):`{组名: [行业代码,...]}`,组内做 winsorize+z-score。GICS 11 组现成映射直接整段复制 `examples/factor_combo_cn_gics.json` 的 `group_by` 字段,不要手编行业列表。
- `filters`(打分前硬过滤):`[{"name","op","value"}]`,op ∈ `>`/`>=`/`<`/`<=`,name 为已注册因子(如 `fund_total_mv`/`fund_circ_mv`/`turnover_rate`)。数值单位跟随原始列(tushare 市值列单位为万元,200000 = 20 亿),用前先确认。
````

- [ ] **Step 5: plan1 —— Phase 3 加预检 + 改日期 + nan 处理 + 中性化变体**

3.1 之前插入:

````
- [ ] **3.0 覆盖率预检**

```bash
uv run trendspec research coverage --spec-file research_out/specs/<假设名>.json \
  --market <us|cn> --start 2010-01-01 --end <当日YYYY-MM-DD>
```

**通过标准:有效日占比 ≥50%**(阈值可由用户调整)。低于 → 按 `data_insufficient` 记 ledger(5.2 模板,stage_failed 填 `data_insufficient`,metrics 粘贴覆盖率数字),停该假设,不进入 IC。
````

3.1 命令原:

```
uv run trendspec research ic --spec-file research_out/specs/<假设名>.json \
  --market us --start 2018-01-01 --horizon 20
```

改为:

```
uv run trendspec research ic --spec-file research_out/specs/<假设名>.json \
  --market <us|cn> --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20
```

3.1 通过标准行后追加:

```
输出含 nan(如 `IC均值=nan`)→ 不是负结论:按 `eval_error` 记 ledger、停该假设(RESEARCH_RULES 第 4 节),留待框架排查。
```

3.2 命令原:

```
uv run trendspec research quantile --spec-file research_out/specs/<假设名>.json \
  --market us --start 2018-01-01 --horizon 20 --n-quantiles 5
```

改为:

```
uv run trendspec research quantile --spec-file research_out/specs/<假设名>.json \
  --market <us|cn> --start 2010-01-01 --end <当日YYYY-MM-DD> --horizon 20 --n-quantiles 5
```

3.3 之后新增:

```
- [ ] **3.4 近失中性化变体(可选,不占假设预算)**:初筛近失(IC/IR 单项达标,或未达标项距门槛差 <20%)时,允许复制该 spec 增加 `group_by`(GICS 11 组,整段抄 examples/factor_combo_cn_gics.json)另存 `research_out/specs/<假设名>_neutral.json`,重跑 3.0-3.2 一次。原始与中性化两组数字都进报告;变体通过则按通过进 Phase 4,变体仍未过则记一条负结论(factors 里注明 group_by)。每个假设最多一次变体,禁止再叠加其他改动。
```

- [ ] **Step 6: plan1 —— Phase 4 日期**

4.1 命令中 `--start 2015-01-01` 改为 `--start 2010-01-01`。

- [ ] **Step 7: plan1 —— 5.2 模板 stage_failed 枚举**

原:`'stage_failed': 'ic|quantile|walkforward',`
改为:`'stage_failed': 'ic|quantile|walkforward|eval_error|data_insufficient',`
其下追加一行说明:

```
`eval_error` = 评估命令报错或输出含 nan(框架故障,非负结论);`data_insufficient` = 覆盖率预检未过。两者不参与 Phase 1.2 判重。
```

- [ ] **Step 8: plan2 —— C2/C3 复现规则**

C2 原命令块与判定行:

````
```bash
uv run trendspec research ic --spec-file research_out/specs/<假设名>.json \
  --market <报告所记市场> --start 2018-01-01 --horizon 20
uv run trendspec research quantile --spec-file research_out/specs/<假设名>.json \
  --market <报告所记市场> --start 2018-01-01 --horizon 20 --n-quantiles 5
```

判定:IC均值/IR/分层价差与报告一致(报告未记 `--end`,重跑时数据可能多几天;允许第 3 位小数内的漂移,数量级或符号不一致 = 复现失败)。
````

改为:

````
从报告第 3 节复制该假设记录的**完整原命令**(含 `--start 2010-01-01 --end <报告所记日期>`)逐字重跑,不得自行改参数。

判定:IC均值/IC标准差/IR/IC胜率/分层各组收益/top-bottom 价差与报告**逐位一致**。`--end` 已锚定区间,不存在数据增量漂移;任何数字不一致 = 复现失败。
````

C3 原:

```
- [ ] **C3 复现失败处理**:先确认是否数据增量导致(用报告日期作 `--end` 重跑一次);仍不一致 → 该因子判「不可信」,记入验收结论,建议 revert 对应 commit。
```

改为:

```
- [ ] **C3 复现失败处理**:先核对 fundamentals 是否发生 restate(对比该区间 parquet 行数与报告第 1 节 ingest status);无 restate 证据仍不一致 → 该因子判「不可信」,记入验收结论,建议 revert 对应 commit。
```

- [ ] **Step 9: plan2 —— D1 补 eval_error 对账**

D1 原:

```
- [ ] **D1 ledger vs 报告**:`research_out/ledger.jsonl` 中每条 `manual_research` 负结论与各轮报告第 3 节一一对应,无报告里有、ledger 里没有(或反之)的假设。
```

改为:

```
- [ ] **D1 ledger vs 报告**:`research_out/ledger.jsonl` 中每条 `manual_research`(含负结论、`eval_error`、`data_insufficient`)与各轮报告第 3 节一一对应,无报告里有、ledger 里没有(或反之)的假设;`eval_error`/`data_insufficient` 条目的 metrics 不得被报告当作负结论引用。
```

- [ ] **Step 10: RESEARCH_RULES —— 第 4、5 节**

第 4 节「失败即停」列表追加一条:

```
- ic/quantile 输出含 nan(如 `IC均值=nan`):按 `eval_error` 记 ledger 并停该假设——这是评估故障,不是负结论,不参与判重;同样禁止绕过(改代码/缩区间/换参数凑数)
```

第 5 节「预算」追加一条:

```
- 同一假设的行业中性化变体(仅加 `group_by`,见 runbook 3.4)不算新假设、不占预算;每假设最多一次
```

- [ ] **Step 11: 全文自查**

```bash
grep -n "2018-01-01\|2015-01-01" strategies/plans/plan1-research-runbook.md strategies/plans/plan2-accept.md RESEARCH_RULES.md
```

Expected: 无输出(起点日期残留清零)。

- [ ] **Step 12: Commit**

```bash
git add strategies/plans/plan1-research-runbook.md strategies/plans/plan2-accept.md RESEARCH_RULES.md
git commit -m "docs(research): unify 2010 start, add coverage precheck, neutralization variants, eval_error category

plan1: coverage precheck (Phase 3.0), explicit --end, group_by/filters in
spec template, near-miss GICS-neutral variant rule, family balance hint,
dedup exemption for eval_error/data_insufficient.
plan2: verbatim-command reproduction with --end anchor, restate check.
RESEARCH_RULES: nan output = eval_error (stop, not negative), neutral
variant exempt from hypothesis budget.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 完成定义(对照 spec 验收标准)

- [ ] `uv run pytest -q` 全绿;新增测试覆盖:零 std 截面剔除、部分并列有限性、NaN 汇总、退化日剔除、qcut 并列不 panic、coverage 函数与 CLI。
- [ ] 重跑 `revenue_cagr_3y`/`ema_alignment` 两 spec 的 coverage/ic/quantile:无 nan、无 panic。
- [ ] `grep "2018-01-01\|2015-01-01"` 三份文档无残留。
- [ ] ledger 无「nan 指标(ic_mean/ir 为 null)+ 非 eval_error/data_insufficient」条目(Task 6 Step 5 脚本验证)。
