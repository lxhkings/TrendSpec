"""研究闭环 CLI：run（跑闭环）/ serve（起面板）。"""

import json
from datetime import date
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

app = typer.Typer(help="AI 自动因子研究闭环")
console = Console()


def _load_factor_spec_json(spec_file: Path) -> dict:
    """Load and validate a factor spec JSON file for research ic/quantile commands.

    Validates factors/filters/group_by/winsorize_pct via parse_research_eval_spec.
    Does not require top_k/rebalance/market. Exits via typer.Exit(1) on missing
    file, invalid JSON, or validation errors.
    """
    if not spec_file.exists():
        console.print(f"[red]--spec-file 不存在: {spec_file}[/red]")
        raise typer.Exit(1)
    try:
        raw = json.loads(spec_file.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]--spec-file 不是合法 JSON: {e}[/red]")
        raise typer.Exit(1) from None

    # 延迟 import：保证调用方已 import trendspec.factors 完成注册
    from pydantic import ValidationError

    from trendspec.combo import parse_research_eval_spec

    try:
        return parse_research_eval_spec(raw)
    except ValidationError as e:
        console.print(f"[red]--spec-file 校验失败: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("run")
def research_run(
    market: str = typer.Option("us", "--market", "-m", help="市场 (us)"),
    start: str = typer.Option("2015-01-01", "--start", help="起始 YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="结束 YYYY-MM-DD，默认今日"),
    rounds: int = typer.Option(10, "--rounds", help="最大假设轮数"),
    max_candidates: int = typer.Option(200, "--max-candidates", help="每轮扫参上限"),
    n_windows: int = typer.Option(4, "--windows", help="walk-forward 窗口数"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="初始资金"),
    out: str | None = typer.Option(None, "--out", help="输出目录"),
    theme: str | None = typer.Option(
        None, "--theme", help="限定假设主题，如'均值回归'；不传则不限定"
    ),
    mock_llm: str | None = typer.Option(
        None, "--mock-llm", help="测试用：注入一段假设 JSON 取代真 LLM"
    ),
) -> None:
    """跑 AI 因子研究闭环，达标策略写成 Markdown 建议书。"""
    import trendspec.factors  # noqa: F401 — 触发因子注册
    import trendspec.strategy.factor_strategy  # noqa: F401 — 触发策略注册
    from trendspec.research.agent import HypothesisAgent
    from trendspec.research.config import ResearchSettings
    from trendspec.research.fast_eval import ResearchEvaluator
    from trendspec.research.llm_client import MockLLMClient, OpenAICompatClient
    from trendspec.research.orchestrator import ResearchOrchestrator

    settings = ResearchSettings()
    out_dir = out or settings.out_dir
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()

    if mock_llm is not None:
        client = MockLLMClient(responses=[mock_llm])
    else:
        client = OpenAICompatClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    agent = HypothesisAgent(client, theme=theme)
    evaluator = ResearchEvaluator(
        market=market,
        start=start_date,
        end=end_date,
        n_windows=n_windows,
        capital=capital,
        parallel=True,
    )
    orch = ResearchOrchestrator(
        agent=agent,
        evaluate_fn=None,
        out_dir=out_dir,
        max_rounds=rounds,
        max_candidates=max_candidates,
        batch_evaluator=evaluator,
    )
    console.print(f"[cyan]研究开始[/cyan] market={market} out={out_dir}")
    orch.run()
    console.print(f"[green]完成。建议书见 {out_dir}[/green]")


@app.command("serve")
def research_serve(
    port: int = typer.Option(8800, "--port", "-p", help="端口"),
    out: str | None = typer.Option(None, "--out", help="研究输出目录(读 state/ledger)"),
) -> None:
    """起实时监控面板（只读输出目录，与研究进程解耦）。"""
    import uvicorn

    from trendspec.research.config import ResearchSettings
    from trendspec.research.dashboard import create_app

    out_dir = out or ResearchSettings().out_dir
    console.print(f"[cyan]面板 http://127.0.0.1:{port}[/cyan] 监控 {out_dir}")
    uvicorn.run(create_app(out_dir), host="127.0.0.1", port=port)


@app.command("ic")
def research_ic(
    spec_file: Path = typer.Option(
        ..., "--spec-file",
        help="FactorSpec JSON 文件路径（只读 factors/filters/group_by/winsorize_pct 字段）",
    ),
    market: str = typer.Option("cn", "--market", "-m", help="市场"),
    start: str = typer.Option(..., "--start", help="起始 YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="结束 YYYY-MM-DD，默认今日"),
    horizon: int = typer.Option(20, "--horizon", help="前瞻收益天数"),
) -> None:
    """算因子 RankIC：逐期序列 + IC均值/标准差/IR/胜率，牛熊都能测因子有效性。"""
    import trendspec.factors  # noqa: F401 — 触发因子注册
    from trendspec.research.factor_eval import compute_rank_ic, summarize_ic
    from trendspec.research.market_panel import MarketPanel

    spec = _load_factor_spec_json(spec_file)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    panel = MarketPanel.load(market, start_date, end_date)

    ic_df = compute_rank_ic(
        panel.data, spec["factors"], market, horizon=horizon,
        group_by=spec.get("group_by"), winsorize_pct=spec.get("winsorize_pct", 0.01),
        filters=spec.get("filters"),
    )
    summary = summarize_ic(ic_df)

    console.print(f"[cyan]RankIC[/cyan] {ic_df.height} 期 (horizon={horizon})")
    if summary["ic_mean"] is None:
        console.print("[yellow]没有可用样本（数据太少或因子分全空）[/yellow]")
        return
    ic_std_str = f"{summary['ic_std']:.4f}" if summary['ic_std'] is not None else "N/A"
    ir_str = f"{summary['ir']:.4f}" if summary['ir'] is not None else "N/A"
    console.print(
        f"IC均值={summary['ic_mean']:.4f}  IC标准差={ic_std_str}  "
        f"IR={ir_str}  IC胜率={summary['ic_win_rate']:.2%}"
    )


@app.command("quantile")
def research_quantile(
    spec_file: Path = typer.Option(
        ..., "--spec-file",
        help="FactorSpec JSON 文件路径（只读 factors/filters/group_by/winsorize_pct 字段）",
    ),
    market: str = typer.Option("cn", "--market", "-m", help="市场"),
    start: str = typer.Option(..., "--start", help="起始 YYYY-MM-DD"),
    end: str = typer.Option(None, "--end", help="结束 YYYY-MM-DD，默认今日"),
    horizon: int = typer.Option(20, "--horizon", help="前瞻收益天数"),
    n_quantiles: int = typer.Option(5, "--n-quantiles", help="分层组数"),
) -> None:
    """分层回测：全market按因子分切 N 组，看各组平均前瞻收益是否单调
    （不跑 BacktestEngine，不含交易成本，只看因子分层有没有信息量）。"""
    import trendspec.factors  # noqa: F401 — 触发因子注册
    from trendspec.research.factor_eval import compute_quantile_returns, compute_top_minus_bottom
    from trendspec.research.market_panel import MarketPanel

    spec = _load_factor_spec_json(spec_file)

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    panel = MarketPanel.load(market, start_date, end_date)

    qr = compute_quantile_returns(
        panel.data, spec["factors"], market, horizon=horizon, n_quantiles=n_quantiles,
        group_by=spec.get("group_by"), winsorize_pct=spec.get("winsorize_pct", 0.01),
        filters=spec.get("filters"),
    )
    if qr.is_empty():
        console.print("[yellow]没有可用样本（数据太少或因子分全空）[/yellow]")
        return

    avg_by_q = (
        qr.group_by("quantile").agg(pl.col("avg_fwd_return").mean().alias("mean_ret")).sort("quantile")
    )
    console.print(f"[cyan]分层回测[/cyan] {n_quantiles} 组 (horizon={horizon})")
    for row in avg_by_q.iter_rows(named=True):
        console.print(f"  组{row['quantile']}: 平均前瞻收益={row['mean_ret']:.4%}")

    tmb = compute_top_minus_bottom(qr, n_quantiles=n_quantiles)
    if tmb.is_empty():
        console.print("[yellow]top-bottom 价差: 无法计算（缺最高或最低组数据）[/yellow]")
    else:
        console.print(f"top-bottom 价差均值: {tmb['top_minus_bottom'].mean():.4%}")


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
