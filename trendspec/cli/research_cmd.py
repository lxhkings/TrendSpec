"""研究闭环 CLI：run（跑闭环）/ serve（起面板）。"""

from datetime import date

import typer
from rich.console import Console

app = typer.Typer(help="AI 自动因子研究闭环")
console = Console()


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
    mock_llm: str | None = typer.Option(
        None, "--mock-llm", help="测试用：注入一段假设 JSON 取代真 LLM"
    ),
) -> None:
    """跑 AI 因子研究闭环，达标策略写成 Markdown 建议书。"""
    import trendspec.factors  # noqa: F401 — 触发因子注册
    import trendspec.strategy.factor_strategy  # noqa: F401 — 触发策略注册
    from trendspec.research.agent import HypothesisAgent
    from trendspec.research.config import ResearchSettings
    from trendspec.research.llm_client import MockLLMClient, OpenAICompatClient
    from trendspec.research.orchestrator import ResearchOrchestrator, default_evaluate_fn

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

    agent = HypothesisAgent(client)
    evaluate_fn = default_evaluate_fn(market, start_date, end_date, n_windows, capital)
    orch = ResearchOrchestrator(
        agent=agent,
        evaluate_fn=evaluate_fn,
        out_dir=out_dir,
        max_rounds=rounds,
        max_candidates=max_candidates,
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
