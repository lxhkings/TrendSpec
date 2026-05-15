"""
TrendSpec CLI module.

Placeholder CLI - full functionality will be implemented in Phase 10.
"""

import typer

app = typer.Typer(help="TrendSpec - 量化回测与选股系统")


@app.callback()
def main() -> None:
    """TrendSpec CLI - 完整功能将在后续版本实现."""
    pass


if __name__ == "__main__":
    app()
