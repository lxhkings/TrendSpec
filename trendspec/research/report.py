"""达标策略 → Markdown 建议书。"""

from datetime import datetime
from pathlib import Path
from typing import Any


def _factor_lines(spec: dict) -> str:
    rows = []
    for f in spec["factors"]:
        rows.append(
            f"| {f['name']} | {f.get('params', {})} | {f['direction']} | {f.get('weight', 1.0)} |"
        )
    return "\n".join(rows)


def write_advice(out_dir: str | Path, winner: dict[str, Any], round_no: int) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = winner["spec"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"strategy-r{round_no}-{ts}.md"
    path = out / fname

    win_sharpes = winner.get("window_sharpes", [])
    win_table = "\n".join(f"| {i + 1} | {s:.2f} |" for i, s in enumerate(win_sharpes))

    md = f"""# 策略建议书 — 第 {round_no} 轮

生成时间: {ts}

## 市场逻辑

{spec.get("rationale", "(无)")}

## 因子组合

| 因子 | 参数 | 方向 | 权重 |
|------|------|------|------|
{_factor_lines(spec)}

## 参数

- 市场: {spec["market"]}
- top_k: {spec["top_k"]}
- 调仓周期(交易日): {spec["rebalance"]}

## 样本外绩效 (walk-forward)

- OOS Sharpe: {winner["oos_sharpe"]:.2f}
- OOS 最大回撤: {winner["oos_max_drawdown"]:.2%}
- OOS 累计收益(各窗口求和): {winner["oos_total_return"]:.2%}

### 各窗口 Sharpe

| 窗口 | Sharpe |
|------|--------|
{win_table}

## 持仓 / 调仓条件

- 每 {spec["rebalance"]} 个交易日按因子合成分截面重排，持有 top {spec["top_k"]}。
- 掉出 top_k 即卖出，新进 top_k 即买入。
"""
    path.write_text(md, encoding="utf-8")
    return str(path)
