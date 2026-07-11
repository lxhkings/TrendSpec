"""因子有效性评估：RankIC + 分层回测(quantile)。

两者共用同一份前瞻收益计算；因子截面分复用
research/factor_cache.py::compute_combo_scores，不重新实现 winsorize/z-score。
不跑 BacktestEngine —— 这是纯截面统计，交易成本/滑点等引擎逻辑无关。
"""

import polars as pl


def _attach_forward_returns(panel: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """给 panel 加一列 fwd_ret_{horizon}d = close.shift(-horizon)/close - 1，
    同一 instrument_id 内计算（向量化，不逐股 load，跟
    analyzer/signal_history.py 的逐股循环版本公式一致、实现不同）。

    注意：必须先按 ["instrument_id", "date"] 排序。Polars .over() 不会自动排序，
    只是按指定列分组；shift 操作在行序上进行，行序错误则会产生错误的前瞻收益。"""
    sorted_panel = panel.sort(["instrument_id", "date"])
    return sorted_panel.with_columns(
        (pl.col("close").shift(-horizon).over("instrument_id") / pl.col("close") - 1)
        .alias(f"fwd_ret_{horizon}d")
    )
