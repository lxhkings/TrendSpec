"""因子有效性评估：RankIC + 分层回测(quantile)。

两者共用同一份前瞻收益计算；因子截面分复用
trendspec.combo.scores.compute_combo_scores，不重新实现 winsorize/z-score。
不跑 BacktestEngine —— 这是纯截面统计，交易成本/滑点等引擎逻辑无关。
"""

from typing import Any

import polars as pl

from trendspec.combo import compute_combo_scores


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


def compute_rank_ic(
    panel: pl.DataFrame,
    factors: list[dict[str, Any]],
    market: str,
    horizon: int = 20,
    group_by: dict[str, list[str]] | None = None,
    winsorize_pct: float = 0.01,
    root: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """逐日截面 RankIC：combo_score 与 fwd_ret_{horizon}d 的秩相关（Spearman，
    用 .rank() 转秩再算 Pearson 相关实现，等价、免 scipy 依赖）。"""
    scores = compute_combo_scores(
        panel, factors, market, group_by, winsorize_pct, root, filters=filters
    )
    fwd = _attach_forward_returns(panel, horizon)
    ret_col = f"fwd_ret_{horizon}d"

    joined = scores.join(
        fwd.select(["instrument_id", "date", ret_col]),
        on=["instrument_id", "date"],
        how="inner",
    ).filter(pl.col("combo_score").is_not_null() & pl.col(ret_col).is_not_null())

    ranked = joined.with_columns(
        pl.col("combo_score").rank().over("date").alias("_score_rank"),
        pl.col(ret_col).rank().over("date").alias("_ret_rank"),
    )

    return (
        ranked.group_by("date")
        .agg(pl.corr("_score_rank", "_ret_rank").alias("rank_ic"))
        .drop_nulls("rank_ic")
        # 退化截面(如当日收益秩零方差)corr 产出 NaN,drop_nulls 拦不住 NaN
        .filter(pl.col("rank_ic").is_finite())
        .sort("date")
    )


def summarize_ic(ic_df: pl.DataFrame) -> dict[str, float | None]:
    """RankIC 序列汇总：ic_mean/ic_std/ir(=mean/std)/ic_win_rate(同号比例)。

    非有限 rank_ic 先剔除,不参与汇总。"""
    if not ic_df.is_empty():
        ic_df = ic_df.filter(pl.col("rank_ic").is_finite())
    if ic_df.is_empty():
        return {"ic_mean": None, "ic_std": None, "ir": None, "ic_win_rate": None}

    ic_mean = ic_df["rank_ic"].mean()
    ic_std = ic_df["rank_ic"].std()
    ir = ic_mean / ic_std if ic_std else None
    win_rate = (ic_df["rank_ic"] > 0).sum() / ic_df.height

    return {"ic_mean": ic_mean, "ic_std": ic_std, "ir": ir, "ic_win_rate": win_rate}


def compute_quantile_returns(
    panel: pl.DataFrame,
    factors: list[dict[str, Any]],
    market: str,
    horizon: int = 20,
    n_quantiles: int = 5,
    group_by: dict[str, list[str]] | None = None,
    winsorize_pct: float = 0.01,
    root: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """逐日按 combo_score 切 n_quantiles 组（qcut，按 date 分组切），
    每组算 fwd_ret_{horizon}d 简单平均。不跑 BacktestEngine，只看因子
    分层是否单调——研究阶段分析工具，不是可执行策略。"""
    scores = compute_combo_scores(
        panel, factors, market, group_by, winsorize_pct, root, filters=filters
    )
    fwd = _attach_forward_returns(panel, horizon)
    ret_col = f"fwd_ret_{horizon}d"

    joined = scores.join(
        fwd.select(["instrument_id", "date", ret_col]),
        on=["instrument_id", "date"],
        how="inner",
    ).filter(pl.col("combo_score").is_not_null() & pl.col(ret_col).is_not_null())

    labels = [str(i) for i in range(n_quantiles)]
    bucketed = joined.with_columns(
        pl.col("combo_score").qcut(n_quantiles, labels=labels).over("date").alias("quantile")
    )

    return (
        bucketed.group_by(["date", "quantile"])
        .agg(pl.col(ret_col).mean().alias("avg_fwd_return"))
        .with_columns(pl.col("quantile").cast(pl.String))
        .sort(["date", "quantile"])
    )


def compute_top_minus_bottom(quantile_df: pl.DataFrame, n_quantiles: int) -> pl.DataFrame:
    """quantile_df: compute_quantile_returns 的输出。返回每日 (最高组-最低组)
    平均前瞻收益之差；只保留当天两组都有数据的日期。"""
    top_label = str(n_quantiles - 1)
    top = quantile_df.filter(pl.col("quantile") == top_label).select(
        ["date", pl.col("avg_fwd_return").alias("top")]
    )
    bottom = quantile_df.filter(pl.col("quantile") == "0").select(
        ["date", pl.col("avg_fwd_return").alias("bottom")]
    )
    return (
        top.join(bottom, on="date", how="inner")
        .with_columns((pl.col("top") - pl.col("bottom")).alias("top_minus_bottom"))
        .select(["date", "top_minus_bottom"])
        .sort("date")
    )
