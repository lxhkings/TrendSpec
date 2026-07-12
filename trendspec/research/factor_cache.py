"""因子面板缓存 + 复合分构建。

`compute_combo_scores` 是 group_by 分组解析 + 去极值(winsorize) + z-score
加权的唯一实现——FactorStrategy.init() 和 research 的快评估路径
（fast_eval.py）都调用这里，不要在别处重新实现一份。历史上这里曾经是一份
没有 winsorize/group_by 的简化重实现，跟 FactorStrategy 内联逻辑对不上，
导致 precomputed_scores 注入路径和批量评估路径算出的结果跟真实回测不一致
（2026-07-11 修复，见 test_factor_strategy_inject.py / test_fast_eval.py）。
"""

from typing import Any

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import scan_parquet
from trendspec.data.sectors import TICKER_GROUP_OVERRIDES
from trendspec.factors.registry import get_factor, get_factor_with_market


_FILTER_OPS = {
    ">": lambda col, v: col > v,
    ">=": lambda col, v: col >= v,
    "<": lambda col, v: col < v,
    "<=": lambda col, v: col <= v,
}


def _apply_filters(
    df: pl.DataFrame, filters: list[dict[str, Any]], market: str
) -> pl.DataFrame:
    """按 filters 逐条 semi-join 剔除不合格 (instrument_id, date) 行。

    Polars 比较遇 null 结果为 null，被 filter 丢弃——缺失值自然落入
    "剔除"分支，与 FilterTerm 的语义一致。
    """
    for term in filters:
        factor = get_factor_with_market(term["name"], term.get("params") or {}, market)
        result = factor.compute_full(df)
        cond = _FILTER_OPS[term["op"]](pl.col(result.name), term["value"])
        passed = result.values.filter(cond).select(["instrument_id", "date"])
        df = df.join(passed, on=["instrument_id", "date"], how="semi")
    return df


def compute_combo_scores(
    df: pl.DataFrame,
    factors: list[dict[str, Any]],
    market: str,
    group_by: dict[str, list[str]] | None = None,
    winsorize_pct: float = 0.01,
    root: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> pl.DataFrame:
    """按 factors 对截面做（可选按行业分组）winsorize + z-score 加权，
    产出 (instrument_id, date, _group, combo_score)。

    与 FactorStrategy.init() 打分逻辑一致（该方法直接调用本函数）：
      lo/hi = quantile(winsorize_pct)/(1-winsorize_pct) over (date, _group)
      z = sign * (clip(x, lo, hi) - mean_over(date,_group)) / std_over(date,_group)
      combo_score = sum(weight_i * fill_null(z_i, 0))
      任一因子 z 无定义（原始值缺失 or 单成员分组 std 为 null）的行整条剔除。

    group_by: {组名: [行业代码, ...]}；为 None 时全部落一个虚拟组 "_all"
        （等价于原全局排名）。设置时需要 root 以读取 sectors 数据集。
    filters: [{name, params, op, value}, ...]；AND 语义，在 winsorize/z-score
        之前按原始因子值剔除不合格行，因子值缺失（null 比较）一并剔除。
    """
    if filters:
        df = _apply_filters(df, filters, market)
    if group_by is not None:
        if root is None:
            raise ValueError("group_by 需要提供 root 以读取 sectors 数据集")
        market_enum = Market(market.upper())
        sectors_df = scan_parquet(root, market_enum, "sectors").collect().sort(
            ["instrument_id", "date"]
        )
        df_sorted = df.sort(["instrument_id", "date"])
        df_with_sector = df_sorted.join_asof(
            sectors_df.select(["instrument_id", "date", "sector"]),
            on="date", by="instrument_id", strategy="backward",
        )

        group_lookup = pl.DataFrame(
            [{"sector": s, "_group": g} for g, members in group_by.items() for s in members]
        )
        df_with_sector = df_with_sector.join(group_lookup, on="sector", how="left")

        override_lookup = pl.DataFrame(
            [{"instrument_id": iid, "_group_override": g}
             for iid, g in TICKER_GROUP_OVERRIDES.items()]
        ) if TICKER_GROUP_OVERRIDES else pl.DataFrame(
            schema={"instrument_id": pl.Utf8, "_group_override": pl.Utf8}
        )
        df_with_sector = df_with_sector.join(
            override_lookup, on="instrument_id", how="left"
        ).with_columns(
            pl.coalesce(["_group_override", "_group"]).alias("_group")
        )
        group_col_df = df_with_sector.select(["instrument_id", "date", "_group"])
    else:
        group_col_df = df.select(["instrument_id", "date"]).with_columns(
            pl.lit("_all").alias("_group")
        )

    score_df = df.select(["instrument_id", "date"]).join(
        group_col_df, on=["instrument_id", "date"], how="left"
    )
    score_df = score_df.filter(pl.col("_group").is_not_null())

    weight_cols: list[pl.Expr] = []
    missing_any = pl.lit(False)
    for i, term in enumerate(factors):
        factor = get_factor_with_market(term["name"], term.get("params") or {}, market)
        result = factor.compute_full(df)
        col = result.name
        sign = 1.0 if term["direction"] == "high" else -1.0
        zcol = f"_z_{i}"
        ncol = f"_null_{i}"

        vals = result.values.join(
            score_df.select(["instrument_id", "date", "_group"]),
            on=["instrument_id", "date"], how="inner",
        )
        lo = pl.col(col).quantile(winsorize_pct).over(["date", "_group"])
        hi = pl.col(col).quantile(1 - winsorize_pct).over(["date", "_group"])
        winsorized = pl.col(col).clip(lo, hi)

        vals = vals.with_columns([
            winsorized.alias("_w"),
            pl.col(col).is_null().alias(ncol),
        ]).with_columns(
            (
                sign * (pl.col("_w") - pl.col("_w").mean().over(["date", "_group"]))
                / pl.col("_w").std().over(["date", "_group"])
            ).alias(zcol)
        ).select(["instrument_id", "date", zcol, ncol])

        score_df = score_df.join(vals, on=["instrument_id", "date"], how="left")
        weight_cols.append(pl.col(zcol).fill_null(0.0) * term.get("weight", 1.0))
        # zcol 为 null 既覆盖原始值缺失（ncol），也覆盖单成员分组下 std 为 null
        # 导致 z-score 无定义的情况——两者都应从排名中剔除，而非按 0 计分。
        missing_any = missing_any | pl.col(ncol).fill_null(True) | pl.col(zcol).is_null()

    score_df = score_df.with_columns(sum(weight_cols).alias("combo_score"))
    return score_df.filter(~missing_any).select(
        ["instrument_id", "date", "_group", "combo_score"]
    )


def _key(name: str, params: dict[str, Any]) -> tuple:
    return (name, tuple(sorted(params.items())))


class FactorCache:
    """按 (name, frozenset(params)) memoize 单因子面板。底层 compute_full 只算一次。"""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._cache: dict[tuple, pl.DataFrame] = {}
        self.compute_count = 0

    def get(self, name: str, params: dict[str, Any]) -> pl.DataFrame:
        k = _key(name, params or {})
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        factor = get_factor(name, params or {})
        result = factor.compute_full(self._df)
        panel = result.values.rename({result.name: "value"})
        self._cache[k] = panel
        self.compute_count += 1
        return panel
