"""因子面板缓存 + 复合分构建。z-score 逻辑与 FactorStrategy.init 逐值一致。"""

from typing import Any

import polars as pl

from trendspec.factors.registry import get_factor


def build_combo_score(df: pl.DataFrame, factors: list[dict[str, Any]]) -> pl.DataFrame:
    """按 spec.factors 对截面做 z-score 加权，产出 (instrument_id,date,combo_score)。

    与 trendspec/strategy/factor_strategy.py:FactorStrategy.init 内联逻辑一致：
      sign = +1 if direction=='high' else -1
      z = sign * (x - mean_over_date(x)) / std_over_date(x)
      combo_score = sum(weight_i * fill_null(z_i, 0))
    """
    score_df = df.select(["instrument_id", "date"])
    weight_cols: list[pl.Expr] = []
    for i, term in enumerate(factors):
        factor = get_factor(term["name"], term.get("params", {}))
        result = factor.compute_full(df)
        col = result.name  # column name in result.values
        sign = 1.0 if term["direction"] == "high" else -1.0
        zcol = f"_z_{i}"
        vals = result.values.with_columns(
            (
                sign
                * (pl.col(col) - pl.col(col).mean().over("date"))
                / pl.col(col).std().over("date")
            ).alias(zcol)
        ).select(["instrument_id", "date", zcol])
        score_df = score_df.join(vals, on=["instrument_id", "date"], how="left")
        weight_cols.append(pl.col(zcol).fill_null(0.0) * term.get("weight", 1.0))
    return score_df.with_columns(sum(weight_cols).alias("combo_score")).select(
        ["instrument_id", "date", "combo_score"]
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
