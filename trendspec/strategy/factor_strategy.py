"""通用声明式因子组合策略。spec 经 params["spec"] 注入。"""

from datetime import date as DateType

import polars as pl

from trendspec.factors.registry import get_factor
from trendspec.research.spec import FactorSpec
from trendspec.strategy.base import BaseStrategy, register_strategy
from trendspec.strategy.context import StrategyContext


@register_strategy("factor_combo")
class FactorStrategy(BaseStrategy):
    """按声明式 spec 截面打分选 top_k、周期调仓。"""

    name = "factor_combo"
    version = "1.0.0"

    def init(self, ctx: StrategyContext) -> None:
        spec = FactorSpec(**self.get_param("spec"))
        self._spec = spec
        df = ctx._data
        if df is None or df.is_empty():
            self._ranked_by_date = {}
            self._score_by_date = {}
            self._date_index = {}
            self._last_rebalance_idx = None
            self._last_processed_date = None
            self._full_data = df
            return

        score_df = df.select(["instrument_id", "date"])
        weight_cols: list[pl.Expr] = []

        for i, term in enumerate(spec.factors):
            factor = get_factor(term.name, term.params)
            result = factor.compute_full(df)  # values: instrument_id, date, <term.name>
            sign = 1.0 if term.direction == "high" else -1.0
            zcol = f"_z_{i}"
            vals = result.values.with_columns(
                (
                    sign
                    * (pl.col(term.name) - pl.col(term.name).mean().over("date"))
                    / pl.col(term.name).std().over("date")
                ).alias(zcol)
            ).select(["instrument_id", "date", zcol])
            score_df = score_df.join(vals, on=["instrument_id", "date"], how="left")
            weight_cols.append(pl.col(zcol).fill_null(0.0) * term.weight)

        score_df = score_df.with_columns(
            sum(weight_cols).alias("combo_score")
        )

        # 缓存：每日按分降序 iid 列表 + (date,iid)->score
        self._ranked_by_date: dict[DateType, list[str]] = {}
        self._score_by_date: dict[tuple, float] = {}
        for (d,), g in score_df.group_by(["date"], maintain_order=True):
            g_sorted = g.sort("combo_score", descending=True, nulls_last=True)
            iids = g_sorted["instrument_id"].to_list()
            self._ranked_by_date[d] = iids
            for iid, sc in zip(iids, g_sorted["combo_score"].to_list()):
                if sc is not None:
                    self._score_by_date[(d, iid)] = sc

        all_dates = sorted(df["date"].unique().to_list())
        self._date_index = {d: i for i, d in enumerate(all_dates)}
        self._last_rebalance_idx: int | None = None
        self._last_processed_date: DateType | None = None
        self._full_data = df

    def next(self, ctx: StrategyContext) -> None:
        pass
