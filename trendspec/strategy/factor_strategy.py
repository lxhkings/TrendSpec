"""通用声明式因子组合策略。spec 经 params["spec"] 注入。"""

from datetime import date as DateType

import polars as pl

from trendspec.factors.registry import get_factor_with_market
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

        precomputed = self.get_param("precomputed_scores")
        if precomputed is not None:
            score_df = precomputed  # (instrument_id,date,combo_score)
        else:
            score_df = df.select(["instrument_id", "date"])
            weight_cols: list[pl.Expr] = []
            for i, term in enumerate(spec.factors):
                factor = get_factor_with_market(term.name, term.params, spec.market)
                result = factor.compute_full(df)
                col = result.name
                sign = 1.0 if term.direction == "high" else -1.0
                zcol = f"_z_{i}"
                vals = result.values.with_columns(
                    (
                        sign
                        * (pl.col(col) - pl.col(col).mean().over("date"))
                        / pl.col(col).std().over("date")
                    ).alias(zcol)
                ).select(["instrument_id", "date", zcol])
                score_df = score_df.join(vals, on=["instrument_id", "date"], how="left")
                weight_cols.append(pl.col(zcol).fill_null(0.0) * term.weight)
            score_df = score_df.with_columns(sum(weight_cols).alias("combo_score"))

        # 缓存：每日按分降序 iid 列表 + (date,iid)->score
        self._ranked_by_date: dict[DateType, list[str]] = {}
        self._score_by_date: dict[tuple, float] = {}
        for (d,), g in score_df.group_by(["date"], maintain_order=True):
            g_sorted = g.sort("combo_score", descending=True, nulls_last=True)
            iids = g_sorted["instrument_id"].to_list()
            self._ranked_by_date[d] = iids
            for iid, sc in zip(iids, g_sorted["combo_score"].to_list(), strict=True):
                if sc is not None:
                    self._score_by_date[(d, iid)] = sc

        all_dates = sorted(df["date"].unique().to_list())
        self._date_index = {d: i for i, d in enumerate(all_dates)}
        self._last_rebalance_idx: int | None = None
        self._last_processed_date: DateType | None = None
        self._full_data = df

    def next(self, ctx: StrategyContext) -> None:
        current_date = ctx.date
        if current_date == self._last_processed_date:
            return  # 一天只处理一次（首个 instrument 调用做全部工作）

        idx = self._date_index.get(current_date)
        if idx is None:
            return

        # 周期调仓闸门
        if (
            self._last_rebalance_idx is not None
            and idx - self._last_rebalance_idx < self._spec.rebalance
        ):
            self._last_processed_date = current_date
            return

        self._last_rebalance_idx = idx
        self._last_processed_date = current_date

        universe = set(ctx.pit_universe(current_date))
        ranked = [iid for iid in self._ranked_by_date.get(current_date, []) if iid in universe]
        top = ranked[: self._spec.top_k]
        top_set = set(top)

        day = self._full_data.filter(pl.col("date") == current_date)
        close_of = {r["instrument_id"]: r["close"] for r in day.iter_rows(named=True)}
        ticker_of = {r["instrument_id"]: r["ticker"] for r in day.iter_rows(named=True)}

        # SELL: 持仓掉出 top_set
        for iid in list(ctx.positions.keys()):
            if iid in top_set:
                continue
            price = close_of.get(iid)
            if price is None:
                continue
            sig = ctx.signal("SELL", iid, price, note="掉出 top_k")
            sig.ticker = ticker_of.get(iid, iid)

        # BUY: top_set 中未持仓
        for rank_pos, iid in enumerate(top, start=1):
            if ctx.has_position(iid):
                continue
            price = close_of.get(iid)
            if price is None or price <= 0:
                continue
            sig = ctx.signal(
                "BUY",
                iid,
                price,
                trigger_value=self._score_by_date.get((current_date, iid)),
                note=f"rank={rank_pos}",
            )
            sig.ticker = ticker_of.get(iid, iid)
