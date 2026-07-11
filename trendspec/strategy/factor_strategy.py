"""通用声明式因子组合策略。spec 经 params["spec"] 注入。"""

from datetime import date as DateType

import polars as pl

from trendspec.research.factor_cache import compute_combo_scores
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
            self._ranked_by_group_date = {}
            self._score_by_date = {}
            self._date_index = {}
            self._last_rebalance_idx = None
            self._last_processed_date = None
            self._full_data = df
            return

        precomputed = self.get_param("precomputed_scores")
        if precomputed is not None:
            score_df = precomputed  # (instrument_id,date,combo_score[,_group])
            if "_group" not in score_df.columns:
                score_df = score_df.with_columns(pl.lit("_all").alias("_group"))
        else:
            score_df = compute_combo_scores(
                df,
                [t.model_dump() for t in spec.factors],
                spec.market,
                group_by=spec.group_by,
                winsorize_pct=spec.winsorize_pct,
                root=ctx._root,
            )

        # 缓存：每 (date, group) 按分降序 iid 列表 + (date,iid)->score
        self._ranked_by_group_date: dict[tuple, list[str]] = {}
        self._score_by_date: dict[tuple, float] = {}
        for (d, g), rows in score_df.group_by(["date", "_group"], maintain_order=True):
            g_sorted = rows.sort("combo_score", descending=True, nulls_last=True)
            iids = g_sorted["instrument_id"].to_list()
            self._ranked_by_group_date[(d, g)] = iids
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
        if self._spec.sector_filter:
            allowed_sectors = set(self._spec.sector_filter)
            universe = {
                iid for iid in universe
                if ctx.sector(iid, current_date) in allowed_sectors
            }

        day = self._full_data.filter(pl.col("date") == current_date)
        close_of = {r["instrument_id"]: r["close"] for r in day.iter_rows(named=True)}
        ticker_of = {r["instrument_id"]: r["ticker"] for r in day.iter_rows(named=True)}

        # 候选集合：group_by 设置时是"各组 top_k 拼接"，否则是全局单一 top_k
        # （"_all" 是唯一的组名，等价于原来的全局排名）。
        top: list[str] = []
        group_of: dict[str, str] = {}
        groups = self._spec.group_by if self._spec.group_by is not None else {"_all": None}
        for group_name in groups:
            group_ranked = [
                iid for iid in self._ranked_by_group_date.get((current_date, group_name), [])
                if iid in universe
            ]
            if self._spec.top_pct is not None:
                cap = max(1, round(len(group_ranked) * self._spec.top_pct))
            else:
                cap = self._spec.top_k
            selected = group_ranked[:cap]
            top.extend(selected)
            for iid in selected:
                group_of[iid] = group_name
        top_set = set(top)

        # SELL: 持仓掉出候选集合 —— 全清，不留残余
        for iid in list(ctx.positions.keys()):
            if iid in top_set:
                continue
            price = close_of.get(iid)
            if price is None:
                continue
            sig = ctx.signal("SELL", iid, price, note="掉出 top_k")
            sig.ticker = ticker_of.get(iid, iid)
            sig.shares = float(ctx.positions[iid])

        # BUY: top 中未持仓，等权资金分配，现金预算递减防超支
        nav = ctx.available_capital
        for iid, qty in ctx.positions.items():
            price = close_of.get(iid)
            if price is not None:
                nav += qty * price

        target_total_positions = len(top)
        available = ctx.available_capital
        per_slot_budget = nav / target_total_positions if target_total_positions > 0 else 0.0

        for rank_pos, iid in enumerate(top, start=1):
            if ctx.has_position(iid):
                continue
            price = close_of.get(iid)
            if price is None or price <= 0:
                continue
            shares = int(min(per_slot_budget, available) / price)
            if shares < 1:
                continue
            group_name = group_of.get(iid, "")
            sig = ctx.signal(
                "BUY",
                iid,
                price,
                trigger_value=self._score_by_date.get((current_date, iid)),
                note=f"rank={rank_pos}",
            )
            sig.ticker = ticker_of.get(iid, iid)
            sig.shares = float(shares)
            if group_name and group_name != "_all":
                sig.extras["group"] = group_name
            available -= shares * price
