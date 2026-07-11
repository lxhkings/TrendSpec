"""通用声明式因子组合策略。spec 经 params["spec"] 注入。"""

from datetime import date as DateType

import polars as pl

from trendspec.data.markets import Market
from trendspec.data.parquet_loader import scan_parquet
from trendspec.data.sectors import TICKER_GROUP_OVERRIDES
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
            self._ranked_by_group_date = {}
            self._score_by_date = {}
            self._date_index = {}
            self._last_rebalance_idx = None
            self._last_processed_date = None
            self._full_data = df
            return

        precomputed = self.get_param("precomputed_scores")
        if precomputed is not None:
            score_df = precomputed  # (instrument_id,date,combo_score)
            score_df = score_df.with_columns(pl.lit("_all").alias("_group"))
        else:
            # 分组归属：group_by 设置时，批量 PIT asof-join sectors 数据集
            # 再展开映射；否则全部落一个虚拟组 "_all"（等价于原全局排名）。
            if spec.group_by is not None:
                market_enum = Market(spec.market.upper())
                sectors_df = scan_parquet(ctx._root, market_enum, "sectors").collect().sort(
                    ["instrument_id", "date"]
                )
                df_sorted = df.sort(["instrument_id", "date"])
                df_with_sector = df_sorted.join_asof(
                    sectors_df.select(["instrument_id", "date", "sector"]),
                    on="date", by="instrument_id", strategy="backward",
                )

                group_lookup = pl.DataFrame(
                    [{"sector": s, "_group": g} for g, members in spec.group_by.items() for s in members]
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
            for i, term in enumerate(spec.factors):
                factor = get_factor_with_market(term.name, term.params, spec.market)
                result = factor.compute_full(df)
                col = result.name
                sign = 1.0 if term.direction == "high" else -1.0
                zcol = f"_z_{i}"
                ncol = f"_null_{i}"

                vals = result.values.join(
                    score_df.select(["instrument_id", "date", "_group"]),
                    on=["instrument_id", "date"], how="inner",
                )
                lo = pl.col(col).quantile(spec.winsorize_pct).over(["date", "_group"])
                hi = pl.col(col).quantile(1 - spec.winsorize_pct).over(["date", "_group"])
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
                weight_cols.append(pl.col(zcol).fill_null(0.0) * term.weight)
                # zcol 为 null 既覆盖原始值缺失（ncol），也覆盖单成员分组下 std 为 null
                # 导致 z-score 无定义的情况——两者都应从排名中剔除，而非按 0 计分。
                missing_any = missing_any | pl.col(ncol).fill_null(True) | pl.col(zcol).is_null()

            score_df = score_df.with_columns(sum(weight_cols).alias("combo_score"))
            score_df = score_df.filter(~missing_any)

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
