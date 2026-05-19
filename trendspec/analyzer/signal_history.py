"""
Signal history builder and cache for TrendSpec.

Generates historical signal statistics by replaying a strategy over past
trading days, computing forward returns, and aggregating per-instrument.
Results are cached as Parquet for fast lookup by the screening report.

Architecture:
    SignalHistoryBuilder.build() → replay signals → attach returns → aggregate
                                              ↓
    data_lake/signal_history/strategy=<n>/market=<m>/agg.parquet

Output schema (per instrument_id):
    instrument_id, n_signals,
    mean_ret_1d, mean_ret_3d, mean_ret_5d, mean_ret_10d, mean_ret_20d,
    hit_rate_5d, hit_rate_20d,
    last_signal_date, last_built_at
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from trendspec.config.settings import get_settings
from trendspec.data.calendar import trading_days_between
from trendspec.data.markets import Market
from trendspec.data.parquet_loader import bars_for_instrument
from trendspec.engine.screening_engine import screen
from trendspec.strategy.base import get_strategy

# =============================================================================
# Cache store
# =============================================================================


class SignalHistoryStore:
    """Read/write signal history aggregates from the data lake."""

    BASE_DIR = "signal_history"

    @classmethod
    def _cache_path(cls, strategy: str, market: Market) -> Path:
        root = get_settings().data_lake.data_lake_root
        return Path(root) / cls.BASE_DIR / f"strategy={strategy}" / f"market={market.value}" / "agg.parquet"

    @classmethod
    def load(cls, strategy: str, market: Market) -> pl.DataFrame | None:
        """Load cached signal history. Returns None if not found."""
        path = cls._cache_path(strategy, market)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    @classmethod
    def save(cls, df: pl.DataFrame, strategy: str, market: Market) -> Path:
        """Write signal history to cache. Returns the path written."""
        path = cls._cache_path(strategy, market)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp file first, then atomic replace to avoid partial writes
        # corrupting the cache if the process is interrupted.
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(path)
        return path


# =============================================================================
# Builder
# =============================================================================


_FORWARD_DAYS = [1, 3, 5, 10, 20]
_PRICE_PAD_CALENDAR_DAYS = 45  # extra calendar days beyond T+20 to ensure price data (covers long holidays)


def _incremental_merge(agg_new: pl.DataFrame, cache_old: pl.DataFrame) -> pl.DataFrame:
    """
    Merge new incremental aggregates into the existing cache with weighted averaging.

    Three categories:
    - Only in old cache (no new signals): keep as-is, aligned to new schema.
    - Only in new agg (never seen before): add as-is.
    - In both: weighted average of n_signals, mean_ret_*, hit_rate_* columns;
      last_signal_date = max; last_built_at = new value.
    """
    log = logging.getLogger(__name__)
    new_ids = set(agg_new["instrument_id"].to_list())
    old_ids = set(cache_old["instrument_id"].to_list())
    overlap_ids = new_ids & old_ids

    # New schema is the target; old cache may be missing newer columns.
    target_cols = agg_new.columns

    def _align_to_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Select target_cols from df, filling missing cols with null."""
        exprs = []
        for col in target_cols:
            if col in df.columns:
                exprs.append(pl.col(col))
            else:
                exprs.append(pl.lit(None).cast(agg_new[col].dtype).alias(col))
        return df.select(exprs)

    if set(target_cols) - set(cache_old.columns):
        log.warning(
            "schema mismatch on incremental merge, missing from cache: %s",
            set(target_cols) - set(cache_old.columns),
        )

    only_old = _align_to_schema(cache_old.filter(~pl.col("instrument_id").is_in(new_ids)))
    only_new = agg_new.filter(~pl.col("instrument_id").is_in(old_ids))

    if not overlap_ids:
        return pl.concat([only_old, only_new])

    # Weighted merge for overlapping instruments
    old_overlap = cache_old.filter(pl.col("instrument_id").is_in(overlap_ids))
    new_overlap = agg_new.filter(pl.col("instrument_id").is_in(overlap_ids))

    j = old_overlap.join(new_overlap, on="instrument_id", how="inner", suffix="_right")

    n_old = pl.col("n_signals")
    n_new = pl.col("n_signals_right")
    n_total = n_old + n_new

    merge_exprs: list[pl.Expr] = [
        pl.col("instrument_id"),
        n_total.alias("n_signals"),
    ]
    for col in ["mean_ret_1d", "mean_ret_3d", "mean_ret_5d", "mean_ret_10d", "mean_ret_20d",
                "hit_rate_5d", "hit_rate_20d"]:
        right = f"{col}_right"
        if col in old_overlap.columns and right in j.columns:
            weighted = (
                (n_old * pl.col(col).fill_null(0.0) + n_new * pl.col(right).fill_null(0.0))
                / n_total
            )
            merge_exprs.append(weighted.alias(col))
        elif right in j.columns:
            # Column exists in new but not old cache (schema evolution): use new value
            merge_exprs.append(pl.col(right).alias(col))

    if "last_signal_date" in j.columns and "last_signal_date_right" in j.columns:
        merge_exprs.append(
            pl.max_horizontal("last_signal_date", "last_signal_date_right").alias("last_signal_date")
        )
    elif "last_signal_date" in j.columns:
        merge_exprs.append(pl.col("last_signal_date"))

    if "last_built_at_right" in j.columns:
        merge_exprs.append(pl.col("last_built_at_right").alias("last_built_at"))
    elif "last_built_at" in j.columns:
        merge_exprs.append(pl.col("last_built_at"))

    merged_overlap = j.select(merge_exprs)

    # All three parts must have identical schemas before concat
    return pl.concat([only_old, only_new, _align_to_schema(merged_overlap)])


class SignalHistoryBuilder:
    """
    Replay a strategy over historical trading days, compute forward returns
    for each signal, and aggregate statistics per instrument.
    """

    def build(
        self,
        strategy_name: str,
        market: Market,
        lookback_years: int = 10,
        rebuild: bool = False,
    ) -> pl.DataFrame:
        """
        Build signal history cache.

        Args:
            strategy_name: Registered strategy name.
            market: Market to replay.
            lookback_years: How many years of history to replay.
            rebuild: If True, ignore existing cache and rebuild from scratch.

        Returns:
            Aggregated DataFrame with signal statistics per instrument.
        """
        strategy_class = get_strategy(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Clip end so all signals have T+20 forward return data available.
        # max(_FORWARD_DAYS)=20 trading days ≈ 28 calendar days; use 45 for safety.
        end = date.today() - timedelta(days=45)
        start = end - timedelta(days=lookback_years * 365)

        # Incremental: check existing cache
        incremental_start = None
        existing_cache = None
        if not rebuild:
            existing_cache = SignalHistoryStore.load(strategy_name, market)
            if existing_cache is not None and not existing_cache.is_empty():
                last_dt = existing_cache["last_signal_date"].max()
                if last_dt is not None:
                    if hasattr(last_dt, "date"):
                        last_dt = last_dt.date()
                    incremental_start = last_dt + timedelta(days=1)

        if incremental_start is not None:
            start = incremental_start

        # Step 1: Replay signals
        signal_records = self._replay_signals(
            strategy_class, market, start, end,
        )
        signal_df = pl.DataFrame(
            signal_records,
            schema={
                "signal_date": pl.Date,
                "instrument_id": pl.String,
                "rank": pl.Float64,
            },
        )

        if signal_df.is_empty():
            return self._empty_aggregate()

        # Step 2: Attach forward returns
        rets_df = self._attach_forward_returns(signal_df, market)

        if rets_df.is_empty():
            return self._empty_aggregate()

        # Step 3: Aggregate
        agg_df = self._aggregate_per_instrument(rets_df)

        # Step 4: Merge with existing cache if incremental update
        if existing_cache is not None and not existing_cache.is_empty():
            agg_df = _incremental_merge(agg_df, existing_cache)

        # Step 5: Save
        SignalHistoryStore.save(agg_df, strategy_name, market)

        return agg_df

    # ----------------------------------------------------------------
    # Internal methods (patchable via patch.object for testing)
    # ----------------------------------------------------------------

    def _get_trading_days(
        self, market: Market, start: date, end: date,
    ) -> list[date]:
        """Get trading days between start and end. Patchable for tests."""
        return trading_days_between(market, start, end)

    def _run_screen(
        self, market: Market, strategy_class: type, target_date: date,
    ):
        """Run a single-day screen. Patchable for tests."""
        return screen(market, strategy_class, target_date)

    def _load_bars(
        self, market: Market, instrument_id: str,
        start_date: date, end_date: date,
    ) -> pl.DataFrame:
        """Load price bars for an instrument. Patchable for tests."""
        return bars_for_instrument(
            market=market,
            instrument_id=instrument_id,
            start_date=start_date,
            end_date=end_date,
            adjustment_mode="forward",
        )

    def _replay_signals(
        self,
        strategy_class: type,
        market: Market,
        start: date,
        end: date,
    ) -> list[dict]:
        """
        Replay the strategy over trading days and collect buy signals.

        Returns list of dicts: {signal_date, instrument_id, rank}.
        """
        trading_days = self._get_trading_days(market, start, end)

        records: list[dict] = []
        n_failed = 0
        log = logging.getLogger(__name__)
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                f"Replaying {strategy_class.name} on {market.value}",
                total=len(trading_days),
            )
            for day in trading_days:
                try:
                    result = self._run_screen(market, strategy_class, day)
                    for sig in result.buy_signals:
                        rank = sig.extras.get("rank", 0.0) if sig.extras else 0.0
                        records.append({
                            "signal_date": day,
                            "instrument_id": sig.instrument_id,
                            "rank": float(rank),
                        })
                except Exception:
                    n_failed += 1
                    log.warning("screen failed on %s, skipping", day)
                progress.advance(task)

        n_days = len(trading_days)
        if n_days >= 3 and n_failed / n_days > 0.5:
            raise RuntimeError(
                f"high failure rate in signal replay: {n_failed}/{n_days} days failed. "
                "Check strategy, market, and data lake configuration."
            )
        if n_failed > 0:
            log.warning("signal replay finished: %d/%d days failed", n_failed, n_days)

        return records

    def _attach_forward_returns(
        self,
        signal_df: pl.DataFrame,
        market: Market,
    ) -> pl.DataFrame:
        """
        Compute forward returns for each signal.

        For each unique instrument_id in signals, load its close price series,
        compute T+N day forward returns via shift(-N), then join back.
        """
        all_rets: list[pl.DataFrame] = []
        unique_instruments = signal_df["instrument_id"].unique().to_list()

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        )

        with progress:
            task = progress.add_task(
                "Computing forward returns",
                total=len(unique_instruments),
            )
            for inst_id in unique_instruments:
                inst_signals = signal_df.filter(pl.col("instrument_id") == inst_id)
                inst_min_date = inst_signals["signal_date"].min()
                inst_max_date = inst_signals["signal_date"].max()

                if inst_min_date is None or inst_max_date is None:
                    progress.advance(task)
                    continue

                end_padded = inst_max_date + timedelta(days=_PRICE_PAD_CALENDAR_DAYS)

                bars = self._load_bars(market, inst_id, inst_min_date, end_padded)

                if bars.is_empty() or "close" not in bars.columns:
                    progress.advance(task)
                    continue

                bars = bars.sort("date").select(["date", "close"])

                # Forward returns: ret_Nd = close_{t+N} / close_t - 1
                ret_exprs = [pl.col("date")]
                for n in _FORWARD_DAYS:
                    ret_exprs.append(
                        (pl.col("close").shift(-n) / pl.col("close") - 1).alias(f"ret_{n}d")
                    )
                bars = bars.select(ret_exprs)

                # Join: match signal_date to bars' date, then drop the date column
                inst_rets = inst_signals.join(
                    bars,
                    left_on="signal_date",
                    right_on="date",
                    how="inner",
                )
                # Polars keeps left key (signal_date) but drops right key (date)
                # when column names differ — no need to drop anything

                if not inst_rets.is_empty():
                    all_rets.append(inst_rets)

                progress.advance(task)

        if not all_rets:
            return pl.DataFrame()

        return pl.concat(all_rets)

    def _aggregate_per_instrument(self, rets_df: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate forward returns per instrument_id.
        """
        ret_cols = [f"ret_{n}d" for n in _FORWARD_DAYS]

        agg_exprs = [
            pl.len().cast(pl.Int64).alias("n_signals"),
            pl.col("signal_date").max().alias("last_signal_date"),
        ]

        for col in ret_cols:
            if col in rets_df.columns:
                agg_exprs.append(pl.col(col).mean().alias(f"mean_{col}"))

        for n in [5, 20]:
            col = f"ret_{n}d"
            if col in rets_df.columns:
                agg_exprs.append(
                    (pl.col(col) > 0).mean().alias(f"hit_rate_{n}d")
                )

        agg = rets_df.group_by("instrument_id").agg(agg_exprs)

        # Cast to naive to stay compatible with existing cache files
        now = datetime.now(timezone.utc).replace(tzinfo=None)  # noqa: UP017
        agg = agg.with_columns(pl.lit(now).alias("last_built_at"))
        agg = agg.filter(pl.col("n_signals") >= 1)
        agg = agg.sort("n_signals", descending=True)

        return agg

    @staticmethod
    def _empty_aggregate() -> pl.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        return pl.DataFrame({
            "instrument_id": pl.Series([], dtype=pl.String),
            "n_signals": pl.Series([], dtype=pl.Int64),
            "mean_ret_1d": pl.Series([], dtype=pl.Float64),
            "mean_ret_3d": pl.Series([], dtype=pl.Float64),
            "mean_ret_5d": pl.Series([], dtype=pl.Float64),
            "mean_ret_10d": pl.Series([], dtype=pl.Float64),
            "mean_ret_20d": pl.Series([], dtype=pl.Float64),
            "hit_rate_5d": pl.Series([], dtype=pl.Float64),
            "hit_rate_20d": pl.Series([], dtype=pl.Float64),
            "last_signal_date": pl.Series([], dtype=pl.Date),
            "last_built_at": pl.Series([], dtype=pl.Datetime),
        })
