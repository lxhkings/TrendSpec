# Changelog

All notable changes to TrendSpec are documented here.

## [0.2.0.0] - 2026-05-19

### Added

- **Signal history cache** (`SignalHistoryBuilder`, `SignalHistoryStore`): replay any strategy over historical trading days, compute T+1/3/5/10/20 forward returns, and store per-instrument win rate and mean return as Parquet under `data_lake/signal_history/`.
- **CLI subcommands** `trendspec signal-history build` and `trendspec signal-history status`: build the cache incrementally (resumes from last cached date) or full-rebuild with `--rebuild`, inspect cache health.
- **Screening report enrichment**: clenow_momentum screening output now includes six historical columns — 历史样本数, 历史 1d/5d/20d 均值收益 %, 历史 5d 胜率 %, 信号置信度 — both in terminal table and CSV export. Gracefully shows `-` when cache is not yet built.
- **Atomic Parquet writes**: cache saves write to a `.parquet.tmp` then `os.replace()` to prevent corruption on interrupted builds.
- **High-failure-rate guard**: `_replay_signals` raises `RuntimeError` if >50% of trading days fail to screen (over ≥3 days), surfacing misconfiguration instead of silently returning empty.
- **gstack skill routing rules** added to CLAUDE.md.

### Fixed

- **Incremental merge now correctly weighted-averages** overlapping instruments instead of replacing old stats with new-window-only values. Old `n_signals=5 + new n_signals=1 → merged n_signals=6` with proper weighted `mean_ret_*` and `hit_rate_*`.
- **`_load_signal_history` market casing bug**: `Market(self.market.lower())` raised `ValueError` for all real CLI invocations (market is always lowercase from CLI). Fixed to `Market(self.market.upper())`.
- **`_load_signal_history` result now cached** on the instance — avoids double Parquet load when both `output()` and `export()` are called.
- **Test isolation**: `mock_settings` fixture now also patches `trendspec.analyzer.signal_history.get_settings` and `trendspec.screening.report.get_settings`, preventing tests from writing to the real `data_lake/`.
- **ColumnNotFoundError** in `_signals_to_clenow_dataframe`: percentage expressions now conditional on column presence in `available`, preventing crashes with older cache schemas.
- **`map_elements(skip_nulls=False)`** on `n_signals` so unmatched instruments correctly get `"-"` confidence star instead of `None` in CSV.
- **Schema evolution in `_incremental_merge`**: columns present in new but absent in old cache are now forwarded from new data rather than silently dropped.
- **`status` command**: added `last_signal_date` column guard alongside existing `last_built_at` guard to prevent `KeyError` on malformed cache files.
- **`_PRICE_PAD_CALENDAR_DAYS` raised to 45** (from 30) to cover long-holiday windows (Chinese New Year, Christmas).
- **Build end date clipped** to `today - 45 days` to ensure all signals have T+20 forward return data available.
