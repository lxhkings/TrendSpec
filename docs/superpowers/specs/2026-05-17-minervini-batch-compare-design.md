# Minervini Trend Template + Batch Backtest Comparison — Design Spec

## Architecture

Five sequential layers:

1. **Indices data layer** — MariaDB `index_prices` table (date, index_id, close) → data_lake/{market}/indices/
2. **StrategyContext.index_close()** — lazy-load + cache indices DataFrame for strategy access
3. **RS_RATING indicator** — IBD-style weighted relative strength, cross-sectional percentile rank 0-100
4. **MinerviniTrendTemplate strategy** — 6-criteria pure-technical filter with 2-day confirmation
5. **ComparisonReport + CLI compare command** — run all strategies head-to-head, sorted table output

## RS_RATING Formula

```
RS_raw = (2 * P/P[63] + P/P[126] + P/P[189] + P/P[252]) / 5
RS_RATING = percentile_rank(RS_raw) within each date, scaled 0-100
```

Weights: recent momentum (63-day) gets 2x weight. Older periods (126, 189, 252) get 1x each. Column name: `RS_RATING_{period}` where period=252.

## Minervini 6 Criteria

All must pass for `confirmation_days` consecutive days:

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | close > MA50 > MA150 > MA200 | Uptrend (all MAs stacked) |
| 2 | MA200[today] > MA200[today-20] | MA200 slope up ≥ 1 month |
| 3 | (close - LL_252) / LL_252 ≥ 0.30 | ≥ 30% above 52-week low |
| 4 | (HH_252 - close) / HH_252 ≤ 0.25 | Within 25% of 52-week high |
| 5 | RS_RATING_252 ≥ 70 | Top 30% relative strength |
| 6 | index_close(SP500) > index_MA50 AND > index_MA200 | Market in uptrend |

## Buy/Sell Logic

- **Event-driven:** Track pass/fail per instrument with `_pass_history` deque (size = `confirmation_days`)
- **BUY:** All `confirmation_days` = True, not already held
- **SELL:** All `confirmation_days` = False (not any), already held
- Default `confirmation_days = 2` — requires 2 consecutive days

## Comparison Report

`ComparisonReport` renders a `rich.Table` with columns: 策略, 总收益, 年化收益, 最大回撤, Sharpe, 交易次数, 耗时(s). Sorted by chosen metric (default: sharpe). Best strategy highlighted in bold green. Strategies with errors shown dimmed with ERROR marker.

Export formats: CSV, JSON, Markdown.

## CLI Commands

```bash
# Ingest indices (one-time, like sectors)
uv run trendspec ingest indices --market us
uv run trendspec ingest indices --market cn

# Backtest single strategy
uv run trendspec backtest run --strategy minervini_trend --market us --start 2022-01-01 --end 2024-12-31

# Compare all strategies
uv run trendspec backtest compare --market us --start 2022-01-01 --end 2024-12-31 --sort sharpe
uv run trendspec backtest compare --market us --start 2022-01-01 --end 2024-12-31 --export csv
uv run trendspec backtest compare --market us --start 2022-01-01 --end 2024-12-31 --exclude rsi_reversal
```

## Strategy Parameters (minervini_trend)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ma_short` | 50 | Short MA period |
| `ma_mid` | 150 | Mid MA period |
| `ma_long` | 200 | Long MA period |
| `ma_slope_lookback` | 20 | Days for MA200 upward slope check |
| `high_low_lookback` | 252 | 52-week high/low window |
| `low_distance_min` | 0.30 | Min distance from 52w low |
| `high_distance_max` | 0.25 | Max distance from 52w high |
| `rs_period` | 252 | RS_RATING lookback |
| `rs_threshold` | 70.0 | Min RS_RATING to qualify |
| `market_index_id` | SP500 | Index for market filter |
| `market_ma_short` | 50 | Index short MA |
| `market_ma_long` | 200 | Index long MA |
| `confirmation_days` | 2 | Consecutive days required |
