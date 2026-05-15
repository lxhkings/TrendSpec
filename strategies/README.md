# TrendSpec Strategy Development Guide

This guide explains how to create custom trading strategies in TrendSpec.

## Quick Start

Every strategy inherits from `BaseStrategy` and implements two methods:

```python
from trendspec.strategy import BaseStrategy, StrategyContext

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    params = {"period": 20}

    def init(self, ctx: StrategyContext) -> None:
        # Precompute indicators (vectorized, called once)
        ctx.precompute_indicator("MA", period=self.params["period"])

    def next(self, ctx: StrategyContext) -> None:
        # Per-bar logic (called for each date)
        ma_val = ctx.indicator_value("MA", ctx.instrument_id, period=self.params["period"])
        if ctx.close > ma_val:
            ctx.signal("BUY", ctx.instrument_id, ctx.close)
```

## Core Concepts

### Strategy Lifecycle

1. **Initialize**: Strategy instantiated with params
2. **init()**: Called once with full data for precomputation
3. **next()**: Called per-bar (backtest) or once for latest date (screening)
4. **Signals collected**: Processed through risk pipeline

### Dual-Mode Design

The same strategy works for both backtest and screening:
- **Backtest**: `next()` called for each historical bar
- **Screening**: `next()` called once for the latest date

No mode-specific code needed - the engine handles the difference.

### PIT (Point-In-Time) Access

All universe/sector/factor lookups are date-parametrized to prevent survivorship bias:

```python
# Correct PIT usage
universe_ids = ctx.pit_universe(as_of_date=date)  # Stocks active at that date
sector = ctx.sector(instrument_id, as_of_date=date)  # Sector at that date

# WRONG - no "current" shortcuts
universe_ids = ctx.pit_universe()  # Uses current date implicitly - OK if set
sector = ctx.sector(instrument_id)  # Uses current date implicitly - OK if set
```

## Available Indicators

Pre-built indicators (computed via Polars expressions):

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| MA | Simple Moving Average | period |
| EMA | Exponential Moving Average | period |
| RSI | Relative Strength Index | period |
| MACD | Moving Average Convergence Divergence | fast_period, slow_period, signal_period |
| ATR | Average True Range | period |
| BB | Bollinger Bands | period, std_dev |
| ROC | Rate of Change (Momentum) | period |
| VOL | Historical Volatility | period |

### Using Indicators

```python
def init(self, ctx: StrategyContext) -> None:
    # Precompute in init() - efficient vectorized computation
    ctx.precompute_indicator("MA", period=20)
    ctx.precompute_indicator("RSI", period=14)

def next(self, ctx: StrategyContext) -> None:
    # Look up precomputed values
    ma20 = ctx.indicator_value("MA", ctx.instrument_id, ctx.date, period=20)
    rsi = ctx.indicator_value("RSI", ctx.instrument_id, ctx.date, period=14)
```

## Signal Generation

Generate signals via `ctx.signal()`:

```python
# Basic signal
ctx.signal("BUY", ctx.instrument_id, ctx.close)

# With trigger value and note
ctx.signal(
    "BUY",
    ctx.instrument_id,
    ctx.close,
    trigger_value=ma_val,  # Indicator value that triggered
    note="Price above MA20"
)
```

Signal directions:
- `BUY`: Open/add position
- `SELL`: Close/reduce position

## Position Management

Check positions via context methods:

```python
# Check if holding a position
if ctx.has_position(ctx.instrument_id):
    ctx.signal("SELL", ctx.instrument_id, ctx.close)

# Check position for specific instrument
if ctx.has_position("SH600036"):
    ...

# Get position quantity
qty = ctx.position(ctx.instrument_id)

# Available capital
capital = ctx.available_capital
```

## PIT Universe and Sector Access

### Universe (Active Stocks at Date)

```python
# Get all active instruments at a date
universe_ids = ctx.pit_universe(as_of_date=date)

# Iterate through universe
for instrument_id in universe_ids:
    sector = ctx.sector(instrument_id, as_of_date=date)
```

### Sector Assignment

```python
# Get sector for current instrument
sector = ctx.sector(ctx.instrument_id, ctx.date)

# Get sector for specific instrument at date
sector = ctx.sector("SH600000", as_of_date=date(2024, 1, 15))

# Get all instruments in a sector
sector_stocks = ctx.sector_universe("Finance", as_of_date=date)
```

## Parameters

### Using Dict Parameters

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"
    params = {"period": 20, "threshold": 0.05}

    def _validate_dict_params(self) -> None:
        period = self.get_param("period", 20)
        if period < 1:
            raise ValueError("period must be >= 1")

    def init(self, ctx: StrategyContext) -> None:
        period = self.get_param("period", 20)
        threshold = self.get_param("threshold", 0.05)
```

### Using StrategyParams Dataclass

```python
from dataclasses import dataclass
from trendspec.strategy import StrategyParams

@dataclass
class MyParams(StrategyParams):
    period: int = 20
    threshold: float = 0.05

    def validate(self) -> None:
        if self.period >= self.threshold:
            raise ValueError("Invalid params")

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def init(self, ctx: StrategyContext) -> None:
        period = self.params.period  # Type-safe access
```

## Risk Rules

Strategies can use risk rules via the risk pipeline. Configure in engine:

```python
from trendspec.risk import RiskPipeline, MaxPositionsCount, SectorConcentrationLimit

pipeline = RiskPipeline([
    MaxPositionsCount(max_count=20),
    SectorConcentrationLimit(max_pct=0.3),
])

# Pipeline applied by engine, not strategy code
```

### Sector Neutral Strategy

For sector-balanced portfolios:

```python
from trendspec.risk import SectorNeutralRule

# Use in pipeline for sector-neutral rebalancing
pipeline = RiskPipeline([
    SectorNeutralRule(target_weight=0.1),  # Equal weight per sector
])
```

## Cross-Sectional Operations

Rank stocks within sectors or across universe:

```python
def next(self, ctx: StrategyContext) -> None:
    # Get momentum for all instruments
    momentum_col = f"ROC_{self._period}"

    # Get current data
    current_data = self._data.filter(pl.col("date") == ctx.date)

    # Group by sector and rank
    for sector in unique_sectors:
        sector_data = current_data.filter(
            pl.col("instrument_id").is_in(ctx.sector_universe(sector, ctx.date))
        )
        # Sort by momentum and select top
        ...
```

## Example Strategies

See `trendspec/strategy/examples/` for complete examples:

1. **MACrossStrategy**: Dual MA crossover
   - Demonstrates: indicator computation, crossover detection

2. **RSIReversalStrategy**: RSI oversold/overbought reversal
   - Demonstrates: RSI indicator, threshold-based signals

3. **SectorMomentumStrategy**: Sector-relative momentum ranking
   - Demonstrates: sector lookup, cross-sectional ranking, factor usage

## Running Strategies

### Backtest

```python
from trendspec.engine import BacktestEngine, EngineConfig
from trendspec.data.markets import Market
from trendspec.strategy.examples import MACrossStrategy

config = EngineConfig(
    market=Market.CN_A,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    initial_capital=100000,
)

engine = BacktestEngine(config)
result = engine.run(MACrossStrategy)

# Access results
print(result.metrics.total_return)
print(result.trades)
```

### Screening

```python
from trendspec.engine.screening_engine import screen

result = screen(Market.CN_A, MACrossStrategy, date(2024, 12, 31))

# Access signals
for signal in result.buy_signals:
    print(f"{signal.ticker}: {signal.direction} @ {signal.price}")
```

## Best Practices

1. **Precompute in init()**: Vectorized computation is efficient
2. **PIT for all lookups**: Use date-parameterized methods
3. **Simple next()**: Keep per-bar logic clean
4. **Validate params**: Use `_validate_dict_params()` or StrategyParams
5. **Log signals**: Add notes for debugging
6. **Position checks**: Use `has_position()` before generating signals

## New Strategy = Zero Engine Changes

The framework design ensures:
- Adding a new strategy requires **only** strategy code
- No engine modifications needed
- Same strategy works for backtest and screening
- Risk pipeline configurable independently