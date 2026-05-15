# TrendSpec

Quantitative backtesting and stock screening system for China A-shares and US stocks.

## Features

- Dual-mode: historical backtesting AND daily stock screening
- PIT (point-in-time) universe to avoid survivorship bias
- Local Parquet cache for fast data access
- Support for China A-shares and US stocks (SP500 + Russell 1000)

## Requirements

- Python >= 3.11
- MariaDB/MySQL (for data source)

## Installation

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

## Usage

```bash
uv run trendspec --help
```