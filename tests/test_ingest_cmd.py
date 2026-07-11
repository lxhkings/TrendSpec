"""Tests for trendspec/cli/ingest_cmd.py `status` command."""

from datetime import date

import polars as pl
from typer.testing import CliRunner

from trendspec.cli.ingest_cmd import app
from trendspec.data.markets import Market
from trendspec.ingest.writer import write_parquet

runner = CliRunner()


def test_status_reports_full_dataset_stats_not_last_batch(mock_settings):
    """status must scan the whole parquet dataset, not just the manifest's
    last-synced-batch watermark (a single day's incremental row count)."""
    df = pl.DataFrame({
        "instrument_id": ["600519.SH", "600519.SH", "000001.SZ"],
        "date": [date(2020, 1, 2), date(2024, 6, 3), date(2024, 6, 3)],
        "close": [100.0, 200.0, 10.0],
    })
    write_parquet(df, Market.CN, "daily", mock_settings.data_lake.data_lake_root)

    result = runner.invoke(app, ["status", "--market", "cn"])

    assert result.exit_code == 0, result.output
    assert "3" in result.output
    assert "2020-01-02" in result.output
    assert "2024-06-03" in result.output


def test_status_shows_unsynced_for_missing_dataset(mock_settings):
    result = runner.invoke(app, ["status", "--market", "cn"])

    assert result.exit_code == 0, result.output
    assert "未同步" in result.output
