"""
Sync manifest for tracking ingest state.

JSON file tracking each dataset's sync state.
Path: data_lake/_manifest/<market>.json

Fields per dataset:
- market: Market code
- dataset: Dataset name
- last_sync_time: Timestamp of last successful sync
- date_range: {start, end} dates
- row_count: Total rows
- instrument_count: Number of unique instrument_ids

Used by CLI `ingest --status` to show sync state.
"""

import json
import os
from datetime import UTC, datetime
from typing import Any

from trendspec.data.markets import Market


class Manifest:
    """
    Sync manifest for tracking ingest state.

    Each market has its own manifest file for atomic updates.
    """

    def __init__(self, market: Market, root: str) -> None:
        """
        Initialize manifest for a market.

        Args:
            market: Market enum
            root: Root directory for data_lake
        """
        self.market = market
        self.root = root
        self.manifest_dir = os.path.join(root, "_manifest")
        self.manifest_path = os.path.join(self.manifest_dir, f"{market.path}.json")
        self.data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load manifest from file if exists."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def _save(self) -> None:
        """Save manifest to file."""
        os.makedirs(self.manifest_dir, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def get_dataset_state(self, dataset: str) -> dict[str, Any] | None:
        """
        Get sync state for a dataset.

        Args:
            dataset: Dataset name (daily, components, sectors)

        Returns:
            Dataset state dict or None if not synced yet
        """
        return self.data.get(dataset)

    def get_last_date(self, dataset: str, instrument_id: str) -> str | None:
        """
        Get last synced date for a specific instrument.

        Args:
            dataset: Dataset name
            instrument_id: Instrument ID

        Returns:
            Last date string (YYYY-MM-DD) or None if not synced
        """
        state = self.get_dataset_state(dataset)
        if state is None:
            return None

        instruments = state.get("instruments", {})
        return instruments.get(instrument_id)

    def update_dataset_state(
        self,
        dataset: str,
        row_count: int,
        date_range: tuple[str, str],
        instrument_count: int,
        instruments: dict[str, str] | None = None,
    ) -> None:
        """
        Update sync state for a dataset.

        Args:
            dataset: Dataset name
            row_count: Number of rows synced
            date_range: (start_date, end_date) tuple
            instrument_count: Number of unique instrument_ids
            instruments: Dict of {instrument_id: last_date} for incremental tracking
        """
        self.data[dataset] = {
            "market": self.market.value,
            "dataset": dataset,
            "path": f"{self.root}/{self.market.path}/{dataset}",
            "last_sync_time": datetime.now(UTC).isoformat(),
            "date_range": {
                "start": date_range[0],
                "end": date_range[1],
            },
            "row_count": row_count,
            "instrument_count": instrument_count,
            "instruments": instruments or {},
        }
        self._save()

    def update_instrument_date(
        self, dataset: str, instrument_id: str, last_date: str, save: bool = True
    ) -> None:
        """
        Update last date for a specific instrument (for incremental sync).

        Args:
            dataset: Dataset name
            instrument_id: Instrument ID
            last_date: Last synced date (YYYY-MM-DD)
            save: If True, save manifest to file. Set False for batch updates.
        """
        if dataset not in self.data:
            self.data[dataset] = {
                "market": self.market.value,
                "dataset": dataset,
                "instruments": {},
            }

        if "instruments" not in self.data[dataset]:
            self.data[dataset]["instruments"] = {}

        self.data[dataset]["instruments"][instrument_id] = last_date
        self.data[dataset]["last_sync_time"] = datetime.now(UTC).isoformat()
        if save:
            self._save()

    def update_instrument_dates_batch(
        self, dataset: str, instruments: dict[str, str]
    ) -> None:
        """
        Update last dates for multiple instruments in a single batch.

        More efficient than calling update_instrument_date for each instrument,
        as it only saves the manifest once after all updates.

        Args:
            dataset: Dataset name
            instruments: Dict of {instrument_id: last_date}
        """
        if dataset not in self.data:
            self.data[dataset] = {
                "market": self.market.value,
                "dataset": dataset,
                "instruments": {},
            }

        if "instruments" not in self.data[dataset]:
            self.data[dataset]["instruments"] = {}

        self.data[dataset]["instruments"].update(instruments)
        self.data[dataset]["last_sync_time"] = datetime.now(UTC).isoformat()
        self._save()

    def get_status_summary(self) -> list[dict[str, Any]]:
        """
        Get summary of all datasets in manifest.

        Returns:
            List of dataset state summaries
        """
        return list(self.data.values())

    def clear_dataset(self, dataset: str) -> None:
        """
        Clear sync state for a dataset.

        Args:
            dataset: Dataset name
        """
        if dataset in self.data:
            del self.data[dataset]
            self._save()


def read_manifest(market: Market, root: str) -> Manifest:
    """
    Read manifest for a market.

    Args:
        market: Market enum
        root: Root directory for data_lake

    Returns:
        Manifest object
    """
    return Manifest(market, root)


def write_manifest(manifest: Manifest) -> None:
    """
    Write manifest to file.

    Args:
        manifest: Manifest object to write

    Note:
        This is a convenience wrapper. Manifest._save() is called automatically
        on updates, so this is mainly for explicit saves.
    """
    manifest._save()


def get_global_status(root: str) -> dict[str, list[dict[str, Any]]]:
    """
    Get status summary for all markets.

    Args:
        root: Root directory for data_lake

    Returns:
        Dict mapping market code to list of dataset summaries
    """
    manifest_dir = os.path.join(root, "_manifest")

    if not os.path.exists(manifest_dir):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}

    for filename in os.listdir(manifest_dir):
        if not filename.endswith(".json"):
            continue

        market_code = filename[:-5]  # Remove .json
        file_path = os.path.join(manifest_dir, filename)

        with open(file_path) as f:
            data = json.load(f)

        result[market_code] = list(data.values())

    return result
