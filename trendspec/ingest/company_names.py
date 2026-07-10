"""Company name lookup from the MariaDB `stocks` table."""

from sqlalchemy import text

from trendspec.ingest.mariadb_client import create_engine_from_settings


def fetch_company_names(tickers: list[str], db_settings) -> dict[str, str]:
    """Ticker -> Chinese name lookup, best-effort. Returns {} on empty input or any failure."""
    if not tickers:
        return {}
    try:
        engine = create_engine_from_settings(db_settings)
        placeholders = ", ".join(f":t{i}" for i in range(len(tickers)))
        sql = text(f"SELECT ticker, name FROM stocks WHERE ticker IN ({placeholders})")
        params = {f"t{i}": t for i, t in enumerate(tickers)}
        with engine.connect() as conn:
            return {row[0]: row[1] for row in conn.execute(sql, params)}
    except Exception:
        return {}
