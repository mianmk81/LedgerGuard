"""
Data storage layer - Medallion architecture (Bronze/Silver/Gold).

Bronze Layer: Raw API responses (JSON, timestamped)
Silver Layer: Validated, normalized entities (Pydantic validated)
Gold Layer: Enriched with metrics, relationships, anomaly scores

All storage uses DuckDB for OLAP workloads.
"""

from functools import lru_cache

from api.config import get_settings

from .base import StorageBackend
from .duckdb_storage import DuckDBStorage


@lru_cache
def get_storage() -> StorageBackend:
    """
    Get cached storage backend instance (singleton).

    Returns the appropriate storage implementation based on configuration.
    Currently supports DuckDB; designed for future Delta Lake swap.

    Returns:
        StorageBackend implementation instance
    """
    settings = get_settings()
    return DuckDBStorage(db_path=settings.db_path)


__all__ = [
    "StorageBackend",
    "DuckDBStorage",
    "get_storage",
]
