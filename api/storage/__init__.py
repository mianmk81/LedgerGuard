"""
Data storage layer - Medallion architecture (Bronze/Silver/Gold).

Bronze Layer: Raw API responses (JSON, timestamped)
Silver Layer: Validated, normalized entities (Pydantic validated)
Gold Layer: Enriched with metrics, relationships, anomaly scores

All storage uses DuckDB for OLAP workloads.
"""

# Storage modules:
# - bronze.py: Raw data storage
# - silver.py: Validated entity storage
# - gold.py: Enriched analytics storage
# - schema.py: DuckDB table schemas
# - migrate.py: Schema migrations
