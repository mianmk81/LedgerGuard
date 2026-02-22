"""
DuckDB storage implementation for the Business Reliability Engine.

Provides a production-grade local storage backend using DuckDB with full support
for the medallion architecture (Bronze/Silver/Gold layers). This implementation
is optimized for development, testing, and small-to-medium production deployments
with efficient JSON handling and columnar storage.

Key features:
- Thread-safe connection pooling
- Automatic schema creation and migration
- Efficient JSON column support for nested data
- Proper indexing for query performance
- Transaction semantics with rollback on error
- Comprehensive error handling with structured logging
"""

import json
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import structlog

from api.models.blast_radius import BlastRadius
from api.models.events import CanonicalEvent
from api.models.incidents import Incident, IncidentCascade
from api.models.monitors import MonitorAlert, MonitorRule
from api.models.postmortem import Postmortem
from api.models.rca import CausalChain
from api.models.simulation import IncidentComparison, WhatIfScenario
from api.models.system import RunManifest

from .base import StorageBackend

logger = structlog.get_logger(__name__)


class StorageError(Exception):
    """Base exception for all storage operation failures."""

    pass


class DuckDBStorage(StorageBackend):
    """
    DuckDB implementation of the storage backend.

    Provides thread-safe access to a local DuckDB database with automatic schema
    management, connection pooling, and optimized query performance. Suitable for
    development, testing, and production deployments up to ~100GB data scale.

    Architecture:
    - Uses DuckDB's native JSON support for complex nested objects
    - Implements proper indexing on commonly queried columns
    - Thread-safe connection management with per-thread connections
    - Automatic table creation on first access
    - Support for bulk operations with efficient batching

    Attributes:
        db_path: Path to the DuckDB database file
        _local: Thread-local storage for per-thread connections
        _lock: Thread lock for schema operations
        _initialized: Flag tracking whether schema is initialized
    """

    def __init__(self, db_path: str = "./data/bre.duckdb"):
        """
        Initialize DuckDB storage backend.

        Args:
            db_path: Path to DuckDB database file (default: ./data/bre.duckdb)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False

        logger.info("duckdb_storage_initialized", db_path=str(self.db_path))

        # Initialize schema
        self._initialize_schema()

    @contextmanager
    def _get_connection(self):
        """
        Get a thread-local DuckDB connection.

        Yields:
            DuckDB connection instance

        Raises:
            StorageError: If connection cannot be established
        """
        if not hasattr(self._local, "connection"):
            try:
                self._local.connection = duckdb.connect(str(self.db_path))
                logger.debug("duckdb_connection_created", thread_id=threading.get_ident())
            except Exception as e:
                logger.error("duckdb_connection_failed", error=str(e))
                raise StorageError(f"Failed to connect to DuckDB: {e}") from e

        try:
            yield self._local.connection
        except Exception as e:
            # Rollback on error
            try:
                self._local.connection.rollback()
            except Exception:
                pass
            raise

    def _initialize_schema(self):
        """
        Initialize all database tables and indexes.

        Creates tables for Bronze, Silver, Gold, and operational layers with
        appropriate schemas, indexes, and constraints. This method is idempotent
        and safe to call multiple times.

        Raises:
            StorageError: If schema creation fails
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                with self._get_connection() as conn:
                    # =========================================================
                    # Bronze Layer Tables
                    # =========================================================

                    # QBO Raw Entities
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS bronze_qbo_raw_entities (
                            record_id VARCHAR PRIMARY KEY,
                            entity_id VARCHAR NOT NULL,
                            entity_type VARCHAR NOT NULL,
                            source VARCHAR NOT NULL,
                            operation VARCHAR NOT NULL,
                            raw_payload JSON NOT NULL,
                            webhook_event_id VARCHAR,
                            api_version VARCHAR,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_qbo_entity_id
                        ON bronze_qbo_raw_entities(entity_id)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_qbo_entity_type
                        ON bronze_qbo_raw_entities(entity_type)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_qbo_source
                        ON bronze_qbo_raw_entities(source)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_qbo_created_at
                        ON bronze_qbo_raw_entities(created_at)
                    """)

                    # Supplemental Raw Data
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS bronze_supplemental_raw (
                            record_id VARCHAR PRIMARY KEY,
                            upload_id VARCHAR NOT NULL,
                            source VARCHAR NOT NULL,
                            file_name VARCHAR NOT NULL,
                            raw_payload JSON NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_supp_upload_id
                        ON bronze_supplemental_raw(upload_id)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_bronze_supp_source
                        ON bronze_supplemental_raw(source)
                    """)

                    # =========================================================
                    # Silver Layer Tables
                    # =========================================================

                    # Canonical Events
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS silver_canonical_events (
                            event_id VARCHAR PRIMARY KEY,
                            event_type VARCHAR NOT NULL,
                            event_time TIMESTAMP NOT NULL,
                            ingested_at TIMESTAMP NOT NULL,
                            source VARCHAR NOT NULL,
                            source_entity_id VARCHAR NOT NULL,
                            entity_type VARCHAR NOT NULL,
                            entity_id VARCHAR NOT NULL,
                            related_entity_ids JSON,
                            amount DOUBLE,
                            currency VARCHAR,
                            attributes JSON,
                            data_quality_flags JSON,
                            schema_version VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_silver_event_type
                        ON silver_canonical_events(event_type)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_silver_event_time
                        ON silver_canonical_events(event_time)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_silver_source
                        ON silver_canonical_events(source)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_silver_entity_id
                        ON silver_canonical_events(entity_id)
                    """)

                    # =========================================================
                    # Gold Layer Tables
                    # =========================================================

                    # Daily Metrics
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS gold_daily_metrics (
                            metric_id VARCHAR PRIMARY KEY,
                            metric_name VARCHAR NOT NULL,
                            metric_date DATE NOT NULL,
                            metric_value DOUBLE NOT NULL,
                            aggregation_period VARCHAR NOT NULL,
                            metadata JSON,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_gold_metric_name
                        ON gold_daily_metrics(metric_name)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_gold_metric_date
                        ON gold_daily_metrics(metric_date)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_gold_metric_name_date
                        ON gold_daily_metrics(metric_name, metric_date)
                    """)

                    # =========================================================
                    # Operational Layer Tables
                    # =========================================================

                    # Incidents
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS incidents (
                            incident_id VARCHAR PRIMARY KEY,
                            incident_type VARCHAR NOT NULL,
                            detected_at TIMESTAMP NOT NULL,
                            incident_window_start TIMESTAMP NOT NULL,
                            incident_window_end TIMESTAMP NOT NULL,
                            severity VARCHAR NOT NULL,
                            confidence VARCHAR NOT NULL,
                            detection_methods JSON NOT NULL,
                            primary_metric VARCHAR NOT NULL,
                            primary_metric_value DOUBLE NOT NULL,
                            primary_metric_baseline DOUBLE NOT NULL,
                            primary_metric_zscore DOUBLE NOT NULL,
                            supporting_metrics JSON,
                            evidence_event_ids JSON,
                            evidence_event_count INTEGER NOT NULL,
                            data_quality_score DOUBLE NOT NULL,
                            run_id VARCHAR NOT NULL,
                            cascade_id VARCHAR,
                            status VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_incidents_type
                        ON incidents(incident_type)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_incidents_severity
                        ON incidents(severity)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_incidents_status
                        ON incidents(status)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_incidents_detected_at
                        ON incidents(detected_at)
                    """)

                    # Incident Cascades
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS incident_cascades (
                            cascade_id VARCHAR PRIMARY KEY,
                            root_incident_id VARCHAR NOT NULL,
                            incident_ids JSON NOT NULL,
                            cascade_path JSON NOT NULL,
                            total_blast_radius JSON NOT NULL,
                            cascade_score DOUBLE NOT NULL,
                            detected_at TIMESTAMP NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cascades_detected_at
                        ON incident_cascades(detected_at)
                    """)

                    # Causal Chains
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS causal_chains (
                            chain_id VARCHAR PRIMARY KEY,
                            incident_id VARCHAR NOT NULL,
                            paths JSON NOT NULL,
                            algorithm_version VARCHAR NOT NULL,
                            causal_window_start TIMESTAMP NOT NULL,
                            causal_window_end TIMESTAMP NOT NULL,
                            dependency_graph_version VARCHAR NOT NULL,
                            run_id VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_causal_chains_incident_id
                        ON causal_chains(incident_id)
                    """)

                    # Blast Radii
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS blast_radii (
                            incident_id VARCHAR PRIMARY KEY,
                            customers_affected INTEGER NOT NULL,
                            orders_affected INTEGER NOT NULL,
                            products_affected INTEGER NOT NULL,
                            vendors_involved INTEGER NOT NULL,
                            estimated_revenue_exposure DOUBLE NOT NULL,
                            estimated_refund_exposure DOUBLE NOT NULL,
                            estimated_churn_exposure INTEGER NOT NULL,
                            downstream_incidents_triggered JSON,
                            blast_radius_severity VARCHAR NOT NULL,
                            narrative TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Postmortems
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS postmortems (
                            postmortem_id VARCHAR PRIMARY KEY,
                            incident_id VARCHAR NOT NULL,
                            cascade_id VARCHAR,
                            generated_at TIMESTAMP NOT NULL,
                            title TEXT NOT NULL,
                            severity VARCHAR NOT NULL,
                            duration VARCHAR NOT NULL,
                            status VARCHAR NOT NULL,
                            one_line_summary TEXT NOT NULL,
                            timeline JSON NOT NULL,
                            causal_chain JSON NOT NULL,
                            root_cause_summary TEXT NOT NULL,
                            blast_radius JSON NOT NULL,
                            contributing_factors JSON NOT NULL,
                            monitors JSON NOT NULL,
                            recommendations JSON NOT NULL,
                            data_quality_score DOUBLE NOT NULL,
                            confidence_note TEXT NOT NULL,
                            algorithm_version VARCHAR NOT NULL,
                            run_id VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_postmortems_incident_id
                        ON postmortems(incident_id)
                    """)

                    # Monitor Rules
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS monitor_rules (
                            monitor_id VARCHAR PRIMARY KEY,
                            name VARCHAR NOT NULL,
                            description TEXT NOT NULL,
                            source_incident_id VARCHAR NOT NULL,
                            metric_name VARCHAR NOT NULL,
                            condition TEXT NOT NULL,
                            baseline_window_days INTEGER NOT NULL,
                            check_frequency VARCHAR NOT NULL,
                            severity_if_triggered VARCHAR NOT NULL,
                            enabled BOOLEAN NOT NULL,
                            alert_message_template TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitor_rules_enabled
                        ON monitor_rules(enabled)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitor_rules_metric_name
                        ON monitor_rules(metric_name)
                    """)

                    # Monitor Alerts
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS monitor_alerts (
                            alert_id VARCHAR PRIMARY KEY,
                            monitor_id VARCHAR NOT NULL,
                            triggered_at TIMESTAMP NOT NULL,
                            metric_name VARCHAR NOT NULL,
                            current_value DOUBLE NOT NULL,
                            baseline_value DOUBLE NOT NULL,
                            threshold TEXT NOT NULL,
                            severity VARCHAR NOT NULL,
                            message TEXT NOT NULL,
                            related_incident_id VARCHAR NOT NULL,
                            status VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitor_alerts_monitor_id
                        ON monitor_alerts(monitor_id)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitor_alerts_status
                        ON monitor_alerts(status)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_monitor_alerts_triggered_at
                        ON monitor_alerts(triggered_at)
                    """)

                    # Run Manifests
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS run_manifests (
                            run_id VARCHAR PRIMARY KEY,
                            started_at TIMESTAMP NOT NULL,
                            completed_at TIMESTAMP,
                            events_processed INTEGER NOT NULL,
                            config_version VARCHAR NOT NULL,
                            schema_versions JSON NOT NULL,
                            detection_methods_used JSON NOT NULL,
                            models_invoked JSON NOT NULL,
                            incidents_detected INTEGER NOT NULL,
                            cascades_detected INTEGER NOT NULL,
                            postmortems_generated INTEGER NOT NULL,
                            monitors_created INTEGER NOT NULL,
                            data_quality_summary JSON NOT NULL,
                            status VARCHAR NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_run_manifests_started_at
                        ON run_manifests(started_at)
                    """)

                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_run_manifests_status
                        ON run_manifests(status)
                    """)

                    # What-If Scenarios
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS whatif_scenarios (
                            scenario_id VARCHAR PRIMARY KEY,
                            perturbations JSON NOT NULL,
                            simulated_metrics JSON NOT NULL,
                            triggered_incidents JSON NOT NULL,
                            triggered_cascades JSON NOT NULL,
                            narrative TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Incident Comparisons
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS incident_comparisons (
                            comparison_id VARCHAR PRIMARY KEY,
                            incident_a_id VARCHAR NOT NULL,
                            incident_b_id VARCHAR NOT NULL,
                            incident_type VARCHAR NOT NULL,
                            shared_root_causes JSON NOT NULL,
                            unique_to_a JSON NOT NULL,
                            unique_to_b JSON NOT NULL,
                            severity_comparison JSON NOT NULL,
                            blast_radius_comparison JSON NOT NULL,
                            narrative TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.commit()
                    logger.info("duckdb_schema_initialized", table_count=18)
                    self._initialized = True

            except Exception as e:
                logger.error("duckdb_schema_initialization_failed", error=str(e))
                raise StorageError(f"Failed to initialize schema: {e}") from e

    def clear_for_testing(self) -> None:
        """
        Truncate all tables. For testing only — use when TESTING=true.
        Allows each test to start with a clean slate.
        """
        import os
        if not os.environ.get("TESTING"):
            return
        tables = [
            "blast_radii", "causal_chains", "incident_cascades", "incidents",
            "postmortems", "monitor_alerts", "monitor_rules",
            "gold_daily_metrics", "silver_canonical_events",
            "run_manifests", "whatif_scenarios", "incident_comparisons",
        ]
        try:
            with self._get_connection() as conn:
                for t in tables:
                    try:
                        conn.execute(f"DELETE FROM {t}")
                    except Exception:
                        pass
                conn.commit()
        except Exception:
            pass

    # =========================================================================
    # Bronze Layer Implementation
    # =========================================================================

    def write_raw_entity(
        self,
        entity_id: str,
        entity_type: str,
        source: str,
        operation: str,
        raw_payload: dict,
        webhook_event_id: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> str:
        """Write raw entity to Bronze layer."""
        from uuid import uuid4

        record_id = str(uuid4())

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO bronze_qbo_raw_entities (
                        record_id, entity_id, entity_type, source, operation,
                        raw_payload, webhook_event_id, api_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record_id,
                        entity_id,
                        entity_type,
                        source,
                        operation,
                        json.dumps(raw_payload),
                        webhook_event_id,
                        api_version,
                    ],
                )
                conn.commit()
                logger.debug(
                    "raw_entity_written",
                    record_id=record_id,
                    entity_type=entity_type,
                    source=source,
                )
                return record_id

        except Exception as e:
            logger.error(
                "write_raw_entity_failed",
                entity_id=entity_id,
                entity_type=entity_type,
                error=str(e),
            )
            raise StorageError(f"Failed to write raw entity: {e}") from e

    def write_supplemental_raw(
        self,
        upload_id: str,
        source: str,
        file_name: str,
        raw_payload: dict,
    ) -> str:
        """Write supplemental raw data to Bronze layer."""
        from uuid import uuid4

        record_id = str(uuid4())

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO bronze_supplemental_raw (
                        record_id, upload_id, source, file_name, raw_payload
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        record_id,
                        upload_id,
                        source,
                        file_name,
                        json.dumps(raw_payload),
                    ],
                )
                conn.commit()
                logger.debug(
                    "supplemental_raw_written",
                    record_id=record_id,
                    upload_id=upload_id,
                )
                return record_id

        except Exception as e:
            logger.error(
                "write_supplemental_raw_failed",
                upload_id=upload_id,
                error=str(e),
            )
            raise StorageError(f"Failed to write supplemental raw data: {e}") from e

    def read_raw_entities(
        self,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Read raw entities from Bronze layer."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT record_id, entity_id, entity_type, source, operation,
                           raw_payload, webhook_event_id, api_version, created_at
                    FROM bronze_qbo_raw_entities
                    WHERE 1=1
                """
                params = []

                if source:
                    query += " AND source = ?"
                    params.append(source)

                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type)

                if since:
                    query += " AND created_at >= ?"
                    params.append(since)

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                result = conn.execute(query, params).fetchall()

                entities = []
                for row in result:
                    entities.append({
                        "record_id": row[0],
                        "entity_id": row[1],
                        "entity_type": row[2],
                        "source": row[3],
                        "operation": row[4],
                        "raw_payload": json.loads(row[5]),
                        "webhook_event_id": row[6],
                        "api_version": row[7],
                        "created_at": row[8].isoformat() if row[8] else None,
                    })

                logger.debug("raw_entities_read", count=len(entities))
                return entities

        except Exception as e:
            logger.error("read_raw_entities_failed", error=str(e))
            raise StorageError(f"Failed to read raw entities: {e}") from e

    def read_supplemental_raw(
        self,
        source: Optional[str] = None,
        upload_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Read supplemental raw data from Bronze layer."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT record_id, upload_id, source, file_name,
                           raw_payload, created_at
                    FROM bronze_supplemental_raw
                    WHERE 1=1
                """
                params = []

                if source:
                    query += " AND source = ?"
                    params.append(source)

                if upload_id:
                    query += " AND upload_id = ?"
                    params.append(upload_id)

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                result = conn.execute(query, params).fetchall()

                records = []
                for row in result:
                    records.append({
                        "record_id": row[0],
                        "upload_id": row[1],
                        "source": row[2],
                        "file_name": row[3],
                        "raw_payload": json.loads(row[4]),
                        "created_at": row[5].isoformat() if row[5] else None,
                    })

                logger.debug("supplemental_raw_read", count=len(records))
                return records

        except Exception as e:
            logger.error("read_supplemental_raw_failed", error=str(e))
            raise StorageError(f"Failed to read supplemental raw data: {e}") from e

    # =========================================================================
    # Silver Layer Implementation
    # =========================================================================

    def write_canonical_events(self, events: list[CanonicalEvent]) -> int:
        """Write canonical events to Silver layer in batch."""
        if not events:
            return 0

        try:
            with self._get_connection() as conn:
                written = 0
                for event in events:
                    try:
                        conn.execute(
                            """
                            INSERT INTO silver_canonical_events (
                                event_id, event_type, event_time, ingested_at, source,
                                source_entity_id, entity_type, entity_id, related_entity_ids,
                                amount, currency, attributes, data_quality_flags, schema_version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            [
                                event.event_id,
                                event.event_type.value,
                                event.event_time,
                                event.ingested_at,
                                event.source,
                                event.source_entity_id,
                                event.entity_type.value,
                                event.entity_id,
                                json.dumps(event.related_entity_ids),
                                event.amount,
                                event.currency,
                                json.dumps(event.attributes),
                                json.dumps(event.data_quality_flags),
                                event.schema_version,
                            ],
                        )
                        written += 1
                    except Exception as e:
                        # Skip duplicates silently
                        if "PRIMARY KEY" in str(e) or "UNIQUE" in str(e):
                            logger.debug("duplicate_event_skipped", event_id=event.event_id)
                        else:
                            raise

                conn.commit()
                logger.info("canonical_events_written", count=written)
                return written

        except Exception as e:
            logger.error("write_canonical_events_failed", error=str(e))
            raise StorageError(f"Failed to write canonical events: {e}") from e

    def read_canonical_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 10000,
    ) -> list[CanonicalEvent]:
        """Read canonical events from Silver layer."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT event_id, event_type, event_time, ingested_at, source,
                           source_entity_id, entity_type, entity_id, related_entity_ids,
                           amount, currency, attributes, data_quality_flags, schema_version
                    FROM silver_canonical_events
                    WHERE 1=1
                """
                params = []

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)

                if source:
                    query += " AND source = ?"
                    params.append(source)

                if start_time:
                    query += " AND event_time >= ?"
                    params.append(start_time)

                if end_time:
                    query += " AND event_time <= ?"
                    params.append(end_time)

                if entity_id:
                    query += " AND entity_id = ?"
                    params.append(entity_id)

                query += " ORDER BY event_time ASC LIMIT ?"
                params.append(limit)

                result = conn.execute(query, params).fetchall()

                events = []
                for row in result:
                    events.append(
                        CanonicalEvent(
                            event_id=row[0],
                            event_type=row[1],
                            event_time=row[2],
                            ingested_at=row[3],
                            source=row[4],
                            source_entity_id=row[5],
                            entity_type=row[6],
                            entity_id=row[7],
                            related_entity_ids=json.loads(row[8]) if row[8] else {},
                            amount=row[9],
                            currency=row[10],
                            attributes=json.loads(row[11]) if row[11] else {},
                            data_quality_flags=json.loads(row[12]) if row[12] else [],
                            schema_version=row[13],
                        )
                    )

                logger.debug("canonical_events_read", count=len(events))
                return events

        except Exception as e:
            logger.error("read_canonical_events_failed", error=str(e))
            raise StorageError(f"Failed to read canonical events: {e}") from e

    # =========================================================================
    # Gold Layer Implementation
    # =========================================================================

    def write_gold_metrics(self, metrics: list[dict]) -> int:
        """Write aggregated metrics to Gold layer."""
        if not metrics:
            return 0

        try:
            with self._get_connection() as conn:
                from uuid import uuid4

                written = 0
                for metric in metrics:
                    metric_id = str(uuid4())
                    conn.execute(
                        """
                        INSERT INTO gold_daily_metrics (
                            metric_id, metric_name, metric_date, metric_value,
                            aggregation_period, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            metric_id,
                            metric["metric_name"],
                            metric["metric_date"],
                            metric["metric_value"],
                            metric.get("aggregation_period", "daily"),
                            json.dumps(metric.get("metadata", {})),
                        ],
                    )
                    written += 1

                conn.commit()
                logger.info("gold_metrics_written", count=written)
                return written

        except Exception as e:
            logger.error("write_gold_metrics_failed", error=str(e))
            raise StorageError(f"Failed to write gold metrics: {e}") from e

    def read_gold_metrics(
        self,
        metric_names: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Read aggregated metrics from Gold layer."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT metric_id, metric_name, metric_date, metric_value,
                           aggregation_period, metadata, created_at
                    FROM gold_daily_metrics
                    WHERE 1=1
                """
                params = []

                if metric_names:
                    placeholders = ",".join(["?"] * len(metric_names))
                    query += f" AND metric_name IN ({placeholders})"
                    params.extend(metric_names)

                if start_date:
                    query += " AND metric_date >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND metric_date <= ?"
                    params.append(end_date)

                query += " ORDER BY metric_date ASC, metric_name ASC"

                result = conn.execute(query, params).fetchall()

                metrics = []
                for row in result:
                    metrics.append({
                        "metric_id": row[0],
                        "metric_name": row[1],
                        "metric_date": row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]),
                        "metric_value": row[3],
                        "aggregation_period": row[4],
                        "metadata": json.loads(row[5]) if row[5] else {},
                        "created_at": row[6].isoformat() if row[6] else None,
                    })

                logger.debug("gold_metrics_read", count=len(metrics))
                return metrics

        except Exception as e:
            logger.error("read_gold_metrics_failed", error=str(e))
            raise StorageError(f"Failed to read gold metrics: {e}") from e

    # =========================================================================
    # Operational Layer - Incident Implementation
    # =========================================================================

    def write_incident(self, incident: Incident) -> str:
        """Write incident to operational storage."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO incidents (
                        incident_id, incident_type, detected_at, incident_window_start,
                        incident_window_end, severity, confidence, detection_methods,
                        primary_metric, primary_metric_value, primary_metric_baseline,
                        primary_metric_zscore, supporting_metrics, evidence_event_ids,
                        evidence_event_count, data_quality_score, run_id, cascade_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        incident.incident_id,
                        incident.incident_type.value,
                        incident.detected_at,
                        incident.incident_window_start,
                        incident.incident_window_end,
                        incident.severity.value,
                        incident.confidence.value,
                        json.dumps([m.value for m in incident.detection_methods]),
                        incident.primary_metric,
                        incident.primary_metric_value,
                        incident.primary_metric_baseline,
                        incident.primary_metric_zscore,
                        json.dumps(incident.supporting_metrics),
                        json.dumps(incident.evidence_event_ids),
                        incident.evidence_event_count,
                        incident.data_quality_score,
                        incident.run_id,
                        incident.cascade_id,
                        incident.status.value,
                    ],
                )
                conn.commit()
                logger.info("incident_written", incident_id=incident.incident_id)
                return incident.incident_id

        except Exception as e:
            logger.error(
                "write_incident_failed",
                incident_id=incident.incident_id,
                error=str(e),
            )
            raise StorageError(f"Failed to write incident: {e}") from e

    def read_incidents(
        self,
        incident_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[Incident]:
        """Read incidents from operational storage."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT incident_id, incident_type, detected_at, incident_window_start,
                           incident_window_end, severity, confidence, detection_methods,
                           primary_metric, primary_metric_value, primary_metric_baseline,
                           primary_metric_zscore, supporting_metrics, evidence_event_ids,
                           evidence_event_count, data_quality_score, run_id, cascade_id, status
                    FROM incidents
                    WHERE 1=1
                """
                params = []

                if incident_type:
                    query += " AND incident_type = ?"
                    params.append(incident_type)

                if severity:
                    query += " AND severity = ?"
                    params.append(severity)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                if start:
                    query += " AND detected_at >= ?"
                    params.append(start)

                if end:
                    query += " AND detected_at <= ?"
                    params.append(end)

                query += " ORDER BY detected_at DESC"

                result = conn.execute(query, params).fetchall()

                incidents = []
                for row in result:
                    incidents.append(
                        Incident(
                            incident_id=row[0],
                            incident_type=row[1],
                            detected_at=row[2],
                            incident_window_start=row[3],
                            incident_window_end=row[4],
                            severity=row[5],
                            confidence=row[6],
                            detection_methods=json.loads(row[7]),
                            primary_metric=row[8],
                            primary_metric_value=row[9],
                            primary_metric_baseline=row[10],
                            primary_metric_zscore=row[11],
                            supporting_metrics=json.loads(row[12]) if row[12] else [],
                            evidence_event_ids=json.loads(row[13]) if row[13] else [],
                            evidence_event_count=row[14],
                            data_quality_score=row[15],
                            run_id=row[16],
                            cascade_id=row[17],
                            status=row[18],
                        )
                    )

                logger.debug("incidents_read", count=len(incidents))
                return incidents

        except Exception as e:
            logger.error("read_incidents_failed", error=str(e))
            raise StorageError(f"Failed to read incidents: {e}") from e

    def write_cascade(self, cascade: IncidentCascade) -> str:
        """Write incident cascade."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO incident_cascades (
                        cascade_id, root_incident_id, incident_ids, cascade_path,
                        total_blast_radius, cascade_score, detected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cascade.cascade_id,
                        cascade.root_incident_id,
                        json.dumps(cascade.incident_ids),
                        json.dumps(cascade.cascade_path),
                        json.dumps(cascade.total_blast_radius),
                        cascade.cascade_score,
                        cascade.detected_at,
                    ],
                )
                conn.commit()
                logger.info("cascade_written", cascade_id=cascade.cascade_id)
                return cascade.cascade_id

        except Exception as e:
            logger.error("write_cascade_failed", error=str(e))
            raise StorageError(f"Failed to write cascade: {e}") from e

    def read_cascades(self) -> list[IncidentCascade]:
        """Read all incident cascades."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT cascade_id, root_incident_id, incident_ids, cascade_path,
                           total_blast_radius, cascade_score, detected_at
                    FROM incident_cascades
                    ORDER BY detected_at DESC
                    """
                ).fetchall()

                cascades = []
                for row in result:
                    cascades.append(
                        IncidentCascade(
                            cascade_id=row[0],
                            root_incident_id=row[1],
                            incident_ids=json.loads(row[2]),
                            cascade_path=json.loads(row[3]),
                            total_blast_radius=json.loads(row[4]),
                            cascade_score=row[5],
                            detected_at=row[6],
                        )
                    )

                logger.debug("cascades_read", count=len(cascades))
                return cascades

        except Exception as e:
            logger.error("read_cascades_failed", error=str(e))
            raise StorageError(f"Failed to read cascades: {e}") from e

    # =========================================================================
    # Operational Layer - RCA Implementation
    # =========================================================================

    def write_causal_chain(self, chain: CausalChain) -> str:
        """Write causal chain RCA result."""
        try:
            with self._get_connection() as conn:
                # Serialize paths to JSON
                paths_json = json.dumps([
                    {
                        "rank": path.rank,
                        "overall_score": path.overall_score,
                        "nodes": [
                            {
                                "metric_name": node.metric_name,
                                "contribution_score": node.contribution_score,
                                "anomaly_magnitude": node.anomaly_magnitude,
                                "temporal_precedence": node.temporal_precedence,
                                "graph_proximity": node.graph_proximity,
                                "data_quality_weight": node.data_quality_weight,
                                "metric_value": node.metric_value,
                                "metric_baseline": node.metric_baseline,
                                "metric_zscore": node.metric_zscore,
                                "anomaly_window": [
                                    node.anomaly_window[0].isoformat(),
                                    node.anomaly_window[1].isoformat(),
                                ],
                                "evidence_clusters": [
                                    cluster.model_dump() for cluster in node.evidence_clusters
                                ],
                            }
                            for node in path.nodes
                        ],
                    }
                    for path in chain.paths
                ])

                conn.execute(
                    """
                    INSERT INTO causal_chains (
                        chain_id, incident_id, paths, algorithm_version,
                        causal_window_start, causal_window_end,
                        dependency_graph_version, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chain.chain_id,
                        chain.incident_id,
                        paths_json,
                        chain.algorithm_version,
                        chain.causal_window[0],
                        chain.causal_window[1],
                        chain.dependency_graph_version,
                        chain.run_id,
                    ],
                )
                conn.commit()
                logger.info("causal_chain_written", chain_id=chain.chain_id)
                return chain.chain_id

        except Exception as e:
            logger.error("write_causal_chain_failed", error=str(e))
            raise StorageError(f"Failed to write causal chain: {e}") from e

    def read_causal_chain(self, incident_id: str) -> Optional[CausalChain]:
        """Read causal chain for specific incident."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT chain_id, incident_id, paths, algorithm_version,
                           causal_window_start, causal_window_end,
                           dependency_graph_version, run_id
                    FROM causal_chains
                    WHERE incident_id = ?
                    LIMIT 1
                    """,
                    [incident_id],
                ).fetchone()

                if not result:
                    return None

                # Deserialize paths from JSON
                from api.models.rca import CausalNode, CausalPath, EvidenceCluster

                paths_data = json.loads(result[2])
                paths = []
                for path_data in paths_data:
                    nodes = []
                    for node_data in path_data["nodes"]:
                        evidence_clusters = [
                            EvidenceCluster(**cluster) for cluster in node_data["evidence_clusters"]
                        ]
                        nodes.append(
                            CausalNode(
                                metric_name=node_data["metric_name"],
                                contribution_score=node_data["contribution_score"],
                                anomaly_magnitude=node_data["anomaly_magnitude"],
                                temporal_precedence=node_data["temporal_precedence"],
                                graph_proximity=node_data["graph_proximity"],
                                data_quality_weight=node_data["data_quality_weight"],
                                metric_value=node_data["metric_value"],
                                metric_baseline=node_data["metric_baseline"],
                                metric_zscore=node_data["metric_zscore"],
                                anomaly_window=(
                                    datetime.fromisoformat(node_data["anomaly_window"][0]),
                                    datetime.fromisoformat(node_data["anomaly_window"][1]),
                                ),
                                evidence_clusters=evidence_clusters,
                            )
                        )
                    paths.append(
                        CausalPath(
                            rank=path_data["rank"],
                            overall_score=path_data["overall_score"],
                            nodes=nodes,
                        )
                    )

                chain = CausalChain(
                    chain_id=result[0],
                    incident_id=result[1],
                    paths=paths,
                    algorithm_version=result[3],
                    causal_window=(result[4], result[5]),
                    dependency_graph_version=result[6],
                    run_id=result[7],
                )

                logger.debug("causal_chain_read", chain_id=chain.chain_id)
                return chain

        except Exception as e:
            logger.error("read_causal_chain_failed", incident_id=incident_id, error=str(e))
            raise StorageError(f"Failed to read causal chain: {e}") from e

    # =========================================================================
    # Operational Layer - Blast Radius Implementation
    # =========================================================================

    def write_blast_radius(self, blast_radius: BlastRadius) -> str:
        """Write blast radius impact assessment."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO blast_radii (
                        incident_id, customers_affected, orders_affected,
                        products_affected, vendors_involved, estimated_revenue_exposure,
                        estimated_refund_exposure, estimated_churn_exposure,
                        downstream_incidents_triggered, blast_radius_severity, narrative
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        blast_radius.incident_id,
                        blast_radius.customers_affected,
                        blast_radius.orders_affected,
                        blast_radius.products_affected,
                        blast_radius.vendors_involved,
                        blast_radius.estimated_revenue_exposure,
                        blast_radius.estimated_refund_exposure,
                        blast_radius.estimated_churn_exposure,
                        json.dumps(blast_radius.downstream_incidents_triggered),
                        blast_radius.blast_radius_severity.value,
                        blast_radius.narrative,
                    ],
                )
                conn.commit()
                logger.info("blast_radius_written", incident_id=blast_radius.incident_id)
                return blast_radius.incident_id

        except Exception as e:
            logger.error("write_blast_radius_failed", error=str(e))
            raise StorageError(f"Failed to write blast radius: {e}") from e

    def read_blast_radius(self, incident_id: str) -> Optional[BlastRadius]:
        """Read blast radius for specific incident."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT incident_id, customers_affected, orders_affected,
                           products_affected, vendors_involved, estimated_revenue_exposure,
                           estimated_refund_exposure, estimated_churn_exposure,
                           downstream_incidents_triggered, blast_radius_severity, narrative
                    FROM blast_radii
                    WHERE incident_id = ?
                    LIMIT 1
                    """,
                    [incident_id],
                ).fetchone()

                if not result:
                    return None

                blast_radius = BlastRadius(
                    incident_id=result[0],
                    customers_affected=result[1],
                    orders_affected=result[2],
                    products_affected=result[3],
                    vendors_involved=result[4],
                    estimated_revenue_exposure=result[5],
                    estimated_refund_exposure=result[6],
                    estimated_churn_exposure=result[7],
                    downstream_incidents_triggered=json.loads(result[8]) if result[8] else [],
                    blast_radius_severity=result[9],
                    narrative=result[10],
                )

                logger.debug("blast_radius_read", incident_id=incident_id)
                return blast_radius

        except Exception as e:
            logger.error("read_blast_radius_failed", incident_id=incident_id, error=str(e))
            raise StorageError(f"Failed to read blast radius: {e}") from e

    # =========================================================================
    # Operational Layer - Postmortem Implementation
    # =========================================================================

    def write_postmortem(self, postmortem: Postmortem) -> str:
        """Write postmortem report."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO postmortems (
                        postmortem_id, incident_id, cascade_id, generated_at, title,
                        severity, duration, status, one_line_summary, timeline,
                        causal_chain, root_cause_summary, blast_radius,
                        contributing_factors, monitors, recommendations,
                        data_quality_score, confidence_note, algorithm_version, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        postmortem.postmortem_id,
                        postmortem.incident_id,
                        postmortem.cascade_id,
                        postmortem.generated_at,
                        postmortem.title,
                        postmortem.severity.value,
                        postmortem.duration,
                        postmortem.status.value,
                        postmortem.one_line_summary,
                        json.dumps([t.model_dump() for t in postmortem.timeline]),
                        postmortem.causal_chain.model_dump_json(),
                        postmortem.root_cause_summary,
                        postmortem.blast_radius.model_dump_json(),
                        json.dumps(postmortem.contributing_factors),
                        json.dumps([m.model_dump() for m in postmortem.monitors]),
                        json.dumps(postmortem.recommendations),
                        postmortem.data_quality_score,
                        postmortem.confidence_note,
                        postmortem.algorithm_version,
                        postmortem.run_id,
                    ],
                )
                conn.commit()
                logger.info("postmortem_written", postmortem_id=postmortem.postmortem_id)
                return postmortem.postmortem_id

        except Exception as e:
            logger.error("write_postmortem_failed", error=str(e))
            raise StorageError(f"Failed to write postmortem: {e}") from e

    def read_postmortem(self, incident_id: str) -> Optional[Postmortem]:
        """Read postmortem for specific incident."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT postmortem_id, incident_id, cascade_id, generated_at, title,
                           severity, duration, status, one_line_summary, timeline,
                           causal_chain, root_cause_summary, blast_radius,
                           contributing_factors, monitors, recommendations,
                           data_quality_score, confidence_note, algorithm_version, run_id
                    FROM postmortems
                    WHERE incident_id = ?
                    LIMIT 1
                    """,
                    [incident_id],
                ).fetchone()

                if not result:
                    return None

                from api.models.postmortem import TimelineEntry

                postmortem = Postmortem(
                    postmortem_id=result[0],
                    incident_id=result[1],
                    cascade_id=result[2],
                    generated_at=result[3],
                    title=result[4],
                    severity=result[5],
                    duration=result[6],
                    status=result[7],
                    one_line_summary=result[8],
                    timeline=[TimelineEntry(**t) for t in json.loads(result[9])],
                    causal_chain=CausalChain.model_validate_json(result[10]),
                    root_cause_summary=result[11],
                    blast_radius=BlastRadius.model_validate_json(result[12]),
                    contributing_factors=json.loads(result[13]),
                    monitors=[MonitorRule(**m) for m in json.loads(result[14])],
                    recommendations=json.loads(result[15]),
                    data_quality_score=result[16],
                    confidence_note=result[17],
                    algorithm_version=result[18],
                    run_id=result[19],
                )

                logger.debug("postmortem_read", incident_id=incident_id)
                return postmortem

        except Exception as e:
            logger.error("read_postmortem_failed", incident_id=incident_id, error=str(e))
            raise StorageError(f"Failed to read postmortem: {e}") from e

    # =========================================================================
    # Operational Layer - Monitor Implementation
    # =========================================================================

    def write_monitor(self, monitor: MonitorRule) -> str:
        """Write monitor rule."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO monitor_rules (
                        monitor_id, name, description, source_incident_id, metric_name,
                        condition, baseline_window_days, check_frequency,
                        severity_if_triggered, enabled, alert_message_template, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        monitor.monitor_id,
                        monitor.name,
                        monitor.description,
                        monitor.source_incident_id,
                        monitor.metric_name,
                        monitor.condition,
                        monitor.baseline_window_days,
                        monitor.check_frequency,
                        monitor.severity_if_triggered.value,
                        monitor.enabled,
                        monitor.alert_message_template,
                        monitor.created_at,
                    ],
                )
                conn.commit()
                logger.info("monitor_written", monitor_id=monitor.monitor_id)
                return monitor.monitor_id

        except Exception as e:
            logger.error("write_monitor_failed", error=str(e))
            raise StorageError(f"Failed to write monitor: {e}") from e

    def read_monitors(self, enabled: Optional[bool] = None) -> list[MonitorRule]:
        """Read monitor rules with optional enabled filter."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT monitor_id, name, description, source_incident_id, metric_name,
                           condition, baseline_window_days, check_frequency,
                           severity_if_triggered, enabled, alert_message_template, created_at
                    FROM monitor_rules
                    WHERE 1=1
                """
                params = []

                if enabled is not None:
                    query += " AND enabled = ?"
                    params.append(enabled)

                query += " ORDER BY created_at DESC"

                result = conn.execute(query, params).fetchall()

                monitors = []
                for row in result:
                    monitors.append(
                        MonitorRule(
                            monitor_id=row[0],
                            name=row[1],
                            description=row[2],
                            source_incident_id=row[3],
                            metric_name=row[4],
                            condition=row[5],
                            baseline_window_days=row[6],
                            check_frequency=row[7],
                            severity_if_triggered=row[8],
                            enabled=row[9],
                            alert_message_template=row[10],
                            created_at=row[11],
                        )
                    )

                logger.debug("monitors_read", count=len(monitors))
                return monitors

        except Exception as e:
            logger.error("read_monitors_failed", error=str(e))
            raise StorageError(f"Failed to read monitors: {e}") from e

    def update_monitor(self, monitor_id: str, **updates) -> bool:
        """Update specific fields of a monitor rule."""
        if not updates:
            return False

        # Whitelist of allowed columns to prevent SQL injection
        ALLOWED_MONITOR_COLUMNS = {
            "enabled", "metric_name", "condition", "threshold",
            "frequency_minutes", "severity", "description"
        }

        try:
            with self._get_connection() as conn:
                # Build dynamic UPDATE query
                set_clauses = []
                params = []

                for key, value in updates.items():
                    if key not in ALLOWED_MONITOR_COLUMNS:
                        raise ValueError(f"Invalid column: {key}")
                    set_clauses.append(f"{key} = ?")
                    params.append(value)

                params.append(monitor_id)

                query = f"""
                    UPDATE monitor_rules
                    SET {', '.join(set_clauses)}
                    WHERE monitor_id = ?
                """

                conn.execute(query, params)
                conn.commit()
                # Check if monitor exists (UPDATE affects 0 rows for nonexistent id)
                cur = conn.execute("SELECT 1 FROM monitor_rules WHERE monitor_id = ?", [monitor_id])
                if cur.fetchone() is None:
                    return False
                return True

        except Exception as e:
            logger.error("update_monitor_failed", monitor_id=monitor_id, error=str(e))
            raise StorageError(f"Failed to update monitor: {e}") from e

    def write_monitor_alert(self, alert: MonitorAlert) -> str:
        """Write monitor alert."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO monitor_alerts (
                        alert_id, monitor_id, triggered_at, metric_name, current_value,
                        baseline_value, threshold, severity, message,
                        related_incident_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        alert.alert_id,
                        alert.monitor_id,
                        alert.triggered_at,
                        alert.metric_name,
                        alert.current_value,
                        alert.baseline_value,
                        alert.threshold,
                        alert.severity.value,
                        alert.message,
                        alert.related_incident_id,
                        alert.status.value,
                    ],
                )
                conn.commit()
                logger.info("monitor_alert_written", alert_id=alert.alert_id)
                return alert.alert_id

        except Exception as e:
            logger.error("write_monitor_alert_failed", error=str(e))
            raise StorageError(f"Failed to write monitor alert: {e}") from e

    def read_monitor_alerts(
        self,
        monitor_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[MonitorAlert]:
        """Read monitor alerts with optional filtering."""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT alert_id, monitor_id, triggered_at, metric_name, current_value,
                           baseline_value, threshold, severity, message,
                           related_incident_id, status
                    FROM monitor_alerts
                    WHERE 1=1
                """
                params = []

                if monitor_id:
                    query += " AND monitor_id = ?"
                    params.append(monitor_id)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY triggered_at DESC"

                result = conn.execute(query, params).fetchall()

                alerts = []
                for row in result:
                    alerts.append(
                        MonitorAlert(
                            alert_id=row[0],
                            monitor_id=row[1],
                            triggered_at=row[2],
                            metric_name=row[3],
                            current_value=row[4],
                            baseline_value=row[5],
                            threshold=row[6],
                            severity=row[7],
                            message=row[8],
                            related_incident_id=row[9],
                            status=row[10],
                        )
                    )

                logger.debug("monitor_alerts_read", count=len(alerts))
                return alerts

        except Exception as e:
            logger.error("read_monitor_alerts_failed", error=str(e))
            raise StorageError(f"Failed to read monitor alerts: {e}") from e

    # =========================================================================
    # Operational Layer - System Metadata Implementation
    # =========================================================================

    def write_run_manifest(self, manifest: RunManifest) -> str:
        """Write run execution manifest."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO run_manifests (
                        run_id, started_at, completed_at, events_processed, config_version,
                        schema_versions, detection_methods_used, models_invoked,
                        incidents_detected, cascades_detected, postmortems_generated,
                        monitors_created, data_quality_summary, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        manifest.run_id,
                        manifest.started_at,
                        manifest.completed_at,
                        manifest.events_processed,
                        manifest.config_version,
                        json.dumps(manifest.schema_versions),
                        json.dumps(manifest.detection_methods_used),
                        json.dumps(manifest.models_invoked),
                        manifest.incidents_detected,
                        manifest.cascades_detected,
                        manifest.postmortems_generated,
                        manifest.monitors_created,
                        json.dumps(manifest.data_quality_summary),
                        manifest.status,
                    ],
                )
                conn.commit()
                logger.info("run_manifest_written", run_id=manifest.run_id)
                return manifest.run_id

        except Exception as e:
            logger.error("write_run_manifest_failed", error=str(e))
            raise StorageError(f"Failed to write run manifest: {e}") from e

    def read_run_manifest(self, run_id: str) -> Optional[RunManifest]:
        """Read run manifest for specific run ID."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT run_id, started_at, completed_at, events_processed, config_version,
                           schema_versions, detection_methods_used, models_invoked,
                           incidents_detected, cascades_detected, postmortems_generated,
                           monitors_created, data_quality_summary, status
                    FROM run_manifests
                    WHERE run_id = ?
                    LIMIT 1
                    """,
                    [run_id],
                ).fetchone()

                if not result:
                    return None

                manifest = RunManifest(
                    run_id=result[0],
                    started_at=result[1],
                    completed_at=result[2],
                    events_processed=result[3],
                    config_version=result[4],
                    schema_versions=json.loads(result[5]),
                    detection_methods_used=json.loads(result[6]),
                    models_invoked=json.loads(result[7]),
                    incidents_detected=result[8],
                    cascades_detected=result[9],
                    postmortems_generated=result[10],
                    monitors_created=result[11],
                    data_quality_summary=json.loads(result[12]),
                    status=result[13],
                )

                logger.debug("run_manifest_read", run_id=run_id)
                return manifest

        except Exception as e:
            logger.error("read_run_manifest_failed", run_id=run_id, error=str(e))
            raise StorageError(f"Failed to read run manifest: {e}") from e

    # =========================================================================
    # Operational Layer - Simulation Implementation
    # =========================================================================

    def write_whatif_scenario(self, scenario: WhatIfScenario) -> str:
        """Write what-if scenario analysis result."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO whatif_scenarios (
                        scenario_id, perturbations, simulated_metrics,
                        triggered_incidents, triggered_cascades, narrative
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        scenario.scenario_id,
                        json.dumps(scenario.perturbations),
                        json.dumps(scenario.simulated_metrics),
                        json.dumps(scenario.triggered_incidents),
                        json.dumps(scenario.triggered_cascades),
                        scenario.narrative,
                    ],
                )
                conn.commit()
                logger.info("whatif_scenario_written", scenario_id=scenario.scenario_id)
                return scenario.scenario_id

        except Exception as e:
            logger.error("write_whatif_scenario_failed", error=str(e))
            raise StorageError(f"Failed to write what-if scenario: {e}") from e

    def read_whatif_scenario(self, scenario_id: str) -> Optional[WhatIfScenario]:
        """Read what-if scenario by scenario ID."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT scenario_id, perturbations, simulated_metrics,
                           triggered_incidents, triggered_cascades, narrative
                    FROM whatif_scenarios
                    WHERE scenario_id = ?
                    LIMIT 1
                    """,
                    [scenario_id],
                ).fetchone()

                if not result:
                    return None

                scenario = WhatIfScenario(
                    scenario_id=result[0],
                    perturbations=json.loads(result[1]),
                    simulated_metrics=json.loads(result[2]),
                    triggered_incidents=json.loads(result[3]),
                    triggered_cascades=json.loads(result[4]),
                    narrative=result[5],
                )

                logger.debug("whatif_scenario_read", scenario_id=scenario_id)
                return scenario

        except Exception as e:
            logger.error("read_whatif_scenario_failed", scenario_id=scenario_id, error=str(e))
            raise StorageError(f"Failed to read what-if scenario: {e}") from e

    def write_comparison(self, comparison: IncidentComparison) -> str:
        """Write incident comparison analysis result."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO incident_comparisons (
                        comparison_id, incident_a_id, incident_b_id, incident_type,
                        shared_root_causes, unique_to_a, unique_to_b,
                        severity_comparison, blast_radius_comparison, narrative
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        comparison.comparison_id,
                        comparison.incident_a_id,
                        comparison.incident_b_id,
                        comparison.incident_type,
                        json.dumps(comparison.shared_root_causes),
                        json.dumps(comparison.unique_to_a),
                        json.dumps(comparison.unique_to_b),
                        json.dumps(comparison.severity_comparison),
                        json.dumps(comparison.blast_radius_comparison),
                        comparison.narrative,
                    ],
                )
                conn.commit()
                logger.info("comparison_written", comparison_id=comparison.comparison_id)
                return comparison.comparison_id

        except Exception as e:
            logger.error("write_comparison_failed", error=str(e))
            raise StorageError(f"Failed to write comparison: {e}") from e

    def read_comparison(self, comparison_id: str) -> Optional[IncidentComparison]:
        """Read incident comparison by comparison ID."""
        try:
            with self._get_connection() as conn:
                result = conn.execute(
                    """
                    SELECT comparison_id, incident_a_id, incident_b_id, incident_type,
                           shared_root_causes, unique_to_a, unique_to_b,
                           severity_comparison, blast_radius_comparison, narrative
                    FROM incident_comparisons
                    WHERE comparison_id = ?
                    LIMIT 1
                    """,
                    [comparison_id],
                ).fetchone()

                if not result:
                    return None

                comparison = IncidentComparison(
                    comparison_id=result[0],
                    incident_a_id=result[1],
                    incident_b_id=result[2],
                    incident_type=result[3],
                    shared_root_causes=json.loads(result[4]),
                    unique_to_a=json.loads(result[5]),
                    unique_to_b=json.loads(result[6]),
                    severity_comparison=json.loads(result[7]),
                    blast_radius_comparison=json.loads(result[8]),
                    narrative=result[9],
                )

                logger.debug("comparison_read", comparison_id=comparison_id)
                return comparison

        except Exception as e:
            logger.error("read_comparison_failed", comparison_id=comparison_id, error=str(e))
            raise StorageError(f"Failed to read comparison: {e}") from e
