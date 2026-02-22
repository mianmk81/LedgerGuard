"""
Abstract storage interface for the Business Reliability Engine.

This module defines the storage abstraction layer that enables swapping between
DuckDB (local development) and Databricks Delta Lake (production) without
changing application code. All storage operations are defined as abstract methods
enforcing a consistent contract across implementations.

The storage layer implements a medallion architecture:
- Bronze: Raw data from source systems with full lineage
- Silver: Canonical normalized events with data quality metadata
- Gold: Aggregated metrics and business insights
- Operational: Incidents, RCA results, monitors, and system metadata
"""

from abc import ABC, abstractmethod
from typing import Optional

from api.models.blast_radius import BlastRadius
from api.models.events import CanonicalEvent
from api.models.incidents import Incident, IncidentCascade
from api.models.monitors import MonitorAlert, MonitorRule
from api.models.postmortem import Postmortem
from api.models.rca import CausalChain
from api.models.simulation import IncidentComparison, WhatIfScenario
from api.models.system import RunManifest


class StorageBackend(ABC):
    """
    Abstract base class for all storage implementations.

    Defines the complete contract for persistence operations across Bronze, Silver,
    Gold, and operational data layers. Implementations must provide full CRUD
    capabilities with proper transaction semantics and error handling.

    Storage implementations should ensure:
    - Thread safety for concurrent access
    - Atomic writes with proper rollback on failure
    - Efficient query performance with appropriate indexing
    - Proper connection pooling and resource cleanup
    - Comprehensive error handling with structured logging
    """

    # =========================================================================
    # Bronze Layer - Raw Entity Storage
    # =========================================================================

    @abstractmethod
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
        """
        Write a raw entity from a source system to Bronze layer.

        Stores the complete unmodified payload from source systems for full lineage
        traceability and replay capability. Bronze layer implements immutable append-only
        semantics to preserve complete audit trail.

        Args:
            entity_id: Source system entity identifier
            entity_type: Type of entity (e.g., "invoice", "customer", "order")
            source: Source system identifier (e.g., "qbo", "shopify")
            operation: Operation type (e.g., "create", "update", "delete")
            raw_payload: Complete raw JSON payload from source system
            webhook_event_id: Optional webhook event ID if triggered by webhook
            api_version: Optional API version string from source system

        Returns:
            Unique record ID for the written raw entity

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def write_supplemental_raw(
        self,
        upload_id: str,
        source: str,
        file_name: str,
        raw_payload: dict,
    ) -> str:
        """
        Write supplemental raw data from manual uploads to Bronze layer.

        Handles manual data ingestion from CSV uploads, spreadsheet imports, or
        other supplemental data sources not connected via API/webhook.

        Args:
            upload_id: Unique identifier for this upload batch
            source: Source identifier (e.g., "csv_upload", "manual_entry")
            file_name: Original filename or data source name
            raw_payload: Complete raw data payload

        Returns:
            Unique record ID for the written supplemental data

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_raw_entities(
        self,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Read raw entities from Bronze layer with optional filtering.

        Supports time-travel queries for debugging, replay, and audit scenarios.
        Results are ordered by ingestion timestamp descending (most recent first).

        Args:
            source: Optional filter by source system
            entity_type: Optional filter by entity type
            since: Optional ISO timestamp to filter records ingested after this time
            limit: Maximum number of records to return (default: 1000)

        Returns:
            List of raw entity records with metadata

        Raises:
            StorageError: If read operation fails
        """
        pass

    @abstractmethod
    def read_supplemental_raw(
        self,
        source: Optional[str] = None,
        upload_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Read supplemental raw data from Bronze layer with optional filtering.

        Args:
            source: Optional filter by source identifier
            upload_id: Optional filter by upload batch ID
            limit: Maximum number of records to return (default: 1000)

        Returns:
            List of supplemental raw records with metadata

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Silver Layer - Canonical Event Storage
    # =========================================================================

    @abstractmethod
    def write_canonical_events(self, events: list[CanonicalEvent]) -> int:
        """
        Write canonical events to Silver layer in batch.

        Performs bulk insert of normalized events with automatic deduplication
        based on event_id. Events with duplicate IDs are silently skipped to
        ensure idempotent ingestion.

        Args:
            events: List of CanonicalEvent instances to write

        Returns:
            Count of events successfully written (excludes duplicates)

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_canonical_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 10000,
    ) -> list[CanonicalEvent]:
        """
        Read canonical events from Silver layer with flexible filtering.

        Supports time-range queries, entity filtering, and event type selection
        for incident detection, RCA, and analytics workloads.

        Args:
            event_type: Optional filter by event type
            source: Optional filter by source system
            start_time: Optional ISO timestamp for range start (inclusive)
            end_time: Optional ISO timestamp for range end (inclusive)
            entity_id: Optional filter by specific entity ID
            limit: Maximum number of events to return (default: 10000)

        Returns:
            List of CanonicalEvent instances ordered by event_time

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Gold Layer - Aggregated Metrics Storage
    # =========================================================================

    @abstractmethod
    def write_gold_metrics(self, metrics: list[dict]) -> int:
        """
        Write aggregated metrics to Gold layer in batch.

        Stores pre-computed business metrics at various time granularities
        (hourly, daily, weekly) for efficient querying and visualization.

        Metric dictionary structure:
        {
            "metric_name": str,
            "metric_date": str,  # ISO date
            "metric_value": float,
            "aggregation_period": str,  # "hourly", "daily", "weekly"
            "metadata": dict  # Optional additional context
        }

        Args:
            metrics: List of metric dictionaries to write

        Returns:
            Count of metrics successfully written

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_gold_metrics(
        self,
        metric_names: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Read aggregated metrics from Gold layer with optional filtering.

        Args:
            metric_names: Optional list of metric names to filter
            start_date: Optional ISO date for range start (inclusive)
            end_date: Optional ISO date for range end (inclusive)

        Returns:
            List of metric dictionaries ordered by date and metric name

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Incident Storage
    # =========================================================================

    @abstractmethod
    def write_incident(self, incident: Incident) -> str:
        """
        Write an incident detection result to operational storage.

        Args:
            incident: Incident instance to persist

        Returns:
            The incident_id of the written incident

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_incidents(
        self,
        incident_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[Incident]:
        """
        Read incidents from operational storage with optional filtering.

        Args:
            incident_type: Optional filter by incident type
            severity: Optional filter by severity level
            status: Optional filter by incident status
            start: Optional ISO timestamp for detection range start
            end: Optional ISO timestamp for detection range end

        Returns:
            List of Incident instances ordered by detected_at descending

        Raises:
            StorageError: If read operation fails
        """
        pass

    @abstractmethod
    def write_cascade(self, cascade: IncidentCascade) -> str:
        """
        Write an incident cascade detection result.

        Args:
            cascade: IncidentCascade instance to persist

        Returns:
            The cascade_id of the written cascade

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_cascades(self) -> list[IncidentCascade]:
        """
        Read all incident cascades ordered by detection time descending.

        Returns:
            List of IncidentCascade instances

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Root Cause Analysis Storage
    # =========================================================================

    @abstractmethod
    def write_causal_chain(self, chain: CausalChain) -> str:
        """
        Write a causal chain RCA result.

        Args:
            chain: CausalChain instance to persist

        Returns:
            The chain_id of the written causal chain

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_causal_chain(self, incident_id: str) -> Optional[CausalChain]:
        """
        Read causal chain for a specific incident.

        Args:
            incident_id: Incident ID to lookup causal chain for

        Returns:
            CausalChain instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Blast Radius Storage
    # =========================================================================

    @abstractmethod
    def write_blast_radius(self, blast_radius: BlastRadius) -> str:
        """
        Write a blast radius impact assessment.

        Args:
            blast_radius: BlastRadius instance to persist

        Returns:
            The incident_id of the written blast radius

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_blast_radius(self, incident_id: str) -> Optional[BlastRadius]:
        """
        Read blast radius assessment for a specific incident.

        Args:
            incident_id: Incident ID to lookup blast radius for

        Returns:
            BlastRadius instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Postmortem Storage
    # =========================================================================

    @abstractmethod
    def write_postmortem(self, postmortem: Postmortem) -> str:
        """
        Write a postmortem report.

        Args:
            postmortem: Postmortem instance to persist

        Returns:
            The postmortem_id of the written postmortem

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_postmortem(self, incident_id: str) -> Optional[Postmortem]:
        """
        Read postmortem report for a specific incident.

        Args:
            incident_id: Incident ID to lookup postmortem for

        Returns:
            Postmortem instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Monitor Storage
    # =========================================================================

    @abstractmethod
    def write_monitor(self, monitor: MonitorRule) -> str:
        """
        Write a monitor rule.

        Args:
            monitor: MonitorRule instance to persist

        Returns:
            The monitor_id of the written monitor

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_monitors(self, enabled: Optional[bool] = None) -> list[MonitorRule]:
        """
        Read monitor rules with optional enabled filter.

        Args:
            enabled: Optional filter by enabled status (True/False/None for all)

        Returns:
            List of MonitorRule instances ordered by created_at descending

        Raises:
            StorageError: If read operation fails
        """
        pass

    @abstractmethod
    def update_monitor(self, monitor_id: str, **updates) -> bool:
        """
        Update specific fields of a monitor rule.

        Supports partial updates of monitor configuration including enable/disable.

        Args:
            monitor_id: Monitor ID to update
            **updates: Key-value pairs of fields to update

        Returns:
            True if monitor was found and updated, False if not found

        Raises:
            StorageError: If update operation fails
        """
        pass

    @abstractmethod
    def write_monitor_alert(self, alert: MonitorAlert) -> str:
        """
        Write a monitor alert.

        Args:
            alert: MonitorAlert instance to persist

        Returns:
            The alert_id of the written alert

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_monitor_alerts(
        self,
        monitor_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[MonitorAlert]:
        """
        Read monitor alerts with optional filtering.

        Args:
            monitor_id: Optional filter by specific monitor ID
            status: Optional filter by alert status

        Returns:
            List of MonitorAlert instances ordered by triggered_at descending

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - System Metadata Storage
    # =========================================================================

    @abstractmethod
    def write_run_manifest(self, manifest: RunManifest) -> str:
        """
        Write a run execution manifest.

        Args:
            manifest: RunManifest instance to persist

        Returns:
            The run_id of the written manifest

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_run_manifest(self, run_id: str) -> Optional[RunManifest]:
        """
        Read run manifest for a specific run ID.

        Args:
            run_id: Run ID to lookup manifest for

        Returns:
            RunManifest instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass

    # =========================================================================
    # Operational Layer - Simulation Storage
    # =========================================================================

    @abstractmethod
    def write_whatif_scenario(self, scenario: WhatIfScenario) -> str:
        """
        Write a what-if scenario analysis result.

        Args:
            scenario: WhatIfScenario instance to persist

        Returns:
            The scenario_id of the written scenario

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_whatif_scenario(self, scenario_id: str) -> Optional[WhatIfScenario]:
        """
        Read what-if scenario by scenario ID.

        Args:
            scenario_id: Scenario ID to lookup

        Returns:
            WhatIfScenario instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass

    @abstractmethod
    def write_comparison(self, comparison: IncidentComparison) -> str:
        """
        Write an incident comparison analysis result.

        Args:
            comparison: IncidentComparison instance to persist

        Returns:
            The comparison_id of the written comparison

        Raises:
            StorageError: If write operation fails
        """
        pass

    @abstractmethod
    def read_comparison(self, comparison_id: str) -> Optional[IncidentComparison]:
        """
        Read incident comparison by comparison ID.

        Args:
            comparison_id: Comparison ID to lookup

        Returns:
            IncidentComparison instance if found, None otherwise

        Raises:
            StorageError: If read operation fails
        """
        pass
