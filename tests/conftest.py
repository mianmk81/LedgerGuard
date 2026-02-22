"""
Pytest configuration and shared fixtures for LedgerGuard BRE test suite.

Test-automator agent: Solid framework architecture with proper data factories,
mock storage, environment isolation, and reusable fixtures across all test types
(unit, integration, golden, property-based).
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Set testing environment BEFORE importing app
# Use temp path (must not exist - DuckDB creates the file). :memory: causes
# per-connection DB which breaks multi-threaded tests.
import tempfile
import uuid as _uuid
_test_db_path = os.path.join(tempfile.gettempdir(), f"ledgerguard_test_{_uuid.uuid4().hex[:8]}.duckdb")
os.environ["TESTING"] = "true"
os.environ["DB_PATH"] = _test_db_path


# ---------------------------------------------------------------------------
# Pydantic model factories — reusable across all test suites
# ---------------------------------------------------------------------------

from api.models.enums import (
    AlertStatus,
    BlastRadiusSeverity,
    Confidence,
    DetectionMethod,
    EntityType,
    EventType,
    IncidentStatus,
    IncidentType,
    MonitorStatus,
    Severity,
)
from api.models.events import CanonicalEvent
from api.models.incidents import Incident, IncidentCascade
from api.models.rca import CausalChain, CausalNode, CausalPath, EvidenceCluster
from api.models.blast_radius import BlastRadius
from api.models.monitors import MonitorRule, MonitorAlert
from api.models.postmortem import Postmortem


def make_incident(
    incident_type: IncidentType = IncidentType.REFUND_SPIKE,
    severity: Severity = Severity.HIGH,
    confidence: Confidence = Confidence.HIGH,
    primary_metric: str = "refund_rate",
    primary_metric_value: float = 0.15,
    primary_metric_baseline: float = 0.03,
    primary_metric_zscore: float = 8.5,
    run_id: str = "test_run_001",
    evidence_event_count: int = 45,
    data_quality_score: float = 0.95,
    **overrides,
) -> Incident:
    """Factory function for creating test Incident objects."""
    now = datetime.utcnow()
    defaults = dict(
        incident_id=str(uuid4()),
        incident_type=incident_type,
        detected_at=now,
        incident_window_start=now - timedelta(hours=2),
        incident_window_end=now,
        severity=severity,
        confidence=confidence,
        detection_methods=[DetectionMethod.MAD_ZSCORE],
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        primary_metric_baseline=primary_metric_baseline,
        primary_metric_zscore=primary_metric_zscore,
        supporting_metrics=[],
        evidence_event_ids=[f"evt_{i}" for i in range(5)],
        evidence_event_count=evidence_event_count,
        data_quality_score=data_quality_score,
        run_id=run_id,
        cascade_id=None,
        status=IncidentStatus.OPEN,
    )
    defaults.update(overrides)
    return Incident(**defaults)


def make_causal_node(
    metric_name: str = "supplier_delay_rate",
    contribution_score: float = 0.85,
    anomaly_magnitude: float = 6.0,
    temporal_precedence: float = 0.9,
    graph_proximity: float = 0.8,
    data_quality_weight: float = 0.95,
    **overrides,
) -> CausalNode:
    """Factory function for creating test CausalNode objects."""
    now = datetime.utcnow()
    defaults = dict(
        metric_name=metric_name,
        contribution_score=contribution_score,
        anomaly_magnitude=anomaly_magnitude,
        temporal_precedence=temporal_precedence,
        graph_proximity=graph_proximity,
        data_quality_weight=data_quality_weight,
        metric_value=0.25,
        metric_baseline=0.05,
        metric_zscore=6.0,
        anomaly_window=(now - timedelta(hours=6), now - timedelta(hours=2)),
        evidence_clusters=[],
    )
    defaults.update(overrides)
    return CausalNode(**defaults)


def make_causal_path(rank: int = 1, score: float = 0.85, metric_name: str = "supplier_delay_rate") -> CausalPath:
    """Factory function for creating test CausalPath objects."""
    return CausalPath(
        rank=rank,
        overall_score=score,
        nodes=[make_causal_node(metric_name=metric_name, contribution_score=score)],
    )


def make_causal_chain(incident_id: str = "test_inc_001", n_paths: int = 3) -> CausalChain:
    """Factory function for creating test CausalChain objects."""
    now = datetime.utcnow()
    metrics = ["supplier_delay_rate", "delivery_delay_rate", "ticket_volume", "refund_rate", "churn_proxy"]
    paths = []
    for i in range(n_paths):
        score = round(0.9 - i * 0.15, 2)
        paths.append(make_causal_path(rank=i + 1, score=score, metric_name=metrics[i % len(metrics)]))
    return CausalChain(
        incident_id=incident_id,
        paths=paths,
        causal_window=(now - timedelta(days=30), now),
        dependency_graph_version="dep_graph_v2_test",
        run_id="test_run_001",
    )


def make_blast_radius(
    incident_id: str = "test_inc_001",
    customers_affected: int = 150,
    revenue_exposure: float = 45000.0,
    severity: BlastRadiusSeverity = BlastRadiusSeverity.SIGNIFICANT,
) -> BlastRadius:
    """Factory function for creating test BlastRadius objects."""
    return BlastRadius(
        incident_id=incident_id,
        customers_affected=customers_affected,
        orders_affected=300,
        products_affected=12,
        vendors_involved=3,
        estimated_revenue_exposure=revenue_exposure,
        estimated_refund_exposure=15000.0,
        estimated_churn_exposure=9,
        blast_radius_severity=severity,
        narrative=f"Test blast radius for incident {incident_id}: {customers_affected} customers, ${revenue_exposure:,.0f} revenue exposure.",
    )


def make_canonical_event(
    event_type: EventType = EventType.INVOICE_ISSUED,
    entity_type: EntityType = EntityType.INVOICE,
    amount: float = 1500.0,
    **overrides,
) -> CanonicalEvent:
    """Factory function for creating test CanonicalEvent objects."""
    now = datetime.utcnow()
    entity_id = f"INV-{random.randint(1, 9999):05d}"
    defaults = dict(
        event_id=str(uuid4()),
        event_type=event_type,
        event_time=now,
        entity_type=entity_type,
        entity_id=entity_id,
        source="qbo",
        source_entity_id=entity_id,
        amount=amount,
        currency="USD",
        related_entity_ids={},
    )
    defaults.update(overrides)
    return CanonicalEvent(**defaults)


def make_candidate(
    metric_name: str = "supplier_delay_rate",
    anomaly_magnitude: float = 5.0,
    temporal_precedence: float = 0.8,
    graph_proximity: float = 0.7,
    data_quality_weight: float = 0.9,
    **overrides,
) -> dict:
    """Factory function for creating raw candidate dicts for CausalRanker."""
    now = datetime.utcnow()
    defaults = dict(
        metric_name=metric_name,
        anomaly_magnitude=anomaly_magnitude,
        temporal_precedence=temporal_precedence,
        graph_proximity=graph_proximity,
        data_quality_weight=data_quality_weight,
        metric_value=0.25,
        metric_baseline=0.05,
        metric_zscore=anomaly_magnitude,
        anomaly_window=(now - timedelta(hours=6), now),
        evidence_clusters=[],
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Mock storage — reusable mock for pure unit tests
# ---------------------------------------------------------------------------

class MockStorage:
    """
    In-memory mock of StorageBackend for unit tests.

    Test-automator agent: environment isolation, no I/O dependency.
    """

    def __init__(self):
        self._incidents: list[Incident] = []
        self._causal_chains: dict[str, CausalChain] = {}
        self._blast_radii: dict[str, BlastRadius] = {}
        self._postmortems: dict[str, Postmortem] = {}
        self._monitors: list[MonitorRule] = []
        self._alerts: list[MonitorAlert] = []
        self._cascades: list[IncidentCascade] = []
        self._canonical_events: list[CanonicalEvent] = []
        self._gold_metrics: list[dict] = []
        self._raw_entities: list[dict] = []

    # --- Read methods ---
    def read_incidents(self, severity=None, status=None):
        results = self._incidents
        if severity:
            results = [i for i in results if i.severity.value == severity]
        if status:
            results = [i for i in results if i.status.value == status]
        return results

    def read_causal_chain(self, incident_id):
        return self._causal_chains.get(incident_id)

    def read_blast_radius(self, incident_id):
        return self._blast_radii.get(incident_id)

    def read_postmortem(self, incident_id):
        return self._postmortems.get(incident_id)

    def read_monitors(self, enabled=None):
        results = self._monitors
        if enabled is not None:
            results = [m for m in results if m.enabled == enabled]
        return results

    def read_monitor_alerts(self, status=None):
        results = self._alerts
        if status:
            results = [a for a in results if a.status.value == status]
        return results

    def read_cascades(self):
        return self._cascades

    def read_canonical_events(self, limit=None, event_types=None, start_date=None, end_date=None):
        results = self._canonical_events
        if event_types:
            results = [e for e in results if e.event_type in event_types]
        if limit:
            results = results[:limit]
        return results

    def read_gold_metrics(self, metric_names=None, start_date=None, end_date=None):
        results = self._gold_metrics
        if metric_names:
            results = [m for m in results if m.get("metric_name") in metric_names]
        return results

    # --- Write methods ---
    def write_incident(self, incident):
        self._incidents.append(incident)
        return incident.incident_id

    def write_causal_chain(self, chain):
        self._causal_chains[chain.incident_id] = chain
        return chain.chain_id

    def write_blast_radius(self, blast_radius):
        self._blast_radii[blast_radius.incident_id] = blast_radius
        return blast_radius.incident_id

    def write_postmortem(self, postmortem):
        self._postmortems[postmortem.incident_id] = postmortem
        return postmortem.incident_id

    def write_monitor(self, monitor):
        self._monitors.append(monitor)
        return monitor.monitor_id

    def write_monitor_alert(self, alert):
        self._alerts.append(alert)
        return alert.alert_id

    def write_cascade(self, cascade):
        self._cascades.append(cascade)
        return cascade.cascade_id

    def write_canonical_events(self, events):
        self._canonical_events.extend(events)
        return len(events)

    def write_gold_metrics(self, metrics):
        self._gold_metrics.extend(metrics)
        return len(metrics)

    def write_raw_entity(self, entity_id, entity_type, source, operation, raw_payload, api_version="v3"):
        record = {"entity_id": entity_id, "entity_type": entity_type, "raw_payload": raw_payload}
        self._raw_entities.append(record)
        return entity_id

    def update_monitor(self, monitor_id, enabled=None):
        for m in self._monitors:
            if m.monitor_id == monitor_id:
                if enabled is not None:
                    m.enabled = enabled
                return True
        return False

    def write_comparison(self, comparison):
        return "comp_test"

    def write_whatif_scenario(self, scenario):
        return "sim_test"


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_storage():
    """Fresh MockStorage instance for each test."""
    return MockStorage()


@pytest.fixture
def sample_incident():
    """Single test incident."""
    return make_incident()


@pytest.fixture
def sample_incidents():
    """List of diverse test incidents for multi-incident tests."""
    return [
        make_incident(
            incident_type=IncidentType.REFUND_SPIKE,
            severity=Severity.HIGH,
            primary_metric="refund_rate",
            primary_metric_zscore=8.5,
        ),
        make_incident(
            incident_type=IncidentType.FULFILLMENT_SLA_DEGRADATION,
            severity=Severity.MEDIUM,
            primary_metric="delivery_delay_rate",
            primary_metric_zscore=4.2,
        ),
        make_incident(
            incident_type=IncidentType.SUPPORT_LOAD_SURGE,
            severity=Severity.HIGH,
            primary_metric="ticket_volume",
            primary_metric_zscore=6.1,
        ),
    ]


@pytest.fixture
def sample_candidates():
    """List of candidate dicts for CausalRanker tests."""
    return [
        make_candidate("supplier_delay_rate", anomaly_magnitude=7.0, temporal_precedence=0.9, graph_proximity=0.8),
        make_candidate("delivery_delay_rate", anomaly_magnitude=5.5, temporal_precedence=0.7, graph_proximity=0.6),
        make_candidate("ticket_volume", anomaly_magnitude=4.0, temporal_precedence=0.5, graph_proximity=0.4),
        make_candidate("review_score_avg", anomaly_magnitude=3.0, temporal_precedence=0.3, graph_proximity=0.3),
    ]


@pytest.fixture
def sample_causal_chain(sample_incidents):
    """Sample CausalChain with 3 ranked paths, keyed to first incident."""
    return make_causal_chain(incident_id=sample_incidents[0].incident_id)


@pytest.fixture
def sample_blast_radius(sample_incidents):
    """Sample BlastRadius assessment, keyed to first incident."""
    return make_blast_radius(incident_id=sample_incidents[0].incident_id)


@pytest.fixture
def sample_events():
    """List of diverse canonical events."""
    events = []
    for i in range(20):
        events.append(make_canonical_event(
            event_type=random.choice([EventType.INVOICE_ISSUED, EventType.PAYMENT_RECEIVED, EventType.REFUND_ISSUED]),
            amount=random.uniform(100, 5000),
        ))
    return events


@pytest.fixture
def populated_storage(mock_storage, sample_incidents, sample_causal_chain, sample_blast_radius, sample_events):
    """MockStorage pre-populated with sample data for integration-style tests."""
    for inc in sample_incidents:
        mock_storage.write_incident(inc)
    mock_storage.write_causal_chain(sample_causal_chain)
    mock_storage.write_blast_radius(sample_blast_radius)
    mock_storage.write_canonical_events(sample_events)

    # Add gold metrics for 30 days
    from datetime import date
    for day_offset in range(30):
        d = (datetime.utcnow() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        mock_storage.write_gold_metrics([
            {"metric_name": "refund_rate", "metric_date": d, "metric_value": 0.03 + random.uniform(-0.01, 0.01)},
            {"metric_name": "delivery_delay_rate", "metric_date": d, "metric_value": 0.05 + random.uniform(-0.02, 0.02)},
            {"metric_name": "margin_proxy", "metric_date": d, "metric_value": 0.18 + random.uniform(-0.03, 0.03)},
            {"metric_name": "ticket_volume", "metric_date": d, "metric_value": 25 + random.randint(-5, 5)},
            {"metric_name": "dso_proxy", "metric_date": d, "metric_value": 28 + random.randint(-3, 3)},
        ])

    return mock_storage


@pytest.fixture
def client():
    """FastAPI test client for integration tests."""
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Authenticated request headers for integration tests."""
    return {
        "Authorization": "Bearer test_token_for_integration",
        "X-Request-ID": str(uuid4()),
    }


@pytest.fixture
def sample_realm_id():
    """Sample QuickBooks realm ID."""
    return "9876543210"
