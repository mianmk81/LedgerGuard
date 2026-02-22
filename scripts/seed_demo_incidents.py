#!/usr/bin/env python3
"""
Seed demo incidents, RCA, blast radius, and monitors for LedgerGuard.

Populates the database with realistic incident data so the dashboard
shows a "company under stress" scenario with root cause analysis,
blast radius, and monitors. Run after seed_sandbox.py --mode local.

Usage:
    python scripts/seed_demo_incidents.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import get_settings
from api.models.blast_radius import BlastRadius
from api.models.enums import (
    BlastRadiusSeverity,
    Confidence,
    DetectionMethod,
    IncidentStatus,
    IncidentType,
    Severity,
)
from api.models.incidents import Incident
from api.models.monitors import MonitorRule
from api.models.rca import CausalChain, CausalNode, CausalPath
from api.storage import get_storage
from api.storage.duckdb_storage import DuckDBStorage


def make_incident(
    incident_id: str,
    incident_type: IncidentType,
    severity: Severity,
    primary_metric: str,
    primary_metric_value: float,
    primary_metric_baseline: float,
    primary_metric_zscore: float,
    detected_at: datetime,
) -> Incident:
    """Create incident for demo."""
    return Incident(
        incident_id=incident_id,
        incident_type=incident_type,
        detected_at=detected_at,
        incident_window_start=detected_at - timedelta(hours=6),
        incident_window_end=detected_at,
        severity=severity,
        confidence=Confidence.HIGH,
        detection_methods=[DetectionMethod.MAD_ZSCORE, DetectionMethod.CHANGEPOINT],
        primary_metric=primary_metric,
        primary_metric_value=primary_metric_value,
        primary_metric_baseline=primary_metric_baseline,
        primary_metric_zscore=primary_metric_zscore,
        supporting_metrics=[{"metric": primary_metric, "zscore": primary_metric_zscore}],
        evidence_event_ids=[f"evt_demo_{uuid4().hex[:8]}" for _ in range(8)],
        evidence_event_count=8,
        data_quality_score=0.92,
        run_id="demo_run_001",
        cascade_id=None,
        status=IncidentStatus.OPEN,
    )


def make_causal_chain(incident_id: str, paths_config: list[dict], detected_at: datetime) -> CausalChain:
    """Create causal chain with multiple root cause paths."""
    paths = []
    for i, cfg in enumerate(paths_config):
        nodes = []
        for j, node_cfg in enumerate(cfg["nodes"]):
            win_start = detected_at - timedelta(hours=node_cfg.get("hours_before", 12 + j * 4))
            win_end = win_start + timedelta(hours=4)
            nodes.append(
                CausalNode(
                    metric_name=node_cfg["metric"],
                    contribution_score=node_cfg["score"],
                    anomaly_magnitude=node_cfg.get("zscore", 4.0),
                    temporal_precedence=0.9 - j * 0.1,
                    graph_proximity=0.85,
                    data_quality_weight=0.92,
                    metric_value=node_cfg.get("value", 0.15),
                    metric_baseline=node_cfg.get("baseline", 0.03),
                    metric_zscore=node_cfg.get("zscore", 4.5),
                    anomaly_window=(win_start, win_end),
                    evidence_clusters=[],
                )
            )
        paths.append(
            CausalPath(rank=i + 1, overall_score=cfg["score"], nodes=nodes)
        )
    return CausalChain(
        incident_id=incident_id,
        paths=paths,
        causal_window=(detected_at - timedelta(days=14), detected_at),
        dependency_graph_version="dep_graph_v2",
        run_id="demo_run_001",
    )


def main():
    """Seed demo incidents, RCA, blast radius, monitors."""
    settings = get_settings()
    storage = DuckDBStorage(db_path=settings.db_path)

    # Get some real event IDs from Silver for evidence (optional)
    try:
        events = storage.read_canonical_events(limit=50)
        event_ids = [e.event_id for e in events[:20]] if events else []
    except Exception:
        event_ids = [f"evt_demo_{uuid4().hex[:8]}" for _ in range(20)]

    detected_base = datetime.utcnow() - timedelta(days=2)

    incidents_config = [
        {
            "incident_id": "inc_refund_spike_001",
            "type": IncidentType.REFUND_SPIKE,
            "severity": Severity.HIGH,
            "metric": "refund_rate",
            "value": 0.127,
            "baseline": 0.032,
            "zscore": 7.2,
            "blast": BlastRadius(
                incident_id="inc_refund_spike_001",
                customers_affected=87,
                orders_affected=234,
                products_affected=5,
                vendors_involved=0,
                estimated_revenue_exposure=42500.0,
                estimated_refund_exposure=18500.0,
                estimated_churn_exposure=12,
                blast_radius_severity=BlastRadiusSeverity.SIGNIFICANT,
                narrative=(
                    "Refund spike impacted 87 customers across 234 orders. "
                    "Product quality issue in electronics category drove $18.5K refund exposure. "
                    "12 high-value accounts at risk of churn."
                ),
            ),
            "causal_paths": [
                {
                    "score": 0.91,
                    "nodes": [
                        {"metric": "supplier_delay_rate", "score": 0.88, "zscore": 5.2, "value": 0.18, "baseline": 0.04},
                        {"metric": "delivery_delay_rate", "score": 0.85, "zscore": 4.8, "value": 0.12, "baseline": 0.03},
                        {"metric": "refund_rate", "score": 0.92, "zscore": 7.2, "value": 0.127, "baseline": 0.032},
                    ],
                },
                {
                    "score": 0.72,
                    "nodes": [
                        {"metric": "review_score_avg", "score": 0.68, "zscore": -3.1, "value": 3.2, "baseline": 4.5},
                        {"metric": "ticket_volume", "score": 0.75, "zscore": 5.5, "value": 45, "baseline": 12},
                        {"metric": "refund_rate", "score": 0.72, "zscore": 7.2, "value": 0.127, "baseline": 0.032},
                    ],
                },
            ],
        },
        {
            "incident_id": "inc_fulfillment_002",
            "type": IncidentType.FULFILLMENT_SLA_DEGRADATION,
            "severity": Severity.MEDIUM,
            "metric": "delivery_delay_rate",
            "value": 0.185,
            "baseline": 0.042,
            "zscore": 5.8,
            "blast": BlastRadius(
                incident_id="inc_fulfillment_002",
                customers_affected=156,
                orders_affected=412,
                products_affected=0,
                vendors_involved=3,
                estimated_revenue_exposure=89200.0,
                estimated_refund_exposure=12400.0,
                estimated_churn_exposure=8,
                blast_radius_severity=BlastRadiusSeverity.SIGNIFICANT,
                narrative=(
                    "Delivery delays from Vendor V-001 cascaded to 156 customers. "
                    "412 orders delayed by avg 5.2 days. 3 vendors implicated in supply chain disruption."
                ),
            ),
            "causal_paths": [
                {
                    "score": 0.89,
                    "nodes": [
                        {"metric": "supplier_delay_rate", "score": 0.91, "zscore": 6.1, "value": 0.22, "baseline": 0.05},
                        {"metric": "fulfillment_backlog", "score": 0.86, "zscore": 4.5, "value": 145, "baseline": 32},
                        {"metric": "delivery_delay_rate", "score": 0.89, "zscore": 5.8, "value": 0.185, "baseline": 0.042},
                    ],
                },
            ],
        },
        {
            "incident_id": "inc_margin_003",
            "type": IncidentType.MARGIN_COMPRESSION,
            "severity": Severity.HIGH,
            "metric": "margin_proxy",
            "value": 0.042,
            "baseline": 0.168,
            "zscore": -6.2,
            "blast": BlastRadius(
                incident_id="inc_margin_003",
                customers_affected=0,
                orders_affected=0,
                products_affected=28,
                vendors_involved=12,
                estimated_revenue_exposure=0,
                estimated_refund_exposure=0,
                estimated_churn_exposure=0,
                blast_radius_severity=BlastRadiusSeverity.CONTAINED,
                narrative=(
                    "Margin dropped from 16.8% to 4.2% due to rising costs from 12 vendors. "
                    "28 product SKUs now operating at loss. Immediate pricing review recommended."
                ),
            ),
            "causal_paths": [
                {
                    "score": 0.87,
                    "nodes": [
                        {"metric": "refund_rate", "score": 0.82, "zscore": 5.1, "value": 0.098, "baseline": 0.03},
                        {"metric": "expense_ratio", "score": 0.84, "zscore": 4.2, "value": 0.89, "baseline": 0.65},
                        {"metric": "margin_proxy", "score": 0.87, "zscore": -6.2, "value": 0.042, "baseline": 0.168},
                    ],
                },
            ],
        },
        {
            "incident_id": "inc_liquidity_004",
            "type": IncidentType.LIQUIDITY_CRUNCH_RISK,
            "severity": Severity.CRITICAL,
            "metric": "dso_proxy",
            "value": 67.3,
            "baseline": 28.2,
            "zscore": 8.1,
            "blast": BlastRadius(
                incident_id="inc_liquidity_004",
                customers_affected=42,
                orders_affected=89,
                products_affected=0,
                vendors_involved=0,
                estimated_revenue_exposure=156000.0,
                estimated_refund_exposure=0,
                estimated_churn_exposure=18,
                blast_radius_severity=BlastRadiusSeverity.SEVERE,
                narrative=(
                    "DSO spiked to 67 days (baseline 28). 42 customers with overdue invoices. "
                    "$156K AR aging beyond 60 days. Liquidity risk elevated—immediate collections focus required."
                ),
            ),
            "causal_paths": [
                {
                    "score": 0.94,
                    "nodes": [
                        {"metric": "ar_aging_amount", "score": 0.92, "zscore": 7.5, "value": 156000, "baseline": 42000},
                        {"metric": "dso_proxy", "score": 0.94, "zscore": 8.1, "value": 67.3, "baseline": 28.2},
                    ],
                },
            ],
        },
        {
            "incident_id": "inc_support_005",
            "type": IncidentType.SUPPORT_LOAD_SURGE,
            "severity": Severity.MEDIUM,
            "metric": "ticket_volume",
            "value": 62,
            "baseline": 14,
            "zscore": 6.4,
            "blast": BlastRadius(
                incident_id="inc_support_005",
                customers_affected=98,
                orders_affected=0,
                products_affected=0,
                vendors_involved=0,
                estimated_revenue_exposure=0,
                estimated_refund_exposure=0,
                estimated_churn_exposure=7,
                blast_radius_severity=BlastRadiusSeverity.CONTAINED,
                narrative=(
                    "Support ticket volume surged 4.4x baseline. 98 customers contacted support. "
                    "Correlated with delivery delays—many tickets about shipping status."
                ),
            ),
            "causal_paths": [
                {
                    "score": 0.83,
                    "nodes": [
                        {"metric": "delivery_delay_rate", "score": 0.81, "zscore": 5.2, "value": 0.15, "baseline": 0.04},
                        {"metric": "ticket_volume", "score": 0.83, "zscore": 6.4, "value": 62, "baseline": 14},
                    ],
                },
            ],
        },
    ]

    print("\n" + "=" * 60)
    print("SEEDING DEMO INCIDENTS")
    print("=" * 60)

    for i, cfg in enumerate(incidents_config):
        inc = make_incident(
            incident_id=cfg["incident_id"],
            incident_type=cfg["type"],
            severity=cfg["severity"],
            primary_metric=cfg["metric"],
            primary_metric_value=cfg["value"],
            primary_metric_baseline=cfg["baseline"],
            primary_metric_zscore=cfg["zscore"],
            detected_at=detected_base - timedelta(hours=i * 8),
        )
        storage.write_incident(inc)
        print(f"  [OK] Incident: {cfg['type'].value} ({cfg['severity'].value})")

        chain = make_causal_chain(
            incident_id=cfg["incident_id"],
            paths_config=cfg["causal_paths"],
            detected_at=inc.detected_at,
        )
        storage.write_causal_chain(chain)
        print(f"       RCA: {len(chain.paths)} causal path(s)")

        storage.write_blast_radius(cfg["blast"])
        print(f"       Blast radius: {cfg['blast'].customers_affected} customers, ${cfg['blast'].estimated_revenue_exposure:,.0f} exposure")

    # Create monitors from incidents
    monitors_config = [
        {
            "name": "Refund Rate Spike Alert",
            "description": "Monitors 7-day rolling refund rate. Triggers when exceeding 2x baseline from refund spike incident.",
            "source_incident_id": "inc_refund_spike_001",
            "metric_name": "refund_rate",
            "condition": "value > baseline * 2.0",
            "alert_template": "REFUND SPIKE: Rate is {value:.2%} (baseline {baseline:.2%}). Check product quality and delivery issues.",
        },
        {
            "name": "Delivery Delay Monitor",
            "description": "Detects fulfillment SLA degradation from supplier or backlog issues.",
            "source_incident_id": "inc_fulfillment_002",
            "metric_name": "delivery_delay_rate",
            "condition": "value > baseline * 1.5",
            "alert_template": "DELIVERY DELAY: Rate {value:.2%} exceeds 1.5x baseline. Review supplier performance.",
        },
        {
            "name": "Margin Compression Warning",
            "description": "Alerts when margin drops below 8% (from margin compression incident).",
            "source_incident_id": "inc_margin_003",
            "metric_name": "margin_proxy",
            "condition": "value < 0.08",
            "alert_template": "MARGIN CRITICAL: {value:.1%} below 8% threshold. Cost and pricing review required.",
        },
        {
            "name": "DSO / AR Aging Alert",
            "description": "Warns when days sales outstanding exceeds 45 days (liquidity risk).",
            "source_incident_id": "inc_liquidity_004",
            "metric_name": "dso_proxy",
            "condition": "value > 45",
            "alert_template": "LIQUIDITY RISK: DSO at {value:.0f} days. Accelerate collections.",
        },
    ]

    print("\n  Monitors:")
    for mcfg in monitors_config:
        mon = MonitorRule(
            name=mcfg["name"],
            description=mcfg["description"],
            source_incident_id=mcfg["source_incident_id"],
            metric_name=mcfg["metric_name"],
            condition=mcfg["condition"],
            baseline_window_days=30,
            check_frequency="daily",
            severity_if_triggered=Severity.HIGH,
            enabled=True,
            alert_message_template=mcfg["alert_template"],
        )
        storage.write_monitor(mon)
        print(f"       [OK] {mcfg['name']}")

    print("\n" + "=" * 60)
    print("DEMO SEED COMPLETE")
    print("=" * 60)
    print(f"  Incidents: {len(incidents_config)}")
    print(f"  Causal chains: {len(incidents_config)}")
    print(f"  Blast radii: {len(incidents_config)}")
    print(f"  Monitors: {len(monitors_config)}")
    print("\n  Refresh the dashboard to see the data.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
