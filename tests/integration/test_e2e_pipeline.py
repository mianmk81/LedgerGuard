"""
End-to-end pipeline integration test for LedgerGuard BRE.

Verifies the full pipeline: system health -> incidents -> dashboard -> predictions
all return valid responses with the correct envelope format.
"""

import pytest
from fastapi.testclient import TestClient

from api.auth.dependencies import get_current_realm_id
from api.main import app
from api.storage import get_storage


def mock_realm_id():
    return "demo_realm"


app.dependency_overrides[get_current_realm_id] = mock_realm_id


@pytest.fixture(autouse=True)
def seed_gold_metrics():
    """Seed minimal gold metrics so dashboard/health endpoints work."""
    from datetime import datetime, timedelta

    storage = get_storage()
    if hasattr(storage, "clear_for_testing"):
        storage.clear_for_testing()
    metrics = []
    for day_offset in range(30):
        d = (datetime.utcnow() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        metrics.extend([
            {"metric_name": "daily_revenue", "metric_date": d, "metric_value": 5000 + day_offset * 10},
            {"metric_name": "daily_expenses", "metric_date": d, "metric_value": 3000 + day_offset * 5},
            {"metric_name": "refund_rate", "metric_date": d, "metric_value": 0.03},
            {"metric_name": "margin_proxy", "metric_date": d, "metric_value": 0.18},
            {"metric_name": "dso_proxy", "metric_date": d, "metric_value": 28.0},
            {"metric_name": "order_volume", "metric_date": d, "metric_value": 50},
            {"metric_name": "delivery_delay_rate", "metric_date": d, "metric_value": 0.04},
            {"metric_name": "ticket_volume", "metric_date": d, "metric_value": 15},
            {"metric_name": "review_score_avg", "metric_date": d, "metric_value": 4.2},
        ])
    storage.write_gold_metrics(metrics)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


class TestSystemEndpoints:
    def test_health(self, client):
        resp = client.get("/api/v1/system/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] in ("healthy", "degraded")

    def test_diagnostics(self, client):
        resp = client.get("/api/v1/system/diagnostics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "tables" in body["data"]

    def test_config(self, client):
        resp = client.get("/api/v1/system/config")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_model_status(self, client):
        resp = client.get("/api/v1/system/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert "models" in data
        assert "summary" in data
        assert data["summary"]["total_models"] > 0
        # Every model has a status field
        for model in data["models"]:
            assert model["status"] in ("available", "missing", "loaded", "error")


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------


class TestDashboardEndpoints:
    def test_reports(self, client):
        resp = client.get("/api/v1/dashboard/reports")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_health_score(self, client):
        resp = client.get("/api/v1/metrics/health-score")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Incident endpoints
# ---------------------------------------------------------------------------


class TestIncidentEndpoints:
    def test_list_incidents(self, client):
        resp = client.get("/api/v1/incidents/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)


# ---------------------------------------------------------------------------
# Monitor endpoints
# ---------------------------------------------------------------------------


class TestMonitorEndpoints:
    def test_monitors_health(self, client):
        resp = client.get("/api/v1/metrics/health-score")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Insights endpoints
# ---------------------------------------------------------------------------


class TestInsightsEndpoints:
    def test_cash_runway(self, client):
        resp = client.get("/api/v1/insights/cash-runway")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_support_ticket_sentiment(self, client):
        resp = client.get("/api/v1/insights/support-ticket-sentiment")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True


# ---------------------------------------------------------------------------
# Response envelope contract
# ---------------------------------------------------------------------------


class TestResponseEnvelope:
    """Verify all key endpoints return the standard envelope."""

    ENDPOINTS = [
        "/api/v1/system/health",
        "/api/v1/system/diagnostics",
        "/api/v1/system/config",
        "/api/v1/system/models",
        "/api/v1/dashboard/reports",
        "/api/v1/incidents/",
        "/api/v1/metrics/health-score",
    ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_envelope_format(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.status_code == 200
        body = resp.json()
        assert "success" in body, f"{endpoint} missing 'success' key"
        assert body["success"] is True, f"{endpoint} returned success=False"
