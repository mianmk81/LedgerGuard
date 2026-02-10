"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set testing environment
os.environ["TESTING"] = "true"
os.environ["DB_PATH"] = ":memory:"

from api.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Authenticated request headers."""
    # TODO: Generate valid JWT for testing
    token = "test_token"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_realm_id():
    """Sample QuickBooks realm ID."""
    return "1234567890"


@pytest.fixture
def sample_incident():
    """Sample incident data for testing."""
    return {
        "incident_id": "INC-001",
        "title": "Test incident",
        "severity": "high",
        "confidence": 0.92,
        "detected_at": "2026-02-10T12:00:00Z",
        "affected_entities_count": 10,
        "root_causes_count": 2,
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics data for testing."""
    return [
        {
            "metric_name": "total_revenue",
            "current_value": 150000.0,
            "previous_value": 140000.0,
            "change_percent": 7.14,
            "trend": "up",
            "sparkline": [130000, 135000, 140000, 145000, 150000],
        }
    ]
