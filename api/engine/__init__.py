"""
Business Reliability Engine core components.

This package contains the core analytical engines that power the LedgerGuard
platform, including:

- Event transformation: Bronze → Silver canonical event normalization
- State building: Silver → Gold daily business health metrics
- Anomaly detection: Statistical, changepoint, and ML-based detection
- Root cause analysis: Graph-based causal inference with temporal correlation
- Blast radius mapping: Impact assessment across entity relationships
- Monitor evaluation: SLO compliance checking and alert routing
- Simulation: What-if scenario analysis and incident comparison

All engine components are designed for:
- High performance on analytical workloads (DuckDB optimized)
- Comprehensive observability (structured logging with request IDs)
- Type safety (complete Pydantic validation)
- Testability (pure functions with dependency injection)
"""

__version__ = "1.0.0"

__all__ = [
    "CanonicalEventBuilder",
    "StateBuilder",
    "ChurnClassifier",
]

from api.engine.churn_classifier import ChurnClassifier
from api.engine.event_builder import CanonicalEventBuilder
from api.engine.state_builder import StateBuilder
