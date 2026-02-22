"""
BRE-RCA Root Cause Analysis Engine.

This module exports the complete RCA engine and all sub-components for
dependency graph analysis, temporal correlation, causal ranking, and
root cause explanation generation.

The BRE-RCA algorithm is the core IP of LedgerGuard's reliability engine,
implementing a principled approach to causal inference using:
- Static business dependency graphs (DAG of metric relationships)
- Temporal precedence scoring via cross-correlation
- Contribution scoring combining anomaly magnitude, timing, proximity, and data quality
- Evidence clustering for human-readable explanations

Example:
    >>> from api.engine.rca import RootCauseAnalyzer
    >>> analyzer = RootCauseAnalyzer(storage=storage)
    >>> causal_chain = analyzer.analyze(incident, lookback_days=14, top_k=5)
    >>> print(f"Root cause: {causal_chain.paths[0].nodes[0].metric_name}")
"""

from .analyzer import RootCauseAnalyzer
from .causal_ranker import CausalRanker
from .dependency_graph import BusinessDependencyGraph, INCIDENT_METRIC_MAP
from .explainer import RCAExplainer
from .temporal_correlation import TemporalCorrelator

__all__ = [
    "RootCauseAnalyzer",
    "BusinessDependencyGraph",
    "TemporalCorrelator",
    "CausalRanker",
    "RCAExplainer",
    "INCIDENT_METRIC_MAP",
]
