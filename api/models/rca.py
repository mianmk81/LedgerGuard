"""
Root Cause Analysis models for the Business Reliability Engine.

This module defines causal chain structures representing the algorithmic
analysis of incident root causes through metric correlation, temporal
precedence, and dependency graph analysis.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class EvidenceCluster(BaseModel):
    """
    A cluster of related events supporting a causal hypothesis.

    Groups events that share common characteristics (entity, timing, attributes)
    to provide evidence for a causal link in the chain. Clustering reduces
    noise and highlights meaningful patterns in the event stream.

    Attributes:
        cluster_label: Human-readable label for this cluster
        event_count: Number of events in this cluster
        event_ids: List of canonical event IDs in this cluster
        entity_type: Type of entity these events relate to
        entity_id: Specific entity ID if cluster is entity-specific
        total_amount: Sum of amounts across events, if applicable
        summary: Human-readable summary of this evidence cluster
    """

    cluster_label: str = Field(description="Human-readable label for this cluster")
    event_count: int = Field(description="Number of events in this cluster", ge=0)
    event_ids: list[str] = Field(
        description="List of canonical event IDs in this cluster"
    )
    entity_type: str = Field(description="Type of entity these events relate to")
    entity_id: Optional[str] = Field(
        default=None, description="Specific entity ID if cluster is entity-specific"
    )
    total_amount: Optional[float] = Field(
        default=None, description="Sum of amounts across events, if applicable"
    )
    summary: str = Field(description="Human-readable summary of evidence cluster")

    @field_validator("event_count")
    @classmethod
    def validate_event_count(cls, v: int) -> int:
        """Ensure event count is non-negative."""
        if v < 0:
            raise ValueError("Event count must be non-negative")
        return v

    @field_validator("event_ids")
    @classmethod
    def validate_event_ids_match_count(cls, v: list[str], info) -> list[str]:
        """Ensure event_ids list matches event_count."""
        if "event_count" in info.data and len(v) != info.data["event_count"]:
            raise ValueError("Length of event_ids must match event_count")
        return v


class CausalNode(BaseModel):
    """
    A node in a causal path representing a contributing metric anomaly.

    Each node represents a metric that exhibited anomalous behavior and
    contributed to the incident. Nodes are scored across multiple dimensions
    including temporal precedence, correlation strength, and data quality.

    Attributes:
        metric_name: Name of the metric represented by this node
        contribution_score: Overall contribution score to incident (0.0-1.0)
        anomaly_magnitude: Statistical magnitude of the anomaly (z-score)
        temporal_precedence: Score indicating how early this preceded incident (0.0-1.0)
        graph_proximity: Proximity score in dependency graph (0.0-1.0)
        data_quality_weight: Quality weighting factor for confidence (0.0-1.0)
        metric_value: Actual value of the metric during anomaly window
        metric_baseline: Expected baseline value for comparison
        metric_zscore: Z-score indicating deviation from baseline
        anomaly_window: Time window when this metric was anomalous
        evidence_clusters: Event clusters supporting this causal link
    """

    metric_name: str = Field(description="Name of the metric represented by this node")
    contribution_score: float = Field(
        description="Overall contribution score to incident", ge=0.0, le=1.0
    )
    anomaly_magnitude: float = Field(
        description="Statistical magnitude of the anomaly (z-score)"
    )
    temporal_precedence: float = Field(
        description="Score indicating how early this preceded incident", ge=0.0, le=1.0
    )
    graph_proximity: float = Field(
        description="Proximity score in dependency graph", ge=0.0, le=1.0
    )
    data_quality_weight: float = Field(
        description="Quality weighting factor for confidence", ge=0.0, le=1.0
    )
    metric_value: float = Field(
        description="Actual value of the metric during anomaly window"
    )
    metric_baseline: float = Field(
        description="Expected baseline value for comparison"
    )
    metric_zscore: float = Field(
        description="Z-score indicating deviation from baseline"
    )
    anomaly_window: tuple[datetime, datetime] = Field(
        description="Time window when this metric was anomalous"
    )
    evidence_clusters: list[EvidenceCluster] = Field(
        default_factory=list, description="Event clusters supporting this causal link"
    )

    @field_validator("contribution_score", "temporal_precedence", "graph_proximity", "data_quality_weight")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are in valid 0.0-1.0 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("anomaly_magnitude", "metric_zscore")
    @classmethod
    def validate_zscore(cls, v: float) -> float:
        """Validate z-scores are reasonable."""
        if abs(v) > 100:
            raise ValueError("Z-score exceeds reasonable bounds")
        return round(v, 4)

    @field_validator("anomaly_window")
    @classmethod
    def validate_time_window(cls, v: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
        """Ensure window end is after start."""
        if v[1] <= v[0]:
            raise ValueError("Anomaly window end must be after start")
        return v


class CausalPath(BaseModel):
    """
    A ranked causal path from root cause to incident.

    Represents one plausible causal chain explaining how metric anomalies
    propagated to produce the detected incident. Multiple paths may exist
    for complex incidents with multiple contributing factors.

    Attributes:
        rank: Ranking of this path (1 = most likely root cause)
        overall_score: Combined score across all dimensions (0.0-1.0)
        nodes: Ordered list of causal nodes from root cause to incident
    """

    rank: int = Field(description="Ranking of this path (1 = most likely)", ge=1)
    overall_score: float = Field(
        description="Combined score across all dimensions", ge=0.0, le=1.0
    )
    nodes: list[CausalNode] = Field(
        description="Ordered list of causal nodes from root to incident"
    )

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        """Ensure overall score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v: list[CausalNode]) -> list[CausalNode]:
        """Ensure at least one node in path."""
        if not v:
            raise ValueError("Causal path must contain at least one node")
        return v


class CausalChain(BaseModel):
    """
    Complete root cause analysis for an incident.

    Represents the algorithmic output of the RCA engine, including ranked
    causal paths, algorithm metadata, and the dependency graph context used
    for analysis. This is the primary analytical artifact for understanding
    incident etiology.

    Attributes:
        chain_id: Unique identifier for this causal chain analysis
        incident_id: ID of the incident being analyzed
        paths: Ranked list of causal paths (most likely first)
        algorithm_version: Version of the RCA algorithm used
        causal_window: Time window analyzed for causal relationships
        dependency_graph_version: Version of dependency graph used
        run_id: ID of the analysis run that generated this chain
    """

    chain_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this causal chain analysis",
    )
    incident_id: str = Field(description="ID of the incident being analyzed")
    paths: list[CausalPath] = Field(
        description="Ranked list of causal paths (most likely first)"
    )
    algorithm_version: str = Field(
        default="BRE-RCA-v1", description="Version of the RCA algorithm used"
    )
    causal_window: tuple[datetime, datetime] = Field(
        description="Time window analyzed for causal relationships"
    )
    dependency_graph_version: str = Field(
        description="Version of dependency graph used"
    )
    run_id: str = Field(description="ID of the analysis run that generated this chain")

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v: list[CausalPath]) -> list[CausalPath]:
        """Ensure at least one causal path exists."""
        if not v:
            raise ValueError("At least one causal path must be provided")
        return v

    @field_validator("paths")
    @classmethod
    def validate_path_rankings(cls, v: list[CausalPath]) -> list[CausalPath]:
        """Ensure path rankings are sequential starting from 1."""
        if v:
            expected_ranks = list(range(1, len(v) + 1))
            actual_ranks = [path.rank for path in v]
            if actual_ranks != expected_ranks:
                raise ValueError("Path rankings must be sequential starting from 1")
        return v

    @field_validator("causal_window")
    @classmethod
    def validate_causal_window(cls, v: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
        """Ensure causal window is valid."""
        if v[1] <= v[0]:
            raise ValueError("Causal window end must be after start")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "chain_id": "chain_123e4567-e89b-12d3-a456-426614174000",
                "incident_id": "inc_123e4567",
                "paths": [
                    {
                        "rank": 1,
                        "overall_score": 0.92,
                        "nodes": [
                            {
                                "metric_name": "product_defect_rate",
                                "contribution_score": 0.95,
                                "anomaly_magnitude": 12.5,
                                "temporal_precedence": 0.98,
                                "graph_proximity": 0.85,
                                "data_quality_weight": 0.97,
                                "metric_value": 0.08,
                                "metric_baseline": 0.01,
                                "metric_zscore": 12.5,
                                "anomaly_window": (
                                    "2026-02-10T10:00:00Z",
                                    "2026-02-10T12:00:00Z",
                                ),
                                "evidence_clusters": [],
                            }
                        ],
                    }
                ],
                "algorithm_version": "BRE-RCA-v1",
                "causal_window": ("2026-02-10T08:00:00Z", "2026-02-10T14:00:00Z"),
                "dependency_graph_version": "dep_graph_v2",
                "run_id": "run_20260210_150000",
            }
        }
