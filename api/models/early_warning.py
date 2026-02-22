"""
Early Warning models for forward prediction.

Agent: backend-developer
Pydantic models for pre-incident risk warnings: trend detection,
forward causal chain, and prevention narrative.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EarlyWarning(BaseModel):
    """
    A pre-incident risk warning from trend detection + forward chain analysis.

    Represents a metric trending toward an incident threshold, with
    the causal path and prevention recommendation.

    Attributes:
        warning_id: Unique identifier
        metric: Metric that is degrading
        current_value: Current metric value
        baseline: Expected baseline value
        slope: Trend slope (change per day)
        projected_value: Value projected in 3-5 days
        threshold: Incident trigger threshold
        threshold_below: True if incident triggers when metric goes below threshold
        forward_chain: Causal path metric1 → metric2 → ... → incident_type
        incident_type: Predicted incident type at end of chain
        prevention_steps: Natural language prevention recommendation
        severity: Warning severity (low, medium, high, critical)
        created_at: When warning was generated
    """

    warning_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique warning identifier",
    )
    metric: str = Field(description="Metric that is degrading")
    current_value: float = Field(description="Current metric value")
    baseline: Optional[float] = Field(default=None, description="Expected baseline")
    slope: float = Field(description="Trend slope (change per day)")
    projected_value: float = Field(description="Projected value in projection_days")
    projected_ci_lower: Optional[float] = Field(default=None, description="95% CI lower bound (data-scientist)")
    projected_ci_upper: Optional[float] = Field(default=None, description="95% CI upper bound (data-scientist)")
    p_value: Optional[float] = Field(default=None, description="Regression p-value for slope significance")
    projection_days: int = Field(default=5, description="Days ahead for projection")
    threshold: float = Field(description="Incident trigger threshold")
    threshold_below: bool = Field(
        default=False,
        description="True if incident triggers when metric goes below threshold",
    )
    forward_chain: list[str] = Field(
        default_factory=list,
        description="Causal path: metric1 → metric2 → ... → incident_type",
    )
    incident_type: Optional[str] = Field(
        default=None,
        description="Predicted incident type at end of chain",
    )
    prevention_steps: str = Field(
        default="",
        description="Natural language prevention recommendation",
    )
    severity: str = Field(default="medium", description="low, medium, high, critical")
    days_to_threshold: Optional[float] = Field(
        default=None,
        description="Estimated days until metric crosses threshold",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When warning was generated",
    )
