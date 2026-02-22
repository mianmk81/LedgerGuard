"""
Simulation and comparison models for the Business Reliability Engine.

This module defines what-if scenario analysis and incident comparison
structures for predictive testing and pattern analysis.
"""

from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class WhatIfScenario(BaseModel):
    """
    A what-if scenario analysis result.

    Represents a simulated perturbation of business metrics and the predicted
    incidents, cascades, and downstream effects. What-if scenarios enable
    proactive capacity planning and risk assessment.

    Example use cases:
    - "What if order volume doubles next quarter?"
    - "What if vendor delivery SLA degrades by 20%?"
    - "What if customer support capacity is reduced by 15%?"

    Attributes:
        scenario_id: Unique identifier for this scenario
        perturbations: List of metric perturbations applied (e.g., [{"metric": "order_volume", "change": "+100%"}])
        simulated_metrics: Predicted metric values under the scenario
        triggered_incidents: List of incident types predicted to trigger
        triggered_cascades: List of cascade patterns predicted to occur
        narrative: Human-readable summary of scenario and predictions
    """

    scenario_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this scenario",
    )
    perturbations: list[dict] = Field(
        description="List of metric perturbations applied in the scenario"
    )
    simulated_metrics: dict = Field(
        description="Predicted metric values under the scenario"
    )
    triggered_incidents: list[str] = Field(
        description="List of incident types predicted to trigger"
    )
    triggered_cascades: list[str] = Field(
        description="List of cascade patterns predicted to occur"
    )
    narrative: str = Field(
        description="Human-readable summary of scenario and predictions"
    )
    ml_insights: dict = Field(
        default_factory=dict,
        description="Additional predictions from ML models (churn risk, health score impact)",
    )
    models_used: list[str] = Field(
        default_factory=list,
        description="List of ML models used in this simulation",
    )

    @field_validator("perturbations")
    @classmethod
    def validate_perturbations_not_empty(cls, v: list[dict]) -> list[dict]:
        """Ensure at least one perturbation is specified."""
        if not v:
            raise ValueError("At least one perturbation must be specified")
        return v

    @field_validator("perturbations")
    @classmethod
    def validate_perturbation_structure(cls, v: list[dict]) -> list[dict]:
        """Ensure each perturbation has required fields."""
        for perturbation in v:
            if "metric" not in perturbation or "change" not in perturbation:
                raise ValueError(
                    "Each perturbation must have 'metric' and 'change' fields"
                )
        return v

    @field_validator("narrative")
    @classmethod
    def validate_narrative_not_empty(cls, v: str) -> str:
        """Ensure narrative is meaningful."""
        if not v or len(v.strip()) < 20:
            raise ValueError("Narrative must be at least 20 characters")
        return v.strip()

    @field_validator("simulated_metrics")
    @classmethod
    def validate_simulated_metrics_not_empty(cls, v: dict) -> dict:
        """Ensure simulated metrics are provided."""
        if not v:
            raise ValueError("Simulated metrics cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "scenario_id": "scenario_123e4567-e89b-12d3-a456-426614174000",
                "perturbations": [
                    {"metric": "order_volume", "change": "+50%"},
                    {"metric": "support_capacity", "change": "-10%"},
                ],
                "simulated_metrics": {
                    "order_volume_daily": 1500,
                    "support_tickets_daily": 450,
                    "avg_response_time_hours": 8.5,
                    "predicted_refund_rate": 0.045,
                },
                "triggered_incidents": [
                    "support_load_surge",
                    "fulfillment_sla_degradation",
                ],
                "triggered_cascades": ["support_surge -> satisfaction_regression"],
                "narrative": (
                    "Simulated 50% order volume increase with 10% support capacity "
                    "reduction predicts HIGH severity support load surge (8.5hr "
                    "response time vs 4.0hr baseline) and MEDIUM fulfillment "
                    "degradation. Cascade risk: support surge likely triggers "
                    "customer satisfaction regression within 48 hours."
                ),
            }
        }


class IncidentComparison(BaseModel):
    """
    A comparative analysis between two incidents.

    Identifies similarities and differences between incidents to reveal
    patterns, common root causes, and systematic vulnerabilities. Comparison
    analysis enables learning from historical incidents and improvement of
    detection algorithms.

    Attributes:
        comparison_id: Unique identifier for this comparison
        incident_a_id: ID of the first incident being compared
        incident_b_id: ID of the second incident being compared
        incident_type: Type classification (same for both or "cross-type")
        shared_root_causes: Root causes common to both incidents
        unique_to_a: Root causes unique to incident A
        unique_to_b: Root causes unique to incident B
        severity_comparison: Comparison of severity metrics between incidents
        blast_radius_comparison: Comparison of impact metrics between incidents
        narrative: Human-readable summary of comparison insights
    """

    comparison_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this comparison",
    )
    incident_a_id: str = Field(description="ID of the first incident being compared")
    incident_b_id: str = Field(description="ID of the second incident being compared")
    incident_type: str = Field(
        description="Type classification (same for both or 'cross-type')"
    )
    shared_root_causes: list[str] = Field(
        description="Root causes common to both incidents"
    )
    unique_to_a: list[str] = Field(description="Root causes unique to incident A")
    unique_to_b: list[str] = Field(description="Root causes unique to incident B")
    severity_comparison: dict = Field(
        description="Comparison of severity metrics between incidents"
    )
    blast_radius_comparison: dict = Field(
        description="Comparison of impact metrics between incidents"
    )
    narrative: str = Field(
        description="Human-readable summary of comparison insights"
    )

    @field_validator("incident_a_id", "incident_b_id")
    @classmethod
    def validate_incident_ids_not_empty(cls, v: str) -> str:
        """Ensure incident IDs are not empty."""
        if not v or len(v.strip()) < 1:
            raise ValueError("Incident ID cannot be empty")
        return v.strip()

    @field_validator("incident_b_id")
    @classmethod
    def validate_different_incidents(cls, v: str, info) -> str:
        """Ensure incidents A and B are different."""
        if "incident_a_id" in info.data and v == info.data["incident_a_id"]:
            raise ValueError("Cannot compare an incident to itself")
        return v

    @field_validator("narrative")
    @classmethod
    def validate_narrative_not_empty(cls, v: str) -> str:
        """Ensure narrative is meaningful."""
        if not v or len(v.strip()) < 20:
            raise ValueError("Narrative must be at least 20 characters")
        return v.strip()

    @field_validator("incident_type")
    @classmethod
    def validate_incident_type_not_empty(cls, v: str) -> str:
        """Ensure incident type is not empty."""
        if not v or len(v.strip()) < 3:
            raise ValueError("Incident type must be at least 3 characters")
        return v.strip()

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "comparison_id": "comp_123e4567-e89b-12d3-a456-426614174000",
                "incident_a_id": "inc_feb10_refund_spike",
                "incident_b_id": "inc_jan15_refund_spike",
                "incident_type": "refund_spike",
                "shared_root_causes": [
                    "product_quality_defect",
                    "vendor_quality_control_gap",
                ],
                "unique_to_a": ["shipping_damage_during_transit"],
                "unique_to_b": ["incorrect_product_specifications"],
                "severity_comparison": {
                    "incident_a_severity": "high",
                    "incident_b_severity": "medium",
                    "incident_a_zscore": 8.5,
                    "incident_b_zscore": 4.2,
                },
                "blast_radius_comparison": {
                    "incident_a_customers": 1250,
                    "incident_b_customers": 320,
                    "incident_a_revenue_exposure": 125000.00,
                    "incident_b_revenue_exposure": 28000.00,
                },
                "narrative": (
                    "Both incidents stem from vendor quality control gaps, but Feb 10 "
                    "incident was 4x more severe due to shipping damage amplifying the "
                    "product defect. Jan 15 was contained to incorrect specs. Pattern: "
                    "Vendor-42 requires enhanced quality processes at both manufacturing "
                    "and shipping stages. Recommend: dual-stage inspection protocol."
                ),
            }
        }
