"""
Blast radius impact models for the Business Reliability Engine.

This module defines impact assessment structures quantifying the business
scope and magnitude of detected incidents across customers, revenue, and
operational metrics.
"""

from pydantic import BaseModel, Field, field_validator

from .enums import BlastRadiusSeverity


class BlastRadius(BaseModel):
    """
    Quantified business impact assessment for an incident.

    Measures the scope and magnitude of an incident's impact across multiple
    dimensions: customer reach, order volume, revenue exposure, and downstream
    cascade effects. Blast radius informs incident prioritization and response
    resource allocation.

    The severity classification combines quantitative thresholds with qualitative
    assessment to categorize impact from CONTAINED to CATASTROPHIC.

    Attributes:
        incident_id: ID of the incident being assessed
        customers_affected: Count of unique customers impacted
        orders_affected: Count of orders involved in the incident
        products_affected: Count of unique products involved
        vendors_involved: Count of unique vendors/suppliers involved
        estimated_revenue_exposure: Estimated revenue at risk (USD)
        estimated_refund_exposure: Estimated refund liability (USD)
        estimated_churn_exposure: Estimated number of at-risk customers
        downstream_incidents_triggered: IDs of incidents triggered by this one
        blast_radius_severity: Categorical severity classification
        narrative: Human-readable impact summary and business context
    """

    incident_id: str = Field(description="ID of the incident being assessed")
    customers_affected: int = Field(
        description="Count of unique customers impacted", ge=0
    )
    orders_affected: int = Field(description="Count of orders involved", ge=0)
    products_affected: int = Field(description="Count of unique products involved", ge=0)
    vendors_involved: int = Field(
        description="Count of unique vendors/suppliers involved", ge=0
    )
    estimated_revenue_exposure: float = Field(
        description="Estimated revenue at risk (USD)", ge=0.0
    )
    estimated_refund_exposure: float = Field(
        description="Estimated refund liability (USD)", ge=0.0
    )
    estimated_churn_exposure: int = Field(
        description="Estimated number of at-risk customers", ge=0
    )
    downstream_incidents_triggered: list[str] = Field(
        default_factory=list,
        description="IDs of incidents triggered by this one in cascade",
    )
    blast_radius_severity: BlastRadiusSeverity = Field(
        description="Categorical severity classification"
    )
    narrative: str = Field(
        description="Human-readable impact summary and business context"
    )

    @field_validator("customers_affected", "orders_affected", "products_affected", "vendors_involved", "estimated_churn_exposure")
    @classmethod
    def validate_counts_non_negative(cls, v: int) -> int:
        """Ensure count fields are non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v

    @field_validator("estimated_revenue_exposure", "estimated_refund_exposure")
    @classmethod
    def validate_monetary_values(cls, v: float) -> float:
        """Ensure monetary values are non-negative and reasonable."""
        if v < 0.0:
            raise ValueError("Monetary value must be non-negative")
        if v > 1e12:
            raise ValueError("Monetary value exceeds maximum allowed")
        return round(v, 2)

    @field_validator("narrative")
    @classmethod
    def validate_narrative_not_empty(cls, v: str) -> str:
        """Ensure narrative provides meaningful content."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Narrative must be at least 10 characters")
        return v.strip()

    @field_validator("blast_radius_severity")
    @classmethod
    def validate_severity_consistency(cls, v: BlastRadiusSeverity, info) -> BlastRadiusSeverity:
        """Validate severity is consistent with impact metrics."""
        # This is a business logic validation that could be enhanced
        # with specific threshold checks
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "incident_id": "inc_123e4567-e89b-12d3-a456-426614174000",
                "customers_affected": 1250,
                "orders_affected": 2100,
                "products_affected": 45,
                "vendors_involved": 3,
                "estimated_revenue_exposure": 125000.00,
                "estimated_refund_exposure": 45000.00,
                "estimated_churn_exposure": 75,
                "downstream_incidents_triggered": [
                    "inc_downstream_001",
                    "inc_downstream_002",
                ],
                "blast_radius_severity": "severe",
                "narrative": (
                    "SEVERE impact: 1,250 customers affected with $125K revenue exposure. "
                    "Refund spike originated from quality issues in Product SKU-945, affecting "
                    "2,100 orders. Triggered downstream support surge and potential churn of "
                    "75 high-value customers. Immediate attention required to prevent cascade."
                ),
            }
        }
