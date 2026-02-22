"""
Enumeration types for the Business Reliability Engine.

This module defines all enum types used across the system for type safety
and consistent validation. All enums inherit from str to ensure JSON
serialization compatibility.
"""

from enum import Enum


class EventType(str, Enum):
    """
    Canonical event types representing business activities across the organization.

    These event types are normalized from various source systems (QBO, Shopify, etc.)
    into a unified taxonomy for consistent analysis and incident detection.
    """

    # Invoice lifecycle events
    INVOICE_ISSUED = "invoice_issued"
    INVOICE_PAID = "invoice_paid"
    INVOICE_OVERDUE = "invoice_overdue"

    # Payment events
    PAYMENT_RECEIVED = "payment_received"

    # Expense events
    EXPENSE_POSTED = "expense_posted"
    EXPENSE_PAID = "expense_paid"

    # Credit and refund events
    REFUND_ISSUED = "refund_issued"
    CREDIT_MEMO_ISSUED = "credit_memo_issued"

    # Order lifecycle events
    ORDER_PLACED = "order_placed"
    ORDER_DELIVERED = "order_delivered"
    ORDER_LATE = "order_late"

    # Procurement events
    PURCHASE_ORDER_PLACED = "purchase_order_placed"

    # Inventory events
    INVENTORY_LOW = "inventory_low"

    # Shipping events
    SHIPMENT_DELAYED = "shipment_delayed"

    # Customer lifecycle events
    CUSTOMER_CREATED = "customer_created"
    CUSTOMER_UPDATED = "customer_updated"
    CUSTOMER_CHURNED = "customer_churned"

    # Support events
    SUPPORT_TICKET_OPENED = "support_ticket_opened"
    SUPPORT_TICKET_CLOSED = "support_ticket_closed"

    # Feedback events
    REVIEW_SUBMITTED = "review_submitted"


class EntityType(str, Enum):
    """
    Entity types that can be referenced in events and incidents.

    These represent the primary business objects tracked by the system.
    """

    CUSTOMER = "customer"
    ORDER = "order"
    INVOICE = "invoice"
    TICKET = "ticket"
    PRODUCT = "product"
    VENDOR = "vendor"
    PAYMENT = "payment"
    EXPENSE = "expense"


class IncidentType(str, Enum):
    """
    Types of business reliability incidents detected by the system.

    Each incident type represents a distinct pattern of operational degradation
    with specific detection algorithms and remediation approaches.
    """

    REFUND_SPIKE = "refund_spike"
    FULFILLMENT_SLA_DEGRADATION = "fulfillment_sla_degradation"
    SUPPORT_LOAD_SURGE = "support_load_surge"
    CHURN_ACCELERATION = "churn_acceleration"
    MARGIN_COMPRESSION = "margin_compression"
    LIQUIDITY_CRUNCH_RISK = "liquidity_crunch_risk"
    SUPPLIER_DEPENDENCY_FAILURE = "supplier_dependency_failure"
    CUSTOMER_SATISFACTION_REGRESSION = "customer_satisfaction_regression"


class Severity(str, Enum):
    """
    Severity levels for incidents, monitors, and alerts.

    Severity determines escalation paths, notification urgency, and
    required response SLAs.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(str, Enum):
    """
    Confidence levels for detections and predictions.

    Represents the statistical and algorithmic certainty of an incident
    detection or causal inference.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class IncidentStatus(str, Enum):
    """
    Lifecycle status for incidents.

    Tracks the operational state of an incident from initial detection
    through acknowledgment to resolution.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class MonitorStatus(str, Enum):
    """
    Status values for monitor rules.

    Indicates the current operational state of a monitor and whether
    it is actively checking conditions.
    """

    ACTIVE = "active"
    TRIGGERED = "triggered"
    DISABLED = "disabled"


class AlertStatus(str, Enum):
    """
    Status values for monitor alerts.

    Tracks the lifecycle of an alert from generation through human
    acknowledgment or dismissal.
    """

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"


class DetectionMethod(str, Enum):
    """
    Anomaly detection algorithms used for incident detection.

    Each method represents a specific statistical or machine learning
    approach to identifying unusual patterns in business metrics.
    """

    MAD_ZSCORE = "mad_zscore"
    ISOLATION_FOREST = "isolation_forest"
    CHANGEPOINT = "changepoint"


class BlastRadiusSeverity(str, Enum):
    """
    Severity classification for incident blast radius impact.

    Represents the scope and magnitude of business impact across
    customers, revenue, and operations.
    """

    CONTAINED = "contained"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"
