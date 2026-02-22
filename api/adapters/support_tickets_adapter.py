"""
Support Tickets Dataset Adapter.

This adapter transforms customer support ticket datasets into canonical events
for analyzing support load patterns, resolution times, and customer satisfaction.

Handles various ticket dataset formats with flexible column name matching.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import structlog

from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue

from .base_adapter import BaseAdapter

logger = structlog.get_logger()


class SupportTicketAdapter(BaseAdapter):
    """
    Adapts customer support ticket data into canonical events.

    Transforms ticket lifecycle events including creation, status changes,
    and closure with resolution metrics. Flexible column name matching
    handles various dataset formats.

    Supported event types:
    - SUPPORT_TICKET_OPENED: When ticket is created
    - SUPPORT_TICKET_CLOSED: When ticket is resolved/closed
    """

    # Column name mappings for flexible dataset handling
    COLUMN_MAPPINGS = {
        "ticket_id": ["ticket_id", "id", "ticket_number", "case_id", "case_number"],
        "customer_id": ["customer_id", "user_id", "client_id", "requester_id"],
        "created_at": [
            "created_at",
            "created_date",
            "creation_date",
            "opened_at",
            "submitted_at",
            "timestamp",
        ],
        "closed_at": [
            "closed_at",
            "closed_date",
            "resolved_at",
            "resolution_date",
            "completed_at",
        ],
        "status": ["status", "state", "ticket_status", "case_status"],
        "category": ["category", "type", "ticket_type", "issue_type", "department"],
        "priority": ["priority", "severity", "urgency"],
        "channel": ["channel", "source", "origin", "contact_method"],
        "subject": ["subject", "title", "summary", "description"],
        "satisfaction_score": [
            "satisfaction_score",
            "csat",
            "rating",
            "customer_rating",
            "feedback_score",
        ],
    }

    def __init__(self):
        """Initialize Support Tickets adapter."""
        super().__init__(source_name="support_tickets")

    def _find_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        """
        Find the actual column name in DataFrame using flexible matching.

        Args:
            df: DataFrame to search
            field: Field type to find (from COLUMN_MAPPINGS)

        Returns:
            Actual column name if found, None otherwise
        """
        possible_names = self.COLUMN_MAPPINGS.get(field, [])
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        return None

    def ingest(
        self, tickets_df: pd.DataFrame, **kwargs
    ) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Transform support ticket dataset into canonical events.

        Args:
            tickets_df: DataFrame containing support ticket data
            **kwargs: Additional configuration options

        Returns:
            Tuple of (canonical events, data quality report)

        Expected columns (flexible matching):
        - ticket_id/id: Unique ticket identifier
        - customer_id/user_id: Customer identifier
        - created_at/created_date: Ticket creation timestamp
        - closed_at/resolved_at: Ticket closure timestamp (optional)
        - status: Current ticket status
        - category/type: Ticket category or type
        - priority/severity: Priority level
        - channel/source: Contact channel (email, phone, chat)
        - subject/title: Ticket subject
        - satisfaction_score/rating: Customer satisfaction score (optional)
        """
        self.logger.info(
            "support_tickets_ingestion_started", tickets_count=len(tickets_df)
        )

        events = []
        quality_issues = []
        rejected_count = 0

        # Track quality metrics
        missing_ticket_ids = 0
        missing_created_timestamps = 0
        missing_customer_ids = 0
        invalid_resolution_times = 0
        missing_satisfaction_scores = 0

        # Map column names
        col_ticket_id = self._find_column(tickets_df, "ticket_id")
        col_customer_id = self._find_column(tickets_df, "customer_id")
        col_created_at = self._find_column(tickets_df, "created_at")
        col_closed_at = self._find_column(tickets_df, "closed_at")
        col_status = self._find_column(tickets_df, "status")
        col_category = self._find_column(tickets_df, "category")
        col_priority = self._find_column(tickets_df, "priority")
        col_channel = self._find_column(tickets_df, "channel")
        col_subject = self._find_column(tickets_df, "subject")
        col_satisfaction = self._find_column(tickets_df, "satisfaction_score")

        if not col_ticket_id:
            raise ValueError(
                "Cannot find ticket_id column. Expected one of: "
                + ", ".join(self.COLUMN_MAPPINGS["ticket_id"])
            )

        if not col_created_at:
            raise ValueError(
                "Cannot find created_at column. Expected one of: "
                + ", ".join(self.COLUMN_MAPPINGS["created_at"])
            )

        # Process each ticket
        for idx, row in tickets_df.iterrows():
            ticket_id = self._safe_str(row.get(col_ticket_id))
            customer_id = self._safe_str(row.get(col_customer_id)) if col_customer_id else None

            if not ticket_id:
                missing_ticket_ids += 1
                rejected_count += 1
                continue

            # Extract timestamps
            created_at = self._safe_datetime(row.get(col_created_at))
            closed_at = (
                self._safe_datetime(row.get(col_closed_at)) if col_closed_at else None
            )

            if not created_at:
                missing_created_timestamps += 1
                rejected_count += 1
                continue

            # Track missing customer IDs
            if not customer_id:
                missing_customer_ids += 1

            # Extract attributes
            status = self._safe_str(row.get(col_status)) if col_status else "unknown"
            category = self._safe_str(row.get(col_category)) if col_category else None
            priority = self._safe_str(row.get(col_priority)) if col_priority else None
            channel = self._safe_str(row.get(col_channel)) if col_channel else None
            subject = self._safe_str(row.get(col_subject)) if col_subject else None
            satisfaction_score = (
                self._safe_float(row.get(col_satisfaction))
                if col_satisfaction
                else None
            )

            # Build related entities
            related_entities = {}
            if customer_id:
                related_entities["customer"] = self._create_entity_id(
                    "customer", customer_id
                )

            # EVENT 1: SUPPORT_TICKET_OPENED
            opened_flags = []
            if not customer_id:
                opened_flags.append("missing_customer_id")
            if not category:
                opened_flags.append("missing_category")
            if not priority:
                opened_flags.append("missing_priority")

            opened_event = CanonicalEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.SUPPORT_TICKET_OPENED,
                event_time=created_at,
                source=self.source_name,
                source_entity_id=ticket_id,
                entity_type=EntityType.TICKET,
                entity_id=self._create_entity_id("ticket", ticket_id),
                related_entity_ids=related_entities,
                amount=None,
                currency="USD",
                attributes={
                    "ticket_id": ticket_id,
                    "customer_id": customer_id,
                    "category": category,
                    "priority": priority,
                    "channel": channel,
                    "subject": subject,
                    "status": status,
                },
                data_quality_flags=opened_flags,
                schema_version="canonical_v1",
            )
            events.append(opened_event)

            # EVENT 2: SUPPORT_TICKET_CLOSED (if ticket is closed)
            if closed_at:
                closed_flags = []

                # Calculate resolution time
                resolution_time_hours = None
                if closed_at > created_at:
                    resolution_time_seconds = (closed_at - created_at).total_seconds()
                    resolution_time_hours = round(resolution_time_seconds / 3600, 2)
                else:
                    invalid_resolution_times += 1
                    closed_flags.append("invalid_resolution_time")

                # Check satisfaction score
                if satisfaction_score is None and col_satisfaction:
                    missing_satisfaction_scores += 1
                    closed_flags.append("missing_satisfaction_score")

                closed_event = CanonicalEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.SUPPORT_TICKET_CLOSED,
                    event_time=closed_at,
                    source=self.source_name,
                    source_entity_id=ticket_id,
                    entity_type=EntityType.TICKET,
                    entity_id=self._create_entity_id("ticket", ticket_id),
                    related_entity_ids=related_entities,
                    amount=None,
                    currency="USD",
                    attributes={
                        "ticket_id": ticket_id,
                        "resolution_time_hours": resolution_time_hours,
                        "satisfaction_score": satisfaction_score,
                        "category": category,
                        "priority": priority,
                        "final_status": status,
                    },
                    data_quality_flags=closed_flags,
                    schema_version="canonical_v1",
                )
                events.append(closed_event)

        # Compile quality issues
        if missing_ticket_ids > 0:
            quality_issues.append(
                QualityIssue(
                    field="ticket_id",
                    issue_type="missing",
                    count=missing_ticket_ids,
                    description=f"Missing ticket ID in {missing_ticket_ids} records",
                )
            )

        if missing_created_timestamps > 0:
            quality_issues.append(
                QualityIssue(
                    field="created_at",
                    issue_type="missing",
                    count=missing_created_timestamps,
                    description=f"Missing creation timestamp in {missing_created_timestamps} records",
                )
            )

        if missing_customer_ids > 0:
            quality_issues.append(
                QualityIssue(
                    field="customer_id",
                    issue_type="missing",
                    count=missing_customer_ids,
                    description=f"Missing customer ID in {missing_customer_ids} ticket records",
                )
            )

        if invalid_resolution_times > 0:
            quality_issues.append(
                QualityIssue(
                    field="resolution_time",
                    issue_type="invalid_value",
                    count=invalid_resolution_times,
                    description=f"Invalid resolution time (closed before created) in {invalid_resolution_times} records",
                )
            )

        if missing_satisfaction_scores > 0:
            quality_issues.append(
                QualityIssue(
                    field="satisfaction_score",
                    issue_type="missing",
                    count=missing_satisfaction_scores,
                    description=f"Missing satisfaction score in {missing_satisfaction_scores} closed ticket records",
                )
            )

        # Generate quality report
        total_records = len(tickets_df)
        quality_report = self._compute_quality_scores(
            events=events,
            total_records=total_records,
            rejected_records=rejected_count,
            quality_issues=quality_issues,
        )

        self.logger.info(
            "support_tickets_ingestion_completed",
            total_events=len(events),
            total_records=total_records,
            rejected_records=rejected_count,
            quality_score=quality_report.overall_quality_score,
        )

        return events, quality_report
