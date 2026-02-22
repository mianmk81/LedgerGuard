"""
Logistics and Supply Chain Dataset Adapter.

This adapter transforms logistics and supply chain datasets into canonical
events for analyzing delivery performance, supplier reliability, and
procurement patterns.

Handles shipment tracking, delivery delays, and purchase order data.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import structlog

from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue

from .base_adapter import BaseAdapter

logger = structlog.get_logger()


class LogisticsAdapter(BaseAdapter):
    """
    Adapts logistics and supply chain data into canonical events.

    Processes shipment tracking, delivery performance, and purchase order
    information to enable supply chain reliability analysis.

    Supported event types:
    - SHIPMENT_DELAYED: When actual delivery exceeds expected delivery
    - PURCHASE_ORDER_PLACED: When purchase order is created
    """

    # Column name mappings for flexible dataset handling
    COLUMN_MAPPINGS = {
        "shipment_id": [
            "shipment_id",
            "tracking_id",
            "tracking_number",
            "shipment_number",
            "delivery_id",
        ],
        "order_id": ["order_id", "purchase_order", "po_number", "order_number"],
        "carrier": ["carrier", "shipping_carrier", "courier", "transporter"],
        "route": ["route", "shipping_route", "origin_destination", "lane"],
        "origin": ["origin", "source", "pickup_location", "from_location"],
        "destination": [
            "destination",
            "delivery_location",
            "to_location",
            "destination_city",
        ],
        "expected_delivery": [
            "expected_delivery",
            "scheduled_delivery",
            "planned_delivery",
            "eta",
            "estimated_delivery_date",
        ],
        "actual_delivery": [
            "actual_delivery",
            "delivered_date",
            "actual_delivery_date",
            "delivery_date",
        ],
        "shipment_date": [
            "shipment_date",
            "shipped_date",
            "dispatch_date",
            "pickup_date",
        ],
        "vendor_id": ["vendor_id", "supplier_id", "vendor", "supplier"],
        "amount": ["amount", "total_amount", "order_value", "po_value"],
        "line_items": ["line_items", "items", "products", "item_count"],
    }

    def __init__(self):
        """Initialize Logistics adapter."""
        super().__init__(source_name="logistics")

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
        self, shipments_df: pd.DataFrame, **kwargs
    ) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Transform logistics dataset into canonical events.

        Args:
            shipments_df: DataFrame containing shipment and/or purchase order data
            **kwargs: Additional configuration options

        Returns:
            Tuple of (canonical events, data quality report)

        Expected columns (flexible matching):
        - shipment_id/tracking_id: Unique shipment identifier
        - order_id/purchase_order: Associated order/PO number
        - carrier/shipping_carrier: Carrier name
        - route/shipping_route: Shipping route description
        - origin/source: Origin location
        - destination/delivery_location: Destination location
        - expected_delivery/eta: Expected delivery timestamp
        - actual_delivery/delivered_date: Actual delivery timestamp
        - shipment_date/shipped_date: Shipment dispatch timestamp
        - vendor_id/supplier_id: Vendor/supplier identifier
        - amount/order_value: Order monetary value
        - line_items/items: Number or description of line items

        The adapter detects:
        1. SHIPMENT_DELAYED events when actual > expected delivery
        2. PURCHASE_ORDER_PLACED events when PO/order data is present
        """
        self.logger.info(
            "logistics_ingestion_started", shipments_count=len(shipments_df)
        )

        events = []
        quality_issues = []
        rejected_count = 0

        # Track quality metrics
        missing_ids = 0
        missing_timestamps = 0
        missing_locations = 0
        invalid_delays = 0
        missing_vendors = 0

        # Map column names
        col_shipment_id = self._find_column(shipments_df, "shipment_id")
        col_order_id = self._find_column(shipments_df, "order_id")
        col_carrier = self._find_column(shipments_df, "carrier")
        col_route = self._find_column(shipments_df, "route")
        col_origin = self._find_column(shipments_df, "origin")
        col_destination = self._find_column(shipments_df, "destination")
        col_expected = self._find_column(shipments_df, "expected_delivery")
        col_actual = self._find_column(shipments_df, "actual_delivery")
        col_shipment_date = self._find_column(shipments_df, "shipment_date")
        col_vendor_id = self._find_column(shipments_df, "vendor_id")
        col_amount = self._find_column(shipments_df, "amount")
        col_line_items = self._find_column(shipments_df, "line_items")

        # Determine primary identifier (shipment_id or order_id)
        if not col_shipment_id and not col_order_id:
            raise ValueError(
                "Cannot find shipment_id or order_id column. Expected one of: "
                + ", ".join(
                    self.COLUMN_MAPPINGS["shipment_id"]
                    + self.COLUMN_MAPPINGS["order_id"]
                )
            )

        # Process each record
        for idx, row in shipments_df.iterrows():
            shipment_id = (
                self._safe_str(row.get(col_shipment_id)) if col_shipment_id else None
            )
            order_id = self._safe_str(row.get(col_order_id)) if col_order_id else None

            # Use shipment_id as primary, fallback to order_id
            record_id = shipment_id or order_id
            if not record_id:
                missing_ids += 1
                rejected_count += 1
                continue

            # Extract timestamps
            expected_delivery = (
                self._safe_datetime(row.get(col_expected)) if col_expected else None
            )
            actual_delivery = (
                self._safe_datetime(row.get(col_actual)) if col_actual else None
            )
            shipment_date = (
                self._safe_datetime(row.get(col_shipment_date))
                if col_shipment_date
                else None
            )

            # Extract attributes
            carrier = self._safe_str(row.get(col_carrier)) if col_carrier else None
            route = self._safe_str(row.get(col_route)) if col_route else None
            origin = self._safe_str(row.get(col_origin)) if col_origin else None
            destination = (
                self._safe_str(row.get(col_destination)) if col_destination else None
            )
            vendor_id = (
                self._safe_str(row.get(col_vendor_id)) if col_vendor_id else None
            )
            amount = self._safe_float(row.get(col_amount)) if col_amount else None
            line_items = (
                self._safe_str(row.get(col_line_items)) if col_line_items else None
            )

            # Build related entities
            related_entities = {}
            if order_id:
                related_entities["order"] = self._create_entity_id("order", order_id)
            if vendor_id:
                related_entities["vendor"] = self._create_entity_id("vendor", vendor_id)

            # Track missing critical data
            if not origin or not destination:
                missing_locations += 1
            if not vendor_id and col_vendor_id:
                missing_vendors += 1

            # EVENT 1: SHIPMENT_DELAYED (if actual > expected delivery)
            if expected_delivery and actual_delivery:
                if actual_delivery > expected_delivery:
                    days_delayed = (
                        actual_delivery - expected_delivery
                    ).total_seconds() / 86400

                    event_flags = []
                    if not carrier:
                        event_flags.append("missing_carrier")
                    if not origin or not destination:
                        event_flags.append("missing_location_data")

                    delayed_event = CanonicalEvent(
                        event_id=self._generate_event_id(),
                        event_type=EventType.SHIPMENT_DELAYED,
                        event_time=actual_delivery,
                        source=self.source_name,
                        source_entity_id=record_id,
                        entity_type=EntityType.ORDER,
                        entity_id=self._create_entity_id("shipment", record_id),
                        related_entity_ids=related_entities,
                        amount=amount,
                        currency="USD",
                        attributes={
                            "shipment_id": shipment_id,
                            "order_id": order_id,
                            "carrier": carrier,
                            "route": route,
                            "days_delayed": round(days_delayed, 2),
                            "origin": origin,
                            "destination": destination,
                            "expected_delivery": expected_delivery.isoformat(),
                            "actual_delivery": actual_delivery.isoformat(),
                        },
                        data_quality_flags=event_flags,
                        schema_version="canonical_v1",
                    )
                    events.append(delayed_event)
                elif actual_delivery < expected_delivery:
                    # Actual delivery before expected is unusual but valid
                    # Could create early delivery event in the future
                    pass
            elif expected_delivery and not actual_delivery:
                # Shipment not yet delivered - skip for now
                pass
            elif expected_delivery or actual_delivery:
                # Only one timestamp available
                missing_timestamps += 1

            # EVENT 2: PURCHASE_ORDER_PLACED (if we have order/PO data)
            # Only create if we have shipment_date or can use expected_delivery as proxy
            if order_id and vendor_id:
                po_date = shipment_date or expected_delivery

                if po_date:
                    event_flags = []
                    if not amount:
                        event_flags.append("missing_amount")
                    if not line_items:
                        event_flags.append("missing_line_items")

                    po_event = CanonicalEvent(
                        event_id=self._generate_event_id(),
                        event_type=EventType.PURCHASE_ORDER_PLACED,
                        event_time=po_date,
                        source=self.source_name,
                        source_entity_id=order_id,
                        entity_type=EntityType.ORDER,
                        entity_id=self._create_entity_id("order", order_id),
                        related_entity_ids=related_entities,
                        amount=amount,
                        currency="USD",
                        attributes={
                            "order_id": order_id,
                            "vendor_id": vendor_id,
                            "line_items": line_items,
                            "expected_date": (
                                expected_delivery.isoformat()
                                if expected_delivery
                                else None
                            ),
                            "carrier": carrier,
                            "destination": destination,
                        },
                        data_quality_flags=event_flags,
                        schema_version="canonical_v1",
                    )
                    events.append(po_event)

        # Compile quality issues
        if missing_ids > 0:
            quality_issues.append(
                QualityIssue(
                    field="record_id",
                    issue_type="missing",
                    count=missing_ids,
                    description=f"Missing shipment_id and order_id in {missing_ids} records",
                )
            )

        if missing_timestamps > 0:
            quality_issues.append(
                QualityIssue(
                    field="delivery_timestamps",
                    issue_type="missing",
                    count=missing_timestamps,
                    description=f"Missing or incomplete delivery timestamps in {missing_timestamps} records",
                )
            )

        if missing_locations > 0:
            quality_issues.append(
                QualityIssue(
                    field="location_data",
                    issue_type="missing",
                    count=missing_locations,
                    description=f"Missing origin or destination in {missing_locations} shipment records",
                )
            )

        if invalid_delays > 0:
            quality_issues.append(
                QualityIssue(
                    field="delivery_delay",
                    issue_type="invalid_value",
                    count=invalid_delays,
                    description=f"Invalid delay calculation in {invalid_delays} records",
                )
            )

        if missing_vendors > 0:
            quality_issues.append(
                QualityIssue(
                    field="vendor_id",
                    issue_type="missing",
                    count=missing_vendors,
                    description=f"Missing vendor ID in {missing_vendors} order records",
                )
            )

        # Generate quality report
        total_records = len(shipments_df)
        quality_report = self._compute_quality_scores(
            events=events,
            total_records=total_records,
            rejected_records=rejected_count,
            quality_issues=quality_issues,
        )

        self.logger.info(
            "logistics_ingestion_completed",
            total_events=len(events),
            total_records=total_records,
            rejected_records=rejected_count,
            quality_score=quality_report.overall_quality_score,
        )

        return events, quality_report
