"""
Olist E-Commerce Dataset Adapter.

This adapter transforms the Brazilian E-Commerce Public Dataset by Olist
into canonical events for business reliability analysis. The dataset includes
orders, order items, reviews, customers, products, and sellers.

Dataset source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import structlog

from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue

from .base_adapter import BaseAdapter

logger = structlog.get_logger()


class OlistAdapter(BaseAdapter):
    """
    Adapts Olist e-commerce dataset into canonical events.

    Transforms order lifecycle, delivery performance, and customer review data
    into standardized events for anomaly detection and incident analysis.

    Supported event types:
    - ORDER_PLACED: When customer places an order
    - ORDER_DELIVERED: When order is successfully delivered
    - ORDER_LATE: When order delivered after estimated date
    - REVIEW_SUBMITTED: When customer submits a review
    """

    def __init__(self):
        """Initialize Olist adapter."""
        super().__init__(source_name="olist")

    def ingest(
        self,
        orders_df: pd.DataFrame,
        order_items_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Transform Olist dataset into canonical events.

        Args:
            orders_df: Main orders DataFrame with order lifecycle timestamps
            order_items_df: Optional order items DataFrame for product details
            reviews_df: Optional reviews DataFrame for customer feedback
            **kwargs: Additional configuration options

        Returns:
            Tuple of (canonical events, data quality report)

        Expected columns in orders_df:
        - order_id: Unique order identifier
        - customer_id: Customer identifier
        - order_purchase_timestamp: When order was placed
        - order_delivered_customer_date: Actual delivery timestamp
        - order_estimated_delivery_date: Expected delivery date
        - order_status: Order status (delivered, shipped, etc.)

        Expected columns in order_items_df (optional):
        - order_id: Order identifier
        - product_id: Product identifier
        - seller_id: Seller identifier
        - price: Item price
        - freight_value: Shipping cost

        Expected columns in reviews_df (optional):
        - review_id: Review identifier
        - order_id: Order identifier
        - review_score: Score 1-5
        - review_comment_title: Review title
        - review_comment_message: Review text
        - review_creation_date: When review was created
        """
        self.logger.info(
            "olist_ingestion_started",
            orders_count=len(orders_df),
            has_items=order_items_df is not None,
            has_reviews=reviews_df is not None,
        )

        events = []
        quality_issues = []
        rejected_count = 0

        # Track quality metrics
        missing_timestamps = 0
        missing_customer_ids = 0
        invalid_amounts = 0
        missing_review_scores = 0

        # Merge order items if provided to get pricing data
        if order_items_df is not None:
            # Aggregate order items by order_id
            order_totals = (
                order_items_df.groupby("order_id")
                .agg(
                    {
                        "price": "sum",
                        "freight_value": "sum",
                        "product_id": lambda x: list(x),
                        "seller_id": "first",  # Take first seller for simplicity
                    }
                )
                .reset_index()
            )
            order_totals["total_amount"] = (
                order_totals["price"] + order_totals["freight_value"]
            )
            orders_df = orders_df.merge(order_totals, on="order_id", how="left")

        # Process each order
        for idx, row in orders_df.iterrows():
            order_id = self._safe_str(row.get("order_id"))
            customer_id = self._safe_str(row.get("customer_id"))

            if not order_id:
                rejected_count += 1
                continue

            # Track missing customer IDs
            if not customer_id:
                missing_customer_ids += 1

            # Extract timestamps
            purchase_timestamp = self._safe_datetime(
                row.get("order_purchase_timestamp")
            )
            delivered_timestamp = self._safe_datetime(
                row.get("order_delivered_customer_date")
            )
            estimated_delivery = self._safe_datetime(
                row.get("order_estimated_delivery_date")
            )

            # Calculate total amount
            total_amount = self._safe_float(row.get("total_amount"))
            if total_amount is not None and total_amount < 0:
                invalid_amounts += 1
                total_amount = None

            # Extract product and seller info
            product_ids = row.get("product_id", [])
            if not isinstance(product_ids, list):
                product_ids = [self._safe_str(product_ids)] if product_ids else []
            seller_id = self._safe_str(row.get("seller_id"))

            # Build related entities
            related_entities = {}
            if customer_id:
                related_entities["customer"] = self._create_entity_id(
                    "customer", customer_id
                )
            if seller_id:
                related_entities["vendor"] = self._create_entity_id("vendor", seller_id)

            # EVENT 1: ORDER_PLACED
            if purchase_timestamp:
                event_flags = []
                if not customer_id:
                    event_flags.append("missing_customer_id")
                if total_amount is None:
                    event_flags.append("missing_amount")

                order_placed_event = CanonicalEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.ORDER_PLACED,
                    event_time=purchase_timestamp,
                    source=self.source_name,
                    source_entity_id=order_id,
                    entity_type=EntityType.ORDER,
                    entity_id=self._create_entity_id("order", order_id),
                    related_entity_ids=related_entities,
                    amount=total_amount,
                    currency="BRL",  # Brazilian Real
                    attributes={
                        "order_id": order_id,
                        "customer_id": customer_id,
                        "product_ids": product_ids,
                        "seller_id": seller_id,
                        "estimated_delivery": (
                            estimated_delivery.isoformat() if estimated_delivery else None
                        ),
                        "order_status": self._safe_str(row.get("order_status")),
                    },
                    data_quality_flags=event_flags,
                    schema_version="canonical_v1",
                )
                events.append(order_placed_event)
            else:
                missing_timestamps += 1
                rejected_count += 1

            # EVENT 2: ORDER_DELIVERED
            if delivered_timestamp and purchase_timestamp:
                event_flags = []

                # Calculate delivery time
                delivery_days = (
                    delivered_timestamp - purchase_timestamp
                ).total_seconds() / 86400

                delivered_event = CanonicalEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.ORDER_DELIVERED,
                    event_time=delivered_timestamp,
                    source=self.source_name,
                    source_entity_id=order_id,
                    entity_type=EntityType.ORDER,
                    entity_id=self._create_entity_id("order", order_id),
                    related_entity_ids=related_entities,
                    amount=total_amount,
                    currency="BRL",
                    attributes={
                        "order_id": order_id,
                        "actual_delivery": delivered_timestamp.isoformat(),
                        "delivery_days": round(delivery_days, 2),
                        "estimated_delivery": (
                            estimated_delivery.isoformat() if estimated_delivery else None
                        ),
                    },
                    data_quality_flags=event_flags,
                    schema_version="canonical_v1",
                )
                events.append(delivered_event)

                # EVENT 3: ORDER_LATE (if delivered after estimated date)
                if estimated_delivery and delivered_timestamp > estimated_delivery:
                    days_late = (
                        delivered_timestamp - estimated_delivery
                    ).total_seconds() / 86400

                    late_event = CanonicalEvent(
                        event_id=self._generate_event_id(),
                        event_type=EventType.ORDER_LATE,
                        event_time=delivered_timestamp,
                        source=self.source_name,
                        source_entity_id=order_id,
                        entity_type=EntityType.ORDER,
                        entity_id=self._create_entity_id("order", order_id),
                        related_entity_ids=related_entities,
                        amount=total_amount,
                        currency="BRL",
                        attributes={
                            "order_id": order_id,
                            "days_late": round(days_late, 2),
                            "estimated_delivery": estimated_delivery.isoformat(),
                            "actual_delivery": delivered_timestamp.isoformat(),
                        },
                        data_quality_flags=[],
                        schema_version="canonical_v1",
                    )
                    events.append(late_event)

        # Process reviews if provided
        if reviews_df is not None:
            for idx, row in reviews_df.iterrows():
                review_id = self._safe_str(row.get("review_id"))
                order_id = self._safe_str(row.get("order_id"))
                review_creation = self._safe_datetime(row.get("review_creation_date"))
                review_score = self._safe_float(row.get("review_score"))

                if not review_id or not order_id or not review_creation:
                    rejected_count += 1
                    continue

                event_flags = []
                if review_score is None or review_score < 1 or review_score > 5:
                    event_flags.append("invalid_review_score")
                    missing_review_scores += 1

                review_event = CanonicalEvent(
                    event_id=self._generate_event_id(),
                    event_type=EventType.REVIEW_SUBMITTED,
                    event_time=review_creation,
                    source=self.source_name,
                    source_entity_id=review_id,
                    entity_type=EntityType.ORDER,
                    entity_id=self._create_entity_id("review", review_id),
                    related_entity_ids={
                        "order": self._create_entity_id("order", order_id)
                    },
                    amount=None,
                    currency="BRL",
                    attributes={
                        "review_id": review_id,
                        "order_id": order_id,
                        "score": review_score,
                        "comment_title": self._safe_str(row.get("review_comment_title")),
                        "comment_message": self._safe_str(
                            row.get("review_comment_message")
                        ),
                    },
                    data_quality_flags=event_flags,
                    schema_version="canonical_v1",
                )
                events.append(review_event)

        # Compile quality issues
        if missing_timestamps > 0:
            quality_issues.append(
                QualityIssue(
                    field="order_purchase_timestamp",
                    issue_type="missing",
                    count=missing_timestamps,
                    description=f"Missing purchase timestamp in {missing_timestamps} order records",
                )
            )

        if missing_customer_ids > 0:
            quality_issues.append(
                QualityIssue(
                    field="customer_id",
                    issue_type="missing",
                    count=missing_customer_ids,
                    description=f"Missing customer ID in {missing_customer_ids} order records",
                )
            )

        if invalid_amounts > 0:
            quality_issues.append(
                QualityIssue(
                    field="total_amount",
                    issue_type="invalid_value",
                    count=invalid_amounts,
                    description=f"Invalid or negative amount in {invalid_amounts} order records",
                )
            )

        if missing_review_scores > 0:
            quality_issues.append(
                QualityIssue(
                    field="review_score",
                    issue_type="invalid_value",
                    count=missing_review_scores,
                    description=f"Missing or invalid review score in {missing_review_scores} review records",
                )
            )

        # Calculate total records processed
        total_records = len(orders_df)
        if reviews_df is not None:
            total_records += len(reviews_df)

        # Generate quality report
        quality_report = self._compute_quality_scores(
            events=events,
            total_records=total_records,
            rejected_records=rejected_count,
            quality_issues=quality_issues,
        )

        self.logger.info(
            "olist_ingestion_completed",
            total_events=len(events),
            total_records=total_records,
            rejected_records=rejected_count,
            quality_score=quality_report.overall_quality_score,
        )

        return events, quality_report
