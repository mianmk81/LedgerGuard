"""
Canonical Event Builder - Bronze to Silver transformation engine.

This module implements the core data normalization pipeline that transforms
raw QuickBooks Online API responses (Bronze layer) into canonical business
events (Silver layer) with comprehensive data quality tracking.

The transformation process:
1. Read unprocessed raw entities from Bronze layer storage
2. Apply entity-specific transformation rules (Invoice → events, Payment → events, etc.)
3. Validate transformed events against canonical schema
4. Track data quality issues (completeness, consistency, timeliness)
5. Write canonical events to Silver layer with quality metadata

Supported QBO entity types and their event mappings:
- Invoice (new) → INVOICE_ISSUED
- Invoice (paid) → INVOICE_PAID
- Invoice (overdue) → INVOICE_OVERDUE
- Payment → PAYMENT_RECEIVED
- Bill → EXPENSE_POSTED
- BillPayment → EXPENSE_PAID
- CreditMemo → CREDIT_MEMO_ISSUED
- RefundReceipt → REFUND_ISSUED
- Customer (new) → CUSTOMER_CREATED
- Customer (update) → CUSTOMER_UPDATED
- PurchaseOrder → PURCHASE_ORDER_PLACED
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

import structlog

from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class CanonicalEventBuilder:
    """
    Transforms raw QBO objects into canonical business events.

    This is the Bronze → Silver transformation layer. Every raw QBO API
    response is normalized into the canonical event schema with full
    data quality tracking.

    The builder implements a medallion architecture pattern:
    - Bronze: Raw, immutable source data with full lineage
    - Silver: Validated, normalized canonical events ready for analysis
    - Gold: Aggregated metrics (handled by downstream components)

    Attributes:
        storage: Storage backend for reading Bronze and writing Silver data
        logger: Structured logger for observability

    Example:
        >>> builder = CanonicalEventBuilder(storage=duckdb_storage)
        >>> events, quality_report = builder.process_bronze_to_silver(
        ...     source="qbo",
        ...     since="2026-02-01T00:00:00Z"
        ... )
        >>> print(f"Processed {len(events)} events with quality score {quality_report.overall_quality_score}")
    """

    # Expected fields for each QBO entity type
    REQUIRED_FIELDS = {
        "Invoice": ["Id", "TxnDate", "TotalAmt", "CustomerRef"],
        "Payment": ["Id", "TxnDate", "TotalAmt", "CustomerRef"],
        "Bill": ["Id", "TxnDate", "TotalAmt", "VendorRef"],
        "BillPayment": ["Id", "TxnDate", "TotalAmt", "VendorRef"],
        "CreditMemo": ["Id", "TxnDate", "TotalAmt", "CustomerRef"],
        "RefundReceipt": ["Id", "TxnDate", "TotalAmt", "CustomerRef"],
        "Customer": ["Id", "DisplayName"],
        "PurchaseOrder": ["Id", "TxnDate", "VendorRef"],
    }

    def __init__(self, storage: StorageBackend):
        """
        Initialize the canonical event builder.

        Args:
            storage: Storage backend implementing Bronze/Silver layer operations
        """
        self.storage = storage
        self.logger = structlog.get_logger()

    def process_bronze_to_silver(
        self, source: str = "qbo", since: Optional[str] = None
    ) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Process all unprocessed Bronze records into Silver canonical events.

        Reads raw entities from Bronze layer, transforms them into canonical
        events, validates data quality, and writes to Silver layer. This is
        the core ETL operation of the medallion architecture.

        Args:
            source: Source system identifier (default: "qbo")
            since: Optional ISO timestamp to process only records ingested after this time

        Returns:
            Tuple of:
            - List of canonical events created
            - Data quality report with completeness, consistency, and timeliness scores

        Raises:
            ValueError: If source is invalid
            StorageError: If storage operations fail

        Example:
            >>> events, report = builder.process_bronze_to_silver(
            ...     source="qbo",
            ...     since="2026-02-10T00:00:00Z"
            ... )
        """
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        self.logger.info(
            "bronze_to_silver_processing_started",
            batch_id=batch_id,
            source=source,
            since=since,
        )

        # Read raw entities from Bronze layer
        raw_entities = self.storage.read_raw_entities(
            source=source,
            since=since,
            limit=10000,  # Process in batches of 10k
        )

        if not raw_entities:
            self.logger.info(
                "no_bronze_records_to_process",
                batch_id=batch_id,
                source=source,
            )
            return [], self._create_empty_quality_report(batch_id, source)

        all_events: list[CanonicalEvent] = []
        quality_issues: list[QualityIssue] = []
        rejected_count = 0

        # Track quality metrics
        total_records = len(raw_entities)
        completeness_scores: list[float] = []
        consistency_scores: list[float] = []
        timeliness_scores: list[float] = []

        # Transform each raw entity
        for raw_record in raw_entities:
            entity_type = raw_record.get("entity_type")
            raw_payload = raw_record.get("raw_payload", {})
            ingested_at = raw_record.get("ingested_at")

            try:
                # Transform entity to canonical events
                events = self.transform_entity(
                    entity_type=entity_type,
                    raw_payload=raw_payload,
                    source=source,
                )

                # Check data quality for each event
                for event in events:
                    completeness = self._calculate_completeness(event, entity_type)
                    consistency = self._calculate_consistency(event, raw_payload)
                    timeliness = self._calculate_timeliness(event, ingested_at)

                    completeness_scores.append(completeness)
                    consistency_scores.append(consistency)
                    timeliness_scores.append(timeliness)

                    # Collect quality issues
                    if event.data_quality_flags:
                        for flag in event.data_quality_flags:
                            self._add_quality_issue(quality_issues, flag, entity_type)

                all_events.extend(events)

            except Exception as e:
                rejected_count += 1
                self.logger.error(
                    "entity_transformation_failed",
                    batch_id=batch_id,
                    entity_type=entity_type,
                    error=str(e),
                )
                self._add_quality_issue(
                    quality_issues,
                    f"transformation_error_{entity_type}",
                    entity_type,
                )

        # Write canonical events to Silver layer
        if all_events:
            written_count = self.storage.write_canonical_events(all_events)
            self.logger.info(
                "canonical_events_written",
                batch_id=batch_id,
                events_count=written_count,
            )

        # Build quality report
        quality_report = self._build_quality_report(
            batch_id=batch_id,
            source=source,
            total_records=total_records,
            valid_records=total_records - rejected_count,
            rejected_records=rejected_count,
            completeness_scores=completeness_scores,
            consistency_scores=consistency_scores,
            timeliness_scores=timeliness_scores,
            quality_issues=quality_issues,
        )

        self.logger.info(
            "bronze_to_silver_processing_completed",
            batch_id=batch_id,
            total_records=total_records,
            events_created=len(all_events),
            rejected_records=rejected_count,
            overall_quality_score=quality_report.overall_quality_score,
        )

        return all_events, quality_report

    def transform_entity(
        self, entity_type: str, raw_payload: dict, source: str = "qbo"
    ) -> list[CanonicalEvent]:
        """
        Transform a single raw entity into one or more canonical events.

        Applies entity-specific transformation rules to convert raw QBO
        API responses into standardized canonical events. Some entities
        may produce multiple events (e.g., an overdue Invoice produces
        both INVOICE_ISSUED and INVOICE_OVERDUE events).

        Args:
            entity_type: QBO entity type (e.g., "Invoice", "Payment")
            raw_payload: Raw JSON payload from QBO API
            source: Source system identifier (default: "qbo")

        Returns:
            List of canonical events (may be empty if entity cannot be transformed)

        Raises:
            ValueError: If entity_type is unsupported

        Example:
            >>> invoice_payload = {"Id": "123", "TxnDate": "2026-02-10", ...}
            >>> events = builder.transform_entity("Invoice", invoice_payload, "qbo")
        """
        # Dispatch to entity-specific transformer
        transformer_map = {
            "Invoice": self._transform_invoice,
            "Payment": self._transform_payment,
            "Bill": self._transform_bill,
            "BillPayment": self._transform_bill_payment,
            "CreditMemo": self._transform_credit_memo,
            "RefundReceipt": self._transform_refund_receipt,
            "Customer": self._transform_customer,
            "PurchaseOrder": self._transform_purchase_order,
        }

        transformer = transformer_map.get(entity_type)
        if not transformer:
            self.logger.warning(
                "unsupported_entity_type",
                entity_type=entity_type,
                source=source,
            )
            return []

        return transformer(raw_payload, source)

    # =========================================================================
    # Entity-Specific Transformers
    # =========================================================================

    def _transform_invoice(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform Invoice entity into canonical events.

        Invoice can produce multiple events:
        - INVOICE_ISSUED (always, for new invoices)
        - INVOICE_PAID (if Balance = 0)
        - INVOICE_OVERDUE (if past due date and Balance > 0)

        Args:
            raw_payload: Raw Invoice JSON from QBO
            source: Source system identifier

        Returns:
            List of canonical events (1-3 events possible)
        """
        events: list[CanonicalEvent] = []
        data_quality_flags: list[str] = []

        # Extract core fields
        invoice_id = raw_payload.get("Id")
        doc_number = raw_payload.get("DocNumber")
        txn_date = raw_payload.get("TxnDate")
        due_date = raw_payload.get("DueDate")
        total_amt = raw_payload.get("TotalAmt")
        balance = raw_payload.get("Balance", total_amt)
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        # Validate required fields
        if not invoice_id:
            data_quality_flags.append("missing_invoice_id")
            return events

        # Extract customer reference
        customer_id = self._extract_customer_ref(raw_payload)
        if not customer_id:
            data_quality_flags.append("missing_customer_ref")

        # Extract line items
        line_items = self._extract_line_items(raw_payload)

        # Parse dates
        event_time = self._parse_qbo_date(txn_date)
        if not event_time:
            data_quality_flags.append("invalid_txn_date")
            event_time = datetime.utcnow()

        # Build base attributes
        base_attributes = {
            "doc_number": doc_number,
            "due_date": due_date,
            "line_items": line_items,
            "balance": float(balance) if balance is not None else None,
        }

        # Event 1: INVOICE_ISSUED (always created)
        issued_event = CanonicalEvent(
            event_type=EventType.INVOICE_ISSUED,
            event_time=event_time,
            source=source,
            source_entity_id=invoice_id,
            entity_type=EntityType.INVOICE,
            entity_id=f"invoice:{source}:{invoice_id}",
            related_entity_ids={
                "customer": f"customer:{source}:{customer_id}"
            } if customer_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes=base_attributes,
            data_quality_flags=data_quality_flags.copy(),
        )
        events.append(issued_event)

        # Event 2: INVOICE_PAID (if balance is zero)
        if balance is not None and float(balance) == 0.0:
            paid_date = self._extract_paid_date(raw_payload)
            days_to_pay = self._compute_days_to_pay(raw_payload)

            paid_event = CanonicalEvent(
                event_type=EventType.INVOICE_PAID,
                event_time=paid_date or event_time,
                source=source,
                source_entity_id=invoice_id,
                entity_type=EntityType.INVOICE,
                entity_id=f"invoice:{source}:{invoice_id}",
                related_entity_ids={
                    "customer": f"customer:{source}:{customer_id}"
                } if customer_id else {},
                amount=float(total_amt) if total_amt is not None else None,
                currency=currency,
                attributes={
                    "paid_date": paid_date.isoformat() if paid_date else None,
                    "payment_method": self._extract_payment_method(raw_payload),
                    "days_to_pay": days_to_pay,
                    "original_due_date": due_date,
                },
                data_quality_flags=data_quality_flags.copy(),
            )
            events.append(paid_event)

        # Event 3: INVOICE_OVERDUE (if past due and balance > 0)
        if due_date and balance is not None and float(balance) > 0.0:
            due_datetime = self._parse_qbo_date(due_date)
            if due_datetime and datetime.utcnow() > due_datetime:
                days_overdue = (datetime.utcnow() - due_datetime).days

                overdue_event = CanonicalEvent(
                    event_type=EventType.INVOICE_OVERDUE,
                    event_time=datetime.utcnow(),
                    source=source,
                    source_entity_id=invoice_id,
                    entity_type=EntityType.INVOICE,
                    entity_id=f"invoice:{source}:{invoice_id}",
                    related_entity_ids={
                        "customer": f"customer:{source}:{customer_id}"
                    } if customer_id else {},
                    amount=float(balance),
                    currency=currency,
                    attributes={
                        "days_overdue": days_overdue,
                        "balance_remaining": float(balance),
                        "original_due_date": due_date,
                    },
                    data_quality_flags=data_quality_flags.copy(),
                )
                events.append(overdue_event)

        return events

    def _transform_payment(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform Payment entity into PAYMENT_RECEIVED event.

        Args:
            raw_payload: Raw Payment JSON from QBO
            source: Source system identifier

        Returns:
            List containing single PAYMENT_RECEIVED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        payment_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not payment_id:
            data_quality_flags.append("missing_payment_id")
            return []

        # Extract customer reference
        customer_id = self._extract_customer_ref(raw_payload)
        if not customer_id:
            data_quality_flags.append("missing_customer_ref")

        # Extract applied invoices
        applied_to_invoices = []
        for line in raw_payload.get("Line", []):
            if line.get("LinkedTxn"):
                for linked_txn in line["LinkedTxn"]:
                    if linked_txn.get("TxnType") == "Invoice":
                        applied_to_invoices.append({
                            "invoice_id": linked_txn.get("TxnId"),
                            "amount": line.get("Amount"),
                        })

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.PAYMENT_RECEIVED,
            event_time=event_time,
            source=source,
            source_entity_id=payment_id,
            entity_type=EntityType.PAYMENT,
            entity_id=f"payment:{source}:{payment_id}",
            related_entity_ids={
                "customer": f"customer:{source}:{customer_id}"
            } if customer_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "payment_method": self._extract_payment_method(raw_payload),
                "applied_to_invoices": applied_to_invoices,
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_bill(self, raw_payload: dict, source: str) -> list[CanonicalEvent]:
        """
        Transform Bill entity into EXPENSE_POSTED event.

        Args:
            raw_payload: Raw Bill JSON from QBO
            source: Source system identifier

        Returns:
            List containing single EXPENSE_POSTED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        bill_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        due_date = raw_payload.get("DueDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not bill_id:
            data_quality_flags.append("missing_bill_id")
            return []

        # Extract vendor reference
        vendor_id = self._extract_vendor_ref(raw_payload)
        if not vendor_id:
            data_quality_flags.append("missing_vendor_ref")

        # Extract line items with categories
        line_items = self._extract_line_items(raw_payload)

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.EXPENSE_POSTED,
            event_time=event_time,
            source=source,
            source_entity_id=bill_id,
            entity_type=EntityType.EXPENSE,
            entity_id=f"expense:{source}:{bill_id}",
            related_entity_ids={
                "vendor": f"vendor:{source}:{vendor_id}"
            } if vendor_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "due_date": due_date,
                "line_items": line_items,
                "category": self._extract_expense_category(raw_payload),
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_bill_payment(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform BillPayment entity into EXPENSE_PAID event.

        Args:
            raw_payload: Raw BillPayment JSON from QBO
            source: Source system identifier

        Returns:
            List containing single EXPENSE_PAID event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        payment_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not payment_id:
            data_quality_flags.append("missing_payment_id")
            return []

        # Extract vendor reference
        vendor_id = self._extract_vendor_ref(raw_payload)
        if not vendor_id:
            data_quality_flags.append("missing_vendor_ref")

        # Extract paid bill IDs
        bill_ids = []
        for line in raw_payload.get("Line", []):
            if line.get("LinkedTxn"):
                for linked_txn in line["LinkedTxn"]:
                    if linked_txn.get("TxnType") == "Bill":
                        bill_ids.append(linked_txn.get("TxnId"))

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.EXPENSE_PAID,
            event_time=event_time,
            source=source,
            source_entity_id=payment_id,
            entity_type=EntityType.EXPENSE,
            entity_id=f"expense:{source}:{payment_id}",
            related_entity_ids={
                "vendor": f"vendor:{source}:{vendor_id}"
            } if vendor_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "payment_method": self._extract_payment_method(raw_payload),
                "bill_ids": bill_ids,
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_credit_memo(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform CreditMemo entity into CREDIT_MEMO_ISSUED event.

        Args:
            raw_payload: Raw CreditMemo JSON from QBO
            source: Source system identifier

        Returns:
            List containing single CREDIT_MEMO_ISSUED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        credit_memo_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not credit_memo_id:
            data_quality_flags.append("missing_credit_memo_id")
            return []

        # Extract customer reference
        customer_id = self._extract_customer_ref(raw_payload)
        if not customer_id:
            data_quality_flags.append("missing_customer_ref")

        # Extract line items
        line_items = self._extract_line_items(raw_payload)

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.CREDIT_MEMO_ISSUED,
            event_time=event_time,
            source=source,
            source_entity_id=credit_memo_id,
            entity_type=EntityType.INVOICE,
            entity_id=f"invoice:{source}:{credit_memo_id}",
            related_entity_ids={
                "customer": f"customer:{source}:{customer_id}"
            } if customer_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "line_items": line_items,
                "reason": raw_payload.get("CustomerMemo", {}).get("value"),
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_refund_receipt(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform RefundReceipt entity into REFUND_ISSUED event.

        Args:
            raw_payload: Raw RefundReceipt JSON from QBO
            source: Source system identifier

        Returns:
            List containing single REFUND_ISSUED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        refund_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not refund_id:
            data_quality_flags.append("missing_refund_id")
            return []

        # Extract customer reference
        customer_id = self._extract_customer_ref(raw_payload)
        if not customer_id:
            data_quality_flags.append("missing_customer_ref")

        # Extract line items
        line_items = self._extract_line_items(raw_payload)

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.REFUND_ISSUED,
            event_time=event_time,
            source=source,
            source_entity_id=refund_id,
            entity_type=EntityType.PAYMENT,
            entity_id=f"payment:{source}:{refund_id}",
            related_entity_ids={
                "customer": f"customer:{source}:{customer_id}"
            } if customer_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "line_items": line_items,
                "refund_method": self._extract_payment_method(raw_payload),
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_customer(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform Customer entity into CUSTOMER_CREATED or CUSTOMER_UPDATED event.

        Determines event type based on metadata timestamps - if CreateTime and
        LastUpdatedTime are close, it's a new customer, otherwise it's an update.

        Args:
            raw_payload: Raw Customer JSON from QBO
            source: Source system identifier

        Returns:
            List containing single CUSTOMER_CREATED or CUSTOMER_UPDATED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        customer_id = raw_payload.get("Id")
        display_name = raw_payload.get("DisplayName")
        balance = raw_payload.get("Balance", 0.0)

        if not customer_id:
            data_quality_flags.append("missing_customer_id")
            return []

        # Determine if this is create or update
        metadata = raw_payload.get("MetaData", {})
        create_time = self._parse_qbo_date(metadata.get("CreateTime"))
        last_updated = self._parse_qbo_date(metadata.get("LastUpdatedTime"))

        # If timestamps are within 1 minute, consider it a create event
        is_create = False
        if create_time and last_updated:
            time_diff = abs((last_updated - create_time).total_seconds())
            is_create = time_diff < 60
        else:
            is_create = True  # Default to create if metadata missing

        event_type = EventType.CUSTOMER_CREATED if is_create else EventType.CUSTOMER_UPDATED
        event_time = create_time if is_create else last_updated
        if not event_time:
            event_time = datetime.utcnow()

        # Build attributes
        attributes = {
            "display_name": display_name,
            "email": raw_payload.get("PrimaryEmailAddr", {}).get("Address"),
            "balance": float(balance) if balance is not None else 0.0,
        }

        # For updates, track what changed
        if not is_create:
            # In production, we'd compare with previous version from Silver layer
            attributes["changed_fields"] = ["balance"]  # Simplified
            attributes["new_balance"] = float(balance) if balance is not None else 0.0

        event = CanonicalEvent(
            event_type=event_type,
            event_time=event_time,
            source=source,
            source_entity_id=customer_id,
            entity_type=EntityType.CUSTOMER,
            entity_id=f"customer:{source}:{customer_id}",
            related_entity_ids={},
            amount=None,
            currency="USD",
            attributes=attributes,
            data_quality_flags=data_quality_flags,
        )

        return [event]

    def _transform_purchase_order(
        self, raw_payload: dict, source: str
    ) -> list[CanonicalEvent]:
        """
        Transform PurchaseOrder entity into PURCHASE_ORDER_PLACED event.

        Args:
            raw_payload: Raw PurchaseOrder JSON from QBO
            source: Source system identifier

        Returns:
            List containing single PURCHASE_ORDER_PLACED event
        """
        data_quality_flags: list[str] = []

        # Extract core fields
        po_id = raw_payload.get("Id")
        txn_date = raw_payload.get("TxnDate")
        total_amt = raw_payload.get("TotalAmt")
        currency = raw_payload.get("CurrencyRef", {}).get("value", "USD")

        if not po_id:
            data_quality_flags.append("missing_po_id")
            return []

        # Extract vendor reference
        vendor_id = self._extract_vendor_ref(raw_payload)
        if not vendor_id:
            data_quality_flags.append("missing_vendor_ref")

        # Extract line items
        line_items = self._extract_line_items(raw_payload)

        # Parse event time
        event_time = self._parse_qbo_date(txn_date) or datetime.utcnow()

        event = CanonicalEvent(
            event_type=EventType.PURCHASE_ORDER_PLACED,
            event_time=event_time,
            source=source,
            source_entity_id=po_id,
            entity_type=EntityType.ORDER,
            entity_id=f"order:{source}:{po_id}",
            related_entity_ids={
                "vendor": f"vendor:{source}:{vendor_id}"
            } if vendor_id else {},
            amount=float(total_amt) if total_amt is not None else None,
            currency=currency,
            attributes={
                "line_items": line_items,
                "expected_date": raw_payload.get("ShipDate"),
            },
            data_quality_flags=data_quality_flags,
        )

        return [event]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_line_items(self, raw_payload: dict) -> list[dict]:
        """
        Normalize QBO line item format into canonical structure.

        QBO line items have varying structures depending on entity type.
        This method extracts the common fields into a normalized format.

        Args:
            raw_payload: Raw QBO entity payload

        Returns:
            List of normalized line item dictionaries
        """
        line_items = []

        for line in raw_payload.get("Line", []):
            # Skip subtotal and discount lines
            detail_type = line.get("DetailType")
            if detail_type in ["SubTotalLineDetail", "DiscountLineDetail"]:
                continue

            item = {
                "description": line.get("Description"),
                "amount": line.get("Amount"),
                "quantity": None,
                "item_id": None,
            }

            # Extract item reference based on detail type
            if detail_type == "SalesItemLineDetail":
                detail = line.get("SalesItemLineDetail", {})
                item["quantity"] = detail.get("Qty")
                item["item_id"] = detail.get("ItemRef", {}).get("value")
            elif detail_type == "ItemBasedExpenseLineDetail":
                detail = line.get("ItemBasedExpenseLineDetail", {})
                item["quantity"] = detail.get("Qty")
                item["item_id"] = detail.get("ItemRef", {}).get("value")
            elif detail_type == "AccountBasedExpenseLineDetail":
                detail = line.get("AccountBasedExpenseLineDetail", {})
                item["item_id"] = detail.get("AccountRef", {}).get("value")

            line_items.append(item)

        return line_items

    def _extract_customer_ref(self, raw_payload: dict) -> Optional[str]:
        """
        Extract customer ID from various QBO reference formats.

        Args:
            raw_payload: Raw QBO entity payload

        Returns:
            Customer ID string or None if not found
        """
        customer_ref = raw_payload.get("CustomerRef")
        if customer_ref:
            return customer_ref.get("value")
        return None

    def _extract_vendor_ref(self, raw_payload: dict) -> Optional[str]:
        """
        Extract vendor ID from various QBO reference formats.

        Args:
            raw_payload: Raw QBO entity payload

        Returns:
            Vendor ID string or None if not found
        """
        vendor_ref = raw_payload.get("VendorRef")
        if vendor_ref:
            return vendor_ref.get("value")
        return None

    def _extract_payment_method(self, raw_payload: dict) -> Optional[str]:
        """
        Extract payment method from QBO entity.

        Args:
            raw_payload: Raw QBO entity payload

        Returns:
            Payment method string or None if not found
        """
        payment_method_ref = raw_payload.get("PaymentMethodRef")
        if payment_method_ref:
            return payment_method_ref.get("name") or payment_method_ref.get("value")

        # Check for payment type in some entities
        payment_type = raw_payload.get("PaymentType")
        if payment_type:
            return payment_type

        return None

    def _extract_expense_category(self, raw_payload: dict) -> Optional[str]:
        """
        Extract expense category from Bill line items.

        Args:
            raw_payload: Raw Bill payload

        Returns:
            Category name or None if not found
        """
        # Extract from first line item with account reference
        for line in raw_payload.get("Line", []):
            if line.get("DetailType") == "AccountBasedExpenseLineDetail":
                detail = line.get("AccountBasedExpenseLineDetail", {})
                account_ref = detail.get("AccountRef", {})
                return account_ref.get("name") or account_ref.get("value")
        return None

    def _extract_paid_date(self, raw_payload: dict) -> Optional[datetime]:
        """
        Extract the date when an invoice was paid.

        Args:
            raw_payload: Raw Invoice payload

        Returns:
            Paid date as datetime or None if not paid
        """
        # Check MetaData.LastUpdatedTime as proxy for payment date
        # In production, this would come from Payment entity linkage
        metadata = raw_payload.get("MetaData", {})
        last_updated = metadata.get("LastUpdatedTime")
        if last_updated:
            return self._parse_qbo_date(last_updated)
        return None

    def _compute_days_to_pay(self, raw_payload: dict) -> Optional[int]:
        """
        Calculate payment delay in days (TxnDate to paid date).

        Args:
            raw_payload: Raw Invoice payload

        Returns:
            Number of days to pay or None if not calculable
        """
        txn_date = self._parse_qbo_date(raw_payload.get("TxnDate"))
        paid_date = self._extract_paid_date(raw_payload)

        if txn_date and paid_date:
            return (paid_date - txn_date).days
        return None

    def _parse_qbo_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse QBO date string to datetime object.

        QBO returns dates in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).

        Args:
            date_str: Date string from QBO

        Returns:
            Parsed datetime or None if parsing fails
        """
        if not date_str:
            return None

        try:
            # Try with time component first
            if "T" in date_str:
                # Remove timezone if present
                date_str = date_str.split("+")[0].split("-")[0].split("Z")[0]
                return datetime.fromisoformat(date_str)
            else:
                # Date only, set time to midnight
                return datetime.fromisoformat(f"{date_str}T00:00:00")
        except (ValueError, AttributeError) as e:
            self.logger.warning(
                "date_parse_failed",
                date_str=date_str,
                error=str(e),
            )
            return None

    # =========================================================================
    # Data Quality Scoring
    # =========================================================================

    def _calculate_completeness(
        self, event: CanonicalEvent, entity_type: str
    ) -> float:
        """
        Calculate completeness score for an event (0.0-1.0).

        Checks what percentage of expected fields are populated.

        Args:
            event: Canonical event to evaluate
            entity_type: Original QBO entity type

        Returns:
            Completeness score between 0.0 and 1.0
        """
        required = self.REQUIRED_FIELDS.get(entity_type, [])
        if not required:
            return 1.0

        # Map event fields back to QBO field names
        field_mapping = {
            "event_id": "Id",
            "amount": "TotalAmt",
            "event_time": "TxnDate",
        }

        populated = 0
        for field in required:
            # Check if field is in attributes or mapped fields
            if field in event.attributes or field_mapping.get(field.lower()) in event.model_dump():
                populated += 1

        return populated / len(required) if required else 1.0

    def _calculate_consistency(
        self, event: CanonicalEvent, raw_payload: dict
    ) -> float:
        """
        Calculate consistency score for an event (0.0-1.0).

        Checks for cross-field validation:
        - Amount matches sum of line items
        - Dates are chronological
        - Balance <= TotalAmt for invoices

        Args:
            event: Canonical event to evaluate
            raw_payload: Original raw payload

        Returns:
            Consistency score between 0.0 and 1.0
        """
        checks_passed = 0
        total_checks = 0

        # Check 1: Amount matches line items sum (if applicable)
        if event.amount is not None and event.attributes.get("line_items"):
            total_checks += 1
            line_total = sum(
                float(item.get("amount", 0))
                for item in event.attributes["line_items"]
            )
            # Allow 1% tolerance for rounding
            if abs(event.amount - line_total) / event.amount < 0.01:
                checks_passed += 1

        # Check 2: Dates are not in the future
        total_checks += 1
        if event.event_time <= datetime.utcnow():
            checks_passed += 1

        # Check 3: Balance consistency for invoices
        if event.entity_type == EntityType.INVOICE:
            balance = event.attributes.get("balance")
            if balance is not None and event.amount is not None:
                total_checks += 1
                if balance <= event.amount:
                    checks_passed += 1

        return checks_passed / total_checks if total_checks > 0 else 1.0

    def _calculate_timeliness(
        self, event: CanonicalEvent, ingested_at: Optional[str]
    ) -> float:
        """
        Calculate timeliness score for an event (0.0-1.0).

        Checks if event was ingested within expected time window:
        - Same day: 1.0
        - Within 7 days: 0.8
        - Within 30 days: 0.6
        - Older: 0.4

        Args:
            event: Canonical event to evaluate
            ingested_at: Timestamp when raw entity was ingested

        Returns:
            Timeliness score between 0.0 and 1.0
        """
        if not ingested_at:
            return 0.8  # Default moderate score if ingestion time unknown

        try:
            ingestion_time = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
            age_days = (ingestion_time - event.event_time).days

            if age_days < 1:
                return 1.0
            elif age_days < 7:
                return 0.8
            elif age_days < 30:
                return 0.6
            else:
                return 0.4
        except (ValueError, AttributeError):
            return 0.8

    def _add_quality_issue(
        self, issues_list: list[QualityIssue], flag: str, entity_type: str
    ) -> None:
        """
        Add or increment a quality issue in the issues list.

        Args:
            issues_list: List of quality issues to update
            flag: Quality flag identifier
            entity_type: Entity type where issue occurred
        """
        # Check if issue already exists
        for issue in issues_list:
            if issue.field == flag:
                issue.count += 1
                return

        # Create new issue
        description_map = {
            "missing_invoice_id": "Invoice ID missing in raw payload",
            "missing_customer_ref": "Customer reference missing",
            "missing_vendor_ref": "Vendor reference missing",
            "invalid_txn_date": "Transaction date invalid or unparsable",
            "transformation_error": f"Failed to transform {entity_type} entity",
        }

        issues_list.append(
            QualityIssue(
                field=flag,
                issue_type="missing" if "missing" in flag else "invalid",
                count=1,
                description=description_map.get(
                    flag, f"Data quality issue: {flag}"
                ),
            )
        )

    def _build_quality_report(
        self,
        batch_id: str,
        source: str,
        total_records: int,
        valid_records: int,
        rejected_records: int,
        completeness_scores: list[float],
        consistency_scores: list[float],
        timeliness_scores: list[float],
        quality_issues: list[QualityIssue],
    ) -> DataQualityReport:
        """
        Build comprehensive data quality report for a batch.

        Args:
            batch_id: Unique batch identifier
            source: Source system
            total_records: Total records processed
            valid_records: Successfully transformed records
            rejected_records: Failed transformation records
            completeness_scores: List of completeness scores
            consistency_scores: List of consistency scores
            timeliness_scores: List of timeliness scores
            quality_issues: List of quality issues found

        Returns:
            Comprehensive data quality report
        """
        # Calculate average scores
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 1.0
        )
        avg_consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores
            else 1.0
        )
        avg_timeliness = (
            sum(timeliness_scores) / len(timeliness_scores)
            if timeliness_scores
            else 1.0
        )

        # Overall quality score (weighted average)
        overall_score = (
            avg_completeness * 0.4 + avg_consistency * 0.4 + avg_timeliness * 0.2
        )

        # Generate impact advisory
        if overall_score >= 0.95:
            impact = "Excellent data quality. All metrics exceed thresholds."
        elif overall_score >= 0.85:
            impact = "Good data quality. Minor issues do not impact analysis accuracy."
        elif overall_score >= 0.70:
            impact = "Moderate quality. Some missing fields may affect detection confidence."
        else:
            impact = "Low quality. Significant data issues detected. Review ingestion pipeline."

        return DataQualityReport(
            batch_id=batch_id,
            source=source,
            total_records=total_records,
            valid_records=valid_records,
            rejected_records=rejected_records,
            completeness_score=round(avg_completeness, 4),
            consistency_score=round(avg_consistency, 4),
            timeliness_score=round(avg_timeliness, 4),
            overall_quality_score=round(overall_score, 4),
            quality_issues=quality_issues,
            impact_advisory=impact,
        )

    def _create_empty_quality_report(
        self, batch_id: str, source: str
    ) -> DataQualityReport:
        """
        Create an empty quality report when no records are processed.

        Args:
            batch_id: Unique batch identifier
            source: Source system

        Returns:
            Empty data quality report with perfect scores
        """
        return DataQualityReport(
            batch_id=batch_id,
            source=source,
            total_records=0,
            valid_records=0,
            rejected_records=0,
            completeness_score=1.0,
            consistency_score=1.0,
            timeliness_score=1.0,
            overall_quality_score=1.0,
            quality_issues=[],
            impact_advisory="No records to process in this batch.",
        )
