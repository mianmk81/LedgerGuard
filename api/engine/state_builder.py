"""
State & Feature Builder (Gold Layer) - Business Health Metrics Engine.

This module computes daily business health metrics from canonical Silver layer
events. These metrics are the "vital signs" of the business that the incident
detection engine monitors for anomalies.

The StateBuilder aggregates 27 daily metrics across three domains:
- Financial Health (11 metrics): revenue, expenses, margins, cash flow
- Operational Health (8 metrics): order fulfillment, delivery performance
- Customer Health (8 metrics): support, satisfaction, retention

Computation patterns:
- Point-in-time metrics: Computed from events on a specific date
- Rolling window metrics: Computed from trailing N-day windows
- Cumulative metrics: Running totals over defined periods
- Trend metrics: Linear regression slopes over historical data

Critical path component: All incident detection depends on these metrics.
Quality and timeliness are paramount.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import structlog
from scipy import stats

from api.models.enums import EventType
from api.models.events import CanonicalEvent
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class StateBuilder:
    """
    Computes daily business health metrics from canonical events.

    Transforms Silver layer canonical events into Gold layer aggregated
    metrics using point-in-time, rolling window, and trend computations.
    All metrics are versioned with schema_version for evolution tracking.

    Attributes:
        storage: Storage backend for reading events and writing metrics
        logger: Structured logger for observability
        metric_schema_version: Version identifier for metric schema

    Example:
        >>> builder = StateBuilder(storage=duckdb_storage)
        >>> metrics = builder.compute_daily_metrics(date(2026, 2, 10))
        >>> print(f"Computed {len(metrics)} metrics for 2026-02-10")
    """

    # Schema version for metric evolution tracking
    METRIC_SCHEMA_VERSION = "gold_v1"

    # Domain health thresholds (used for health classification)
    FINANCIAL_THRESHOLDS = {
        "critical": {"margin_proxy": 0.10, "expense_ratio": 0.85},
        "degraded": {"margin_proxy": 0.20, "expense_ratio": 0.75},
    }

    OPERATIONAL_THRESHOLDS = {
        "critical": {"delivery_delay_rate": 0.30, "fulfillment_backlog": 100},
        "degraded": {"delivery_delay_rate": 0.15, "fulfillment_backlog": 50},
    }

    CUSTOMER_THRESHOLDS = {
        "critical": {"review_score_avg": 2.5, "ticket_backlog": 50},
        "degraded": {"review_score_avg": 3.5, "ticket_backlog": 25},
    }

    def __init__(self, storage: StorageBackend):
        """
        Initialize the state builder.

        Args:
            storage: Storage backend implementing Silver/Gold layer operations
        """
        self.storage = storage
        self.logger = structlog.get_logger()

    def compute_daily_metrics(
        self, target_date: date, events: Optional[list[CanonicalEvent]] = None
    ) -> dict:
        """
        Compute all 27 daily metrics for a given date.

        Aggregates canonical events from the target date plus necessary
        historical context windows (7-day, 14-day, 30-day) to compute
        point-in-time and rolling window metrics.

        Args:
            target_date: Date to compute metrics for
            events: Optional pre-fetched events (if None, will query storage)

        Returns:
            Dictionary containing all computed metrics with metadata:
            {
                "metric_date": "2026-02-10",
                "financial": {...},  # 11 financial metrics
                "operational": {...},  # 8 operational metrics
                "customer": {...},  # 8 customer metrics
                "metadata": {
                    "computed_at": "2026-02-10T15:30:00Z",
                    "events_processed": 1234,
                    "metric_schema_version": "gold_v1"
                }
            }

        Example:
            >>> metrics = builder.compute_daily_metrics(date(2026, 2, 10))
            >>> print(f"Daily revenue: ${metrics['financial']['daily_revenue']:.2f}")
        """
        self.logger.info(
            "computing_daily_metrics",
            target_date=str(target_date),
        )

        # Determine date range needed for rolling windows
        # Need 30 days prior for longest rolling window
        start_date = target_date - timedelta(days=30)
        end_date = target_date

        # Fetch events if not provided
        if events is None:
            events = self._get_events_in_window(
                event_types=None,  # Get all event types
                start=start_date,
                end=end_date,
            )

        # Compute metrics by domain
        financial_metrics = self._compute_financial_metrics(target_date, events)
        operational_metrics = self._compute_operational_metrics(target_date, events)
        customer_metrics = self._compute_customer_metrics(target_date, events)

        # Build result
        result = {
            "metric_date": str(target_date),
            "financial": financial_metrics,
            "operational": operational_metrics,
            "customer": customer_metrics,
            "metadata": {
                "computed_at": datetime.utcnow().isoformat(),
                "events_processed": len(events),
                "metric_schema_version": self.METRIC_SCHEMA_VERSION,
            },
        }

        self.logger.info(
            "daily_metrics_computed",
            target_date=str(target_date),
            events_processed=len(events),
            financial_count=len(financial_metrics),
            operational_count=len(operational_metrics),
            customer_count=len(customer_metrics),
        )

        return result

    def compute_date_range(
        self, start_date: date, end_date: date
    ) -> list[dict]:
        """
        Compute metrics for a range of dates. Used for backfill.

        Efficiently computes metrics for multiple dates by fetching
        events once with appropriate buffer window and reusing for
        overlapping date computations.

        Args:
            start_date: First date to compute metrics for (inclusive)
            end_date: Last date to compute metrics for (inclusive)

        Returns:
            List of daily metric dictionaries, one per date

        Example:
            >>> metrics_list = builder.compute_date_range(
            ...     date(2026, 2, 1),
            ...     date(2026, 2, 28)
            ... )
            >>> print(f"Computed metrics for {len(metrics_list)} days")
        """
        self.logger.info(
            "computing_date_range",
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Fetch events for entire range plus 30-day buffer
        buffer_start = start_date - timedelta(days=30)
        events = self._get_events_in_window(
            event_types=None,
            start=buffer_start,
            end=end_date,
        )

        results = []
        current = start_date

        while current <= end_date:
            daily_metrics = self.compute_daily_metrics(current, events)
            results.append(daily_metrics)
            current += timedelta(days=1)

        self.logger.info(
            "date_range_computed",
            start_date=str(start_date),
            end_date=str(end_date),
            days_computed=len(results),
            total_events=len(events),
        )

        return results

    def get_current_state(self) -> dict:
        """
        Get the latest computed business state.

        Returns the most recently computed daily metrics, or computes
        metrics for yesterday if not yet available (today's metrics are
        incomplete until end of day).

        Returns:
            Dictionary containing latest daily metrics

        Example:
            >>> state = builder.get_current_state()
            >>> print(f"Latest state for {state['metric_date']}")
        """
        # Use yesterday as "current" since today is incomplete
        yesterday = date.today() - timedelta(days=1)

        # Try to fetch from Gold layer first
        stored_metrics = self.storage.read_gold_metrics(
            metric_names=None,
            start_date=str(yesterday),
            end_date=str(yesterday),
        )

        if stored_metrics:
            self.logger.info(
                "current_state_retrieved_from_storage",
                metric_date=str(yesterday),
            )
            # Reconstruct metrics dict from stored format
            return self._reconstruct_metrics_from_storage(stored_metrics)
        else:
            # Compute fresh metrics
            self.logger.info(
                "computing_current_state",
                metric_date=str(yesterday),
            )
            return self.compute_daily_metrics(yesterday)

    def get_health_summary(self) -> dict:
        """
        Aggregate health across all three domains into a dashboard summary.

        Computes current state and classifies health status for each domain
        based on predefined thresholds. Returns actionable dashboard view.

        Returns:
            Dictionary containing health summary:
            {
                "financial": {
                    "status": "healthy|degraded|critical",
                    "score": 0.0-1.0,
                    "key_metrics": {...}
                },
                "operational": {...},
                "customer": {...},
                "overall": {
                    "status": "healthy|degraded|critical",
                    "score": 0.0-1.0,
                    "active_incidents": 0,
                    "active_monitors": 0
                }
            }

        Example:
            >>> summary = builder.get_health_summary()
            >>> print(f"Overall health: {summary['overall']['status']}")
        """
        # Get current state
        state = self.get_current_state()

        # Classify domain health
        financial_health = self._classify_domain_health(
            state["financial"], "financial"
        )
        operational_health = self._classify_domain_health(
            state["operational"], "operational"
        )
        customer_health = self._classify_domain_health(
            state["customer"], "customer"
        )

        # Compute overall health (worst of three domains)
        domain_scores = [
            financial_health["score"],
            operational_health["score"],
            customer_health["score"],
        ]
        overall_score = min(domain_scores)

        # Map score to status
        if overall_score >= 0.8:
            overall_status = "healthy"
        elif overall_score >= 0.6:
            overall_status = "degraded"
        else:
            overall_status = "critical"

        # Count active incidents and monitors (query storage)
        active_incidents = len(
            self.storage.read_incidents(status="open")
        )
        active_monitors = len(
            self.storage.read_monitors(enabled=True)
        )

        result = {
            "financial": financial_health,
            "operational": operational_health,
            "customer": customer_health,
            "overall": {
                "status": overall_status,
                "score": round(overall_score, 4),
                "active_incidents": active_incidents,
                "active_monitors": active_monitors,
            },
            "metadata": {
                "computed_at": datetime.utcnow().isoformat(),
                "metric_date": state["metric_date"],
            },
        }

        self.logger.info(
            "health_summary_computed",
            overall_status=overall_status,
            overall_score=overall_score,
            active_incidents=active_incidents,
        )

        return result

    # =========================================================================
    # Domain Metric Computations
    # =========================================================================

    def _compute_financial_metrics(
        self, target_date: date, events: list[CanonicalEvent]
    ) -> dict:
        """
        Compute all 11 financial health metrics.

        Metrics:
        - daily_revenue: Revenue from invoices paid and payments received
        - daily_expenses: Expenses posted
        - daily_refunds: Refunds and credit memos issued
        - refund_rate: 7-day rolling refunds / revenue
        - net_cash_proxy: 30-day rolling revenue - expenses - refunds
        - expense_ratio: 7-day rolling expenses / revenue
        - margin_proxy: 7-day rolling (revenue - expenses - refunds) / revenue
        - dso_proxy: Average days to pay for invoices in window
        - ar_aging_amount: Total overdue invoice balances
        - ar_overdue_count: Count of overdue invoices
        - dpo_proxy: Average days from expense posted to paid

        Args:
            target_date: Date to compute metrics for
            events: All events in required time window

        Returns:
            Dictionary of financial metrics
        """
        # Filter events by date ranges
        target_events = self._filter_events_by_date(events, target_date, target_date)
        window_7d = self._filter_events_by_date(
            events, target_date - timedelta(days=6), target_date
        )
        window_30d = self._filter_events_by_date(
            events, target_date - timedelta(days=29), target_date
        )

        # Daily revenue (INVOICE_PAID + PAYMENT_RECEIVED)
        revenue_events = [
            e for e in target_events
            if e.event_type in [EventType.INVOICE_PAID, EventType.PAYMENT_RECEIVED]
        ]
        daily_revenue = sum(e.amount or 0.0 for e in revenue_events)

        # Daily expenses (EXPENSE_POSTED)
        expense_events = [
            e for e in target_events
            if e.event_type == EventType.EXPENSE_POSTED
        ]
        daily_expenses = sum(e.amount or 0.0 for e in expense_events)

        # Daily refunds (REFUND_ISSUED + CREDIT_MEMO_ISSUED)
        refund_events = [
            e for e in target_events
            if e.event_type in [EventType.REFUND_ISSUED, EventType.CREDIT_MEMO_ISSUED]
        ]
        daily_refunds = sum(e.amount or 0.0 for e in refund_events)

        # 7-day refund rate
        revenue_7d = sum(
            e.amount or 0.0 for e in window_7d
            if e.event_type in [EventType.INVOICE_PAID, EventType.PAYMENT_RECEIVED]
        )
        refunds_7d = sum(
            e.amount or 0.0 for e in window_7d
            if e.event_type in [EventType.REFUND_ISSUED, EventType.CREDIT_MEMO_ISSUED]
        )
        refund_rate = self._safe_divide(refunds_7d, revenue_7d)

        # 30-day net cash proxy
        revenue_30d = sum(
            e.amount or 0.0 for e in window_30d
            if e.event_type in [EventType.INVOICE_PAID, EventType.PAYMENT_RECEIVED]
        )
        expenses_30d = sum(
            e.amount or 0.0 for e in window_30d
            if e.event_type == EventType.EXPENSE_POSTED
        )
        refunds_30d = sum(
            e.amount or 0.0 for e in window_30d
            if e.event_type in [EventType.REFUND_ISSUED, EventType.CREDIT_MEMO_ISSUED]
        )
        net_cash_proxy = revenue_30d - expenses_30d - refunds_30d

        # 7-day expense ratio
        expenses_7d = sum(
            e.amount or 0.0 for e in window_7d
            if e.event_type == EventType.EXPENSE_POSTED
        )
        expense_ratio = self._safe_divide(expenses_7d, revenue_7d)

        # 7-day margin proxy
        margin_numerator = revenue_7d - expenses_7d - refunds_7d
        margin_proxy = self._safe_divide(margin_numerator, revenue_7d)

        # DSO proxy (average days to pay for invoices paid in 30-day window)
        paid_invoices = [
            e for e in window_30d
            if e.event_type == EventType.INVOICE_PAID
            and e.attributes.get("days_to_pay") is not None
        ]
        days_to_pay_list = [
            e.attributes["days_to_pay"] for e in paid_invoices
        ]
        dso_proxy = self._compute_rolling_average(days_to_pay_list, len(days_to_pay_list))

        # AR aging (overdue invoices on target date)
        overdue_events = [
            e for e in target_events
            if e.event_type == EventType.INVOICE_OVERDUE
        ]
        ar_aging_amount = sum(
            e.attributes.get("balance_remaining", 0.0) for e in overdue_events
        )
        ar_overdue_count = len(overdue_events)

        # DPO proxy (average days from EXPENSE_POSTED to EXPENSE_PAID in 30-day window)
        # This requires matching posted to paid events - simplified here
        expense_paid_events = [
            e for e in window_30d
            if e.event_type == EventType.EXPENSE_PAID
        ]
        # Estimate DPO as 30 days (would need event linkage for accuracy)
        dpo_proxy = 30.0 if expense_paid_events else 0.0

        return {
            "daily_revenue": round(daily_revenue, 2),
            "daily_expenses": round(daily_expenses, 2),
            "daily_refunds": round(daily_refunds, 2),
            "refund_rate": round(refund_rate, 4),
            "net_cash_proxy": round(net_cash_proxy, 2),
            "expense_ratio": round(expense_ratio, 4),
            "margin_proxy": round(margin_proxy, 4),
            "dso_proxy": round(dso_proxy, 2) if dso_proxy is not None else None,
            "ar_aging_amount": round(ar_aging_amount, 2),
            "ar_overdue_count": ar_overdue_count,
            "dpo_proxy": round(dpo_proxy, 2),
        }

    def _compute_operational_metrics(
        self, target_date: date, events: list[CanonicalEvent]
    ) -> dict:
        """
        Compute all 8 operational health metrics.

        Metrics:
        - order_volume: Count of ORDER_PLACED events
        - delivery_count: Count of ORDER_DELIVERED events
        - late_delivery_count: Count of ORDER_LATE events
        - delivery_delay_rate: 7-day rolling late / delivered
        - fulfillment_backlog: 14-day rolling placed - delivered
        - avg_delivery_delay_days: Average days late for ORDER_LATE
        - supplier_delay_rate: 7-day rolling SHIPMENT_DELAYED / PURCHASE_ORDER_PLACED
        - supplier_delay_severity: Average days delayed for SHIPMENT_DELAYED

        Args:
            target_date: Date to compute metrics for
            events: All events in required time window

        Returns:
            Dictionary of operational metrics
        """
        # Filter events by date ranges
        target_events = self._filter_events_by_date(events, target_date, target_date)
        window_7d = self._filter_events_by_date(
            events, target_date - timedelta(days=6), target_date
        )
        window_14d = self._filter_events_by_date(
            events, target_date - timedelta(days=13), target_date
        )

        # Order volume
        order_volume = len([
            e for e in target_events
            if e.event_type == EventType.ORDER_PLACED
        ])

        # Delivery count
        delivery_count = len([
            e for e in target_events
            if e.event_type == EventType.ORDER_DELIVERED
        ])

        # Late delivery count
        late_delivery_count = len([
            e for e in target_events
            if e.event_type == EventType.ORDER_LATE
        ])

        # 7-day delivery delay rate
        deliveries_7d = len([
            e for e in window_7d
            if e.event_type == EventType.ORDER_DELIVERED
        ])
        late_7d = len([
            e for e in window_7d
            if e.event_type == EventType.ORDER_LATE
        ])
        delivery_delay_rate = self._safe_divide(late_7d, deliveries_7d)

        # 14-day fulfillment backlog (cumulative placed - delivered)
        placed_14d = len([
            e for e in window_14d
            if e.event_type == EventType.ORDER_PLACED
        ])
        delivered_14d = len([
            e for e in window_14d
            if e.event_type == EventType.ORDER_DELIVERED
        ])
        fulfillment_backlog = max(0, placed_14d - delivered_14d)

        # Average delivery delay days
        late_events = [
            e for e in target_events
            if e.event_type == EventType.ORDER_LATE
            and e.attributes.get("days_late") is not None
        ]
        days_late_list = [e.attributes["days_late"] for e in late_events]
        avg_delivery_delay_days = self._compute_rolling_average(
            days_late_list, len(days_late_list)
        )

        # 7-day supplier delay rate
        po_placed_7d = len([
            e for e in window_7d
            if e.event_type == EventType.PURCHASE_ORDER_PLACED
        ])
        shipment_delayed_7d = len([
            e for e in window_7d
            if e.event_type == EventType.SHIPMENT_DELAYED
        ])
        supplier_delay_rate = self._safe_divide(shipment_delayed_7d, po_placed_7d)

        # Supplier delay severity
        delayed_events = [
            e for e in target_events
            if e.event_type == EventType.SHIPMENT_DELAYED
            and e.attributes.get("days_delayed") is not None
        ]
        days_delayed_list = [e.attributes["days_delayed"] for e in delayed_events]
        supplier_delay_severity = self._compute_rolling_average(
            days_delayed_list, len(days_delayed_list)
        )

        return {
            "order_volume": order_volume,
            "delivery_count": delivery_count,
            "late_delivery_count": late_delivery_count,
            "delivery_delay_rate": round(delivery_delay_rate, 4),
            "fulfillment_backlog": fulfillment_backlog,
            "avg_delivery_delay_days": round(avg_delivery_delay_days, 2) if avg_delivery_delay_days is not None else None,
            "supplier_delay_rate": round(supplier_delay_rate, 4),
            "supplier_delay_severity": round(supplier_delay_severity, 2) if supplier_delay_severity is not None else None,
        }

    def _compute_customer_metrics(
        self, target_date: date, events: list[CanonicalEvent]
    ) -> dict:
        """
        Compute all 8 customer health metrics.

        Metrics:
        - ticket_volume: Count of SUPPORT_TICKET_OPENED
        - ticket_close_volume: Count of SUPPORT_TICKET_CLOSED
        - ticket_backlog: 14-day rolling opened - closed
        - avg_resolution_time: Average resolution time for closed tickets
        - review_score_avg: 7-day rolling average of review scores
        - review_score_trend: 14-day linear regression slope of review scores
        - churn_proxy: 30-day rolling CUSTOMER_CHURNED / total customers
        - customer_concentration: 30-day revenue from top 10% / total revenue

        Args:
            target_date: Date to compute metrics for
            events: All events in required time window

        Returns:
            Dictionary of customer metrics
        """
        # Filter events by date ranges
        target_events = self._filter_events_by_date(events, target_date, target_date)
        window_7d = self._filter_events_by_date(
            events, target_date - timedelta(days=6), target_date
        )
        window_14d = self._filter_events_by_date(
            events, target_date - timedelta(days=13), target_date
        )
        window_30d = self._filter_events_by_date(
            events, target_date - timedelta(days=29), target_date
        )

        # Ticket volume
        ticket_volume = len([
            e for e in target_events
            if e.event_type == EventType.SUPPORT_TICKET_OPENED
        ])

        # Ticket close volume
        ticket_close_volume = len([
            e for e in target_events
            if e.event_type == EventType.SUPPORT_TICKET_CLOSED
        ])

        # 14-day ticket backlog
        opened_14d = len([
            e for e in window_14d
            if e.event_type == EventType.SUPPORT_TICKET_OPENED
        ])
        closed_14d = len([
            e for e in window_14d
            if e.event_type == EventType.SUPPORT_TICKET_CLOSED
        ])
        ticket_backlog = max(0, opened_14d - closed_14d)

        # Average resolution time
        closed_tickets = [
            e for e in target_events
            if e.event_type == EventType.SUPPORT_TICKET_CLOSED
            and e.attributes.get("resolution_time_hours") is not None
        ]
        resolution_times = [e.attributes["resolution_time_hours"] for e in closed_tickets]
        avg_resolution_time = self._compute_rolling_average(
            resolution_times, len(resolution_times)
        )

        # 7-day review score average
        reviews_7d = [
            e for e in window_7d
            if e.event_type == EventType.REVIEW_SUBMITTED
            and e.attributes.get("score") is not None
        ]
        scores_7d = [e.attributes["score"] for e in reviews_7d]
        review_score_avg = self._compute_rolling_average(scores_7d, len(scores_7d))

        # 14-day review score trend (linear regression slope)
        reviews_14d = [
            e for e in window_14d
            if e.event_type == EventType.REVIEW_SUBMITTED
            and e.attributes.get("score") is not None
        ]
        # Group by day and compute daily averages for trend
        daily_scores = self._compute_daily_averages_for_trend(
            reviews_14d, target_date - timedelta(days=13), target_date
        )
        review_score_trend = self._compute_trend_slope(daily_scores, 14)

        # 30-day churn proxy
        churned_30d = len([
            e for e in window_30d
            if e.event_type == EventType.CUSTOMER_CHURNED
        ])
        # Count unique customers in 30-day window
        customer_ids = set()
        for e in window_30d:
            if e.entity_type.value == "customer":
                customer_ids.add(e.entity_id)
            if "customer" in e.related_entity_ids:
                customer_ids.add(e.related_entity_ids["customer"])
        total_customers = max(len(customer_ids), 1)  # Avoid division by zero
        churn_proxy = self._safe_divide(churned_30d, total_customers)

        # 30-day customer concentration
        # Revenue by customer in 30-day window
        revenue_by_customer = {}
        for e in window_30d:
            if e.event_type in [EventType.INVOICE_PAID, EventType.PAYMENT_RECEIVED]:
                customer_id = e.related_entity_ids.get("customer")
                if customer_id and e.amount:
                    revenue_by_customer[customer_id] = revenue_by_customer.get(customer_id, 0.0) + e.amount

        if revenue_by_customer:
            total_revenue = sum(revenue_by_customer.values())
            # Sort by revenue and take top 10%
            sorted_revenues = sorted(revenue_by_customer.values(), reverse=True)
            top_10_pct_count = max(1, len(sorted_revenues) // 10)
            top_revenue = sum(sorted_revenues[:top_10_pct_count])
            customer_concentration = self._safe_divide(top_revenue, total_revenue)
        else:
            customer_concentration = 0.0

        return {
            "ticket_volume": ticket_volume,
            "ticket_close_volume": ticket_close_volume,
            "ticket_backlog": ticket_backlog,
            "avg_resolution_time": round(avg_resolution_time, 2) if avg_resolution_time is not None else None,
            "review_score_avg": round(review_score_avg, 2) if review_score_avg is not None else None,
            "review_score_trend": round(review_score_trend, 4) if review_score_trend is not None else None,
            "churn_proxy": round(churn_proxy, 4),
            "customer_concentration": round(customer_concentration, 4),
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _filter_events_by_date(
        self, events: list[CanonicalEvent], start_date: date, end_date: date
    ) -> list[CanonicalEvent]:
        """
        Filter events to only those within date range (inclusive).

        Args:
            events: List of canonical events
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            Filtered list of events
        """
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        return [
            e for e in events
            if start_dt <= e.event_time <= end_dt
        ]

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide two numbers, returning 0.0 if denominator is zero.

        Args:
            numerator: Numerator value
            denominator: Denominator value

        Returns:
            Division result or 0.0 if denominator is zero
        """
        if denominator == 0.0 or denominator is None:
            return 0.0
        return numerator / denominator

    def _compute_rolling_average(
        self, values: list[float], window: int
    ) -> Optional[float]:
        """
        Compute rolling average over values.

        Args:
            values: List of numeric values
            window: Window size (not used, uses len(values))

        Returns:
            Average value or None if no values
        """
        if not values:
            return None
        return sum(values) / len(values)

    def _compute_trend_slope(
        self, values: list[float], window: int
    ) -> Optional[float]:
        """
        Compute linear regression slope over values.

        Uses scipy.stats.linregress to compute the trend slope.
        Positive slope indicates improving trend, negative indicates declining.

        Args:
            values: List of numeric values (daily averages)
            window: Expected window size (for validation)

        Returns:
            Slope of linear regression or None if insufficient data
        """
        if not values or len(values) < 2:
            return None

        # Create x-axis (day indices)
        x = list(range(len(values)))
        y = values

        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        except Exception as e:
            self.logger.warning(
                "trend_slope_computation_failed",
                error=str(e),
                values_count=len(values),
            )
            return None

    def _compute_daily_averages_for_trend(
        self, events: list[CanonicalEvent], start_date: date, end_date: date
    ) -> list[float]:
        """
        Compute daily average scores for trend analysis.

        Groups events by day and computes average score for each day.
        Missing days are filled with previous day's value or 0.0.

        Args:
            events: List of review events with scores
            start_date: Start of window
            end_date: End of window

        Returns:
            List of daily average scores
        """
        # Group scores by date
        daily_scores: dict[date, list[float]] = {}
        for event in events:
            event_date = event.event_time.date()
            score = event.attributes.get("score")
            if score is not None:
                if event_date not in daily_scores:
                    daily_scores[event_date] = []
                daily_scores[event_date].append(score)

        # Compute daily averages
        result = []
        current = start_date
        last_avg = 3.0  # Default middle score

        while current <= end_date:
            if current in daily_scores:
                avg = sum(daily_scores[current]) / len(daily_scores[current])
                result.append(avg)
                last_avg = avg
            else:
                # Forward fill with last average
                result.append(last_avg)
            current += timedelta(days=1)

        return result

    def _get_events_in_window(
        self,
        event_types: Optional[list[EventType]],
        start: date,
        end: date,
    ) -> list[CanonicalEvent]:
        """
        Fetch events from storage within date window.

        Args:
            event_types: Optional list of event types to filter (None for all)
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of canonical events
        """
        # Convert dates to ISO strings
        start_time = datetime.combine(start, datetime.min.time()).isoformat()
        end_time = datetime.combine(end, datetime.max.time()).isoformat()

        # Fetch events for each type or all if None
        all_events = []

        if event_types is None:
            # Fetch all event types
            events = self.storage.read_canonical_events(
                start_time=start_time,
                end_time=end_time,
                limit=100000,  # High limit for large windows
            )
            all_events.extend(events)
        else:
            # Fetch specific event types
            for event_type in event_types:
                events = self.storage.read_canonical_events(
                    event_type=event_type.value,
                    start_time=start_time,
                    end_time=end_time,
                    limit=100000,
                )
                all_events.extend(events)

        return all_events

    def _classify_domain_health(
        self, metrics: dict, domain: str
    ) -> dict:
        """
        Classify health status for a domain based on thresholds.

        Compares key metrics against predefined thresholds to determine
        if domain is healthy, degraded, or critical.

        Args:
            metrics: Dictionary of domain metrics
            domain: Domain name ("financial", "operational", "customer")

        Returns:
            Dictionary with status, score, and key metrics:
            {
                "status": "healthy|degraded|critical",
                "score": 0.0-1.0,
                "key_metrics": {...}
            }
        """
        # Get domain thresholds
        if domain == "financial":
            thresholds = self.FINANCIAL_THRESHOLDS
            key_metric_names = ["margin_proxy", "expense_ratio", "refund_rate"]
        elif domain == "operational":
            thresholds = self.OPERATIONAL_THRESHOLDS
            key_metric_names = ["delivery_delay_rate", "fulfillment_backlog"]
        elif domain == "customer":
            thresholds = self.CUSTOMER_THRESHOLDS
            key_metric_names = ["review_score_avg", "ticket_backlog"]
        else:
            return {
                "status": "healthy",
                "score": 1.0,
                "key_metrics": {},
            }

        # Extract key metrics
        key_metrics = {
            name: metrics.get(name)
            for name in key_metric_names
        }

        # Evaluate health status
        # Start with healthy assumption
        status = "healthy"
        score = 1.0

        # Check critical thresholds
        critical_threshold = thresholds["critical"]
        degraded_threshold = thresholds["degraded"]

        violations = 0
        for metric_name, threshold_value in critical_threshold.items():
            actual_value = metrics.get(metric_name)
            if actual_value is None:
                continue

            # Different comparison logic per metric type
            if metric_name in ["margin_proxy", "review_score_avg"]:
                # Lower is worse
                if actual_value < threshold_value:
                    status = "critical"
                    violations += 1
            else:
                # Higher is worse
                if actual_value > threshold_value:
                    status = "critical"
                    violations += 1

        # Check degraded thresholds if not already critical
        if status != "critical":
            for metric_name, threshold_value in degraded_threshold.items():
                actual_value = metrics.get(metric_name)
                if actual_value is None:
                    continue

                if metric_name in ["margin_proxy", "review_score_avg"]:
                    if actual_value < threshold_value:
                        status = "degraded"
                        violations += 1
                else:
                    if actual_value > threshold_value:
                        status = "degraded"
                        violations += 1

        # Compute score (1.0 = healthy, 0.5 = degraded, 0.0 = critical)
        if status == "healthy":
            score = 1.0
        elif status == "degraded":
            score = 0.7
        else:  # critical
            score = 0.3

        return {
            "status": status,
            "score": round(score, 4),
            "key_metrics": key_metrics,
        }

    def _reconstruct_metrics_from_storage(
        self, stored_metrics: list[dict]
    ) -> dict:
        """
        Reconstruct metrics dictionary from stored flat format.

        Gold layer stores metrics as flat list of dicts. This method
        reconstructs the structured format with domains.

        Args:
            stored_metrics: List of metric dicts from storage

        Returns:
            Structured metrics dictionary
        """
        # Group metrics by domain
        financial = {}
        operational = {}
        customer = {}
        metadata = {}
        metric_date = str(date.today())

        for metric in stored_metrics:
            metric_name = metric.get("metric_name")
            metric_value = metric.get("metric_value")
            metric_date = metric.get("metric_date")

            # Classify into domain based on metric name
            if metric_name in [
                "daily_revenue", "daily_expenses", "daily_refunds", "refund_rate",
                "net_cash_proxy", "expense_ratio", "margin_proxy", "dso_proxy",
                "ar_aging_amount", "ar_overdue_count", "dpo_proxy"
            ]:
                financial[metric_name] = metric_value
            elif metric_name in [
                "order_volume", "delivery_count", "late_delivery_count",
                "delivery_delay_rate", "fulfillment_backlog", "avg_delivery_delay_days",
                "supplier_delay_rate", "supplier_delay_severity"
            ]:
                operational[metric_name] = metric_value
            elif metric_name in [
                "ticket_volume", "ticket_close_volume", "ticket_backlog",
                "avg_resolution_time", "review_score_avg", "review_score_trend",
                "churn_proxy", "customer_concentration"
            ]:
                customer[metric_name] = metric_value

        return {
            "metric_date": metric_date or str(date.today()),
            "financial": financial,
            "operational": operational,
            "customer": customer,
            "metadata": {
                "computed_at": datetime.utcnow().isoformat(),
                "events_processed": 0,
                "metric_schema_version": self.METRIC_SCHEMA_VERSION,
            },
        }
