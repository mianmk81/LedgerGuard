"""
State Builder Demo - Gold Layer Metrics Computation

Demonstrates the StateBuilder computing all 27 daily business health metrics
from canonical Silver layer events.

This script shows:
1. Creating synthetic canonical events for demonstration
2. Computing daily metrics for a target date
3. Computing metrics for a date range (backfill)
4. Getting current business state
5. Generating health summary dashboard

Usage:
    python demo_state_builder.py
"""

from datetime import date, datetime, timedelta
from uuid import uuid4

from api.engine.state_builder import StateBuilder
from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent
from api.storage.duckdb_storage import DuckDBStorage


def create_synthetic_events(target_date: date, num_days: int = 30) -> list[CanonicalEvent]:
    """
    Create synthetic canonical events for demonstration.

    Generates realistic business events across all three domains:
    - Financial: invoices, payments, expenses, refunds
    - Operational: orders, deliveries, supplier shipments
    - Customer: support tickets, reviews, churn

    Args:
        target_date: Target date to generate events around
        num_days: Number of days of historical data to generate

    Returns:
        List of canonical events
    """
    events = []
    start_date = target_date - timedelta(days=num_days - 1)

    print(f"Generating synthetic events from {start_date} to {target_date}...")

    for day_offset in range(num_days):
        current_date = start_date + timedelta(days=day_offset)
        event_time = datetime.combine(current_date, datetime.min.time())

        # Financial events (10-20 per day)
        # Invoices paid
        for i in range(10, 15):
            events.append(CanonicalEvent(
                event_type=EventType.INVOICE_PAID,
                event_time=event_time + timedelta(hours=i),
                source="qbo",
                source_entity_id=f"INV-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.INVOICE,
                entity_id=f"invoice:qbo:INV-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"customer": f"customer:qbo:CUST-{i % 10}"},
                amount=1000.0 + (i * 100),
                currency="USD",
                attributes={
                    "days_to_pay": 15 + (i % 5),
                    "payment_method": "credit_card",
                },
            ))

        # Expenses posted
        for i in range(5, 8):
            events.append(CanonicalEvent(
                event_type=EventType.EXPENSE_POSTED,
                event_time=event_time + timedelta(hours=i),
                source="qbo",
                source_entity_id=f"BILL-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.EXPENSE,
                entity_id=f"expense:qbo:BILL-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"vendor": f"vendor:qbo:VEND-{i % 3}"},
                amount=500.0 + (i * 50),
                currency="USD",
                attributes={"category": "operations"},
            ))

        # Refunds (spike on certain days for testing)
        refund_count = 5 if day_offset % 7 == 0 else 1
        for i in range(refund_count):
            events.append(CanonicalEvent(
                event_type=EventType.REFUND_ISSUED,
                event_time=event_time + timedelta(hours=14 + i),
                source="qbo",
                source_entity_id=f"REFUND-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.PAYMENT,
                entity_id=f"payment:qbo:REFUND-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"customer": f"customer:qbo:CUST-{i % 10}"},
                amount=150.0 + (i * 25),
                currency="USD",
            ))

        # Overdue invoices
        if day_offset % 3 == 0:
            for i in range(2):
                events.append(CanonicalEvent(
                    event_type=EventType.INVOICE_OVERDUE,
                    event_time=event_time + timedelta(hours=10 + i),
                    source="qbo",
                    source_entity_id=f"OVERDUE-{current_date.strftime('%Y%m%d')}-{i}",
                    entity_type=EntityType.INVOICE,
                    entity_id=f"invoice:qbo:OVERDUE-{current_date.strftime('%Y%m%d')}-{i}",
                    related_entity_ids={"customer": f"customer:qbo:CUST-{i % 10}"},
                    amount=0.0,
                    attributes={
                        "balance_remaining": 500.0 + (i * 100),
                        "days_overdue": 10 + i,
                    },
                ))

        # Operational events (15-25 per day)
        # Orders placed
        for i in range(20):
            events.append(CanonicalEvent(
                event_type=EventType.ORDER_PLACED,
                event_time=event_time + timedelta(hours=8 + (i % 12)),
                source="shopify",
                source_entity_id=f"ORD-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.ORDER,
                entity_id=f"order:shopify:ORD-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"customer": f"customer:shopify:CUST-{i % 15}"},
                amount=200.0 + (i * 10),
            ))

        # Orders delivered (some delayed)
        for i in range(18):
            events.append(CanonicalEvent(
                event_type=EventType.ORDER_DELIVERED,
                event_time=event_time + timedelta(hours=16 + (i % 6)),
                source="shopify",
                source_entity_id=f"ORD-{(current_date - timedelta(days=2)).strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.ORDER,
                entity_id=f"order:shopify:ORD-{(current_date - timedelta(days=2)).strftime('%Y%m%d')}-{i}",
            ))

        # Late deliveries
        if day_offset % 5 == 0:
            for i in range(3):
                events.append(CanonicalEvent(
                    event_type=EventType.ORDER_LATE,
                    event_time=event_time + timedelta(hours=18 + i),
                    source="shopify",
                    source_entity_id=f"LATE-{current_date.strftime('%Y%m%d')}-{i}",
                    entity_type=EntityType.ORDER,
                    entity_id=f"order:shopify:LATE-{current_date.strftime('%Y%m%d')}-{i}",
                    attributes={"days_late": 2 + i},
                ))

        # Purchase orders
        for i in range(3):
            events.append(CanonicalEvent(
                event_type=EventType.PURCHASE_ORDER_PLACED,
                event_time=event_time + timedelta(hours=9 + i),
                source="qbo",
                source_entity_id=f"PO-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.ORDER,
                entity_id=f"order:qbo:PO-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"vendor": f"vendor:qbo:VEND-{i % 3}"},
                amount=2000.0 + (i * 500),
            ))

        # Shipment delays
        if day_offset % 4 == 0:
            events.append(CanonicalEvent(
                event_type=EventType.SHIPMENT_DELAYED,
                event_time=event_time + timedelta(hours=15),
                source="fedex",
                source_entity_id=f"SHIP-{current_date.strftime('%Y%m%d')}-DELAYED",
                entity_type=EntityType.ORDER,
                entity_id=f"order:fedex:SHIP-{current_date.strftime('%Y%m%d')}-DELAYED",
                attributes={"days_delayed": 3},
            ))

        # Customer events (10-15 per day)
        # Support tickets opened
        for i in range(8):
            events.append(CanonicalEvent(
                event_type=EventType.SUPPORT_TICKET_OPENED,
                event_time=event_time + timedelta(hours=9 + i),
                source="zendesk",
                source_entity_id=f"TICKET-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.TICKET,
                entity_id=f"ticket:zendesk:TICKET-{current_date.strftime('%Y%m%d')}-{i}",
                related_entity_ids={"customer": f"customer:qbo:CUST-{i % 10}"},
            ))

        # Support tickets closed
        for i in range(6):
            events.append(CanonicalEvent(
                event_type=EventType.SUPPORT_TICKET_CLOSED,
                event_time=event_time + timedelta(hours=15 + i),
                source="zendesk",
                source_entity_id=f"TICKET-{(current_date - timedelta(days=1)).strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.TICKET,
                entity_id=f"ticket:zendesk:TICKET-{(current_date - timedelta(days=1)).strftime('%Y%m%d')}-{i}",
                attributes={"resolution_time_hours": 24.0 + (i * 4)},
            ))

        # Reviews submitted (varying quality)
        for i in range(5):
            # Score degrades over time for trend testing
            base_score = 4.5 - (day_offset * 0.02)
            score = max(1.0, min(5.0, base_score + (i * 0.1 - 0.2)))
            events.append(CanonicalEvent(
                event_type=EventType.REVIEW_SUBMITTED,
                event_time=event_time + timedelta(hours=17 + i),
                source="trustpilot",
                source_entity_id=f"REVIEW-{current_date.strftime('%Y%m%d')}-{i}",
                entity_type=EntityType.CUSTOMER,
                entity_id=f"customer:qbo:CUST-{i % 10}",
                attributes={"score": round(score, 1)},
            ))

        # Customer churn (occasional)
        if day_offset % 10 == 0:
            events.append(CanonicalEvent(
                event_type=EventType.CUSTOMER_CHURNED,
                event_time=event_time + timedelta(hours=23),
                source="qbo",
                source_entity_id=f"CHURN-{current_date.strftime('%Y%m%d')}",
                entity_type=EntityType.CUSTOMER,
                entity_id=f"customer:qbo:CUST-CHURNED-{day_offset}",
            ))

    print(f"Generated {len(events)} synthetic events")
    return events


def print_metrics_summary(metrics: dict):
    """
    Pretty print metrics summary.

    Args:
        metrics: Metrics dictionary from StateBuilder
    """
    print("\n" + "=" * 80)
    print(f"METRICS SUMMARY FOR {metrics['metric_date']}")
    print("=" * 80)

    print("\nFINANCIAL HEALTH (11 metrics)")
    print("-" * 80)
    financial = metrics["financial"]
    print(f"  Daily Revenue:        ${financial['daily_revenue']:>12,.2f}")
    print(f"  Daily Expenses:       ${financial['daily_expenses']:>12,.2f}")
    print(f"  Daily Refunds:        ${financial['daily_refunds']:>12,.2f}")
    print(f"  Refund Rate (7d):      {financial['refund_rate']:>11.2%}")
    print(f"  Net Cash Proxy (30d): ${financial['net_cash_proxy']:>12,.2f}")
    print(f"  Expense Ratio (7d):    {financial['expense_ratio']:>11.2%}")
    print(f"  Margin Proxy (7d):     {financial['margin_proxy']:>11.2%}")
    dso = financial['dso_proxy']
    print(f"  DSO Proxy:             {dso if dso is None else f'{dso:.1f} days':>12}")
    print(f"  AR Aging Amount:      ${financial['ar_aging_amount']:>12,.2f}")
    print(f"  AR Overdue Count:      {financial['ar_overdue_count']:>12}")
    print(f"  DPO Proxy:             {financial['dpo_proxy']:>9.1f} days")

    print("\nOPERATIONAL HEALTH (8 metrics)")
    print("-" * 80)
    operational = metrics["operational"]
    print(f"  Order Volume:          {operational['order_volume']:>12}")
    print(f"  Delivery Count:        {operational['delivery_count']:>12}")
    print(f"  Late Delivery Count:   {operational['late_delivery_count']:>12}")
    print(f"  Delivery Delay Rate:   {operational['delivery_delay_rate']:>11.2%}")
    print(f"  Fulfillment Backlog:   {operational['fulfillment_backlog']:>12}")
    delay_days = operational['avg_delivery_delay_days']
    print(f"  Avg Delivery Delay:    {delay_days if delay_days is None else f'{delay_days:.1f} days':>12}")
    print(f"  Supplier Delay Rate:   {operational['supplier_delay_rate']:>11.2%}")
    severity = operational['supplier_delay_severity']
    print(f"  Supplier Delay Sev:    {severity if severity is None else f'{severity:.1f} days':>12}")

    print("\nCUSTOMER HEALTH (8 metrics)")
    print("-" * 80)
    customer = metrics["customer"]
    print(f"  Ticket Volume:         {customer['ticket_volume']:>12}")
    print(f"  Ticket Close Volume:   {customer['ticket_close_volume']:>12}")
    print(f"  Ticket Backlog:        {customer['ticket_backlog']:>12}")
    res_time = customer['avg_resolution_time']
    print(f"  Avg Resolution Time:   {res_time if res_time is None else f'{res_time:.1f} hours':>12}")
    score = customer['review_score_avg']
    print(f"  Review Score Avg:      {score if score is None else f'{score:.2f}/5.00':>12}")
    trend = customer['review_score_trend']
    trend_str = "N/A" if trend is None else f"{trend:+.4f}"
    print(f"  Review Score Trend:    {trend_str:>12}")
    print(f"  Churn Proxy (30d):     {customer['churn_proxy']:>11.2%}")
    print(f"  Customer Concentration:{customer['customer_concentration']:>11.2%}")

    print("\nMETADATA")
    print("-" * 80)
    metadata = metrics["metadata"]
    print(f"  Computed At:           {metadata['computed_at']}")
    print(f"  Events Processed:      {metadata['events_processed']:>12}")
    print(f"  Schema Version:        {metadata['metric_schema_version']:>12}")
    print("=" * 80 + "\n")


def print_health_summary(summary: dict):
    """
    Pretty print health summary.

    Args:
        summary: Health summary from get_health_summary()
    """
    print("\n" + "=" * 80)
    print("BUSINESS HEALTH DASHBOARD")
    print("=" * 80)

    # Overall health
    overall = summary["overall"]
    status_emoji = {
        "healthy": "✓",
        "degraded": "⚠",
        "critical": "✗",
    }
    print(f"\nOVERALL STATUS: {overall['status'].upper()} {status_emoji.get(overall['status'], '')}")
    print(f"Overall Score:     {overall['score']:.2f}/1.00")
    print(f"Active Incidents:  {overall['active_incidents']}")
    print(f"Active Monitors:   {overall['active_monitors']}")

    # Domain health
    for domain_name in ["financial", "operational", "customer"]:
        domain = summary[domain_name]
        print(f"\n{domain_name.upper()} DOMAIN: {domain['status'].upper()} {status_emoji.get(domain['status'], '')}")
        print(f"  Score: {domain['score']:.2f}/1.00")
        print(f"  Key Metrics:")
        for metric_name, metric_value in domain["key_metrics"].items():
            if metric_value is not None:
                if isinstance(metric_value, float) and metric_value < 1.0:
                    print(f"    {metric_name}: {metric_value:.2%}")
                else:
                    print(f"    {metric_name}: {metric_value}")
            else:
                print(f"    {metric_name}: N/A")

    print("\nMETADATA")
    print("-" * 80)
    metadata = summary["metadata"]
    print(f"  Computed At:  {metadata['computed_at']}")
    print(f"  Metric Date:  {metadata['metric_date']}")
    print("=" * 80 + "\n")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("STATE BUILDER DEMO - Gold Layer Metrics Computation")
    print("=" * 80 + "\n")

    # Initialize storage and state builder
    print("Initializing DuckDB storage and StateBuilder...")
    storage = DuckDBStorage(db_path=":memory:")  # In-memory for demo
    builder = StateBuilder(storage=storage)

    # Generate synthetic events
    target_date = date.today() - timedelta(days=1)
    events = create_synthetic_events(target_date, num_days=30)

    # Write events to Silver layer
    print(f"\nWriting {len(events)} events to Silver layer...")
    storage.write_canonical_events(events)

    # Demo 1: Compute daily metrics for target date
    print("\n" + "=" * 80)
    print("DEMO 1: Compute Daily Metrics for Single Date")
    print("=" * 80)
    metrics = builder.compute_daily_metrics(target_date)
    print_metrics_summary(metrics)

    # Demo 2: Compute date range (backfill)
    print("\n" + "=" * 80)
    print("DEMO 2: Compute Date Range (Backfill)")
    print("=" * 80)
    start_date = target_date - timedelta(days=6)
    end_date = target_date
    print(f"\nComputing metrics for {start_date} to {end_date} (7 days)...")
    metrics_range = builder.compute_date_range(start_date, end_date)
    print(f"Computed metrics for {len(metrics_range)} days")

    # Show first and last day
    print("\nFirst day metrics:")
    print_metrics_summary(metrics_range[0])

    print("\nLast day metrics:")
    print_metrics_summary(metrics_range[-1])

    # Demo 3: Get current state
    print("\n" + "=" * 80)
    print("DEMO 3: Get Current Business State")
    print("=" * 80)
    current_state = builder.get_current_state()
    print_metrics_summary(current_state)

    # Demo 4: Get health summary
    print("\n" + "=" * 80)
    print("DEMO 4: Get Health Summary Dashboard")
    print("=" * 80)
    health_summary = builder.get_health_summary()
    print_health_summary(health_summary)

    # Write metrics to Gold layer
    print("\n" + "=" * 80)
    print("Writing Metrics to Gold Layer")
    print("=" * 80)

    # Flatten metrics for storage
    flat_metrics = []
    for day_metrics in metrics_range:
        metric_date = day_metrics["metric_date"]

        # Financial metrics
        for metric_name, metric_value in day_metrics["financial"].items():
            flat_metrics.append({
                "metric_name": metric_name,
                "metric_date": metric_date,
                "metric_value": metric_value or 0.0,
                "aggregation_period": "daily",
                "metadata": {"domain": "financial"},
            })

        # Operational metrics
        for metric_name, metric_value in day_metrics["operational"].items():
            flat_metrics.append({
                "metric_name": metric_name,
                "metric_date": metric_date,
                "metric_value": metric_value or 0.0,
                "aggregation_period": "daily",
                "metadata": {"domain": "operational"},
            })

        # Customer metrics
        for metric_name, metric_value in day_metrics["customer"].items():
            flat_metrics.append({
                "metric_name": metric_name,
                "metric_date": metric_date,
                "metric_value": metric_value or 0.0,
                "aggregation_period": "daily",
                "metadata": {"domain": "customer"},
            })

    written_count = storage.write_gold_metrics(flat_metrics)
    print(f"Wrote {written_count} metric records to Gold layer")

    # Verify read back
    print("\nReading back from Gold layer...")
    read_metrics = storage.read_gold_metrics(
        start_date=str(start_date),
        end_date=str(end_date),
    )
    print(f"Read {len(read_metrics)} metric records from Gold layer")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. StateBuilder computes 27 daily metrics across 3 domains")
    print("2. Metrics use point-in-time, rolling window, and trend computations")
    print("3. Health summary classifies business status using thresholds")
    print("4. All metrics stored in Gold layer for incident detection")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
