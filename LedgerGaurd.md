# Business Reliability Engine — Complete Principal-Level System Spec v2.0

## Written as a senior staff/principal engineer with 15 years at Amazon/Netflix/Google would write an internal design doc

---

# SECTION 0: EXECUTIVE SUMMARY

## What you are building

A deployable reliability platform for businesses that connects to QuickBooks (Intuit), ingests business activity into a governed Databricks lakehouse, normalizes it into a canonical event stream, detects cross-domain business incidents (financial, operational, customer), reconstructs evidence-linked root-cause chains using a formalized scoring algorithm called **BRE-RCA (Business Reliability Engine Root Cause Attribution)**, generates postmortems with blast radius calculations, produces prevention monitors that actually run and catch recurrences, supports what-if simulation, incident cascade correlation, incident comparison across time periods, and exposes all of this through a clean admin UI with an interactive causal chain visualization as the centerpiece.

## One sentence for judges

> "I built a Business Reliability Engine: an event-sourced platform with a formalized causal chain extraction algorithm, closed-loop prevention monitors, cascade detection, blast radius scoring, and what-if simulation — deployed on Databricks with Intuit QuickBooks as the live data source."

---

# SECTION 1: THE PROBLEM

Most business tooling operates at four levels, and almost all tools stop at level 1 or 2:

**Level 1 — Descriptive:** "Your profit is down 12% this month." (Every dashboard does this.)

**Level 2 — Predictive:** "Cash might be low next month based on trends." (Some forecasting tools do this.)

**Level 3 — Diagnostic:** "This specific chain of events caused the profit drop: a supplier delay on Feb 3 led to stockouts on 14 SKUs, which caused 340 customers to request refunds between Feb 8-15, which compressed your margin by 8 points." (Almost nothing does this.)

**Level 4 — Preventive:** "Here are three monitors that would have caught this cascade at step 1, before the damage propagated. Monitor A is now running and will alert you if this pattern begins again." (Nothing does this.)

Your system operates at all four levels. The innovation is not any single ML model — it is the **system design** that connects event sourcing, incident detection, root cause attribution, postmortem generation, and closed-loop prevention into a single workflow.

### Real cascading failure examples this system catches

- Marketing campaign spike → fulfillment backlog → delivery delays → support ticket surge → review score collapse → churn acceleration → revenue drop
- Supplier delay → stockouts on key SKUs → repeat customer dropout → revenue concentration risk → cash flow stress
- AR delays (invoices not paid on time) → cash position weakens → can't pay vendors on time → vendor relationship risk → supply disruption
- Product quality regression → refund spike → margin compression → expense ratio drift → liquidity crunch risk
- Support team overwhelmed → ticket backlog grows → resolution time increases → customer satisfaction drops → churn → revenue loss

---

# SECTION 2: WHAT THE SYSTEM DOES (Inputs → Outputs)

## Inputs

1. **Primary: QuickBooks Online data** via Intuit OAuth + API + Webhooks (Invoices, Payments, Expenses, Customers, Items, Vendors, Purchase Orders, Credit Memos, Refund Receipts)
2. **Secondary: Supplemental datasets** for domains QBO doesn't cover well (support tickets, logistics, churn labels) — ingested via CSV/JSON/Parquet upload
3. **Tertiary (optional future): External context** — holiday calendars, macro indicators, industry benchmarks

## Outputs (8 distinct output types)

1. **Canonical Event Stream** — every business activity normalized into a single schema
2. **Business State Views** — daily/weekly health metrics across financial, operational, and customer domains
3. **Incidents** — detected anomalies with type, severity, confidence, evidence links, and time range
4. **Incident Cascades** — groups of related incidents linked by temporal proximity and shared entities
5. **Root Cause Chains** — ranked causal paths from the BRE-RCA algorithm with contribution scores and evidence event IDs
6. **Blast Radius Reports** — quantified downstream impact (entities affected, dollar exposure, customer count)
7. **Postmortems** — structured documents with timeline, contributing factors, blast radius, prevention monitors, and exportable as PDF/JSON/Markdown
8. **Prevention Monitors** — live rules/thresholds generated from the causal chain that actually run against incoming data and fire alerts on recurrence

---

# SECTION 3: CORE DESIGN PRINCIPLES

These are the architectural commitments that make this "production-grade" rather than "hackathon demo":

### 3.1 Event Sourcing
Every piece of business activity is stored as an immutable event. All derived state (metrics, incidents, RCA results) is computed from the event log. This means any analysis is reproducible: given the same events and the same config version, you get the same incidents and the same causal chains. No hidden state. No "it worked yesterday but I don't know why."

### 3.2 Data Contracts
Every event type has a strict Pydantic schema with versioning. If a field is added or changed, the schema version increments. Ingestion rejects events that don't conform. This means the incident engine and RCA never operate on garbage data without knowing it.

### 3.3 Explainability as First-Class Output
Every incident links to the exact metric values and time windows that triggered it. Every causal chain link includes the contribution score formula inputs, the evidence event IDs, and the metric values at the time. A human can independently verify every claim the system makes.

### 3.4 Deterministic Fallbacks
If ML models are uncertain or unavailable, the system still works via robust statistics (MAD z-scores) and rule-based detection. ML is an enhancement layer, not a dependency. The system never says "model failed, no results."

### 3.5 Auditability
Every run produces a `run_manifest` that records: which events were processed, which config/schema versions were used, which detection methods fired, which models (if any) were invoked, and what their confidence was. This is the "audit trail" a principal engineer would demand.

### 3.6 Closed-Loop Prevention
The system doesn't just report — it generates monitors, runs them, and proves they work. This is the SRE principle: every postmortem must produce action items that prevent recurrence, and those action items must be verified.

### 3.7 Data Quality Awareness
Every ingestion run produces a Data Quality Score (completeness, consistency, timeliness). RCA confidence is adjusted based on data quality. If 20% of shipping dates are missing, the system says "logistics-related RCA confidence is reduced" rather than silently producing unreliable results.

---

# SECTION 4: SYSTEM ARCHITECTURE (Complete)

## 4.1 High-Level Data Flow

```
Intuit QuickBooks Online
        │
        ▼
┌─────────────────────┐
│  QBO Connector       │ ← OAuth 2.0 + Webhooks + REST API polling
│  Service (Admin)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Bronze Delta Tables │ ← Raw API payloads, append-only
│  (Databricks)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Canonical Event     │ ← Schema validation + normalization
│  Builder (Silver)    │    QBO objects → CanonicalEvent schema
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Canonical Event     │ ← Immutable event log in Delta
│  Store (Silver)      │    + supplemental dataset events
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  State & Feature     │ ← Daily/weekly business health metrics
│  Builder (Gold)      │    (financial, operational, customer)
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────┐
│  Incident Detection  │          │  What-If Simulation    │
│  Engine              │          │  Engine                │
└────────┬────────────┘          └────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Incident Cascade    │ ← Groups co-occurring incidents
│  Correlator          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Root Cause Analyzer │ ← BRE-RCA algorithm
│  (RCA)               │    (formalized scoring)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Blast Radius        │ ← Downstream impact quantification
│  Calculator          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Postmortem +        │ ← Timeline + narrative + monitors
│  Monitor Generator   │    PDF/JSON/Markdown export
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────────┐          ┌────────────────────────┐
│  Monitor Runtime     │          │  Incident Comparator   │
│  Engine              │          │                        │
│  (closed-loop)       │          └────────────────────────┘
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  API Service         │ ← FastAPI, admin-only auth
│  (FastAPI)           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Admin UI            │ ← Databricks App (React or Streamlit)
│                      │    Causal chain viz as centerpiece
└─────────────────────┘
```

## 4.2 Component-by-Component Detail

---

### COMPONENT A: QBO Connector Service

**What it does:** Handles all communication with Intuit QuickBooks Online. This is the only component that knows about QBO's API shape.

**Tech:**
- Python 3.11+
- `intuitlib` (Intuit's official Python OAuth client)
- `httpx` for async REST calls to QBO API
- Webhook receiver endpoint (FastAPI route)

**How it works:**
1. Admin clicks "Connect QuickBooks" in the UI
2. App redirects to Intuit OAuth 2.0 authorization URL
3. Admin authorizes, Intuit redirects back with auth code
4. App exchanges code for access_token + refresh_token
5. Tokens stored encrypted in Databricks Secrets (or env vars for dev)
6. App registers webhook subscriptions for entity change notifications (Invoice, Payment, Customer, etc.)
7. Two ingestion modes:
   - **Webhook-driven (near real-time):** Intuit sends POST to your webhook endpoint when entities change. App fetches the changed entity via REST API and writes to Bronze.
   - **Batch pull (scheduled):** A scheduled job (Databricks Job or cron) pulls all entities modified since last sync via QBO's CDC query parameter (`?minorversion=65&query=select * from Invoice where Metadata.LastUpdatedTime > 'YYYY-MM-DD'`)

**QBO entities to pull (minimum set):**
- Invoice (→ InvoiceIssued, InvoicePaid, InvoiceOverdue events)
- Payment (→ PaymentReceived event)
- Bill (→ ExpensePosted event)
- BillPayment (→ ExpensePaid event)
- CreditMemo (→ RefundIssued event)
- RefundReceipt (→ RefundIssued event)
- Customer (→ CustomerCreated, CustomerUpdated events)
- Item (→ for product/SKU reference in incidents)
- Vendor (→ for supplier reference in RCA)
- PurchaseOrder (→ PurchaseOrderPlaced event)
- Estimate (→ optional, for pipeline visibility)

**Webhook payload handling:**
- Intuit webhook payload contains entity type + entity ID + operation (Create/Update/Delete)
- App fetches full entity from REST API (webhook payload doesn't contain the full object)
- Writes raw JSON response to Bronze Delta table with metadata: `{source: "qbo", entity_type: "Invoice", raw_payload: {...}, ingested_at: timestamp, webhook_event_id: "..."}`

**Token refresh:**
- Access tokens expire in 1 hour
- Refresh tokens expire in 100 days
- App implements automatic refresh with retry logic
- If refresh fails, marks connection as "stale" and alerts admin in UI

**Sandbox support:**
- For development/demo, use Intuit Sandbox company
- Toggle via environment variable: `INTUIT_ENV=sandbox|production`
- Sandbox base URL: `https://sandbox-quickbooks.api.intuit.com`
- Production base URL: `https://quickbooks.api.intuit.com`

---

### COMPONENT B: Bronze Delta Tables (Raw Storage)

**What it does:** Stores raw API responses from QBO exactly as received. This is the "source of truth" layer — never modified, only appended.

**Tech:**
- Databricks Delta Lake
- Tables partitioned by `source` and `ingested_date`

**Tables:**

```
bronze.qbo_raw_entities
├── entity_id (string)
├── entity_type (string: Invoice, Payment, Bill, etc.)
├── source (string: "qbo")
├── operation (string: Create, Update, Delete)
├── raw_payload (JSON string — full QBO API response)
├── ingested_at (timestamp)
├── webhook_event_id (string, nullable — null for batch pulls)
├── api_version (string)
├── schema_version (string: "bronze_v1")
└── _partition_date (date, derived from ingested_at)

bronze.supplemental_raw
├── upload_id (uuid)
├── source (string: "olist", "support_tickets", "telco_churn", "logistics")
├── file_name (string)
├── raw_payload (JSON string — one row per source record)
├── ingested_at (timestamp)
└── schema_version (string: "bronze_v1")
```

**Why this matters:** If anything goes wrong downstream, you can always reprocess from Bronze. This is the Netflix/Amazon pattern — never throw away raw data.

---

### COMPONENT C: Canonical Event Builder (Silver)

**What it does:** Transforms raw QBO objects and supplemental data into the canonical event schema. This is where all the normalization logic lives.

**Tech:**
- Python transformation functions
- Pydantic v2 for schema validation
- Runs as Databricks Job or DLT pipeline

**The Canonical Event Schema (version 1.0):**

```python
class CanonicalEvent(BaseModel):
    event_id: str                    # UUID v4, generated at creation
    event_type: EventType            # Enum (see below)
    event_time: datetime             # When the business event occurred
    ingested_at: datetime            # When we processed it
    source: str                      # "qbo", "olist", "support_tickets", etc.
    source_entity_id: str            # Original ID from source system
    entity_type: EntityType          # Enum: customer, order, invoice, ticket, product, vendor, payment, expense
    entity_id: str                   # Normalized entity ID
    related_entity_ids: dict         # {"customer_id": "...", "product_ids": [...], "vendor_id": "..."}
    amount: Optional[float]          # Dollar amount (nullable)
    currency: Optional[str]          # ISO 4217 (default "USD" for QBO)
    attributes: dict                 # Event-type-specific fields (see below)
    data_quality_flags: list[str]    # Any quality issues detected during normalization
    schema_version: str              # "canonical_v1"
```

**Event Type Enum (complete list):**

```python
class EventType(str, Enum):
    # Financial domain
    INVOICE_ISSUED = "invoice_issued"
    INVOICE_PAID = "invoice_paid"
    INVOICE_OVERDUE = "invoice_overdue"         # derived: due_date < today and not paid
    PAYMENT_RECEIVED = "payment_received"
    EXPENSE_POSTED = "expense_posted"
    EXPENSE_PAID = "expense_paid"
    REFUND_ISSUED = "refund_issued"
    CREDIT_MEMO_ISSUED = "credit_memo_issued"

    # Operational domain
    ORDER_PLACED = "order_placed"
    ORDER_DELIVERED = "order_delivered"
    ORDER_LATE = "order_late"                   # derived: delivered after estimated date
    PURCHASE_ORDER_PLACED = "purchase_order_placed"
    INVENTORY_LOW = "inventory_low"             # derived: threshold-based
    SHIPMENT_DELAYED = "shipment_delayed"       # from logistics data

    # Customer domain
    CUSTOMER_CREATED = "customer_created"
    CUSTOMER_UPDATED = "customer_updated"
    SUPPORT_TICKET_OPENED = "support_ticket_opened"
    SUPPORT_TICKET_CLOSED = "support_ticket_closed"
    REVIEW_SUBMITTED = "review_submitted"       # from Olist
    CUSTOMER_CHURNED = "customer_churned"       # from churn labels or derived proxy
```

**QBO → Canonical Event Mapping (exact transformations):**

| QBO Object | Trigger | Canonical Event | Key Attributes |
|---|---|---|---|
| Invoice (new) | Create webhook or batch detect | INVOICE_ISSUED | `{due_date, line_items: [{item_id, qty, amount}], customer_id, total}` |
| Invoice (Balance=0) | Update where Balance goes to 0 | INVOICE_PAID | `{paid_date, payment_method, days_to_pay, original_due_date}` |
| Invoice (past due) | Derived: DueDate < today AND Balance > 0 | INVOICE_OVERDUE | `{days_overdue, balance_remaining, customer_id}` |
| Payment | Create | PAYMENT_RECEIVED | `{payment_method, applied_to_invoices: [...], customer_id}` |
| Bill | Create | EXPENSE_POSTED | `{vendor_id, line_items, due_date, category}` |
| BillPayment | Create | EXPENSE_PAID | `{vendor_id, bill_ids, payment_method}` |
| CreditMemo | Create | CREDIT_MEMO_ISSUED | `{customer_id, line_items, reason (if available)}` |
| RefundReceipt | Create | REFUND_ISSUED | `{customer_id, line_items, amount, refund_method}` |
| Customer | Create | CUSTOMER_CREATED | `{display_name, email, balance}` |
| Customer | Update | CUSTOMER_UPDATED | `{changed_fields, new_balance}` |
| PurchaseOrder | Create | PURCHASE_ORDER_PLACED | `{vendor_id, line_items, expected_date}` |

**Supplemental Dataset → Canonical Event Mapping:**

| Dataset | Source Record | Canonical Event | Key Attributes |
|---|---|---|---|
| Olist | orders table | ORDER_PLACED | `{order_id, customer_id, product_ids, seller_id, estimated_delivery}` |
| Olist | orders (delivered) | ORDER_DELIVERED | `{actual_delivery, days_late}` |
| Olist | orders (late) | ORDER_LATE | `{days_late, estimated_delivery, actual_delivery}` |
| Olist | reviews | REVIEW_SUBMITTED | `{score, comment_text, order_id}` |
| Support Tickets | ticket record | SUPPORT_TICKET_OPENED | `{category, priority, channel, subject}` |
| Support Tickets | resolved ticket | SUPPORT_TICKET_CLOSED | `{resolution_time_hours, satisfaction_score}` |
| Telco Churn | customer record | CUSTOMER_CHURNED | `{tenure, contract_type, monthly_charges, churn_label}` |
| Logistics | shipment record | SHIPMENT_DELAYED | `{carrier, route, days_delayed, origin, destination}` |

**Data Quality Scoring at Ingestion:**

Every batch of events gets a quality report:

```python
class DataQualityReport(BaseModel):
    batch_id: str
    source: str
    total_records: int
    valid_records: int
    rejected_records: int
    completeness_score: float        # % of required fields present
    consistency_score: float         # % of records passing cross-field validation
    timeliness_score: float          # % of records with event_time within expected range
    overall_quality_score: float     # weighted average
    quality_issues: list[QualityIssue]  # specific problems found
    impact_advisory: str             # "Logistics RCA confidence may be reduced due to 18% missing shipping dates"
```

Quality issues are attached to individual events via `data_quality_flags` so the RCA engine knows which evidence is strong vs. weak.

---

### COMPONENT D: State & Feature Builder (Gold)

**What it does:** Aggregates canonical events into daily business health metrics. These are the "vital signs" that the incident engine monitors.

**Tech:**
- PySpark or plain Python (pandas for smaller datasets)
- Output: Gold Delta tables
- Runs daily (scheduled) or on-demand ("Reliability Scan")

**Business State Views (complete list with exact computation):**

#### Financial Health Metrics (daily)

| Metric | Computation | Unit |
|---|---|---|
| `daily_revenue` | SUM(amount) WHERE event_type IN (INVOICE_PAID, PAYMENT_RECEIVED) | USD |
| `daily_expenses` | SUM(amount) WHERE event_type IN (EXPENSE_POSTED) | USD |
| `daily_refunds` | SUM(amount) WHERE event_type IN (REFUND_ISSUED, CREDIT_MEMO_ISSUED) | USD |
| `refund_rate` | daily_refunds / daily_revenue (7-day rolling) | ratio |
| `net_cash_proxy` | daily_revenue - daily_expenses - daily_refunds (cumulative rolling 30-day) | USD |
| `expense_ratio` | daily_expenses / daily_revenue (7-day rolling) | ratio |
| `margin_proxy` | (daily_revenue - daily_expenses - daily_refunds) / daily_revenue (7-day rolling) | ratio |
| `dso_proxy` | AVG(days_to_pay) for invoices paid in window | days |
| `ar_aging_amount` | SUM(balance_remaining) WHERE event_type = INVOICE_OVERDUE | USD |
| `ar_overdue_count` | COUNT WHERE event_type = INVOICE_OVERDUE | count |
| `dpo_proxy` | AVG(days from EXPENSE_POSTED to EXPENSE_PAID) | days |

#### Operational Health Metrics (daily)

| Metric | Computation | Unit |
|---|---|---|
| `order_volume` | COUNT WHERE event_type = ORDER_PLACED | count |
| `delivery_count` | COUNT WHERE event_type = ORDER_DELIVERED | count |
| `late_delivery_count` | COUNT WHERE event_type = ORDER_LATE | count |
| `delivery_delay_rate` | late_delivery_count / delivery_count (7-day rolling) | ratio |
| `fulfillment_backlog` | cumulative(ORDER_PLACED) - cumulative(ORDER_DELIVERED) over rolling 14-day window | count |
| `avg_delivery_delay_days` | AVG(days_late) WHERE event_type = ORDER_LATE | days |
| `supplier_delay_rate` | COUNT(SHIPMENT_DELAYED) / COUNT(PURCHASE_ORDER_PLACED) (7-day rolling) | ratio |
| `supplier_delay_severity` | AVG(days_delayed) WHERE event_type = SHIPMENT_DELAYED | days |

#### Customer Health Metrics (daily)

| Metric | Computation | Unit |
|---|---|---|
| `ticket_volume` | COUNT WHERE event_type = SUPPORT_TICKET_OPENED | count |
| `ticket_close_volume` | COUNT WHERE event_type = SUPPORT_TICKET_CLOSED | count |
| `ticket_backlog` | cumulative(OPENED) - cumulative(CLOSED) over rolling 14-day window | count |
| `avg_resolution_time` | AVG(resolution_time_hours) WHERE event_type = SUPPORT_TICKET_CLOSED | hours |
| `review_score_avg` | AVG(score) WHERE event_type = REVIEW_SUBMITTED (7-day rolling) | 1-5 |
| `review_score_trend` | slope of review_score_avg over trailing 14 days | float |
| `churn_proxy` | predicted churn probability (from model) OR repeat_purchase_dropoff_rate | ratio |
| `customer_concentration` | revenue from top 10% customers / total revenue (30-day rolling) | ratio |

**Schema versioning:** All Gold tables include a `metric_schema_version` column. If a metric computation changes, the version increments and downstream systems know to re-baseline.

---

### COMPONENT E: Incident Detection Engine

**What it does:** Monitors Gold metrics and fires incidents when significant deviations occur.

**Tech:**
- Python
- scikit-learn (Isolation Forest)
- scipy.stats (for statistical tests)
- Custom anomaly detection module

**How detection works (three layers, run in order):**

#### Layer 1: Robust Statistical Detection (always runs)

For each metric, compute:
- **Baseline window:** trailing 30 days (configurable per incident type)
- **Median** and **MAD (Median Absolute Deviation)** of the baseline
- **Modified z-score:** `z = 0.6745 * (current_value - median) / MAD`
- **Threshold:** `|z| > 3.0` = anomaly detected (configurable)

Why MAD instead of standard deviation: MAD is robust to outliers. A single spike in the baseline won't inflate your threshold and hide the next spike. This is standard practice in SRE alerting (Netflix, Google).

#### Layer 2: Isolation Forest (runs if enabled)

- Trained on trailing 90 days of multi-metric feature vectors (all Gold metrics for a given domain)
- `contamination=0.05` (expect 5% anomalies)
- If a day's feature vector has anomaly_score < threshold, flag as anomalous
- Registered in MLflow/Unity Catalog with version, training window, and hyperparameters

#### Layer 3: Change-Point Detection (runs if enabled)

- Bayesian Online Change-Point Detection (BOCPD) on key metrics
- Detects regime changes (not just spikes): "the mean delivery delay shifted from 2 days to 5 days starting Feb 12"
- Uses `ruptures` library (Python) with Pelt algorithm as a simpler alternative

**Detection method fusion:**
- If Layer 1 fires alone → incident with confidence "MEDIUM"
- If Layer 1 + Layer 2 agree → confidence "HIGH"
- If Layer 1 + Layer 2 + Layer 3 agree → confidence "VERY_HIGH"
- If only Layer 2 or Layer 3 fires without Layer 1 → confidence "LOW" (flagged for review)

**Incident Taxonomy (complete definitions):**

#### Incident 1: Refund Spike
- **Metric:** `refund_rate`
- **Baseline:** 30-day trailing median
- **Trigger:** modified z-score > 3.0
- **Severity:** LOW if z ∈ (3, 4), MEDIUM if z ∈ (4, 6), HIGH if z > 6, CRITICAL if z > 8 AND refund_amount > 10% of trailing revenue
- **Evidence:** list of REFUND_ISSUED and CREDIT_MEMO_ISSUED event IDs in the anomaly window

#### Incident 2: Fulfillment SLA Degradation
- **Metric:** `delivery_delay_rate` AND `fulfillment_backlog`
- **Baseline:** 30-day trailing median for each
- **Trigger:** EITHER metric z-score > 3.0
- **Severity:** based on max(z_delay_rate, z_backlog) using same scale
- **Evidence:** ORDER_LATE event IDs + ORDER_PLACED event IDs in backlog

#### Incident 3: Support Load Surge
- **Metric:** `ticket_volume` AND `ticket_backlog`
- **Baseline:** 30-day trailing median
- **Trigger:** EITHER metric z-score > 3.0
- **Severity:** same scale, upgraded one level if `avg_resolution_time` also anomalous
- **Evidence:** SUPPORT_TICKET_OPENED event IDs

#### Incident 4: Churn Acceleration
- **Metric:** `churn_proxy` (model-based or repeat_purchase_dropoff)
- **Baseline:** 30-day trailing median
- **Trigger:** z-score > 2.5 (lower threshold because churn is high-impact and slow-moving)
- **Severity:** based on z-score AND absolute churn rate AND customer_concentration (if top customers are churning, severity upgrades)
- **Evidence:** CUSTOMER_CHURNED event IDs or customer IDs with declining purchase frequency

#### Incident 5: Margin Compression / Expense Drift
- **Metric:** `margin_proxy` AND `expense_ratio`
- **Baseline:** 30-day trailing median
- **Trigger:** margin_proxy z-score < -3.0 (margin dropping) OR expense_ratio z-score > 3.0 (expenses rising relative to revenue)
- **Severity:** same scale
- **Evidence:** EXPENSE_POSTED event IDs during anomaly window + REFUND_ISSUED if contributing

#### Incident 6: Liquidity Crunch Risk
- **Metric:** `net_cash_proxy` AND `ar_aging_amount`
- **Baseline:** 30-day trailing
- **Trigger:** net_cash_proxy z-score < -3.0 OR ar_aging_amount z-score > 3.0
- **Severity:** CRITICAL if both fire simultaneously
- **Evidence:** INVOICE_OVERDUE event IDs + large EXPENSE_POSTED events

#### Incident 7: Supplier Dependency Failure
- **Metric:** `supplier_delay_rate` AND `supplier_delay_severity`
- **Baseline:** 30-day trailing
- **Trigger:** either z-score > 3.0
- **Severity:** upgraded if specific vendor concentration is high
- **Evidence:** SHIPMENT_DELAYED and PURCHASE_ORDER_PLACED event IDs

#### Incident 8: Customer Satisfaction Regression
- **Metric:** `review_score_avg` AND `review_score_trend`
- **Baseline:** 30-day trailing
- **Trigger:** review_score_avg z-score < -2.5 OR review_score_trend slope < -0.1
- **Severity:** based on magnitude of decline
- **Evidence:** REVIEW_SUBMITTED event IDs with low scores

**Incident data model:**

```python
class Incident(BaseModel):
    incident_id: str                     # UUID
    incident_type: IncidentType          # Enum from taxonomy above
    detected_at: datetime                # When detection fired
    incident_window_start: datetime      # Start of anomaly window
    incident_window_end: datetime        # End of anomaly window
    severity: Severity                   # LOW, MEDIUM, HIGH, CRITICAL
    confidence: Confidence               # LOW, MEDIUM, HIGH, VERY_HIGH
    detection_methods: list[str]         # ["mad_zscore", "isolation_forest", "changepoint"]
    primary_metric: str                  # The metric that triggered
    primary_metric_value: float          # Current value
    primary_metric_baseline: float       # Baseline median
    primary_metric_zscore: float         # Modified z-score
    supporting_metrics: list[dict]       # Other metrics that corroborate
    evidence_event_ids: list[str]        # Canonical event IDs
    evidence_event_count: int
    data_quality_score: float            # Quality score of underlying data
    run_id: str                          # Links to run_manifest
    cascade_id: Optional[str]           # If part of a cascade (see Component F)
    status: str                         # "open", "acknowledged", "resolved"
```

---

### COMPONENT F: Incident Cascade Correlator

**What it does:** Detects when multiple incidents are actually part of the same cascading failure rather than independent events.

**How it works:**

1. When new incidents are detected, look at all incidents within a **correlation window** (default: 7 days, configurable)
2. For each pair of incidents (A, B) where A.detected_at < B.detected_at, compute a **cascade score:**

```
cascade_score(A, B) = 
    temporal_weight(A, B)           # higher if A precedes B by 1-5 days
    × entity_overlap(A, B)          # higher if they share customer/product/vendor IDs
    × causal_plausibility(A, B)     # from the dependency graph: is A→B a known causal path?
```

**temporal_weight:** `exp(-|days_between| / decay_constant)` where decay_constant = 3.0. Peaks when B follows A by 1-3 days, drops off for longer gaps.

**entity_overlap:** `|evidence_entities(A) ∩ evidence_entities(B)| / |evidence_entities(A) ∪ evidence_entities(B)|` (Jaccard similarity on entity IDs).

**causal_plausibility:** Binary {0, 1} from a predefined **Incident Dependency Graph:**

```
FULFILLMENT_SLA_DEGRADATION → SUPPORT_LOAD_SURGE → CHURN_ACCELERATION
FULFILLMENT_SLA_DEGRADATION → CUSTOMER_SATISFACTION_REGRESSION → CHURN_ACCELERATION
SUPPLIER_DEPENDENCY_FAILURE → FULFILLMENT_SLA_DEGRADATION
REFUND_SPIKE → MARGIN_COMPRESSION → LIQUIDITY_CRUNCH_RISK
CHURN_ACCELERATION → MARGIN_COMPRESSION → LIQUIDITY_CRUNCH_RISK
SUPPORT_LOAD_SURGE → CUSTOMER_SATISFACTION_REGRESSION
```

3. If cascade_score > 0.3 (tunable), group incidents into a **Cascade:**

```python
class IncidentCascade(BaseModel):
    cascade_id: str                    # UUID
    root_incident_id: str              # The earliest incident in the chain
    incident_ids: list[str]            # All incidents in the cascade
    cascade_path: list[str]            # Ordered: ["SUPPLIER_DELAY", "FULFILLMENT_SLA", "SUPPORT_SURGE"]
    total_blast_radius: dict           # Aggregated blast radius
    cascade_score: float               # Overall confidence
    detected_at: datetime
```

**Why this matters:** Judges will see five separate incidents and think "so what?" But seeing those five incidents grouped as a single cascade with a clear propagation path — that's the "oh wow" moment. It proves the system understands the business as a connected system, not just independent metrics.

---

### COMPONENT G: Root Cause Analyzer — The BRE-RCA Algorithm (CORE IP)

**This is the most important component. This is what you defend at the whiteboard.**

**What it does:** Given a detected incident, it traces backwards through the event stream and metric history to produce a ranked list of contributing causes with evidence.

**The algorithm is called BRE-RCA (Business Reliability Engine Root Cause Attribution).**

#### Step 0: Define the Business Dependency Graph (static, configured)

This is a directed acyclic graph (DAG) where nodes are metrics and edges represent known causal relationships. This is defined by domain knowledge, not learned from data.

```
Nodes (metrics):
  supplier_delay_rate
  order_volume
  fulfillment_backlog
  delivery_delay_rate
  ticket_volume
  ticket_backlog
  avg_resolution_time
  review_score_avg
  churn_proxy
  refund_rate
  daily_revenue
  daily_expenses
  expense_ratio
  margin_proxy
  net_cash_proxy
  dso_proxy
  ar_aging_amount

Edges (causal direction):
  supplier_delay_rate → delivery_delay_rate
  supplier_delay_rate → fulfillment_backlog
  order_volume → fulfillment_backlog
  fulfillment_backlog → delivery_delay_rate
  delivery_delay_rate → ticket_volume
  delivery_delay_rate → review_score_avg
  ticket_volume → ticket_backlog
  ticket_backlog → avg_resolution_time
  avg_resolution_time → review_score_avg
  review_score_avg → churn_proxy
  ticket_volume → churn_proxy
  churn_proxy → daily_revenue
  refund_rate → margin_proxy
  refund_rate → daily_revenue
  daily_revenue → margin_proxy
  daily_expenses → expense_ratio
  expense_ratio → margin_proxy
  margin_proxy → net_cash_proxy
  dso_proxy → ar_aging_amount
  ar_aging_amount → net_cash_proxy
```

#### Step 1: Identify the incident metric node

Example: Incident is "Fulfillment SLA Degradation" → incident metric node = `delivery_delay_rate`

#### Step 2: Select the causal window

`causal_window = [incident_window_start - lookback_days, incident_window_end]`

Default `lookback_days = 14` (configurable per incident type). This is where we look for causes.

#### Step 3: Find all upstream nodes in the dependency graph

Using the DAG, traverse backwards from the incident metric node. For `delivery_delay_rate`, upstream nodes are: `supplier_delay_rate`, `fulfillment_backlog`, `order_volume`.

For deeper incidents like `net_cash_proxy`, the upstream set is much larger (the whole graph feeds into it).

#### Step 4: For each upstream metric, compute a Contribution Score

```
contribution_score(metric_i, incident) = 
    anomaly_magnitude(metric_i)
    × temporal_precedence(metric_i, incident)
    × graph_proximity(metric_i, incident)
    × data_quality_weight(metric_i)
```

**anomaly_magnitude:** `min(|modified_z_score(metric_i)| / 3.0, 3.0)` — normalized so that z=3 gives score 1.0, z=9 gives 3.0 (capped). This measures "how abnormal was this metric?"

**temporal_precedence:** We compute the cross-correlation between metric_i and the incident metric with lag bounds [0, lookback_days]. `temporal_precedence = max_cross_correlation_at_optimal_lag × lag_decay`. `lag_decay = exp(-optimal_lag / decay_constant)` where decay_constant = 7.0 days. This rewards metrics that moved abnormally BEFORE the incident metric moved, and penalizes metrics that moved after (those are effects, not causes).

**graph_proximity:** `1.0 / (shortest_path_length + 1)` from metric_i to the incident metric in the DAG. Direct parents get proximity 0.5, grandparents get 0.33, etc. This rewards direct causes over indirect ones.

**data_quality_weight:** `data_quality_score` for events underlying this metric (from the data quality report). If shipping dates are 80% complete, supplier_delay metrics get weight 0.8.

#### Step 5: Rank candidate causes by contribution_score

Sort all upstream metrics by descending contribution_score. Take top K (default K=5).

#### Step 6: For each top cause, extract evidence events

For each top contributing metric:
1. Find the time window where this metric was most anomalous
2. Retrieve the canonical events that contributed to this metric in that window
3. Cluster events by entity (e.g., "14 ORDER_LATE events from vendor V-123", "87 REFUND_ISSUED events for product category 'Electronics'")
4. These become the evidence clusters

#### Step 7: Construct the causal chain

The output is a ranked list of causal paths from root cause to incident:

```python
class CausalChain(BaseModel):
    chain_id: str
    incident_id: str
    paths: list[CausalPath]          # Ranked by overall score
    algorithm_version: str            # "BRE-RCA-v1"
    causal_window: tuple[datetime, datetime]
    dependency_graph_version: str
    run_id: str

class CausalPath(BaseModel):
    rank: int
    overall_score: float              # Product of contribution scores along path
    nodes: list[CausalNode]           # Ordered from root cause → incident
    
class CausalNode(BaseModel):
    metric_name: str
    contribution_score: float
    anomaly_magnitude: float
    temporal_precedence: float
    graph_proximity: float
    data_quality_weight: float
    metric_value: float               # Value during anomaly
    metric_baseline: float            # Baseline value
    metric_zscore: float
    anomaly_window: tuple[datetime, datetime]
    evidence_clusters: list[EvidenceCluster]

class EvidenceCluster(BaseModel):
    cluster_label: str                # "Vendor V-123 shipment delays"
    event_count: int
    event_ids: list[str]              # Actual canonical event IDs
    entity_type: str
    entity_id: Optional[str]          # If cluster is entity-specific
    total_amount: Optional[float]     # Dollar impact
    summary: str                      # Human-readable: "14 late deliveries from vendor V-123 averaging 4.2 days late"
```

**Example output for a Fulfillment SLA Degradation incident:**

```
Causal Chain (ranked):
  Path 1 (score: 2.84):
    [supplier_delay_rate (z=4.1, 3 days before)]
      → [fulfillment_backlog (z=3.8, 1 day before)]
        → [delivery_delay_rate (z=5.2, incident)]
    Evidence: 
      - 23 SHIPMENT_DELAYED events from vendor "AcmeSupplier" (avg 3.8 days late)
      - Fulfillment backlog grew from 45 to 128 orders between Feb 5-10

  Path 2 (score: 1.92):
    [order_volume (z=3.2, 5 days before)]
      → [fulfillment_backlog (z=3.8, 1 day before)]
        → [delivery_delay_rate (z=5.2, incident)]
    Evidence:
      - ORDER_PLACED count increased 85% (Feb 3-5) — potential campaign spike
      - 340 orders entered pipeline while fulfillment capacity was flat
```

**Why this is defensible:** You're not claiming true causality. You're saying "here are the statistically abnormal upstream metrics that preceded this incident, ranked by a transparent scoring formula, with links to the actual events." A judge can verify every number. That's the standard at Amazon/Netflix for postmortem analysis.

---

### COMPONENT H: Blast Radius Calculator

**What it does:** For each incident, quantifies the downstream impact in business terms.

**How it works:**

1. From the incident's evidence events, extract all affected entity IDs (customers, orders, products, vendors)
2. For each entity type, compute:

```python
class BlastRadius(BaseModel):
    incident_id: str
    customers_affected: int
    orders_affected: int
    products_affected: int
    vendors_involved: int
    estimated_revenue_exposure: float    # SUM of amounts on affected orders/invoices
    estimated_refund_exposure: float     # Based on historical refund rate for affected entities
    estimated_churn_exposure: int        # Customers with >50% churn probability in affected set
    downstream_incidents_triggered: list[str]  # Other incident IDs in the cascade
    blast_radius_severity: str           # "contained" (<1% of monthly revenue), "significant" (1-5%), "severe" (5-15%), "catastrophic" (>15%)
    narrative: str                       # "This incident affected 342 orders across 127 customers, with estimated revenue exposure of $18,400 and potential churn risk for 23 high-value customers."
```

**Why this matters:** This is what turns a technical alert into a business-impact statement. When the admin sees "342 orders, 127 customers, $18,400 at risk" — that's actionable. That's what Intuit would want their small business customers to see.

---

### COMPONENT I: Postmortem + Monitor Generator

**What it does:** Generates two outputs: a structured postmortem document and a set of runnable prevention monitors.

#### Postmortem Structure:

```python
class Postmortem(BaseModel):
    postmortem_id: str
    incident_id: str
    cascade_id: Optional[str]
    generated_at: datetime

    # Section 1: Summary
    title: str                          # "Fulfillment SLA Degradation — Feb 8-15, 2025"
    severity: str
    duration: str                       # "7 days"
    status: str                         # "resolved" / "ongoing"
    one_line_summary: str               # "Supplier delays from AcmeSupplier caused a fulfillment backlog..."

    # Section 2: Timeline
    timeline: list[TimelineEntry]       # Ordered events with timestamps and descriptions

    # Section 3: Root Cause
    causal_chain: CausalChain           # Full BRE-RCA output
    root_cause_summary: str             # Human-readable paragraph

    # Section 4: Blast Radius
    blast_radius: BlastRadius

    # Section 5: Contributing Factors
    contributing_factors: list[str]     # Bullet-style factors beyond root cause

    # Section 6: Prevention Monitors
    monitors: list[MonitorRule]         # Generated monitors (see below)

    # Section 7: Recommendations
    recommendations: list[str]          # Action items

    # Section 8: Metadata
    data_quality_score: float
    confidence_note: str                # "RCA confidence is HIGH based on..."
    algorithm_version: str
    run_id: str

class TimelineEntry(BaseModel):
    timestamp: datetime
    event_description: str              # "Supplier delay rate exceeded 3σ threshold"
    metric_name: Optional[str]
    metric_value: Optional[float]
    evidence_event_ids: list[str]
```

**Export formats:**
- JSON (for programmatic consumption)
- Markdown (for human reading)
- PDF (for formal sharing — generated via `weasyprint` or `reportlab`)

**Optional LLM enhancement:** If enabled, pass the structured postmortem data to an LLM (Claude API) to generate a polished narrative summary. But the evidence, numbers, and causal chain all come from BRE-RCA — the LLM only improves readability, never invents facts.

#### Monitor Rules:

```python
class MonitorRule(BaseModel):
    monitor_id: str
    name: str                           # "Supplier Delay Rate Watch"
    description: str
    source_incident_id: str             # Which incident generated this monitor
    metric_name: str                    # "supplier_delay_rate"
    condition: str                      # "zscore > 2.5" or "value > 0.15"
    baseline_window_days: int
    check_frequency: str                # "daily"
    severity_if_triggered: str
    enabled: bool                       # Default True
    created_at: datetime
    alert_message_template: str         # "Supplier delay rate has exceeded threshold: {value} (baseline: {baseline}). This pattern preceded a fulfillment SLA degradation incident on {source_incident_date}."
```

**How monitors are generated:**
For each node in the top causal path of the RCA output:
1. Take the metric name
2. Set threshold at 2.5σ (slightly lower than incident detection at 3.0σ — this is an early warning)
3. Set baseline to the same window used in detection
4. Generate the monitor rule

This means the monitors are directly derived from what actually caused the last incident — not generic "watch everything" alerts.

---

### COMPONENT J: Monitor Runtime Engine (Closed-Loop)

**What it does:** This is the component that RUNS the generated monitors and proves they catch recurrences. This is the closed-loop that makes the system a true reliability platform.

**How it works:**

1. Active monitors are stored in a `monitors` table
2. On each "Reliability Scan" (daily or on-demand), after computing Gold metrics, the Monitor Runtime checks all enabled monitors
3. For each monitor:
   - Compute the metric's current value
   - Compute the baseline from the specified window
   - Evaluate the condition
   - If triggered: create a **Monitor Alert** (not a full incident — a pre-incident warning)

```python
class MonitorAlert(BaseModel):
    alert_id: str
    monitor_id: str
    triggered_at: datetime
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: str
    severity: str
    message: str                        # From template
    related_incident_id: str            # The original incident this monitor was born from
    status: str                         # "active", "acknowledged", "dismissed"
```

4. **The demo moment:** Inject a synthetic recurrence (or use a later time window in the data where a similar pattern occurs). Show that the monitor fires BEFORE the incident detection engine would have caught it (because the monitor threshold is 2.5σ vs. 3.0σ). This proves the prevention loop works.

**UI integration:** Monitors and their alerts get their own section in the admin UI — a list of active monitors with green/yellow/red status, and a feed of recent alerts.

---

### COMPONENT K: What-If Simulation Engine

**What it does:** Lets the admin ask "what would happen if [metric] changed by X%?" and see which incidents would trigger.

**How it works:**

1. Admin selects a metric and a perturbation: e.g., "increase delivery_delay_rate by 30%"
2. System takes the current Gold metrics snapshot
3. Applies the perturbation to the selected metric
4. Propagates the perturbation through the dependency graph:
   - For each downstream metric, estimate the impact using the historical correlation between the perturbed metric and the downstream metric
   - Apply the estimated change to downstream metrics
5. Run the Incident Detection Engine on the simulated metrics
6. Return: which incidents would fire, at what severity, and which cascade would form

```python
class WhatIfScenario(BaseModel):
    scenario_id: str
    perturbations: list[dict]           # [{"metric": "delivery_delay_rate", "change_pct": 30}]
    simulated_metrics: dict             # Full Gold metrics after perturbation
    triggered_incidents: list[Incident] # Incidents that would fire
    triggered_cascades: list[IncidentCascade]
    narrative: str                      # "A 30% increase in delivery delays would trigger Fulfillment SLA Degradation (HIGH) and Support Load Surge (MEDIUM) within an estimated 3-5 days."
```

**Implementation detail:** The propagation uses simple linear estimates from historical correlation. This is not a perfect causal model — and you should say that. It's a "stress test" tool, not a prediction engine.

---

### COMPONENT L: Incident Comparator

**What it does:** Lets the admin compare two incidents of the same type from different time periods to see if they had different root causes.

**How it works:**

1. Admin selects two incidents of the same type (e.g., two "Refund Spike" incidents from March and June)
2. System displays side-by-side:
   - Causal chains for both
   - Metrics comparison
   - Evidence cluster comparison
   - Blast radius comparison
3. Highlights differences: "March refund spike was driven by product quality (evidence: review scores down for SKU-X). June refund spike was driven by shipping delays (evidence: ORDER_LATE events up 200%)."

```python
class IncidentComparison(BaseModel):
    comparison_id: str
    incident_a_id: str
    incident_b_id: str
    incident_type: str
    shared_root_causes: list[str]       # Metrics that appear in both causal chains
    unique_to_a: list[str]              # Causes only in A
    unique_to_b: list[str]              # Causes only in B
    severity_comparison: dict           # {"a": "HIGH", "b": "MEDIUM"}
    blast_radius_comparison: dict
    narrative: str                      # "These two refund spikes had different root causes..."
```

**Why this impresses:** It proves your RCA is actually specific, not generic. If every incident produces the same "everything is correlated" output, judges will be skeptical. Showing that different incidents of the same type get different causal chains proves the algorithm works.

---

### COMPONENT M: API Service

**Tech:** FastAPI, Python 3.11+, Pydantic v2 for request/response models, admin-only JWT auth

**Complete endpoint list:**

```
# Authentication
POST   /v1/auth/token                    # Get JWT (admin-only, simple secret-based)

# Connection
POST   /v1/connect/quickbooks            # Initiate QBO OAuth flow
GET    /v1/connect/quickbooks/callback    # OAuth callback
GET    /v1/connect/status                 # Connection health

# Ingestion
POST   /v1/ingest/qbo                    # Trigger QBO data pull
POST   /v1/ingest/upload                  # Upload supplemental CSV/JSON/Parquet
GET    /v1/ingest/status/{job_id}        # Check ingestion job status
GET    /v1/ingest/quality/{batch_id}     # Get data quality report

# Analysis
POST   /v1/scan                           # Run full Reliability Scan (state → detect → RCA → postmortem)
GET    /v1/scan/status/{run_id}          # Check scan status
GET    /v1/scan/manifest/{run_id}        # Get run manifest (auditability)

# Incidents
GET    /v1/incidents                      # List incidents (filterable by type, severity, date range, cascade)
GET    /v1/incidents/{id}                # Get incident detail
GET    /v1/incidents/{id}/causal-chain   # Get RCA result
GET    /v1/incidents/{id}/blast-radius   # Get blast radius
GET    /v1/incidents/{id}/postmortem     # Get postmortem (JSON)
GET    /v1/incidents/{id}/postmortem/pdf # Get postmortem (PDF export)
GET    /v1/incidents/{id}/postmortem/md  # Get postmortem (Markdown export)
GET    /v1/incidents/{id}/graph          # Get causal chain as nodes/edges for visualization
PATCH  /v1/incidents/{id}/status         # Update incident status (open/acknowledged/resolved)

# Cascades
GET    /v1/cascades                      # List incident cascades
GET    /v1/cascades/{id}                 # Get cascade detail with all linked incidents

# Monitors
GET    /v1/monitors                      # List all monitors
GET    /v1/monitors/{id}                 # Get monitor detail
PATCH  /v1/monitors/{id}                 # Enable/disable monitor, adjust threshold
GET    /v1/monitors/alerts               # List monitor alerts
PATCH  /v1/monitors/alerts/{id}/status   # Acknowledge/dismiss alert

# Comparison
POST   /v1/compare                       # Compare two incidents
GET    /v1/compare/{comparison_id}       # Get comparison result

# Simulation
POST   /v1/simulate                      # Run what-if scenario
GET    /v1/simulate/{scenario_id}        # Get simulation result

# Metrics
GET    /v1/metrics/current               # Current Gold metric values
GET    /v1/metrics/history               # Historical metrics (filterable by metric name, date range)
GET    /v1/metrics/health-summary        # Overall business health dashboard data

# System
GET    /v1/health                        # Health check
GET    /v1/config                        # Current system config (schema versions, thresholds, etc.)
```

**Request ID correlation:** Every request gets a `X-Request-ID` header. All logs, DB writes, and background job outputs include this ID. If something goes wrong, you can trace end-to-end.

**Error responses:** Standard format:

```json
{
    "error": {
        "code": "INCIDENT_NOT_FOUND",
        "message": "No incident with id abc-123",
        "request_id": "req-456",
        "timestamp": "2025-02-09T10:30:00Z"
    }
}
```

---

### COMPONENT N: Admin UI

**Tech options (choose one):**

**Option A (recommended for Databricks alignment):** Databricks App using Streamlit or Dash
- Pros: deployed inside Databricks, no separate hosting, sponsors love it
- Cons: less flexible for custom visualizations

**Option B (recommended for best demo impact):** React + Vite + Tailwind, deployed as standalone container or static site
- Pros: beautiful, custom causal chain visualization with D3/Cytoscape, responsive
- Cons: separate deployment

**Option C (hybrid, recommended):** React frontend deployed as a Databricks App using the custom framework option, or served alongside the FastAPI backend

**UI Screens (7 screens):**

#### Screen 1: Connection & Setup
- "Connect QuickBooks" button (initiates OAuth flow)
- Connection status indicator (green/yellow/red)
- Dataset upload area for supplemental data (drag-and-drop CSV/JSON)
- Data quality dashboard showing completeness scores per source

#### Screen 2: Business Health Dashboard
- Summary cards: Financial Health (green/yellow/red), Operational Health, Customer Health
- Sparklines for key metrics (last 30 days)
- Active monitor count and alert count
- "Run Reliability Scan" button

#### Screen 3: Incidents List
- Table: incident type, severity (color-coded), confidence, detected date, status, cascade indicator
- Sortable by any column
- Filterable by type, severity, date range
- Cascade grouping toggle (show incidents grouped by cascade or flat)

#### Screen 4: Incident Detail (THE CENTERPIECE)
- **Header:** Incident type, severity badge, confidence badge, time range
- **Timeline chart:** Interactive line chart showing the incident metric over time, with the anomaly window highlighted, baseline shown as a band, and threshold shown as a line. Users can hover to see exact values.
- **Causal Chain Visualization:** THIS IS THE MONEY SHOT.
  - Interactive directed graph (D3 force-directed or Cytoscape)
  - Nodes = metrics, sized by contribution score
  - Edges = causal links from the dependency graph
  - Node color = anomaly severity (green/yellow/orange/red)
  - Click a node → sidebar shows: metric name, z-score, baseline vs. actual value, anomaly window, evidence cluster summary
  - Click an evidence cluster → drilldown table of actual event IDs with details
  - Animate the cascade: show nodes lighting up in temporal order to visualize how the problem propagated
- **Evidence Table:** Sortable table of all evidence events with event_id, type, time, entity, amount, attributes
- **Blast Radius Card:** customers affected, orders affected, revenue exposure, churn exposure
- **Related Incidents:** If part of a cascade, show links to other incidents in the chain

#### Screen 5: Postmortem View
- Rendered Markdown/HTML of the full postmortem
- Sections clearly delineated: Summary, Timeline, Root Cause, Blast Radius, Contributing Factors, Prevention Monitors, Recommendations
- Export buttons: PDF, Markdown, JSON
- Monitor rules displayed as cards with enable/disable toggles

#### Screen 6: Monitors Dashboard
- List of all active monitors with status (green = healthy, yellow = warning, red = triggered)
- For each monitor: metric name, threshold, last checked, source incident
- Alert feed: recent alerts with timestamps and messages
- Click alert → links to the related incident and postmortem

#### Screen 7: Comparison & Simulation
- **Comparison tab:** Select two incidents of the same type, see side-by-side causal chains, metrics, blast radius, and difference narrative
- **Simulation tab:** Select a metric, set perturbation %, click "Run Simulation," see which incidents would fire and the propagation path

**Graph visualization tech:** Use Cytoscape.js (more control, better for DAGs) or D3.js (more flexible, steeper learning curve). Cytoscape is recommended because it has built-in layout algorithms for DAGs (dagre layout) and click/hover interaction out of the box.

---

# SECTION 5: COMPLETE TECH STACK

| Layer | Technology | Why |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem, team familiarity, Databricks native |
| **API Framework** | FastAPI | Async, auto-docs, Pydantic integration |
| **Schema/Validation** | Pydantic v2 | Data contracts, versioning, serialization |
| **Data Source (Primary)** | Intuit QuickBooks Online API | Real financial data, OAuth, webhooks |
| **Auth (QBO)** | intuitlib (Intuit OAuth client) | Official SDK |
| **HTTP Client** | httpx | Async HTTP for QBO API calls |
| **Storage** | Databricks Delta Lake | Append-only event store, time travel, ACID |
| **Compute** | PySpark (on Databricks) or pandas (local dev) | Metric computation |
| **SQL** | DuckDB (local dev) / Spark SQL (Databricks) | Querying Gold tables |
| **Anomaly Detection** | scipy.stats, scikit-learn | MAD z-scores, Isolation Forest |
| **Change Point Detection** | ruptures | Pelt algorithm |
| **Supervised ML** | LightGBM, scikit-learn | Churn classifier (where labels exist) |
| **Model Registry** | MLflow + Unity Catalog (Databricks) | Model versioning, governance |
| **Graph Data Structure** | networkx | Dependency graph, path computation |
| **Background Jobs** | Databricks Jobs or Celery + Redis (local) | Async scan processing |
| **PDF Generation** | weasyprint or reportlab | Postmortem PDF export |
| **Frontend** | React 18 + Vite + Tailwind CSS | Fast, modern, clean |
| **Graph Visualization** | Cytoscape.js (with dagre layout) | Interactive causal chain DAG |
| **Charts** | Recharts (React) or Chart.js | Timeline metrics charts |
| **App Hosting** | Databricks Apps | Admin UI hosting (Databricks-native) |
| **Containerization** | Docker + Docker Compose | Local dev, portable deployment |
| **Testing** | pytest, hypothesis (property tests) | Golden tests, schema validation |
| **Logging** | structlog | Structured JSON logs |
| **Docs** | OpenAPI (auto from FastAPI), Markdown | API docs, architecture docs |

---

# SECTION 6: DATASETS AND INGESTION PLAN

## Primary: QuickBooks Online Sandbox

For development and demo, use Intuit's Sandbox company. Seed it with realistic data using QBO's API (create invoices, payments, expenses, customers programmatically).

**Sandbox seeding script:** Create a Python script that generates 6 months of realistic business activity:
- ~500 customers
- ~2000 invoices with varying payment times (some on time, some late, some unpaid)
- ~800 expenses across categories
- ~100 refunds (clustered to create synthetic incident patterns)
- ~50 vendors with varying reliability

This gives you full control over the data AND lets you inject known incidents for testing.

## Supplemental datasets:

| Dataset | Source | Events Generated | Purpose |
|---|---|---|---|
| Olist Brazilian E-Commerce | Kaggle | ORDER_PLACED, ORDER_DELIVERED, ORDER_LATE, REVIEW_SUBMITTED | Fulfillment + customer satisfaction |
| Customer Support Tickets | Kaggle | SUPPORT_TICKET_OPENED, SUPPORT_TICKET_CLOSED | Support operations |
| Telco Customer Churn | Kaggle | CUSTOMER_CHURNED | Supervised churn modeling |
| Logistics & Supply Chain | Kaggle | SHIPMENT_DELAYED, PURCHASE_ORDER_PLACED | Supplier reliability |

**Ingestion strategy:** Each supplemental dataset has a dedicated adapter class that reads the source format and produces CanonicalEvent objects. Example:

```python
class OlistAdapter:
    def ingest(self, orders_df, reviews_df, ...) -> list[CanonicalEvent]:
        events = []
        for _, row in orders_df.iterrows():
            events.append(CanonicalEvent(
                event_type=EventType.ORDER_PLACED,
                event_time=row['order_purchase_timestamp'],
                source="olist",
                ...
            ))
        return events
```

---

# SECTION 7: TRAINING AND EVALUATION

## 7.1 Time-Based Splits

All evaluation uses time-based splits (not random) because this is sequential business data:

- **Train:** First 70% of the time range
- **Validation:** Next 15%
- **Test:** Final 15%

## 7.2 Anomaly Detection Evaluation

**Challenge:** No ground-truth "incident" labels exist for most of the data.

**Solution:** Two evaluation approaches:

**Approach A: Synthetic Incident Injection**
1. Take the test period
2. Inject known anomalies: multiply refund_rate by 3x for a 5-day window, add 50% to delivery_delay_rate for a week, etc.
3. Measure: Did the system detect the injected incidents? How quickly? (detection delay in days)
4. Measure false positives: How many incidents were detected in periods with no injection?

**Approach B: Known-Event Validation**
1. In the Olist dataset, there are known periods of high review dissatisfaction and delivery delays
2. Manually label these periods as "expected incidents"
3. Measure detection recall and precision

**Metrics reported:**
- **Detection precision:** % of detected incidents that are real (or injected)
- **Detection recall:** % of real/injected incidents that were detected
- **Detection delay:** Average days between incident onset and detection
- **False positive rate:** Spurious incidents per month

## 7.3 Churn Classifier Evaluation

For the Telco Churn dataset with labels:
- **Model:** LightGBM with hyperparameter tuning via Optuna
- **Baseline:** Logistic Regression
- **Metrics:** AUROC, F1, Precision, Recall (all on test set)
- **Calibration:** Platt scaling to get reliable probabilities
- **Registered in MLflow** with all hyperparameters, metrics, and the training dataset version

## 7.4 RCA Evaluation

**Golden test approach:**
1. Construct 5 synthetic scenarios with known causal chains (e.g., "supplier delay → backlog → delivery delay")
2. Inject the corresponding event patterns into the system
3. Verify that the BRE-RCA output matches the expected causal chain (top-ranked path includes the known root cause)
4. Measure: % of golden tests where the true root cause appears in top-3 ranked paths

## 7.5 Ablation Study

Compare three configurations on the same test data:
1. **Rules-only:** MAD z-score detection + BRE-RCA
2. **Rules + Isolation Forest:** Add anomaly detection layer
3. **Rules + IF + Change-Point:** Full hybrid

Report detection performance and RCA accuracy for each. This shows you understand what each component contributes.

---

# SECTION 8: DEPLOYMENT

## Local Development

```yaml
# docker-compose.yml
services:
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      - INTUIT_CLIENT_ID=...
      - INTUIT_CLIENT_SECRET=...
      - INTUIT_ENV=sandbox
      - DB_TYPE=duckdb
      - DB_PATH=/data/bre.duckdb
  
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  worker:
    build: ./api
    command: celery -A worker worker --loglevel=info
```

## Databricks Deployment

1. **Delta Tables:** Created in a Unity Catalog schema: `bre_catalog.bronze`, `bre_catalog.silver`, `bre_catalog.gold`, `bre_catalog.incidents`
2. **Jobs:** Databricks Workflows for scheduled ingestion and scans
3. **Model Registry:** MLflow models in Unity Catalog under `bre_catalog.models`
4. **App:** Admin UI deployed as Databricks App
5. **Secrets:** QBO OAuth tokens in Databricks Secrets

## One-Command Demo

```bash
# Local
docker-compose up -d
python scripts/seed_sandbox.py    # Seed QBO sandbox with test data
python scripts/demo_run.py        # Trigger full scan and open browser

# Output:
# ✅ Connected to QBO sandbox
# ✅ Ingested 3,847 events
# ✅ Computed 45 days of business state
# ✅ Detected 4 incidents (1 cascade)
# ✅ Generated 4 postmortems with 12 monitors
# ✅ UI available at http://localhost:3000
```

---

# SECTION 9: TESTING STRATEGY

## Unit Tests
- Schema validation: every event type round-trips through Pydantic
- Metric computation: given known events, verify exact metric values
- Anomaly detection: given a known z-score, verify correct severity classification
- BRE-RCA scoring: given known metric values and graph, verify contribution scores

## Integration Tests
- QBO OAuth flow (against sandbox)
- Ingestion pipeline: raw QBO JSON → Bronze → Silver canonical events
- Full scan: events → Gold metrics → incidents → RCA → postmortem

## Golden Tests (most important)
- 5 complete scenarios with known inputs and expected outputs:

```
Scenario 1: "Supplier Delay Cascade"
  Input: 30 days of normal events, then 7 days of SHIPMENT_DELAYED events from vendor V-001
  Expected: 
    - Incident: SUPPLIER_DEPENDENCY_FAILURE (severity >= MEDIUM)
    - Incident: FULFILLMENT_SLA_DEGRADATION (severity >= MEDIUM) detected 2-4 days later
    - Cascade linking both
    - RCA top path includes supplier_delay_rate as root
    - Postmortem generated with monitor for supplier_delay_rate
    - Monitor threshold set at 2.5σ

Scenario 2: "Refund Spike"
  Input: Normal events + 5-day cluster of REFUND_ISSUED (3x normal rate)
  Expected: 
    - Incident: REFUND_SPIKE (severity >= HIGH)
    - RCA evidence includes the refund event cluster
    - Blast radius includes affected customer count

Scenario 3: "Order Volume Surge Cascade"
  Input: Normal events + 3-day 200% order volume spike + no fulfillment capacity increase
  Expected:
    - Incident: FULFILLMENT_SLA_DEGRADATION
    - Incident: SUPPORT_LOAD_SURGE (3-5 days later)
    - Cascade linking both
    - RCA identifies order_volume as root cause

Scenario 4: "Churn Acceleration from Support Failure"
  Input: Normal events + ticket backlog grows for 14 days + resolution time doubles
  Expected:
    - Incident: SUPPORT_LOAD_SURGE
    - Incident: CHURN_ACCELERATION (7-10 days later)
    - RCA chain: ticket_backlog → avg_resolution_time → churn_proxy

Scenario 5: "Liquidity Crunch from AR Aging"
  Input: Normal events + 20 invoices go overdue simultaneously
  Expected:
    - Incident: LIQUIDITY_CRUNCH_RISK
    - RCA identifies ar_aging_amount as primary cause
    - Blast radius shows overdue dollar amount
```

## Property-Based Tests (using Hypothesis)
- "For any valid set of canonical events, the state builder never produces negative order counts"
- "For any incident, the postmortem always contains at least one monitor rule"
- "For any two identical event streams with identical config, the system produces identical incidents"

---

# SECTION 10: OBSERVABILITY

- **Structured logging:** Every log entry is JSON with `request_id`, `run_id`, `component`, `event`, `duration_ms`
- **Metrics:** Track key operational metrics:
  - `bre.ingestion.events_processed` (counter)
  - `bre.scan.duration_seconds` (histogram)
  - `bre.incidents.detected` (counter, by type)
  - `bre.rca.computation_seconds` (histogram)
  - `bre.monitors.alerts_fired` (counter)
- **Health endpoint:** `/v1/health` returns component status (QBO connection, last scan time, active monitors, data freshness)
- **Run manifest:** Every scan produces a manifest with exactly what was processed, which versions were used, and what was produced. This is the audit trail.

---

# SECTION 11: USE CASES

## Use Case A: Marketing spike → operational failure → churn risk

1. Campaign starts → orders surge
2. Fulfillment delays rise → tickets spike → review score drops
3. System detects "Fulfillment SLA degradation" and "Support surge"
4. Cascade correlator links them
5. RCA chain ties campaign start (order_volume spike) + shipping delay clusters + backlog growth
6. Postmortem suggests monitors: order/fulfillment ratio, delay thresholds, ticket queue alarms
7. Monitor catches similar pattern 2 months later, fires early warning

## Use Case B: Supply chain delay → stockouts → revenue concentration exposure

1. Supplier delays increase
2. Certain SKUs go out of stock
3. Repeat customers stop buying → churn proxy rises
4. System detects "Supplier Dependency Failure" and "Churn Acceleration"
5. RCA points to supplier nodes/routes in logistics data and SKU impact concentration
6. Blast radius shows 23 high-value customers at risk, $42K revenue exposure

## Use Case C: Quality regression → refund spike → margin compression

1. Refund events spike
2. Review sentiment declines
3. Expense ratio rises due to returns/shipping
4. Incident: "Refund spike" + "Margin compression"
5. RCA chain ties specific product categories + shipping delay correlation
6. Incident comparator shows this is different from last quarter's refund spike (which was shipping-driven)

## Use Case D: AR delays → liquidity crunch

1. Multiple large invoices go overdue simultaneously
2. Cash position weakens
3. System detects "Liquidity Crunch Risk"
4. RCA identifies the specific customers and invoice amounts
5. Blast radius: $67K in overdue receivables, vendor payment obligations at risk
6. What-if simulation: "If 50% of overdue invoices are collected within 7 days, liquidity risk drops to LOW"

---

# SECTION 12: WHAT YOU TELL JUDGES

## The elevator pitch (30 seconds)

"I built a Business Reliability Engine — it's like PagerDuty for your business operations. It connects to QuickBooks, turns accounting activity into an event stream, detects business incidents like refund spikes or cash flow risks, traces the root cause using a formalized algorithm I designed called BRE-RCA, generates postmortems with blast radius calculations, and creates prevention monitors that actually run and catch recurrences. The ML is supporting infrastructure — the core innovation is the reliability system itself."

## When asked about the RCA algorithm (2 minutes)

"BRE-RCA builds a directed acyclic graph of business metric dependencies — like supplier delays cause fulfillment backlogs which cause delivery delays. When an incident fires, I look at every upstream metric in the graph, compute a contribution score based on four factors: how abnormal was it, did it precede the incident temporally, how close is it in the dependency graph, and how trustworthy is the underlying data. The output is a ranked list of causal paths, each with specific evidence event IDs that a human can verify. I'm not claiming true causality — I'm providing a debugger-quality explanation that's transparent and auditable."

## When asked about the closed loop (1 minute)

"The system doesn't just detect and report — it generates monitors derived from the actual causal chain and runs them against incoming data. I can demo this: here's an incident from March, here's the monitor it generated, and here's the monitor firing on a similar pattern in June, three days before it would have become a full incident. That's the SRE principle: every postmortem produces action items that prevent recurrence."

## When asked "what's novel?" (30 seconds)

"Three things no existing business tool does: first, cross-domain incident detection that connects financial, operational, and customer signals. Second, evidence-linked root cause attribution with a transparent scoring algorithm. Third, a closed-loop prevention system where monitors are born from actual incidents and verified against real data."

## What you tell Intuit sponsors

"This connects to QuickBooks via OAuth and webhooks, turns accounting activity into an event stream, detects business incidents, and generates a postmortem + prevention monitors — like SRE for SMB operations."

## What you tell Databricks sponsors

"This is lakehouse-native: raw QBO objects into bronze Delta, normalized events in silver, business health metrics in gold, governed models in Unity Catalog, and an admin app deployed as a Databricks App."

---

# SECTION 13: COMPLETE CHECKLIST

- [x] Clear data contracts & schema versioning (Pydantic v2, schema_version field)
- [x] Idempotent ingestion & reproducible runs (event sourcing, run manifests)
- [x] Evidence-linked RCA outputs with formalized BRE-RCA algorithm
- [x] Incident cascade correlation (temporal + entity overlap + causal plausibility)
- [x] Blast radius calculation (customers, orders, revenue exposure, churn exposure)
- [x] Closed-loop monitor generation AND execution with demo
- [x] What-if simulation engine
- [x] Incident comparison across time periods
- [x] Data quality scoring with confidence adjustment
- [x] Confidence calibration framework (HIGH/MEDIUM/LOW with explicit criteria)
- [x] Golden tests (5 complete scenarios with expected outputs)
- [x] Property-based tests (Hypothesis)
- [x] Ablation study (rules-only vs. hybrid)
- [x] Time-based train/test splits
- [x] MLflow model registry integration
- [x] Deployment story (Docker Compose local + Databricks production)
- [x] One-command demo
- [x] Structured JSON logging with request_id correlation
- [x] Health endpoint and operational metrics
- [x] Run manifests for auditability
- [x] PDF/Markdown/JSON export for postmortems
- [x] Interactive causal chain visualization (Cytoscape.js) as UI centerpiece
- [x] QBO OAuth + Webhooks integration
- [x] Delta Lake event store
- [x] Admin-only auth (JWT)
- [x] API documentation (auto-generated OpenAPI from FastAPI)
- [x] Architecture documentation with diagrams
- [x] 8 incident types across 3 domains (financial, operational, customer)
- [x] Incident Dependency Graph for cascade detection
- [x] Business Dependency Graph for RCA
- [x] 7 UI screens with centerpiece causal chain visualization
- [x] Complete API with 30+ endpoints
- [x] 4 supplemental datasets mapped to canonical events
- [x] QBO sandbox seeding script for controlled testing

---

# SECTION 14: IMPROVEMENTS ADDED BEYOND ORIGINAL SPEC

1. **BRE-RCA formalized algorithm** with contribution scoring formula (anomaly_magnitude × temporal_precedence × graph_proximity × data_quality_weight)
2. **Incident Cascade Correlator** with cascade scoring (temporal_weight × entity_overlap × causal_plausibility)
3. **Blast Radius Calculator** with dollar exposure, customer count, churn exposure, and severity classification
4. **Closed-Loop Monitor Runtime** that actually runs generated monitors and catches recurrences
5. **What-If Simulation Engine** with perturbation propagation through dependency graph
6. **Incident Comparator** for cross-period root cause comparison
7. **Data Quality Scoring** at ingestion with per-metric confidence adjustment in RCA
8. **Confidence Calibration Framework** with explicit criteria per level (MEDIUM/HIGH/VERY_HIGH based on detection method agreement)
9. **Timeline Reconstruction** in postmortems with timestamped event descriptions
10. **Golden Test Strategy** with 5 complete end-to-end scenarios
11. **Ablation Study** plan comparing rules-only vs. hybrid detection
12. **Interactive Causal Chain Visualization** spec with Cytoscape.js dagre layout and cascade animation
13. **8 incident types** (up from 5) including Supplier Dependency Failure and Customer Satisfaction Regression
14. **4 use cases** with complete walkthrough including closed-loop demo moments
15. **30+ API endpoints** covering all features including simulation, comparison, and monitor management
16. **7 UI screens** (up from 4) including health dashboard, monitors dashboard, and comparison/simulation
17. **QBO sandbox seeding script** for deterministic, controlled testing
18. **Export formats** (PDF, Markdown, JSON) for postmortems
19. **Incident Dependency Graph** (separate from Business Dependency Graph) for cascade plausibility scoring
20. **Run manifests** for complete auditability of every scan
