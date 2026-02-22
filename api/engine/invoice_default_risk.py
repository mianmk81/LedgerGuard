"""
Invoice Default Risk Scoring (H5) - Data Scientist / Fintech Role.

Scores top receivables by default/late risk using payment history,
aging, and amount. Classification-style scoring for SMB credit risk.
"""

import math
from datetime import date, datetime, timedelta
from typing import Optional

import structlog

from api.models.events import CanonicalEvent
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class InvoiceDefaultRiskScorer:
    """
    Scores invoices/receivables by default and late payment risk.

    Uses:
    - Aging (days overdue) - primary driver
    - Amount (larger = higher exposure)
    - Customer payment history (DSO, prior late payments) when available

    Output: Top N receivables with risk score and explainability.
    """

    def __init__(
        self,
        storage: StorageBackend,
        lookback_days: int = 90,
        top_n: int = 10,
    ):
        self.storage = storage
        self.lookback_days = lookback_days
        self.top_n = top_n

    def score_receivables(
        self,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Score top receivables by default/late risk.

        Returns:
            {
                "receivables": [{
                    "entity_id": str,
                    "customer_id": str | None,
                    "amount": float,
                    "aging_days": int,
                    "risk_score": float,
                    "risk_bucket": str,
                    "factors": [str],
                }],
                "summary": {...}
            }
        """
        target = target_date or date.today()
        end_str = (target + timedelta(days=1)).isoformat()
        start_str = (target - timedelta(days=self.lookback_days)).isoformat()

        events = self.storage.read_canonical_events(
            event_type="invoice_overdue",
            start_time=start_str,
            end_time=end_str,
            limit=5000,
        )

        issued = self.storage.read_canonical_events(
            event_type="invoice_issued",
            start_time=start_str,
            end_time=end_str,
            limit=5000,
        )

        invoice_state: dict[str, dict] = {}

        for e in events:
            if not isinstance(e, CanonicalEvent):
                continue
            eid = e.entity_id
            amt = e.amount or 0.0
            bal = e.attributes.get("balance_remaining", amt)
            due = e.attributes.get("due_date")
            cust = e.related_entity_ids.get("customer") if e.related_entity_ids else None
            days_overdue = e.attributes.get("days_overdue", 30)

            invoice_state[eid] = {
                "entity_id": eid,
                "customer_id": cust,
                "amount": amt,
                "balance": bal,
                "days_overdue": int(days_overdue) if isinstance(days_overdue, (int, float)) else 30,
                "due_date": due,
            }

        for e in issued:
            if not isinstance(e, CanonicalEvent):
                continue
            eid = e.entity_id
            if eid in invoice_state:
                continue
            amt = e.amount or 0.0
            bal = e.attributes.get("balance_remaining", amt)
            due = e.attributes.get("due_date")
            cust = e.related_entity_ids.get("customer") if e.related_entity_ids else None
            due_dt = None
            if due:
                try:
                    s = str(due)[:10]
                    due_dt = datetime.strptime(s, "%Y-%m-%d").date()
                except Exception:
                    pass
            evt_date = e.event_time.date() if hasattr(e.event_time, "date") else target
            days_overdue = (target - due_dt).days if due_dt and due_dt < target else 0

            if bal > 0:
                invoice_state[eid] = {
                    "entity_id": eid,
                    "customer_id": cust,
                    "amount": amt,
                    "balance": bal,
                    "days_overdue": max(0, days_overdue),
                    "due_date": due,
                }

        paid = self.storage.read_canonical_events(
            event_type="invoice_paid",
            start_time=start_str,
            end_time=end_str,
            limit=2000,
        )
        customer_dso: dict[str, float] = {}
        customer_late_count: dict[str, int] = {}
        cust_counts: dict[str, int] = {}
        for e in paid:
            if not isinstance(e, CanonicalEvent):
                continue
            cust = e.related_entity_ids.get("customer") if e.related_entity_ids else None
            if not cust:
                continue
            days = e.attributes.get("days_to_pay") or 30
            days = int(days) if isinstance(days, (int, float)) else 30
            customer_dso[cust] = customer_dso.get(cust, 0) + days
            if days > 30:
                customer_late_count[cust] = customer_late_count.get(cust, 0) + 1
            cust_counts[cust] = cust_counts.get(cust, 0) + 1

        customer_dso_avg: dict[str, float] = {}
        for c, total in customer_dso.items():
            n = cust_counts.get(c, 1)
            customer_dso_avg[c] = total / n

        scored = []
        for inv in invoice_state.values():
            bal = inv["balance"]
            if bal <= 0:
                continue
            aging = inv["days_overdue"]
            cust = inv.get("customer_id")
            dso_avg = customer_dso_avg.get(cust, 30.0) if cust else 30.0
            late_cnt = customer_late_count.get(cust, 0) if cust else 0

            aging_score = min(90, aging / 30 * 25)
            amt_score = min(30, math.log10(max(1, bal / 100)) * 10)
            hist_score = min(25, (dso_avg - 30) / 30 * 15 + late_cnt * 5)

            risk_score = min(100, aging_score + amt_score + hist_score)

            if risk_score < 25:
                bucket = "low"
            elif risk_score < 50:
                bucket = "medium"
            elif risk_score < 75:
                bucket = "high"
            else:
                bucket = "critical"

            factors = []
            if aging > 30:
                factors.append(f"{aging} days overdue")
            if bal > 5000:
                factors.append(f"High balance: ${bal:,.0f}")
            if dso_avg > 45:
                factors.append(f"Customer DSO: {dso_avg:.0f}d")
            if late_cnt > 0:
                factors.append(f"{late_cnt} prior late payment(s)")
            if not factors:
                factors.append("Baseline risk")

            scored.append({
                "entity_id": inv["entity_id"],
                "customer_id": cust,
                "amount": round(bal, 2),
                "aging_days": aging,
                "risk_score": round(risk_score, 1),
                "risk_bucket": bucket,
                "factors": factors,
            })

        scored.sort(key=lambda x: (-x["risk_score"], -x["amount"]))
        top = scored[: self.top_n]

        total_at_risk = sum(s["amount"] for s in top)
        high_risk_count = sum(1 for s in top if s["risk_bucket"] in ("high", "critical"))

        logger.info(
            "invoice_default_risk_computed",
            receivables_count=len(top),
            total_at_risk=total_at_risk,
            high_risk_count=high_risk_count,
        )

        return {
            "receivables": top,
            "summary": {
                "total_receivables_scored": len(scored),
                "top_n": self.top_n,
                "total_amount_at_risk": round(total_at_risk, 2),
                "high_risk_count": high_risk_count,
                "target_date": target.isoformat(),
            },
        }
