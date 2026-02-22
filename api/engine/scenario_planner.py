"""
Scenario Planning Engine — What-If for Expenses and Invoices.

Agents: data-scientist, fintech-engineer
Answers: "If I cut expenses by 10%, what happens to runway?"
"If I collect these 5 invoices, cash runway improves to X months."
Connects to existing simulation engine and cash runway predictor.
"""

from datetime import date
from typing import Optional

import structlog

from api.storage.base import StorageBackend

logger = structlog.get_logger()


class ScenarioPlannerService:
    """
    What-if scenario planning for cash flow.

    Supports expense cuts and invoice collection scenarios.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.logger = structlog.get_logger()

    def run_expense_scenario(
        self,
        expense_cut_pct: float,
        lookback_days: int = 60,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Simulate: "If I cut expenses by X%, what happens to runway?"

        Returns:
            {
                "current_runway_months": float | None,
                "scenario_runway_months": float | None,
                "expense_cut_pct": float,
                "monthly_burn_before": float,
                "monthly_burn_after": float,
                "runway_improvement_months": float | None,
                "narrative": str,
            }
        """
        from api.engine.prediction.cash_runway import CashRunwayPredictor

        predictor = CashRunwayPredictor(storage=self.storage, lookback_days=lookback_days)
        pred = predictor.predict(target_date=target_date)

        net_cash = pred.get("net_cash_proxy", 0.0)
        monthly_burn = pred.get("monthly_burn_rate", 0.0)
        current_runway = pred.get("runway_months")

        # Assume ~70% of burn is expenses (rest is revenue shortfall)
        expense_portion = 0.7
        burn_reduction = monthly_burn * expense_portion * (expense_cut_pct / 100)
        new_burn = monthly_burn - burn_reduction

        scenario_runway = None
        if new_burn > 0 and net_cash > 0:
            scenario_runway = net_cash / new_burn
        elif new_burn <= 0:
            scenario_runway = float("inf")

        improvement = None
        if current_runway and scenario_runway and scenario_runway != float("inf"):
            improvement = scenario_runway - current_runway
        elif current_runway is None and scenario_runway and scenario_runway != float("inf"):
            improvement = scenario_runway

        narrative = (
            f"If you cut expenses by {expense_cut_pct:.0f}%, "
            f"monthly burn drops from ${monthly_burn:,.0f} to ${new_burn:,.0f}. "
        )
        if improvement and improvement > 0:
            narrative += f"Cash runway extends by {improvement:.1f} months."
        elif scenario_runway == float("inf"):
            narrative += "You would achieve positive cash flow."
        else:
            narrative += f"Runway would be {scenario_runway:.1f} months."

        self.logger.info(
            "expense_scenario_computed",
            expense_cut_pct=expense_cut_pct,
            runway_before=current_runway,
            runway_after=scenario_runway,
        )

        return {
            "current_runway_months": current_runway,
            "scenario_runway_months": scenario_runway if scenario_runway != float("inf") else None,
            "runway_infinite_after": scenario_runway == float("inf"),
            "expense_cut_pct": expense_cut_pct,
            "monthly_burn_before": round(monthly_burn, 2),
            "monthly_burn_after": round(new_burn, 2),
            "runway_improvement_months": round(improvement, 1) if improvement else None,
            "narrative": narrative,
        }

    def run_invoice_collection_scenario(
        self,
        invoice_ids: Optional[list[str]] = None,
        collection_amount: Optional[float] = None,
        top_n_invoices: int = 5,
        lookback_days: int = 60,
        target_date: Optional[date] = None,
    ) -> dict:
        """
        Simulate: "If I collect these invoices, cash runway improves to X months."

        Either pass specific invoice_ids, collection_amount, or use top_n
        from invoice default risk (highest priority to chase).
        """
        from api.engine.invoice_default_risk import InvoiceDefaultRiskScorer
        from api.engine.prediction.cash_runway import CashRunwayPredictor

        predictor = CashRunwayPredictor(storage=self.storage, lookback_days=lookback_days)
        pred = predictor.predict(target_date=target_date)

        net_cash = pred.get("net_cash_proxy", 0.0)
        monthly_burn = pred.get("monthly_burn_rate", 0.0)
        current_runway = pred.get("runway_months")

        if collection_amount is not None:
            amount = collection_amount
            invoices_used = []
        else:
            scorer = InvoiceDefaultRiskScorer(
                storage=self.storage,
                lookback_days=90,
                top_n=top_n_invoices,
            )
            result = scorer.score_receivables(target_date=target_date)
            receivables = result.get("receivables", [])
            if invoice_ids:
                amount = sum(r["amount"] for r in receivables if r.get("entity_id") in invoice_ids)
                invoices_used = [r for r in receivables if r.get("entity_id") in invoice_ids]
            else:
                amount = sum(r["amount"] for r in receivables[:top_n_invoices])
                invoices_used = receivables[:top_n_invoices]

        new_cash = net_cash + amount
        scenario_runway = None
        if monthly_burn > 0 and new_cash > 0:
            scenario_runway = new_cash / monthly_burn
        elif monthly_burn <= 0:
            scenario_runway = float("inf")

        improvement = None
        if current_runway and scenario_runway and scenario_runway != float("inf"):
            improvement = scenario_runway - current_runway
        elif current_runway is None and scenario_runway and scenario_runway != float("inf"):
            improvement = scenario_runway

        narrative = (
            f"Collecting ${amount:,.0f} in receivables would add to your cash position. "
            f"Runway improves from {current_runway or 'N/A'} months to "
            f"{scenario_runway:.1f if scenario_runway != float('inf') else '∞'} months."
        )
        if improvement and improvement > 0:
            narrative += f" That's {improvement:.1f} additional months of runway."
        if invoices_used:
            narrative += f" (Top {len(invoices_used)} prioritized invoices.)"

        self.logger.info(
            "invoice_collection_scenario_computed",
            collection_amount=amount,
            runway_before=current_runway,
            runway_after=scenario_runway,
        )

        return {
            "current_runway_months": current_runway,
            "scenario_runway_months": scenario_runway if scenario_runway != float("inf") else None,
            "runway_infinite_after": scenario_runway == float("inf"),
            "collection_amount": round(amount, 2),
            "invoices_count": len(invoices_used),
            "runway_improvement_months": round(improvement, 1) if improvement else None,
            "narrative": narrative,
            "invoices": [{"entity_id": r.get("entity_id"), "amount": r.get("amount")} for r in invoices_used[:5]],
        }
