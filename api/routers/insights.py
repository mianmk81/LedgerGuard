"""
Business Insights Router (H4-H7 + extended features).

Exposes hackathon competitiveness features:
- H4: Cash Runway prediction
- H5: Invoice default risk
- H6: Support ticket sentiment
- H7: Tax liability estimate
- Recommendations, cash forecast curve, invoice follow-up priorities
- Period comparison, scenario planning, recurring patterns
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth.dependencies import get_current_realm_id
from api.engine.invoice_default_risk import InvoiceDefaultRiskScorer
from api.engine.period_comparison import PeriodComparisonService
from api.engine.prediction.cash_runway import CashRunwayPredictor
from api.engine.recommendation_feed import RecommendationFeedService
from api.engine.recurring_patterns import RecurringPatternsService
from api.engine.scenario_planner import ScenarioPlannerService
from api.engine.sentiment import TicketSentimentAnalyzer
from api.engine.tax_liability import TaxLiabilityEstimator
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/cash-runway")
async def get_cash_runway(
    realm_id: str = Depends(get_current_realm_id),
    lookback_days: int = Query(60, ge=7, le=365),
):
    """
    H4: Cash Runway prediction.

    Returns months until cash runs out based on Gold metrics burn rate.
    Includes 95% confidence intervals (data-scientist methodology).
    """
    logger.info("insights_cash_runway", realm_id=realm_id)
    storage = get_storage()
    predictor = CashRunwayPredictor(storage=storage, lookback_days=lookback_days)
    result = predictor.predict()
    return {"success": True, "data": result}


@router.get("/invoice-default-risk")
async def get_invoice_default_risk(
    realm_id: str = Depends(get_current_realm_id),
    top_n: int = Query(10, ge=1, le=50),
    lookback_days: int = Query(90, ge=30, le=365),
):
    """
    H5: Invoice default risk scoring.

    Returns top receivables by default/late risk with scores and factors.
    Uses aging, amount, and customer payment history.
    """
    logger.info("insights_invoice_risk", realm_id=realm_id)
    storage = get_storage()
    scorer = InvoiceDefaultRiskScorer(
        storage=storage,
        lookback_days=lookback_days,
        top_n=top_n,
    )
    result = scorer.score_receivables()
    return {"success": True, "data": result}


@router.get("/support-ticket-sentiment")
async def get_support_ticket_sentiment(
    realm_id: str = Depends(get_current_realm_id),
    top_n: int = Query(20, ge=1, le=100),
    lookback_days: int = Query(30, ge=7, le=90),
):
    """
    H6: Support ticket sentiment analysis.

    NLP on ticket subject → risk signal for churn/health.
    Flags "Escalating frustration" tickets.
    """
    logger.info("insights_ticket_sentiment", realm_id=realm_id)
    storage = get_storage()
    analyzer = TicketSentimentAnalyzer(
        storage=storage,
        lookback_days=lookback_days,
        top_n=top_n,
    )
    result = analyzer.analyze()
    return {"success": True, "data": result}


@router.get("/tax-liability-estimate")
async def get_tax_liability_estimate(
    realm_id: str = Depends(get_current_realm_id),
    lookback_days: int = Query(365, ge=30, le=730),
    effective_rate: Optional[float] = Query(None, ge=0, le=1),
):
    """
    H7: Tax liability estimate.

    From P&L (revenue, expenses) → Estimated tax liability + audit risk flag.
    """
    logger.info("insights_tax_liability", realm_id=realm_id)
    storage = get_storage()
    estimator = TaxLiabilityEstimator(storage=storage, lookback_days=lookback_days)
    result = estimator.estimate(effective_rate=effective_rate)
    return {"success": True, "data": result}


# --- Extended features ---


@router.get("/recommendations")
async def get_recommendations(
    realm_id: str = Depends(get_current_realm_id),
    top_n: int = Query(5, ge=1, le=10),
):
    """
    Top N actions to improve business health score.

    Driven by Credit Pulse weak metrics, postmortem recommendations, blast radius.
    """
    logger.info("insights_recommendations", realm_id=realm_id)
    storage = get_storage()
    service = RecommendationFeedService(storage=storage, top_n=top_n)
    result = service.get_recommendations(realm_id=realm_id)
    return {"success": True, "data": result}


@router.get("/cash-forecast-curve")
async def get_cash_forecast_curve(
    realm_id: str = Depends(get_current_realm_id),
    projection_months: int = Query(6, ge=3, le=12),
    lookback_days: int = Query(60, ge=7, le=365),
):
    """
    Cash flow forecast curve over next N months.

    Returns baseline, best-case, worst-case monthly projections (95% CI bands).
    """
    logger.info("insights_cash_forecast", realm_id=realm_id)
    storage = get_storage()
    predictor = CashRunwayPredictor(storage=storage, lookback_days=lookback_days)
    result = predictor.forecast_curve(projection_months=projection_months)
    return {"success": True, "data": result}


@router.get("/invoice-follow-up-priorities")
async def get_invoice_follow_up_priorities(
    realm_id: str = Depends(get_current_realm_id),
    top_n: int = Query(5, ge=1, le=20),
    lookback_days: int = Query(90, ge=30, le=365),
):
    """
    Prioritized list of invoices to chase today.

    Ranked by aging, amount, and customer default risk.
    """
    logger.info("insights_invoice_follow_up", realm_id=realm_id)
    storage = get_storage()
    scorer = InvoiceDefaultRiskScorer(
        storage=storage,
        lookback_days=lookback_days,
        top_n=top_n,
    )
    result = scorer.score_receivables()
    result["action_label"] = f"Top {top_n} to follow up today"
    return {"success": True, "data": result}


@router.get("/period-comparison")
async def get_period_comparison(
    realm_id: str = Depends(get_current_realm_id),
    period: str = Query("month", pattern="^(month|quarter)$"),
):
    """
    This month vs last month, or this quarter vs last quarter.

    Compares health score, key metrics, incident count.
    """
    logger.info("insights_period_comparison", realm_id=realm_id, period=period)
    storage = get_storage()
    service = PeriodComparisonService(storage=storage)
    result = service.compare(period=period)
    return {"success": True, "data": result}


@router.get("/scenario-expense")
async def run_expense_scenario(
    realm_id: str = Depends(get_current_realm_id),
    expense_cut_pct: float = Query(10, ge=1, le=50),
    lookback_days: int = Query(60, ge=7, le=365),
):
    """
    What-if: If I cut expenses by X%, what happens to runway?
    """
    logger.info("insights_scenario_expense", realm_id=realm_id)
    storage = get_storage()
    service = ScenarioPlannerService(storage=storage)
    result = service.run_expense_scenario(
        expense_cut_pct=expense_cut_pct,
        lookback_days=lookback_days,
    )
    return {"success": True, "data": result}


@router.get("/scenario-invoice-collection")
async def run_invoice_collection_scenario(
    realm_id: str = Depends(get_current_realm_id),
    top_n: int = Query(5, ge=1, le=20),
    lookback_days: int = Query(60, ge=7, le=365),
):
    """
    What-if: If I collect these top invoices, cash runway improves to X months.
    """
    logger.info("insights_scenario_invoice", realm_id=realm_id)
    storage = get_storage()
    service = ScenarioPlannerService(storage=storage)
    result = service.run_invoice_collection_scenario(
        top_n_invoices=top_n,
        lookback_days=lookback_days,
    )
    return {"success": True, "data": result}


@router.get("/recurring-patterns")
async def get_recurring_patterns(
    realm_id: str = Depends(get_current_realm_id),
    lookback_days: int = Query(365, ge=90, le=730),
):
    """
    Recurring incident patterns — seasonality detection.

    E.g. "Refund spikes tend to recur every quarter."
    """
    logger.info("insights_recurring_patterns", realm_id=realm_id)
    storage = get_storage()
    service = RecurringPatternsService(storage=storage, lookback_days=lookback_days)
    result = service.get_patterns()
    return {"success": True, "data": result}
