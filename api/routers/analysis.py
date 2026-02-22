"""
On-demand analysis router - Trigger anomaly detection and RCA.

Wired to:
- EnsembleDetector for anomaly detection
- CascadeCorrelator for cascade analysis
- RootCauseAnalyzer for RCA
- BlastRadiusMapper for impact assessment
- PostmortemGenerator for report generation
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _restructure_metrics(flat_metrics: list[dict]) -> dict:
    """
    Restructure flat Gold metrics into nested dict by domain.

    Args:
        flat_metrics: List of flat metric dicts from storage

    Returns:
        Nested dict with financial, operational, customer domains
    """
    financial_names = {
        "daily_revenue", "daily_expenses", "daily_refunds", "refund_rate",
        "net_cash_proxy", "expense_ratio", "margin_proxy", "dso_proxy",
        "ar_aging_amount", "ar_overdue_count", "dpo_proxy"
    }
    operational_names = {
        "order_volume", "delivery_count", "late_delivery_count",
        "delivery_delay_rate", "fulfillment_backlog", "avg_delivery_delay_days",
        "supplier_delay_rate", "supplier_delay_severity"
    }
    customer_names = {
        "ticket_volume", "ticket_close_volume", "ticket_backlog",
        "avg_resolution_time", "review_score_avg", "review_score_trend",
        "churn_proxy", "customer_concentration"
    }

    result = {"financial": {}, "operational": {}, "customer": {}}

    for metric in flat_metrics:
        metric_name = metric.get("metric_name")
        metric_value = metric.get("metric_value")

        if metric_name in financial_names:
            result["financial"][metric_name] = metric_value
        elif metric_name in operational_names:
            result["operational"][metric_name] = metric_value
        elif metric_name in customer_names:
            result["customer"][metric_name] = metric_value

    return result


class AnalysisRequest(BaseModel):
    """
    Analysis request with validated parameters.

    Backend-developer agent: Input validation and sanitization.
    """

    lookback_days: int = 30
    min_zscore: float = 3.0
    run_rca: bool = True
    run_blast_radius: bool = True
    run_postmortem: bool = True

    @property
    def validated_lookback_days(self) -> int:
        """Clamp lookback_days to [1, 365] for safety."""
        return max(1, min(365, self.lookback_days))

    @property
    def validated_min_zscore(self) -> float:
        """Clamp min_zscore to [1.0, 10.0] for practical detection."""
        return max(1.0, min(10.0, self.min_zscore))


@router.post("/run")
async def run_analysis(
    request: AnalysisRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Run full BRE analysis pipeline: detection → cascade → RCA → blast radius → postmortem.

    This is the primary endpoint that orchestrates the complete analysis flow.
    """
    from datetime import datetime
    from uuid import uuid4

    run_id = f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    logger.info(
        "analysis_started",
        realm_id=realm_id,
        run_id=run_id,
        lookback_days=request.lookback_days,
    )

    storage = get_storage()

    results = {
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "incidents_detected": 0,
        "cascades_detected": 0,
        "rca_completed": 0,
        "blast_radii_computed": 0,
        "postmortems_generated": 0,
        "incidents": [],
    }

    try:
        # Step 1: Build state (Gold metrics) from canonical events
        from datetime import date, timedelta
        from api.engine.state_builder import StateBuilder

        builder = StateBuilder(storage=storage)

        # Get Gold metrics for the lookback window
        end_date = date.today()
        start_date = end_date - timedelta(days=request.validated_lookback_days)

        gold_metrics = storage.read_gold_metrics(
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Restructure flat Gold metrics into nested dict format
        current_metrics = {"financial": {}, "operational": {}, "customer": {}}
        historical_metrics = []

        if gold_metrics:
            # Group by date
            by_date = {}
            for metric in gold_metrics:
                metric_date = metric.get("metric_date", str(end_date))
                if metric_date not in by_date:
                    by_date[metric_date] = []
                by_date[metric_date].append(metric)

            # Convert to nested dicts
            sorted_dates = sorted(by_date.keys())
            for metric_date in sorted_dates[:-1]:  # All except last are historical
                historical_metrics.append(_restructure_metrics(by_date[metric_date]))

            # Last date is current
            if sorted_dates:
                current_metrics = _restructure_metrics(by_date[sorted_dates[-1]])
        else:
            # No Gold metrics exist, compute on-the-fly
            current_metrics = builder.compute_daily_metrics(end_date - timedelta(days=1))
            for i in range(request.validated_lookback_days - 1):
                hist_date = end_date - timedelta(days=i + 2)
                historical_metrics.append(builder.compute_daily_metrics(hist_date))

        # Read canonical events
        events = storage.read_canonical_events(limit=10000)

        # Step 2: Run detection
        from api.engine.detection.ensemble import EnsembleDetector

        detector = EnsembleDetector()
        incidents = detector.run_full_detection(
            current_metrics=current_metrics,
            historical_metrics=historical_metrics,
            events=events,
            run_id=run_id,
        )

        # Store detected incidents
        for incident in incidents:
            storage.write_incident(incident)

        results["incidents_detected"] = len(incidents)

        # Step 3: Cascade correlation
        if len(incidents) >= 2:
            from api.engine.cascade_correlator import CascadeCorrelator

            correlator = CascadeCorrelator()
            cascades = correlator.correlate(incidents)

            # Store cascades
            for cascade in cascades:
                storage.write_cascade(cascade)

            results["cascades_detected"] = len(cascades)

        # Step 4: RCA for each incident
        if request.run_rca:
            from api.engine.rca import RootCauseAnalyzer

            analyzer = RootCauseAnalyzer(storage=storage)
            for incident in incidents:
                try:
                    chain = analyzer.analyze(
                        incident=incident,
                        lookback_days=request.lookback_days,
                    )
                    results["rca_completed"] += 1
                except Exception as e:
                    logger.warning(
                        "rca_failed",
                        incident_id=incident.incident_id,
                        error=str(e),
                    )

        # Step 5: Blast radius for each incident
        if request.run_blast_radius:
            from api.engine.blast_radius import BlastRadiusMapper

            mapper = BlastRadiusMapper(storage=storage)
            for incident in incidents:
                try:
                    blast = mapper.compute_blast_radius(incident)
                    results["blast_radii_computed"] += 1
                except Exception as e:
                    logger.warning(
                        "blast_radius_failed",
                        incident_id=incident.incident_id,
                        error=str(e),
                    )

        # Step 6: Postmortems
        if request.run_postmortem:
            from api.engine.postmortem_generator import PostmortemGenerator

            generator = PostmortemGenerator(storage=storage)
            for incident in incidents:
                try:
                    chain = storage.read_causal_chain(incident.incident_id)
                    blast = storage.read_blast_radius(incident.incident_id)
                    if chain and blast:
                        pm = generator.generate(
                            incident=incident,
                            causal_chain=chain,
                            blast_radius=blast,
                        )
                        results["postmortems_generated"] += 1
                except Exception as e:
                    logger.warning(
                        "postmortem_failed",
                        incident_id=incident.incident_id,
                        error=str(e),
                    )

        # Summarize incidents
        for inc in incidents:
            results["incidents"].append({
                "incident_id": inc.incident_id,
                "incident_type": inc.incident_type.value,
                "severity": inc.severity.value,
                "primary_metric_zscore": inc.primary_metric_zscore,
            })

        results["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        logger.error("analysis_pipeline_failed", run_id=run_id, error=str(e))
        results["error"] = str(e)

    return {"success": True, "data": results}


@router.get("/result/{run_id}")
async def get_analysis_result(
    run_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get analysis run results by run_id.
    Queries incidents, RCA chains, and blast radii associated with the run.
    """
    logger.info("analysis_result_fetch", realm_id=realm_id, run_id=run_id)

    storage = get_storage()

    # Find incidents from this run
    all_incidents = storage.read_incidents()
    run_incidents = [i for i in all_incidents if i.run_id == run_id]

    if not run_incidents:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for analysis run {run_id}",
        )

    results = []
    for inc in run_incidents:
        chain = storage.read_causal_chain(inc.incident_id)
        blast = storage.read_blast_radius(inc.incident_id)
        postmortem = storage.read_postmortem(inc.incident_id)

        results.append({
            "incident": inc.model_dump(mode="json"),
            "causal_chain": chain.model_dump(mode="json") if chain else None,
            "blast_radius": blast.model_dump(mode="json") if blast else None,
            "postmortem": postmortem.model_dump(mode="json") if postmortem else None,
        })

    return {
        "success": True,
        "data": {
            "run_id": run_id,
            "incidents_count": len(results),
            "results": results,
        },
    }
