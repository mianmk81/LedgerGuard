"""
Incident management router - List, detail, and postmortem generation.

Wired to:
- StorageBackend for incident/RCA/blast radius queries
- PostmortemGenerator for on-demand postmortem assembly
- RootCauseAnalyzer + BlastRadiusMapper for on-demand computation
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from api.auth.dependencies import get_current_realm_id
from api.storage import get_storage
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _ensure_rca_and_blast_radius(storage, incident):
    """
    Compute RCA and blast radius on-the-fly if they don't exist yet.
    Returns (causal_chain, blast_radius) — either from storage or freshly computed.
    """
    causal_chain = storage.read_causal_chain(incident.incident_id)
    blast_radius = storage.read_blast_radius(incident.incident_id)

    if causal_chain is None:
        try:
            from api.engine.rca import RootCauseAnalyzer
            analyzer = RootCauseAnalyzer(storage=storage)
            causal_chain = analyzer.analyze(incident=incident, lookback_days=14)
            logger.info("auto_rca_computed", incident_id=incident.incident_id)
        except Exception as e:
            logger.warning("auto_rca_failed", incident_id=incident.incident_id, error=str(e))

    if blast_radius is None:
        try:
            from api.engine.blast_radius import BlastRadiusMapper
            mapper = BlastRadiusMapper(storage=storage)
            blast_radius = mapper.compute_blast_radius(incident)
            logger.info("auto_blast_radius_computed", incident_id=incident.incident_id)
        except Exception as e:
            logger.warning("auto_blast_radius_failed", incident_id=incident.incident_id, error=str(e))

    return causal_chain, blast_radius


@router.get("/")
async def list_incidents(
    realm_id: str = Depends(get_current_realm_id),
    severity: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
    sort_by: str = "detected_at",
    sort_order: str = "desc",
):
    """
    List incidents with optional filtering and cursor-based pagination.

    API-designer agent: proper pagination, consistent naming, and sort params.

    Args:
        severity: Filter by severity (low, medium, high, critical)
        status: Filter by status (open, acknowledged, resolved)
        page: Page number (1-indexed)
        page_size: Items per page (max 100)
        sort_by: Sort field (detected_at, severity, confidence)
        sort_order: Sort direction (asc, desc)

    Returns:
        Paginated response with metadata for next/prev navigation.
    """
    # Input validation per api-designer agent
    page = max(1, page)
    page_size = max(1, min(100, page_size))

    logger.info(
        "incidents_list",
        realm_id=realm_id,
        severity=severity,
        status=status,
        page=page,
        page_size=page_size,
    )

    storage = get_storage()
    incidents = storage.read_incidents(
        severity=severity,
        status=status,
    )

    # Sort per api-designer agent: support multiple sort fields
    sort_reverse = sort_order.lower() == "desc"
    try:
        incidents.sort(key=lambda i: getattr(i, sort_by, ""), reverse=sort_reverse)
    except (AttributeError, TypeError):
        pass  # Graceful fallback if sort_by is invalid

    # Pagination calculation
    total_count = len(incidents)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_incidents = incidents[offset : offset + page_size]

    # Serialize
    results = []
    for inc in page_incidents:
        results.append({
            "incident_id": inc.incident_id,
            "incident_type": inc.incident_type.value,
            "severity": inc.severity.value,
            "confidence": inc.confidence.value,
            "status": inc.status.value,
            "detected_at": inc.detected_at.isoformat(),
            "primary_metric": inc.primary_metric,
            "primary_metric_zscore": inc.primary_metric_zscore,
            "evidence_event_count": inc.evidence_event_count,
            "data_quality_score": inc.data_quality_score,
            "cascade_id": inc.cascade_id,
        })

    return {
        "success": True,
        "data": results,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }


@router.get("/{incident_id}")
async def get_incident_detail(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get detailed incident information including RCA and blast radius.
    """
    logger.info("incident_detail", realm_id=realm_id, incident_id=incident_id)

    storage = get_storage()

    # Find incident
    incidents = storage.read_incidents()
    incident = next(
        (i for i in incidents if i.incident_id == incident_id), None
    )
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    causal_chain, blast_radius = _ensure_rca_and_blast_radius(storage, incident)

    result = {
        "incident": incident.model_dump(mode="json"),
        "causal_chain": causal_chain.model_dump(mode="json") if causal_chain else None,
        "blast_radius": blast_radius.model_dump(mode="json") if blast_radius else None,
    }

    return {"success": True, "data": result}


@router.get("/{incident_id}/postmortem")
async def get_incident_postmortem(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get or generate incident postmortem report.
    Checks for existing postmortem first; generates on-demand if missing.
    """
    logger.info("postmortem_fetch", realm_id=realm_id, incident_id=incident_id)

    storage = get_storage()

    # Check for existing postmortem
    existing = storage.read_postmortem(incident_id)
    if existing:
        return {"success": True, "data": existing.model_dump(mode="json")}

    # Generate on-demand
    incidents = storage.read_incidents()
    incident = next(
        (i for i in incidents if i.incident_id == incident_id), None
    )
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    causal_chain, blast_radius = _ensure_rca_and_blast_radius(storage, incident)

    if causal_chain is None or blast_radius is None:
        raise HTTPException(
            status_code=422,
            detail="Could not compute root cause analysis. Please try again or run a full analysis first.",
        )

    from api.engine.postmortem_generator import PostmortemGenerator

    generator = PostmortemGenerator(storage=storage)
    postmortem = generator.generate(
        incident=incident,
        causal_chain=causal_chain,
        blast_radius=blast_radius,
    )

    return {"success": True, "data": postmortem.model_dump(mode="json")}


@router.post("/{incident_id}/analyze")
async def analyze_incident(
    incident_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    One-click full analysis for a single incident: RCA + blast radius + postmortem.
    Computes everything from scratch regardless of whether it already exists.
    """
    logger.info("incident_analyze", realm_id=realm_id, incident_id=incident_id)

    storage = get_storage()

    incidents = storage.read_incidents()
    incident = next(
        (i for i in incidents if i.incident_id == incident_id), None
    )
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    results = {"incident_id": incident_id, "rca": False, "blast_radius": False, "postmortem": False}

    # Step 1: RCA
    causal_chain = None
    try:
        from api.engine.rca import RootCauseAnalyzer
        analyzer = RootCauseAnalyzer(storage=storage)
        causal_chain = analyzer.analyze(incident=incident, lookback_days=14)
        results["rca"] = True
    except Exception as e:
        logger.warning("analyze_rca_failed", incident_id=incident_id, error=str(e))

    # Step 2: Blast radius
    blast_radius = None
    try:
        from api.engine.blast_radius import BlastRadiusMapper
        mapper = BlastRadiusMapper(storage=storage)
        blast_radius = mapper.compute_blast_radius(incident)
        results["blast_radius"] = True
    except Exception as e:
        logger.warning("analyze_blast_failed", incident_id=incident_id, error=str(e))

    # Step 3: Postmortem
    if causal_chain and blast_radius:
        try:
            from api.engine.postmortem_generator import PostmortemGenerator
            generator = PostmortemGenerator(storage=storage)
            generator.generate(
                incident=incident,
                causal_chain=causal_chain,
                blast_radius=blast_radius,
            )
            results["postmortem"] = True
        except Exception as e:
            logger.warning("analyze_postmortem_failed", incident_id=incident_id, error=str(e))

    return {"success": True, "data": results}
