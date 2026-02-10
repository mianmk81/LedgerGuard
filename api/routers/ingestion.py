"""
Data ingestion router - Trigger and monitor ingestion jobs.
"""

from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class EntityType(str, Enum):
    """QuickBooks entity types."""

    INVOICE = "invoice"
    PAYMENT = "payment"
    CUSTOMER = "customer"
    VENDOR = "vendor"
    ACCOUNT = "account"
    JOURNAL_ENTRY = "journal_entry"
    ALL = "all"


class IngestionStatus(str, Enum):
    """Ingestion job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionRequest(BaseModel):
    """Request to start ingestion."""

    entity_types: List[EntityType] = [EntityType.ALL]
    full_refresh: bool = False
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class IngestionJob(BaseModel):
    """Ingestion job details."""

    job_id: str
    status: IngestionStatus
    entity_types: List[EntityType]
    started_at: str
    completed_at: Optional[str] = None
    entities_ingested: int = 0
    error: Optional[str] = None


@router.post("/start", response_model=IngestionJob)
async def start_ingestion(
    request: IngestionRequest,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Start data ingestion from QuickBooks.
    Triggers async Celery task.
    """
    import uuid
    from datetime import datetime

    job_id = str(uuid.uuid4())

    logger.info(
        "ingestion_started",
        realm_id=realm_id,
        job_id=job_id,
        entity_types=[et.value for et in request.entity_types],
        full_refresh=request.full_refresh,
    )

    # TODO: Trigger Celery task
    return IngestionJob(
        job_id=job_id,
        status=IngestionStatus.PENDING,
        entity_types=request.entity_types,
        started_at=datetime.utcnow().isoformat(),
    )


@router.get("/status/{job_id}", response_model=IngestionJob)
async def get_ingestion_status(
    job_id: str,
    realm_id: str = Depends(get_current_realm_id),
):
    """
    Get status of ingestion job.
    """
    from datetime import datetime

    logger.info("ingestion_status_check", realm_id=realm_id, job_id=job_id)

    # TODO: Query Celery task status
    return IngestionJob(
        job_id=job_id,
        status=IngestionStatus.COMPLETED,
        entity_types=[EntityType.ALL],
        started_at=datetime.utcnow().isoformat(),
        completed_at=datetime.utcnow().isoformat(),
        entities_ingested=1234,
    )


@router.get("/history", response_model=List[IngestionJob])
async def get_ingestion_history(
    realm_id: str = Depends(get_current_realm_id),
    limit: int = 10,
):
    """
    Get ingestion job history.
    """
    logger.info("ingestion_history", realm_id=realm_id, limit=limit)

    # TODO: Query from database
    return []
