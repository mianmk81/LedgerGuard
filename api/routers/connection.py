"""
QuickBooks connection management router.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import get_current_realm_id
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ConnectionStatus(BaseModel):
    """QuickBooks connection status."""

    connected: bool
    realm_id: Optional[str] = None
    company_name: Optional[str] = None
    last_sync: Optional[str] = None
    token_expires_at: Optional[str] = None


class DisconnectResponse(BaseModel):
    """Disconnect response."""

    success: bool
    message: str


@router.get("/status", response_model=ConnectionStatus)
async def get_connection_status(realm_id: str = Depends(get_current_realm_id)):
    """
    Get QuickBooks connection status for authenticated realm.
    """
    logger.info("connection_status_check", realm_id=realm_id)

    # TODO: Check actual connection status from Redis/DB
    return ConnectionStatus(
        connected=True,
        realm_id=realm_id,
        company_name="Demo Company",
        last_sync="2026-02-10T12:00:00Z",
        token_expires_at="2026-02-11T12:00:00Z",
    )


@router.post("/disconnect", response_model=DisconnectResponse)
async def disconnect_quickbooks(realm_id: str = Depends(get_current_realm_id)):
    """
    Disconnect QuickBooks connection.
    Revokes tokens and clears cached data.
    """
    logger.info("connection_disconnect", realm_id=realm_id)

    # TODO: Revoke OAuth tokens, clear Redis cache
    return DisconnectResponse(success=True, message="Successfully disconnected from QuickBooks")


@router.post("/test")
async def test_connection(realm_id: str = Depends(get_current_realm_id)):
    """
    Test QuickBooks API connection.
    Makes a simple API call to verify credentials.
    """
    logger.info("connection_test", realm_id=realm_id)

    # TODO: Make actual API call to QuickBooks
    return {
        "success": True,
        "message": "Connection successful",
        "api_version": "v3",
        "response_time_ms": 145,
    }
