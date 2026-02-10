"""
Authentication router - OAuth2 flow with QuickBooks and JWT issuance.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from api.auth.jwt import create_access_token
from api.config import get_settings
from api.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class OAuthInitResponse(BaseModel):
    """OAuth2 initialization response."""

    authorization_url: str
    state: str


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str
    expires_in: int
    realm_id: str


@router.get("/authorize", response_model=OAuthInitResponse)
async def initiate_oauth():
    """
    Initiate OAuth2 flow with QuickBooks.
    Returns authorization URL for user to visit.
    """
    settings = get_settings()
    import uuid

    state = str(uuid.uuid4())

    # Construct OAuth2 authorization URL
    auth_url = (
        f"{settings.intuit_auth_url}?"
        f"client_id={settings.intuit_client_id}&"
        f"scope=com.intuit.quickbooks.accounting&"
        f"redirect_uri={settings.intuit_redirect_uri}&"
        f"response_type=code&"
        f"state={state}"
    )

    logger.info("oauth_initiated", state=state)

    return OAuthInitResponse(authorization_url=auth_url, state=state)


@router.get("/callback", response_model=TokenResponse)
async def oauth_callback(
    code: str = Query(..., description="Authorization code from Intuit"),
    state: str = Query(..., description="State parameter for CSRF protection"),
    realmId: str = Query(..., description="QuickBooks company ID"),
):
    """
    OAuth2 callback endpoint.
    Exchanges authorization code for tokens and issues JWT.
    """
    settings = get_settings()

    logger.info("oauth_callback", realm_id=realmId, state=state)

    # TODO: Exchange code for access token with Intuit
    # For now, we'll create a JWT directly (demo mode)

    # Create JWT with realm_id
    access_token = create_access_token(
        data={"sub": realmId, "scopes": ["read:financials", "write:monitors"]},
        expires_delta=timedelta(minutes=settings.jwt_expiration_minutes),
    )

    logger.info("jwt_issued", realm_id=realmId)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expiration_minutes * 60,
        realm_id=realmId,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(realm_id: str):
    """
    Refresh JWT token.
    In production, this would validate a refresh token.
    """
    settings = get_settings()

    access_token = create_access_token(
        data={"sub": realm_id, "scopes": ["read:financials", "write:monitors"]},
        expires_delta=timedelta(minutes=settings.jwt_expiration_minutes),
    )

    logger.info("jwt_refreshed", realm_id=realm_id)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expiration_minutes * 60,
        realm_id=realm_id,
    )


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    In production, would revoke tokens in Redis.
    """
    logger.info("user_logout")
    return {"success": True, "message": "Logged out successfully"}
