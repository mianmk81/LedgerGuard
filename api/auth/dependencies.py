"""
FastAPI dependencies for authentication and authorization.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from api.auth.jwt import decode_access_token
from api.utils.logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_current_realm_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    Extract and validate realm_id from JWT token.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        Realm ID (QuickBooks company ID)

    Raises:
        HTTPException: If token is missing or invalid
    """
    if not credentials:
        logger.warning("auth_failed", reason="missing_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        realm_id: str = payload.get("sub")

        if not realm_id:
            logger.warning("auth_failed", reason="missing_realm_id")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        logger.debug("auth_success", realm_id=realm_id)
        return realm_id

    except JWTError as e:
        logger.warning("auth_failed", reason="invalid_token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_realm_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Extract realm_id from JWT token, but don't require authentication.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        Realm ID if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        return await get_current_realm_id(credentials)
    except HTTPException:
        return None
