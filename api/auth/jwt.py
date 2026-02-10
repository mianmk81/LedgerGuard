"""
JWT token creation and validation.
Uses python-jose for JWT handling.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt

from api.config import get_settings


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    settings = get_settings()
    to_encode = data.copy()

    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)

    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
    )

    # Encode token
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload

    Raises:
        JWTError: If token is invalid or expired
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )

        # Validate token type
        if payload.get("type") != "access":
            raise JWTError("Invalid token type")

        return payload

    except JWTError as e:
        raise JWTError(f"Token validation failed: {str(e)}")


def verify_token(token: str) -> bool:
    """
    Verify if a token is valid without raising exceptions.

    Args:
        token: JWT token string

    Returns:
        True if valid, False otherwise
    """
    try:
        decode_access_token(token)
        return True
    except JWTError:
        return False
