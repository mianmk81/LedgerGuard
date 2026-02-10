"""JWT authentication and authorization module."""

from api.auth.dependencies import get_current_realm_id
from api.auth.jwt import create_access_token, decode_access_token

__all__ = ["create_access_token", "decode_access_token", "get_current_realm_id"]
