"""
QuickBooks Online API client with OAuth2 token management.

This module provides a comprehensive async client for interacting with the
QuickBooks Online API, including:
- OAuth2 authorization flow (3-legged OAuth)
- Automatic token refresh with expiry tracking
- Entity querying with pagination support
- Rate limiting and retry logic
- Comprehensive error handling with structured logging
"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import structlog

from api.config import get_settings

logger = structlog.get_logger()


class QBOAuthError(Exception):
    """Raised when OAuth2 authentication fails."""

    pass


class QBOAPIError(Exception):
    """Raised when QuickBooks API request fails."""

    pass


class QBOClient:
    """
    QuickBooks Online API client with OAuth2 token management.

    Handles complete OAuth2 flow, token lifecycle management, and entity
    retrieval from QuickBooks Online API with automatic pagination, retry
    logic, and rate limiting compliance.

    Attributes:
        client_id: Intuit OAuth2 client ID
        client_secret: Intuit OAuth2 client secret
        redirect_uri: OAuth2 callback URL
        environment: "sandbox" or "production"
        realm_id: QuickBooks company ID (set after authorization)
    """

    # QuickBooks API supported entity types
    SUPPORTED_ENTITY_TYPES = [
        "Invoice",
        "Payment",
        "Bill",
        "BillPayment",
        "CreditMemo",
        "RefundReceipt",
        "Customer",
        "Item",
        "Vendor",
        "PurchaseOrder",
        "Estimate",
        "Account",
        "JournalEntry",
        "Deposit",
        "Transfer",
        "SalesReceipt",
        "Purchase",
    ]

    # Rate limiting: QuickBooks allows 500 requests per minute
    RATE_LIMIT_REQUESTS = 500
    RATE_LIMIT_WINDOW = 60  # seconds

    # Token expiry tracking
    ACCESS_TOKEN_LIFETIME = 3600  # 1 hour
    REFRESH_TOKEN_LIFETIME = 8640000  # 100 days in seconds

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        environment: str = "sandbox",
        realm_id: Optional[str] = None,
    ):
        """
        Initialize QuickBooks Online API client.

        Args:
            client_id: Intuit OAuth2 client ID
            client_secret: Intuit OAuth2 client secret
            redirect_uri: OAuth2 callback URL
            environment: "sandbox" or "production"
            realm_id: QuickBooks company ID (optional, set after auth)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.environment = environment
        self.realm_id = realm_id

        # Token storage (in-memory, should be persisted in Redis for production)
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._refresh_token_expiry: Optional[datetime] = None

        # OAuth state for CSRF protection
        self._oauth_state: Optional[str] = None

        # Rate limiting tracking
        self._request_times: list[datetime] = []

        # HTTP client with connection pooling
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            "qbo_client_initialized",
            environment=environment,
            realm_id=realm_id,
            has_credentials=bool(client_id and client_secret),
        )

    async def __aenter__(self):
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()

    def _get_base_url(self) -> str:
        """
        Get QuickBooks API base URL based on environment.

        Returns:
            Base URL for QuickBooks API
        """
        if self.environment == "production":
            return "https://quickbooks.api.intuit.com"
        return "https://sandbox-quickbooks.api.intuit.com"

    def _get_oauth_base_url(self) -> str:
        """
        Get Intuit OAuth2 base URL.

        Returns:
            Base URL for OAuth2 endpoints
        """
        return "https://oauth.platform.intuit.com/oauth2/v1"

    def get_authorization_url(self, scopes: Optional[list[str]] = None) -> str:
        """
        Generate OAuth2 authorization URL for user consent.

        Creates a secure authorization URL with CSRF protection state parameter.
        User should be redirected to this URL to grant access.

        Args:
            scopes: Optional list of OAuth scopes (defaults to accounting.com.intuit.quickbooks.accounting)

        Returns:
            Complete authorization URL for user redirection

        Example:
            >>> client = QBOClient(client_id="...", client_secret="...", redirect_uri="...")
            >>> auth_url = client.get_authorization_url()
            >>> # Redirect user to auth_url
        """
        if scopes is None:
            scopes = ["com.intuit.quickbooks.accounting"]

        # Generate secure random state for CSRF protection
        self._oauth_state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": " ".join(scopes),
            "redirect_uri": self.redirect_uri,
            "state": self._oauth_state,
        }

        auth_url = f"https://appcenter.intuit.com/connect/oauth2?{urlencode(params)}"

        logger.info(
            "authorization_url_generated",
            redirect_uri=self.redirect_uri,
            scopes=scopes,
            state=self._oauth_state,
        )

        return auth_url

    async def exchange_code(
        self, auth_code: str, state: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.

        Called after user authorizes the application and is redirected back
        with an authorization code. Validates CSRF state and exchanges code
        for OAuth tokens.

        Args:
            auth_code: Authorization code from callback
            state: CSRF state parameter from callback (optional but recommended)

        Returns:
            Dictionary containing token information:
            {
                "access_token": str,
                "refresh_token": str,
                "expires_in": int,
                "token_type": "bearer"
            }

        Raises:
            QBOAuthError: If code exchange fails or state validation fails
        """
        # Validate CSRF state if provided
        if state and self._oauth_state and state != self._oauth_state:
            logger.error(
                "oauth_state_mismatch",
                expected=self._oauth_state,
                received=state,
            )
            raise QBOAuthError("OAuth state mismatch - possible CSRF attack")

        token_url = f"{self._get_oauth_base_url()}/tokens/bearer"

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
        }

        headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}

        try:
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=30.0)

            response = await self._http_client.post(
                token_url,
                data=data,
                headers=headers,
                auth=(self.client_id, self.client_secret),
            )

            response.raise_for_status()
            token_data = response.json()

            # Store tokens
            self._access_token = token_data["access_token"]
            self._refresh_token = token_data["refresh_token"]
            self._token_expiry = datetime.utcnow() + timedelta(
                seconds=token_data.get("expires_in", self.ACCESS_TOKEN_LIFETIME)
            )
            self._refresh_token_expiry = datetime.utcnow() + timedelta(
                seconds=self.REFRESH_TOKEN_LIFETIME
            )

            logger.info(
                "oauth_code_exchanged",
                token_type=token_data.get("token_type"),
                expires_in=token_data.get("expires_in"),
            )

            return token_data

        except httpx.HTTPStatusError as e:
            logger.error(
                "oauth_code_exchange_failed",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise QBOAuthError(f"Failed to exchange authorization code: {e.response.text}")
        except Exception as e:
            logger.error("oauth_code_exchange_error", error=str(e))
            raise QBOAuthError(f"Unexpected error during code exchange: {str(e)}")

    async def refresh_tokens(self) -> dict[str, Any]:
        """
        Refresh access token using refresh token.

        Automatically called when access token expires. Updates internal token
        storage with new access token.

        Returns:
            Dictionary containing new token information

        Raises:
            QBOAuthError: If token refresh fails
        """
        if not self._refresh_token:
            raise QBOAuthError("No refresh token available")

        # Check if refresh token is expired
        if self._refresh_token_expiry and datetime.utcnow() >= self._refresh_token_expiry:
            logger.error("refresh_token_expired", expiry=self._refresh_token_expiry)
            raise QBOAuthError("Refresh token expired - user re-authentication required")

        token_url = f"{self._get_oauth_base_url()}/tokens/bearer"

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }

        headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}

        try:
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=30.0)

            response = await self._http_client.post(
                token_url,
                data=data,
                headers=headers,
                auth=(self.client_id, self.client_secret),
            )

            response.raise_for_status()
            token_data = response.json()

            # Update tokens
            self._access_token = token_data["access_token"]
            self._refresh_token = token_data["refresh_token"]
            self._token_expiry = datetime.utcnow() + timedelta(
                seconds=token_data.get("expires_in", self.ACCESS_TOKEN_LIFETIME)
            )

            logger.info(
                "tokens_refreshed",
                expires_in=token_data.get("expires_in"),
            )

            return token_data

        except httpx.HTTPStatusError as e:
            logger.error(
                "token_refresh_failed",
                status_code=e.response.status_code,
                error=e.response.text,
            )
            raise QBOAuthError(f"Failed to refresh tokens: {e.response.text}")
        except Exception as e:
            logger.error("token_refresh_error", error=str(e))
            raise QBOAuthError(f"Unexpected error during token refresh: {str(e)}")

    async def _ensure_valid_token(self) -> None:
        """
        Ensure access token is valid and refresh if necessary.

        Internal method that checks token expiry and automatically refreshes
        if needed before making API requests.

        Raises:
            QBOAuthError: If no access token available or refresh fails
        """
        if not self._access_token:
            raise QBOAuthError("No access token available - user authentication required")

        # Refresh if token expires within 5 minutes
        if self._token_expiry and datetime.utcnow() >= (self._token_expiry - timedelta(minutes=5)):
            logger.info("access_token_expiring", expiry=self._token_expiry)
            await self.refresh_tokens()

    async def _rate_limit_wait(self) -> None:
        """
        Implement rate limiting to comply with QuickBooks API limits.

        QuickBooks allows 500 requests per minute. This method tracks request
        times and delays execution if rate limit would be exceeded.
        """
        now = datetime.utcnow()

        # Remove requests older than the rate limit window
        cutoff = now - timedelta(seconds=self.RATE_LIMIT_WINDOW)
        self._request_times = [t for t in self._request_times if t > cutoff]

        # If at limit, wait until oldest request falls out of window
        if len(self._request_times) >= self.RATE_LIMIT_REQUESTS:
            sleep_time = (self._request_times[0] - cutoff).total_seconds()
            if sleep_time > 0:
                logger.warning("rate_limit_throttling", sleep_seconds=sleep_time)
                await asyncio.sleep(sleep_time)
                # Clean up again after waiting
                cutoff = datetime.utcnow() - timedelta(seconds=self.RATE_LIMIT_WINDOW)
                self._request_times = [t for t in self._request_times if t > cutoff]

        # Record this request
        self._request_times.append(now)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        retry_count: int = 3,
    ) -> dict[str, Any]:
        """
        Make authenticated API request with retry logic.

        Internal method that handles authentication headers, rate limiting,
        error handling, and automatic retries for transient failures.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            retry_count: Number of retries for transient failures

        Returns:
            JSON response from API

        Raises:
            QBOAPIError: If request fails after retries
            QBOAuthError: If authentication fails
        """
        if not self.realm_id:
            raise QBOAPIError("No realm_id set - QuickBooks company not connected")

        await self._ensure_valid_token()
        await self._rate_limit_wait()

        url = f"{self._get_base_url()}/v3/company/{self.realm_id}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        for attempt in range(retry_count):
            try:
                if not self._http_client:
                    self._http_client = httpx.AsyncClient(timeout=30.0)

                response = await self._http_client.request(
                    method,
                    url,
                    params=params,
                    json=data,
                    headers=headers,
                )

                response.raise_for_status()

                logger.debug(
                    "qbo_api_request_success",
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )

                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    "qbo_api_request_failed",
                    method=method,
                    endpoint=endpoint,
                    status_code=e.response.status_code,
                    error=e.response.text,
                    attempt=attempt + 1,
                )

                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise QBOAPIError(f"API request failed: {e.response.text}")

                # Retry server errors (5xx)
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info("retrying_request", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    raise QBOAPIError(f"API request failed after {retry_count} attempts: {e.response.text}")

            except Exception as e:
                logger.error(
                    "qbo_api_request_error",
                    method=method,
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt + 1,
                )

                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise QBOAPIError(f"Unexpected API error: {str(e)}")

    async def get_entity(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        """
        Retrieve a single entity by ID.

        Args:
            entity_type: Type of entity (e.g., "Invoice", "Customer")
            entity_id: QuickBooks entity ID

        Returns:
            Entity data as dictionary

        Raises:
            QBOAPIError: If entity not found or request fails
        """
        if entity_type not in self.SUPPORTED_ENTITY_TYPES:
            raise QBOAPIError(f"Unsupported entity type: {entity_type}")

        endpoint = f"{entity_type.lower()}/{entity_id}"
        response = await self._make_request("GET", endpoint)

        logger.info(
            "entity_retrieved",
            entity_type=entity_type,
            entity_id=entity_id,
        )

        return response.get(entity_type, {})

    async def query_entities(
        self,
        entity_type: str,
        where_clause: Optional[str] = None,
        since: Optional[datetime] = None,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Query entities with optional filtering.

        Uses QuickBooks Query Language (similar to SQL) to retrieve entities
        matching specific criteria.

        Args:
            entity_type: Type of entity to query
            where_clause: Optional SQL-like WHERE clause
            since: Optional datetime to filter entities modified since
            max_results: Maximum results per page

        Returns:
            List of entity dictionaries

        Raises:
            QBOAPIError: If query fails

        Example:
            >>> entities = await client.query_entities(
            ...     "Invoice",
            ...     where_clause="TotalAmt > '100'",
            ...     since=datetime(2026, 1, 1)
            ... )
        """
        if entity_type not in self.SUPPORTED_ENTITY_TYPES:
            raise QBOAPIError(f"Unsupported entity type: {entity_type}")

        # Build query
        query_parts = [f"SELECT * FROM {entity_type}"]

        if where_clause:
            query_parts.append(f"WHERE {where_clause}")

        if since:
            # QuickBooks requires ISO format without microseconds
            since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
            if where_clause:
                query_parts.append(f"AND MetaData.LastUpdatedTime >= '{since_str}'")
            else:
                query_parts.append(f"WHERE MetaData.LastUpdatedTime >= '{since_str}'")

        query_parts.append(f"MAXRESULTS {max_results}")

        query = " ".join(query_parts)

        params = {"query": query}

        response = await self._make_request("GET", "query", params=params)

        entities = response.get("QueryResponse", {}).get(entity_type, [])

        logger.info(
            "entities_queried",
            entity_type=entity_type,
            count=len(entities),
            query=query,
        )

        return entities

    async def get_all_entities(
        self,
        entity_type: str,
        since: Optional[datetime] = None,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all entities with automatic pagination.

        Fetches all entities of a given type, handling pagination automatically.
        This can be slow for large datasets - consider using incremental sync
        with the 'since' parameter.

        Args:
            entity_type: Type of entity to retrieve
            since: Optional datetime to filter entities modified since
            batch_size: Number of entities per page (max 1000)

        Returns:
            Complete list of all matching entities

        Raises:
            QBOAPIError: If retrieval fails
        """
        if entity_type not in self.SUPPORTED_ENTITY_TYPES:
            raise QBOAPIError(f"Unsupported entity type: {entity_type}")

        all_entities = []
        start_position = 1

        while True:
            # Build query with pagination
            query_parts = [f"SELECT * FROM {entity_type}"]

            if since:
                since_str = since.strftime("%Y-%m-%dT%H:%M:%S")
                query_parts.append(f"WHERE MetaData.LastUpdatedTime >= '{since_str}'")

            query_parts.append(f"STARTPOSITION {start_position}")
            query_parts.append(f"MAXRESULTS {batch_size}")

            query = " ".join(query_parts)
            params = {"query": query}

            response = await self._make_request("GET", "query", params=params)

            entities = response.get("QueryResponse", {}).get(entity_type, [])

            if not entities:
                break

            all_entities.extend(entities)

            logger.info(
                "entity_batch_retrieved",
                entity_type=entity_type,
                batch_size=len(entities),
                total=len(all_entities),
                start_position=start_position,
            )

            # Check if there are more results
            if len(entities) < batch_size:
                break

            start_position += batch_size

        logger.info(
            "all_entities_retrieved",
            entity_type=entity_type,
            total_count=len(all_entities),
        )

        return all_entities

    def is_connected(self) -> bool:
        """
        Check if client is connected with valid tokens.

        Returns:
            True if access token is available and not expired
        """
        if not self._access_token or not self._token_expiry:
            return False

        return datetime.utcnow() < self._token_expiry

    def is_refresh_token_valid(self) -> bool:
        """
        Check if refresh token is still valid.

        Returns:
            True if refresh token exists and not expired
        """
        if not self._refresh_token or not self._refresh_token_expiry:
            return False

        return datetime.utcnow() < self._refresh_token_expiry

    def set_tokens(
        self,
        access_token: str,
        refresh_token: str,
        access_token_expiry: datetime,
        refresh_token_expiry: datetime,
    ) -> None:
        """
        Set tokens from external storage (e.g., Redis).

        Used to restore token state from persistent storage after application
        restart or for distributed deployments.

        Args:
            access_token: OAuth2 access token
            refresh_token: OAuth2 refresh token
            access_token_expiry: Access token expiration datetime
            refresh_token_expiry: Refresh token expiration datetime
        """
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._token_expiry = access_token_expiry
        self._refresh_token_expiry = refresh_token_expiry

        logger.info(
            "tokens_restored",
            access_token_valid=self.is_connected(),
            refresh_token_valid=self.is_refresh_token_valid(),
        )

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get detailed connection status information.

        Returns:
            Dictionary with connection health metrics
        """
        return {
            "is_connected": self.is_connected(),
            "has_access_token": bool(self._access_token),
            "has_refresh_token": bool(self._refresh_token),
            "access_token_expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            "refresh_token_expiry": self._refresh_token_expiry.isoformat()
            if self._refresh_token_expiry
            else None,
            "realm_id": self.realm_id,
            "environment": self.environment,
        }
