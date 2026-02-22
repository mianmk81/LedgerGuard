"""
QuickBooks Online connector for LedgerGuard Business Reliability Engine.

This package provides comprehensive integration with QuickBooks Online API:
- OAuth2 authentication and token management
- Entity fetching with automatic pagination
- Webhook handling for real-time updates
- Batch and incremental data ingestion
- Rate limiting and retry logic
- Connection health monitoring

Main Components:
    QBOClient: Core API client with OAuth2 support
    WebhookHandler: Processes Intuit webhook notifications
    IngestionService: Orchestrates data ingestion pipeline

Usage:
    >>> from api.connectors import QBOClient, IngestionService
    >>> from api.storage.duckdb_storage import DuckDBStorage
    >>>
    >>> # Initialize client
    >>> async with QBOClient(
    ...     client_id="your_client_id",
    ...     client_secret="your_secret",
    ...     redirect_uri="http://localhost:8000/callback"
    ... ) as client:
    ...     # Get authorization URL
    ...     auth_url = client.get_authorization_url()
    ...
    ...     # After user authorizes, exchange code for tokens
    ...     tokens = await client.exchange_code(auth_code="...")
    ...
    ...     # Initialize ingestion service
    ...     storage = DuckDBStorage(db_path="./data/bre.duckdb")
    ...     ingestion = IngestionService(client, storage)
    ...
    ...     # Run batch ingestion
    ...     result = await ingestion.run_batch_ingestion()
"""

from api.connectors.ingestion_service import IngestionError, IngestionService
from api.connectors.qbo_client import QBOAPIError, QBOAuthError, QBOClient
from api.connectors.webhook_handler import (
    WebhookEvent,
    WebhookHandler,
    WebhookVerificationError,
)

__all__ = [
    # Core client
    "QBOClient",
    "QBOAuthError",
    "QBOAPIError",
    # Webhook handling
    "WebhookHandler",
    "WebhookEvent",
    "WebhookVerificationError",
    # Ingestion orchestration
    "IngestionService",
    "IngestionError",
]
