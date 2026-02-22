# QuickBooks Online Connector

Comprehensive integration with QuickBooks Online API for the LedgerGuard Business Reliability Engine.

## Features

### QBOClient (`qbo_client.py`)
Complete QuickBooks Online API client with OAuth2 token management:

- **OAuth2 Flow**
  - Generate authorization URL with CSRF protection
  - Exchange authorization code for tokens
  - Automatic token refresh (access tokens expire after 1 hour)
  - Refresh token lifecycle management (100-day lifetime)

- **Entity Operations**
  - Fetch single entities by ID
  - Query entities with SQL-like WHERE clauses
  - Fetch all entities with automatic pagination
  - Support for 17+ entity types (Invoice, Payment, Customer, etc.)

- **Performance & Reliability**
  - Rate limiting (500 requests/minute compliance)
  - Automatic retry with exponential backoff
  - Connection pooling with httpx
  - Comprehensive error handling
  - Structured logging with structlog

### WebhookHandler (`webhook_handler.py`)
Processes real-time webhook notifications from Intuit:

- **Security**
  - HMAC-SHA256 signature verification
  - CSRF protection with state parameter
  - Constant-time signature comparison

- **Event Processing**
  - Parse webhook payloads into structured events
  - Fetch full entity data from QBO API
  - Write to Bronze layer with full lineage
  - Handle Create, Update, Delete, Merge operations

### IngestionService (`ingestion_service.py`)
Orchestrates the complete data ingestion pipeline:

- **Batch Ingestion**
  - Sync all entity types in single operation
  - Incremental sync (only fetch changes since last run)
  - Force full sync option for backfill scenarios

- **Connection Monitoring**
  - Health status tracking (healthy/degraded/unhealthy)
  - Connection test with response time metrics
  - Last sync time tracking
  - Entity count statistics

- **Error Recovery**
  - Continue on entity-level failures
  - Comprehensive error tracking and reporting
  - Detailed ingestion statistics

## Usage Examples

### 1. OAuth2 Authorization Flow

```python
from api.connectors import QBOClient
from api.config import get_settings

settings = get_settings()

# Initialize client
async with QBOClient(
    client_id=settings.intuit_client_id,
    client_secret=settings.intuit_client_secret,
    redirect_uri=settings.intuit_redirect_uri,
    environment=settings.intuit_env
) as client:
    # Step 1: Get authorization URL
    auth_url = client.get_authorization_url()
    print(f"Redirect user to: {auth_url}")

    # Step 2: User authorizes and is redirected back with code
    # Extract code and state from callback URL

    # Step 3: Exchange code for tokens
    tokens = await client.exchange_code(
        auth_code="QB_AUTH_CODE_FROM_CALLBACK",
        state="STATE_FROM_CALLBACK"
    )

    print(f"Access token expires in: {tokens['expires_in']} seconds")
```

### 2. Fetch Entities

```python
from datetime import datetime, timedelta

# Set realm_id after authorization
client.realm_id = "123146096291789"

# Fetch single invoice
invoice = await client.get_entity("Invoice", "145")
print(f"Invoice total: ${invoice['TotalAmt']}")

# Query invoices over $100
invoices = await client.query_entities(
    entity_type="Invoice",
    where_clause="TotalAmt > '100'"
)
print(f"Found {len(invoices)} invoices over $100")

# Fetch all invoices modified in last 7 days
since = datetime.utcnow() - timedelta(days=7)
recent_invoices = await client.get_all_entities(
    entity_type="Invoice",
    since=since
)
print(f"Found {len(recent_invoices)} recent invoices")
```

### 3. Batch Ingestion

```python
from api.connectors import IngestionService
from api.storage.duckdb_storage import DuckDBStorage

# Initialize storage
storage = DuckDBStorage(db_path="./data/bre.duckdb")

# Initialize ingestion service
ingestion = IngestionService(client, storage)

# Run batch ingestion (incremental sync)
result = await ingestion.run_batch_ingestion()
print(f"Synced {result['total_entities_written']} entities")
print(f"Duration: {result['duration_seconds']} seconds")

# Force full sync
result = await ingestion.run_batch_ingestion(force_full_sync=True)

# Sync specific entity types
result = await ingestion.run_batch_ingestion(
    entity_types=["Invoice", "Payment", "Customer"]
)

# Sync entities modified since specific date
result = await ingestion.run_batch_ingestion(
    since=datetime(2026, 1, 1)
)
```

### 4. Webhook Handling

```python
from api.connectors import WebhookHandler

# Initialize webhook handler
handler = WebhookHandler(verifier_token="YOUR_WEBHOOK_VERIFIER_TOKEN")

# In webhook endpoint handler
@app.post("/api/v1/webhooks/qbo")
async def handle_webhook(request: Request):
    # Get raw payload and signature
    payload_bytes = await request.body()
    payload_str = payload_bytes.decode("utf-8")
    signature = request.headers.get("Intuit-Signature", "")

    # Verify signature
    if not handler.verify_webhook_signature(payload_str, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse and process webhook
    payload = await request.json()
    processed_count = await handler.process_webhook(
        payload=payload,
        qbo_client=client,
        storage=storage
    )

    return handler.generate_webhook_response(success=True)
```

### 5. Connection Health Monitoring

```python
# Get ingestion status
status = await ingestion.get_ingestion_status()
print(f"Health: {status['health']}")
print(f"Last sync: {status['last_sync_time']}")
print(f"Entity counts: {status['entity_counts']}")

# Test connection
test_result = await ingestion.test_connection()
if test_result['success']:
    print(f"Connection OK - Response time: {test_result['response_time_ms']}ms")
else:
    print(f"Connection failed: {test_result['message']}")

# Check connection status
conn_status = client.get_connection_status()
print(f"Connected: {conn_status['is_connected']}")
print(f"Token expires: {conn_status['access_token_expiry']}")
```

### 6. Token Persistence

```python
# Save tokens to Redis for persistence
tokens = {
    "access_token": client._access_token,
    "refresh_token": client._refresh_token,
    "access_token_expiry": client._token_expiry.isoformat(),
    "refresh_token_expiry": client._refresh_token_expiry.isoformat(),
}
redis_client.set(f"qbo_tokens:{realm_id}", json.dumps(tokens))

# Restore tokens from Redis
stored_tokens = json.loads(redis_client.get(f"qbo_tokens:{realm_id}"))
client.set_tokens(
    access_token=stored_tokens["access_token"],
    refresh_token=stored_tokens["refresh_token"],
    access_token_expiry=datetime.fromisoformat(stored_tokens["access_token_expiry"]),
    refresh_token_expiry=datetime.fromisoformat(stored_tokens["refresh_token_expiry"]),
)
```

## Supported Entity Types

The connector supports all major QuickBooks Online entity types:

- **Revenue**: Invoice, Payment, SalesReceipt, Estimate
- **Expenses**: Bill, BillPayment, Purchase, Expense
- **Credits**: CreditMemo, RefundReceipt
- **Entities**: Customer, Vendor, Item
- **Accounting**: Account, JournalEntry, Deposit, Transfer
- **Procurement**: PurchaseOrder

## Rate Limiting

QuickBooks Online enforces a rate limit of **500 requests per minute**. The client automatically:
- Tracks request timestamps
- Implements sliding window rate limiting
- Delays requests when limit is reached
- Logs throttling events for monitoring

## Error Handling

### QBOAuthError
Raised for OAuth2 authentication failures:
- Invalid authorization code
- Expired refresh token
- OAuth state mismatch (CSRF attack)

### QBOAPIError
Raised for API request failures:
- 400 Bad Request (validation errors)
- 404 Not Found (entity doesn't exist)
- 401 Unauthorized (invalid token)
- 500 Internal Server Error (QuickBooks API issues)

### IngestionError
Raised for ingestion pipeline failures:
- Storage write failures
- Entity processing errors

## Security Considerations

### Token Storage
- **Never commit tokens to version control**
- Store tokens in Redis with encryption at rest
- Use short TTLs for access tokens (1 hour)
- Rotate refresh tokens regularly

### Webhook Security
- Always verify HMAC signatures before processing
- Use HTTPS for webhook endpoints
- Implement rate limiting on webhook endpoint
- Log all webhook events for audit trail

### CSRF Protection
- Generate secure random state for OAuth flow
- Validate state parameter on callback
- Use constant-time comparison for signatures

## Performance Optimization

### Batch Operations
- Use `get_all_entities()` for bulk fetches (handles pagination)
- Limit `max_results` for large datasets
- Use `since` parameter for incremental syncs

### Connection Pooling
- httpx AsyncClient maintains connection pool
- Reuse client instance across requests
- Use async context manager for automatic cleanup

### Rate Limit Compliance
- Automatic throttling prevents API blocks
- Exponential backoff for retries
- Request batching where possible

## Monitoring & Observability

### Structured Logging
All operations emit structured logs with:
- Request/response metadata
- Performance metrics (response time, entity counts)
- Error details with stack traces
- Connection health events

### Metrics to Track
- **Ingestion metrics**: entities fetched/written, duration, error rate
- **API metrics**: response time, rate limit hits, error codes
- **Connection metrics**: token refresh rate, connection failures
- **Webhook metrics**: events received, processing time, validation failures

## Development & Testing

### Local Development
```bash
# Set environment variables
export INTUIT_CLIENT_ID="your_client_id"
export INTUIT_CLIENT_SECRET="your_client_secret"
export INTUIT_REDIRECT_URI="http://localhost:8000/api/v1/auth/callback"
export INTUIT_ENV="sandbox"

# Run application
uvicorn api.main:app --reload
```

### Testing
```python
# Unit tests with mocked QBO responses
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fetch_invoice():
    with patch("httpx.AsyncClient.request") as mock_request:
        mock_request.return_value.json.return_value = {
            "Invoice": {"Id": "145", "TotalAmt": 100.00}
        }

        async with QBOClient(...) as client:
            invoice = await client.get_entity("Invoice", "145")
            assert invoice["TotalAmt"] == 100.00
```

## Production Deployment

### Configuration Checklist
- [ ] Use production Intuit app credentials
- [ ] Set `INTUIT_ENV=production`
- [ ] Configure Redis for token persistence
- [ ] Enable HTTPS for webhook endpoint
- [ ] Set up webhook verifier token
- [ ] Configure rate limiting on API endpoints
- [ ] Enable structured logging to centralized system
- [ ] Set up monitoring alerts for connection failures
- [ ] Document token refresh procedures
- [ ] Implement token encryption at rest

### Scaling Considerations
- Use Celery for background ingestion jobs
- Distribute webhook processing across workers
- Implement Redis-based distributed locking for sync jobs
- Monitor token refresh rate across workers
- Use connection pooling at infrastructure level

## References

- [Intuit OAuth2 Documentation](https://developer.intuit.com/app/developer/qbo/docs/develop/authentication-and-authorization/oauth-2.0)
- [QuickBooks Online API Reference](https://developer.intuit.com/app/developer/qbo/docs/api/accounting/all-entities/invoice)
- [Webhook Documentation](https://developer.intuit.com/app/developer/qbo/docs/develop/webhooks)
