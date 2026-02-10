"""
External API connectors.

QuickBooks Online connector handles:
- OAuth2 authentication
- API client (httpx)
- Entity fetching (invoices, payments, customers, etc.)
- Rate limiting and retry logic
- Token refresh
"""

# Connector modules:
# - qbo_client.py: QuickBooks API client
# - qbo_oauth.py: OAuth2 flow handling
# - qbo_entities.py: Entity-specific fetchers
