"""
Supplemental dataset adapters.

This module provides adapters for transforming various supplemental datasets
into canonical events for business reliability analysis. Each adapter handles
a specific data source and ensures consistent data quality standards.

Supported adapters:
- OlistAdapter: Brazilian E-Commerce Olist dataset
- SupportTicketAdapter: Customer support ticket datasets
- ChurnAdapter: Telco customer churn datasets
- LogisticsAdapter: Logistics and supply chain datasets

Usage:
    from api.adapters import get_adapter

    # Get adapter by name
    adapter = get_adapter("olist")

    # Transform data
    events, quality_report = adapter.ingest(data_df)
"""

from typing import Type

from .base_adapter import BaseAdapter
from .churn_adapter import ChurnAdapter
from .logistics_adapter import LogisticsAdapter
from .olist_adapter import OlistAdapter
from .support_tickets_adapter import SupportTicketAdapter

# Adapter registry mapping source names to adapter classes
ADAPTER_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "olist": OlistAdapter,
    "support_tickets": SupportTicketAdapter,
    "telco_churn": ChurnAdapter,
    "logistics": LogisticsAdapter,
}


def get_adapter(source: str) -> BaseAdapter:
    """
    Get adapter instance by source name.

    Args:
        source: Source identifier (e.g., "olist", "support_tickets")

    Returns:
        Initialized adapter instance

    Raises:
        ValueError: If source is not found in registry

    Example:
        >>> adapter = get_adapter("olist")
        >>> events, report = adapter.ingest(orders_df)
    """
    adapter_class = ADAPTER_REGISTRY.get(source)
    if not adapter_class:
        available = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter source: '{source}'. Available adapters: {available}"
        )
    return adapter_class()


def list_adapters() -> list[str]:
    """
    List all available adapter source names.

    Returns:
        List of registered adapter source names

    Example:
        >>> adapters = list_adapters()
        >>> print(adapters)
        ['olist', 'support_tickets', 'telco_churn', 'logistics']
    """
    return list(ADAPTER_REGISTRY.keys())


__all__ = [
    "BaseAdapter",
    "OlistAdapter",
    "SupportTicketAdapter",
    "ChurnAdapter",
    "LogisticsAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "list_adapters",
]
