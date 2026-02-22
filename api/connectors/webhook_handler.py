"""
Webhook handler for Intuit real-time notifications.

This module processes webhook events from QuickBooks Online to enable real-time
data synchronization. Intuit sends webhook notifications when entities are
created, updated, or deleted in QuickBooks, allowing the BRE to ingest changes
immediately without polling.

Webhook flow:
1. Intuit sends POST request to webhook endpoint with entity change notifications
2. Verify webhook signature using HMAC-SHA256
3. Parse notification payload to extract changed entity information
4. Fetch full entity data from QuickBooks API
5. Write to Bronze layer for downstream processing
"""

import hashlib
import hmac
import json
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, field_validator

from api.connectors.qbo_client import QBOClient, QBOAPIError
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class WebhookEvent(BaseModel):
    """
    Parsed webhook event representing a single entity change.

    Attributes:
        event_id: Unique identifier for this webhook event
        entity_type: Type of entity that changed (Invoice, Customer, etc.)
        entity_id: QuickBooks entity ID
        operation: Operation type (Create, Update, Delete, Merge)
        realm_id: QuickBooks company ID where change occurred
        last_updated: Timestamp when entity was last modified
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this webhook event",
    )
    entity_type: str = Field(description="Type of entity that changed")
    entity_id: str = Field(description="QuickBooks entity ID")
    operation: str = Field(description="Operation type (Create, Update, Delete, Merge)")
    realm_id: str = Field(description="QuickBooks company ID")
    last_updated: datetime = Field(description="Entity last modified timestamp")

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is a known type."""
        valid_operations = ["Create", "Update", "Delete", "Merge", "Void"]
        if v not in valid_operations:
            raise ValueError(f"Invalid operation: {v}. Must be one of {valid_operations}")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "event_id": "evt_123e4567-e89b-12d3-a456-426614174000",
                "entity_type": "Invoice",
                "entity_id": "145",
                "operation": "Update",
                "realm_id": "123146096291789",
                "last_updated": "2026-02-10T14:30:00Z",
            }
        }


class WebhookVerificationError(Exception):
    """Raised when webhook signature verification fails."""

    pass


class WebhookHandler:
    """
    Handles Intuit webhook notifications for entity changes.

    Provides webhook signature verification, payload parsing, and automated
    entity synchronization to Bronze layer. Implements Intuit's webhook
    security requirements including HMAC-SHA256 signature verification.
    """

    def __init__(self, verifier_token: str):
        """
        Initialize webhook handler.

        Args:
            verifier_token: Webhook verifier token from Intuit app settings
                           Used for HMAC signature verification
        """
        self.verifier_token = verifier_token

        logger.info(
            "webhook_handler_initialized",
            has_verifier_token=bool(verifier_token),
        )

    def verify_webhook_signature(
        self,
        payload: str,
        signature: str,
    ) -> bool:
        """
        Verify webhook signature using HMAC-SHA256.

        Intuit signs all webhook payloads with HMAC-SHA256 using the webhook
        verifier token as the secret key. This prevents webhook spoofing attacks.

        Args:
            payload: Raw webhook payload string (not parsed JSON)
            signature: Intuit-Signature header value from webhook request

        Returns:
            True if signature is valid, False otherwise

        Raises:
            WebhookVerificationError: If verification process fails

        Example:
            >>> handler = WebhookHandler(verifier_token="abc123")
            >>> is_valid = handler.verify_webhook_signature(
            ...     payload='{"eventNotifications":[...]}',
            ...     signature="sha256=1234567890abcdef..."
            ... )
        """
        try:
            # Extract signature hash (format: "sha256=<hash>")
            if not signature.startswith("sha256="):
                logger.error("invalid_signature_format", signature_prefix=signature[:10])
                return False

            received_hash = signature.replace("sha256=", "")

            # Compute expected signature
            expected_hmac = hmac.new(
                key=self.verifier_token.encode("utf-8"),
                msg=payload.encode("utf-8"),
                digestmod=hashlib.sha256,
            )
            expected_hash = expected_hmac.hexdigest()

            # Constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(expected_hash, received_hash)

            if not is_valid:
                logger.warning(
                    "webhook_signature_mismatch",
                    expected_hash=expected_hash[:10] + "...",
                    received_hash=received_hash[:10] + "...",
                )

            logger.info("webhook_signature_verified", is_valid=is_valid)

            return is_valid

        except Exception as e:
            logger.error("webhook_verification_error", error=str(e))
            raise WebhookVerificationError(f"Failed to verify webhook signature: {str(e)}")

    def parse_webhook_payload(self, payload: dict[str, Any]) -> list[WebhookEvent]:
        """
        Parse webhook payload into structured WebhookEvent objects.

        Intuit webhook payload structure:
        {
            "eventNotifications": [
                {
                    "realmId": "123146096291789",
                    "dataChangeEvent": {
                        "entities": [
                            {
                                "name": "Invoice",
                                "id": "145",
                                "operation": "Update",
                                "lastUpdated": "2016-10-25T16:27:46.000Z"
                            }
                        ]
                    }
                }
            ]
        }

        Args:
            payload: Parsed webhook JSON payload

        Returns:
            List of WebhookEvent objects representing entity changes

        Raises:
            ValueError: If payload structure is invalid
        """
        events = []

        try:
            event_notifications = payload.get("eventNotifications", [])

            if not event_notifications:
                logger.warning("empty_webhook_payload")
                return events

            for notification in event_notifications:
                realm_id = notification.get("realmId")

                if not realm_id:
                    logger.warning("missing_realm_id_in_notification")
                    continue

                # Extract data change events
                data_change_event = notification.get("dataChangeEvent", {})
                entities = data_change_event.get("entities", [])

                for entity in entities:
                    try:
                        # Parse last updated timestamp
                        last_updated_str = entity.get("lastUpdated", "")
                        last_updated = datetime.fromisoformat(
                            last_updated_str.replace("Z", "+00:00")
                        )

                        webhook_event = WebhookEvent(
                            entity_type=entity.get("name", ""),
                            entity_id=entity.get("id", ""),
                            operation=entity.get("operation", ""),
                            realm_id=realm_id,
                            last_updated=last_updated,
                        )

                        events.append(webhook_event)

                        logger.debug(
                            "webhook_event_parsed",
                            entity_type=webhook_event.entity_type,
                            entity_id=webhook_event.entity_id,
                            operation=webhook_event.operation,
                        )

                    except Exception as e:
                        logger.error(
                            "failed_to_parse_entity",
                            entity=entity,
                            error=str(e),
                        )
                        # Continue processing other entities
                        continue

            logger.info("webhook_payload_parsed", event_count=len(events))

            return events

        except Exception as e:
            logger.error("webhook_payload_parse_error", error=str(e))
            raise ValueError(f"Failed to parse webhook payload: {str(e)}")

    async def process_webhook(
        self,
        payload: dict[str, Any],
        qbo_client: QBOClient,
        storage: StorageBackend,
    ) -> int:
        """
        Process complete webhook notification end-to-end.

        Orchestrates the full webhook processing pipeline:
        1. Parse webhook payload into events
        2. Fetch full entity data from QuickBooks API for each event
        3. Write raw entity data to Bronze layer
        4. Return count of successfully processed entities

        This method does NOT verify signatures - that should be done by the
        caller before invoking this method.

        Args:
            payload: Parsed webhook JSON payload
            qbo_client: Authenticated QBO client for entity fetching
            storage: Storage backend for Bronze layer writes

        Returns:
            Count of successfully processed entities

        Raises:
            QBOAPIError: If entity fetching fails
            Exception: If storage write fails

        Example:
            >>> handler = WebhookHandler(verifier_token="...")
            >>> processed_count = await handler.process_webhook(
            ...     payload=webhook_payload,
            ...     qbo_client=qbo_client,
            ...     storage=storage
            ... )
            >>> print(f"Processed {processed_count} entities")
        """
        events = self.parse_webhook_payload(payload)

        if not events:
            logger.info("no_events_to_process")
            return 0

        processed_count = 0

        for event in events:
            try:
                # Set realm_id on client if not already set
                if not qbo_client.realm_id:
                    qbo_client.realm_id = event.realm_id

                # Skip delete operations (we only store creates and updates)
                if event.operation in ["Delete", "Void"]:
                    logger.info(
                        "skipping_delete_operation",
                        entity_type=event.entity_type,
                        entity_id=event.entity_id,
                        operation=event.operation,
                    )

                    # Still write to Bronze layer to maintain audit trail
                    storage.write_raw_entity(
                        entity_id=event.entity_id,
                        entity_type=event.entity_type,
                        source="qbo",
                        operation=event.operation,
                        raw_payload={
                            "operation": event.operation,
                            "deleted_at": event.last_updated.isoformat(),
                        },
                        webhook_event_id=event.event_id,
                    )

                    processed_count += 1
                    continue

                # Fetch full entity data from QuickBooks API
                logger.info(
                    "fetching_entity_from_qbo",
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                )

                entity_data = await qbo_client.get_entity(
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                )

                # Write to Bronze layer
                record_id = storage.write_raw_entity(
                    entity_id=event.entity_id,
                    entity_type=event.entity_type,
                    source="qbo",
                    operation=event.operation,
                    raw_payload=entity_data,
                    webhook_event_id=event.event_id,
                )

                logger.info(
                    "webhook_entity_processed",
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                    operation=event.operation,
                    record_id=record_id,
                )

                processed_count += 1

            except QBOAPIError as e:
                logger.error(
                    "failed_to_fetch_entity",
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                    error=str(e),
                )
                # Continue processing other entities
                continue

            except Exception as e:
                logger.error(
                    "failed_to_process_entity",
                    entity_type=event.entity_type,
                    entity_id=event.entity_id,
                    error=str(e),
                )
                # Continue processing other entities
                continue

        logger.info(
            "webhook_processing_complete",
            total_events=len(events),
            processed_count=processed_count,
        )

        return processed_count

    def generate_webhook_response(self, success: bool = True) -> dict[str, Any]:
        """
        Generate standard webhook response.

        Intuit expects a 200 OK response to acknowledge webhook receipt.
        Non-200 responses trigger retry attempts.

        Args:
            success: Whether webhook processing was successful

        Returns:
            Response dictionary to return to Intuit
        """
        return {
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }
