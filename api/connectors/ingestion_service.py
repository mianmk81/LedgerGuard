"""
Ingestion service orchestrating data pipeline from QBO to Bronze/Silver layers.

This module coordinates the complete data ingestion workflow:
1. Batch ingestion: Pull all entities modified since last sync
2. Incremental sync: Track last sync time and only fetch changes
3. Entity validation: Write to Bronze layer with full lineage
4. Connection health monitoring: Track sync status and entity counts

The ingestion service is designed to be called by:
- Celery workers for scheduled batch ingestion
- API endpoints for manual sync triggers
- Webhook handlers for real-time updates
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

import structlog

from api.connectors.qbo_client import QBOAPIError, QBOAuthError, QBOClient
from api.storage.base import StorageBackend

logger = structlog.get_logger()


class IngestionError(Exception):
    """Raised when ingestion process fails."""

    pass


class IngestionService:
    """
    Orchestrates data ingestion from QuickBooks Online into Bronze/Silver layers.

    Manages the complete data ingestion lifecycle including batch pulls,
    incremental syncs, connection health monitoring, and error recovery.
    Implements medallion architecture pattern for data quality progression.

    Attributes:
        qbo_client: Authenticated QuickBooks Online API client
        storage: Storage backend for data persistence
    """

    # Entity types to sync in batch ingestion
    DEFAULT_ENTITY_TYPES = [
        "Invoice",
        "Payment",
        "Bill",
        "BillPayment",
        "CreditMemo",
        "RefundReceipt",
        "Customer",
        "Vendor",
        "Item",
        "PurchaseOrder",
        "Estimate",
        "Account",
    ]

    # Sync metadata storage key prefix
    SYNC_METADATA_PREFIX = "qbo_sync_metadata"

    def __init__(self, qbo_client: QBOClient, storage: StorageBackend):
        """
        Initialize ingestion service.

        Args:
            qbo_client: Authenticated QBO client
            storage: Storage backend for persistence
        """
        self.qbo_client = qbo_client
        self.storage = storage

        logger.info(
            "ingestion_service_initialized",
            realm_id=qbo_client.realm_id,
            is_connected=qbo_client.is_connected(),
        )

    async def run_batch_ingestion(
        self,
        since: Optional[datetime] = None,
        entity_types: Optional[list[str]] = None,
        force_full_sync: bool = False,
    ) -> dict[str, Any]:
        """
        Run batch ingestion for all configured entity types.

        Performs incremental sync by default (only fetch entities modified since
        last sync). Use force_full_sync=True to fetch all entities regardless of
        last sync time.

        Args:
            since: Optional datetime to fetch entities modified after (overrides last sync)
            entity_types: Optional list of entity types to sync (defaults to all)
            force_full_sync: If True, ignore last sync time and fetch all entities

        Returns:
            Dictionary with ingestion statistics:
            {
                "run_id": str,
                "started_at": str,
                "completed_at": str,
                "duration_seconds": float,
                "entity_types_synced": int,
                "total_entities_fetched": int,
                "total_entities_written": int,
                "entity_counts": {entity_type: count},
                "errors": [error_messages],
                "last_sync_time": str
            }

        Raises:
            IngestionError: If ingestion fails
            QBOAuthError: If authentication fails
        """
        run_id = str(uuid4())
        started_at = datetime.utcnow()

        logger.info(
            "batch_ingestion_started",
            run_id=run_id,
            force_full_sync=force_full_sync,
            since=since.isoformat() if since else None,
        )

        # Determine entity types to sync
        types_to_sync = entity_types or self.DEFAULT_ENTITY_TYPES

        # Determine sync starting point
        sync_since = since
        if not force_full_sync and not sync_since:
            # Use last sync time for incremental sync
            sync_since = await self.get_last_sync_time()

        # Track ingestion statistics
        entity_counts: dict[str, int] = {}
        total_fetched = 0
        total_written = 0
        errors: list[str] = []

        # Sync each entity type
        for entity_type in types_to_sync:
            try:
                logger.info(
                    "syncing_entity_type",
                    run_id=run_id,
                    entity_type=entity_type,
                    since=sync_since.isoformat() if sync_since else None,
                )

                # Fetch entities from QuickBooks
                entities = await self.qbo_client.get_all_entities(
                    entity_type=entity_type,
                    since=sync_since,
                )

                total_fetched += len(entities)
                entity_counts[entity_type] = len(entities)

                # Write entities to Bronze layer
                written_count = 0
                for entity in entities:
                    try:
                        entity_id = entity.get("Id", "")

                        record_id = self.storage.write_raw_entity(
                            entity_id=entity_id,
                            entity_type=entity_type,
                            source="qbo",
                            operation="Create",  # Batch sync doesn't distinguish create/update
                            raw_payload=entity,
                            api_version="v3",
                        )

                        written_count += 1

                        logger.debug(
                            "entity_written_to_bronze",
                            run_id=run_id,
                            entity_type=entity_type,
                            entity_id=entity_id,
                            record_id=record_id,
                        )

                    except Exception as e:
                        error_msg = f"Failed to write {entity_type} {entity.get('Id')}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(
                            "entity_write_failed",
                            run_id=run_id,
                            entity_type=entity_type,
                            entity_id=entity.get("Id"),
                            error=str(e),
                        )

                total_written += written_count

                logger.info(
                    "entity_type_sync_complete",
                    run_id=run_id,
                    entity_type=entity_type,
                    fetched=len(entities),
                    written=written_count,
                )

            except QBOAPIError as e:
                error_msg = f"Failed to fetch {entity_type}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    "entity_type_sync_failed",
                    run_id=run_id,
                    entity_type=entity_type,
                    error=str(e),
                )
                # Continue with other entity types

            except Exception as e:
                error_msg = f"Unexpected error syncing {entity_type}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    "entity_type_sync_error",
                    run_id=run_id,
                    entity_type=entity_type,
                    error=str(e),
                )
                # Continue with other entity types

        # Update last sync time
        current_sync_time = started_at
        await self._update_last_sync_time(current_sync_time)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        result = {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(duration, 2),
            "entity_types_synced": len(types_to_sync),
            "total_entities_fetched": total_fetched,
            "total_entities_written": total_written,
            "entity_counts": entity_counts,
            "errors": errors,
            "last_sync_time": current_sync_time.isoformat(),
            "success": len(errors) == 0,
        }

        logger.info(
            "batch_ingestion_complete",
            run_id=run_id,
            duration_seconds=duration,
            total_fetched=total_fetched,
            total_written=total_written,
            error_count=len(errors),
        )

        return result

    async def ingest_entity(
        self,
        entity_type: str,
        entity_id: str,
        operation: str = "Create",
        webhook_event_id: Optional[str] = None,
    ) -> str:
        """
        Ingest a single entity from QuickBooks into Bronze layer.

        Used for real-time ingestion triggered by webhooks or manual sync
        of specific entities.

        Args:
            entity_type: Type of entity (e.g., "Invoice", "Customer")
            entity_id: QuickBooks entity ID
            operation: Operation type (Create, Update, Delete)
            webhook_event_id: Optional webhook event ID for tracking

        Returns:
            Record ID of written entity in Bronze layer

        Raises:
            QBOAPIError: If entity fetch fails
            IngestionError: If entity write fails
        """
        logger.info(
            "ingesting_single_entity",
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
        )

        try:
            # Fetch entity from QuickBooks
            entity_data = await self.qbo_client.get_entity(
                entity_type=entity_type,
                entity_id=entity_id,
            )

            # Write to Bronze layer
            record_id = self.storage.write_raw_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                source="qbo",
                operation=operation,
                raw_payload=entity_data,
                webhook_event_id=webhook_event_id,
                api_version="v3",
            )

            logger.info(
                "entity_ingested",
                entity_type=entity_type,
                entity_id=entity_id,
                record_id=record_id,
                operation=operation,
            )

            return record_id

        except QBOAPIError as e:
            logger.error(
                "entity_fetch_failed",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e),
            )
            raise

        except Exception as e:
            logger.error(
                "entity_ingestion_failed",
                entity_type=entity_type,
                entity_id=entity_id,
                error=str(e),
            )
            raise IngestionError(f"Failed to ingest entity: {str(e)}")

    async def get_last_sync_time(self) -> Optional[datetime]:
        """
        Get timestamp of last successful batch ingestion.

        Returns:
            Datetime of last sync, or None if never synced

        Note:
            This is a simplified implementation storing sync time in memory.
            Production implementation should use Redis or database for persistence.
        """
        # In production, retrieve from Redis/database
        # For now, return a reasonable default (30 days ago)
        default_since = datetime.utcnow() - timedelta(days=30)

        logger.debug(
            "last_sync_time_retrieved",
            last_sync=default_since.isoformat(),
            note="Using default 30 days - implement persistent storage",
        )

        return default_since

    async def _update_last_sync_time(self, sync_time: datetime) -> None:
        """
        Update last sync timestamp after successful ingestion.

        Args:
            sync_time: Timestamp to record as last sync time

        Note:
            This is a placeholder implementation. Production should persist
            to Redis/database for durability across restarts.
        """
        logger.info(
            "last_sync_time_updated",
            sync_time=sync_time.isoformat(),
            note="Implement persistent storage for production",
        )

        # In production, write to Redis:
        # redis_client.set(f"{SYNC_METADATA_PREFIX}:last_sync", sync_time.isoformat())

    async def get_ingestion_status(self) -> dict[str, Any]:
        """
        Get detailed ingestion and connection status.

        Returns comprehensive status information for monitoring and diagnostics
        including connection health, last sync time, and entity counts.

        Returns:
            Dictionary with status information:
            {
                "connection_status": {
                    "is_connected": bool,
                    "realm_id": str,
                    "environment": str,
                    "access_token_expiry": str,
                    "refresh_token_expiry": str
                },
                "last_sync_time": str,
                "entity_counts": {entity_type: count},
                "health": "healthy" | "degraded" | "unhealthy"
            }
        """
        # Get connection status
        connection_status = self.qbo_client.get_connection_status()

        # Get last sync time
        last_sync = await self.get_last_sync_time()

        # Get entity counts from Bronze layer
        entity_counts = {}
        for entity_type in self.DEFAULT_ENTITY_TYPES:
            try:
                # Query Bronze layer for entity counts
                raw_entities = self.storage.read_raw_entities(
                    source="qbo",
                    entity_type=entity_type,
                    limit=1,  # Just need count, not actual data
                )
                # This is a simplified count - production should use COUNT query
                entity_counts[entity_type] = len(raw_entities)
            except Exception as e:
                logger.warning(
                    "failed_to_count_entities",
                    entity_type=entity_type,
                    error=str(e),
                )
                entity_counts[entity_type] = 0

        # Determine overall health
        health = "healthy"
        if not connection_status["is_connected"]:
            health = "unhealthy"
        elif not connection_status["has_refresh_token"]:
            health = "degraded"

        # Check if last sync is stale (more than 7 days ago)
        if last_sync:
            days_since_sync = (datetime.utcnow() - last_sync).days
            if days_since_sync > 7:
                health = "degraded"

        status = {
            "connection_status": connection_status,
            "last_sync_time": last_sync.isoformat() if last_sync else None,
            "entity_counts": entity_counts,
            "health": health,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            "ingestion_status_retrieved",
            health=health,
            is_connected=connection_status["is_connected"],
            last_sync=last_sync.isoformat() if last_sync else None,
        )

        return status

    async def test_connection(self) -> dict[str, Any]:
        """
        Test QuickBooks connection by fetching a small entity.

        Validates that the connection is working properly by making a minimal
        API request. Useful for health checks and diagnostics.

        Returns:
            Dictionary with test results:
            {
                "success": bool,
                "message": str,
                "response_time_ms": float,
                "realm_id": str
            }

        Raises:
            QBOAuthError: If authentication fails
            QBOAPIError: If API request fails
        """
        logger.info("testing_qbo_connection", realm_id=self.qbo_client.realm_id)

        started_at = datetime.utcnow()

        try:
            # Fetch a small query to test connection
            # Using CompanyInfo query which returns minimal data
            result = await self.qbo_client._make_request(
                method="GET",
                endpoint="companyinfo/" + (self.qbo_client.realm_id or ""),
            )

            completed_at = datetime.utcnow()
            response_time = (completed_at - started_at).total_seconds() * 1000

            logger.info(
                "connection_test_successful",
                response_time_ms=response_time,
                realm_id=self.qbo_client.realm_id,
            )

            return {
                "success": True,
                "message": "Connection test successful",
                "response_time_ms": round(response_time, 2),
                "realm_id": self.qbo_client.realm_id,
                "company_name": result.get("CompanyInfo", {}).get("CompanyName"),
            }

        except QBOAuthError as e:
            logger.error("connection_test_auth_failed", error=str(e))
            return {
                "success": False,
                "message": f"Authentication failed: {str(e)}",
                "response_time_ms": 0,
                "realm_id": self.qbo_client.realm_id,
            }

        except QBOAPIError as e:
            logger.error("connection_test_api_failed", error=str(e))
            return {
                "success": False,
                "message": f"API request failed: {str(e)}",
                "response_time_ms": 0,
                "realm_id": self.qbo_client.realm_id,
            }

        except Exception as e:
            logger.error("connection_test_error", error=str(e))
            return {
                "success": False,
                "message": f"Unexpected error: {str(e)}",
                "response_time_ms": 0,
                "realm_id": self.qbo_client.realm_id,
            }

    async def sync_entity_type(
        self,
        entity_type: str,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Sync a single entity type from QuickBooks.

        Useful for targeted sync of specific entity types without running
        full batch ingestion.

        Args:
            entity_type: Entity type to sync (e.g., "Invoice")
            since: Optional datetime to fetch entities modified after

        Returns:
            Dictionary with sync results:
            {
                "entity_type": str,
                "fetched_count": int,
                "written_count": int,
                "duration_seconds": float,
                "errors": [str]
            }

        Raises:
            ValueError: If entity type is not supported
        """
        if entity_type not in QBOClient.SUPPORTED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        started_at = datetime.utcnow()

        logger.info(
            "syncing_single_entity_type",
            entity_type=entity_type,
            since=since.isoformat() if since else None,
        )

        errors: list[str] = []
        written_count = 0

        try:
            # Fetch entities
            entities = await self.qbo_client.get_all_entities(
                entity_type=entity_type,
                since=since,
            )

            # Write to Bronze layer
            for entity in entities:
                try:
                    entity_id = entity.get("Id", "")

                    self.storage.write_raw_entity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        source="qbo",
                        operation="Create",
                        raw_payload=entity,
                        api_version="v3",
                    )

                    written_count += 1

                except Exception as e:
                    error_msg = f"Failed to write {entity_type} {entity.get('Id')}: {str(e)}"
                    errors.append(error_msg)
                    logger.error("entity_write_failed", entity_id=entity.get("Id"), error=str(e))

            completed_at = datetime.utcnow()
            duration = (completed_at - started_at).total_seconds()

            result = {
                "entity_type": entity_type,
                "fetched_count": len(entities),
                "written_count": written_count,
                "duration_seconds": round(duration, 2),
                "errors": errors,
                "success": len(errors) == 0,
            }

            logger.info(
                "entity_type_sync_complete",
                entity_type=entity_type,
                fetched=len(entities),
                written=written_count,
                errors=len(errors),
            )

            return result

        except QBOAPIError as e:
            logger.error("entity_type_sync_failed", entity_type=entity_type, error=str(e))
            raise IngestionError(f"Failed to sync {entity_type}: {str(e)}")
