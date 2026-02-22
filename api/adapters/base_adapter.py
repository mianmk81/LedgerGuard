"""
Base adapter class for supplemental dataset transformations.

This module provides the abstract base class that all dataset adapters inherit from,
ensuring consistent ingestion patterns, data quality assessment, and event generation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from uuid import uuid4

import structlog

from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue

logger = structlog.get_logger()


class BaseAdapter(ABC):
    """
    Abstract base class for supplemental dataset adapters.

    All adapters must implement the ingest() method to transform source data
    into canonical events with comprehensive quality reporting. This ensures
    consistent data quality standards across all supplemental data sources.

    Attributes:
        source_name: Identifier for the data source (e.g., "olist", "support_tickets")
    """

    def __init__(self, source_name: str):
        """
        Initialize the adapter with a source name.

        Args:
            source_name: Identifier for this data source
        """
        self.source_name = source_name
        self.logger = logger.bind(adapter=source_name)

    @abstractmethod
    def ingest(self, *args, **kwargs) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Transform source data into canonical events with quality report.

        This method must be implemented by all concrete adapters. It should:
        1. Validate and clean input data
        2. Transform records into CanonicalEvent objects
        3. Track quality issues and generate comprehensive report
        4. Return both events and quality assessment

        Returns:
            Tuple of (canonical events list, data quality report)

        Raises:
            ValueError: If input data is fundamentally invalid
        """
        pass

    def _generate_event_id(self) -> str:
        """
        Generate a unique event ID using UUID v4.

        Returns:
            String representation of UUID v4
        """
        return str(uuid4())

    def _safe_str(self, value, default: str = "") -> str:
        """
        Safely convert value to string, handling None and NaN.

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            String representation or default
        """
        if value is None:
            return default
        try:
            # Handle pandas NA/NaN values
            import pandas as pd
            if pd.isna(value):
                return default
        except (ImportError, TypeError):
            pass
        return str(value).strip()

    def _safe_float(self, value, default: Optional[float] = None) -> Optional[float]:
        """
        Safely convert value to float, handling None and NaN.

        Args:
            value: Value to convert
            default: Default value if conversion fails

        Returns:
            Float value or default
        """
        if value is None:
            return default
        try:
            import pandas as pd
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError, ImportError):
            return default

    def _safe_datetime(
        self, value, default: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Safely convert value to datetime, handling various formats.

        Args:
            value: Value to convert (string, datetime, timestamp)
            default: Default value if conversion fails

        Returns:
            Datetime object or default
        """
        if value is None:
            return default

        try:
            import pandas as pd
            if pd.isna(value):
                return default
            # Use pandas to_datetime for robust parsing
            result = pd.to_datetime(value, errors='coerce')
            if pd.isna(result):
                return default
            # Convert to Python datetime if it's a pandas Timestamp
            if hasattr(result, 'to_pydatetime'):
                return result.to_pydatetime()
            return result
        except (ValueError, TypeError, ImportError):
            return default

    def _is_missing(self, value) -> bool:
        """
        Check if a value is missing (None, NaN, empty string).

        Args:
            value: Value to check

        Returns:
            True if value is missing
        """
        if value is None:
            return True
        try:
            import pandas as pd
            if pd.isna(value):
                return True
        except ImportError:
            pass
        if isinstance(value, str) and not value.strip():
            return True
        return False

    def _compute_quality_scores(
        self,
        events: list[CanonicalEvent],
        total_records: int,
        rejected_records: int,
        quality_issues: list[QualityIssue],
    ) -> DataQualityReport:
        """
        Generate comprehensive data quality report for ingested batch.

        Args:
            events: List of successfully created canonical events
            total_records: Total number of input records processed
            rejected_records: Number of records that failed validation
            quality_issues: List of quality issues detected during processing

        Returns:
            Complete DataQualityReport with scores and recommendations
        """
        valid_records = len(events)

        # Compute completeness score based on data quality flags in events
        total_flags = sum(len(event.data_quality_flags) for event in events)
        # Assume 5 key fields per event; penalize for each missing/invalid field
        expected_fields = valid_records * 5
        completeness_score = max(0.0, 1.0 - (total_flags / expected_fields)) if expected_fields > 0 else 1.0

        # Consistency score based on rejection rate
        consistency_score = (valid_records / total_records) if total_records > 0 else 1.0

        # Timeliness score (assume all supplemental data is batch, so 1.0 if processed)
        timeliness_score = 1.0

        # Overall quality score: weighted average
        overall_quality_score = (
            0.4 * completeness_score +
            0.4 * consistency_score +
            0.2 * timeliness_score
        )

        # Generate impact advisory
        if overall_quality_score >= 0.95:
            impact_advisory = "Excellent quality batch. No significant impact on analysis accuracy."
        elif overall_quality_score >= 0.85:
            impact_advisory = "Good quality batch. Minor issues may slightly affect confidence in some detections."
        elif overall_quality_score >= 0.70:
            impact_advisory = "Moderate quality batch. Quality issues may reduce detection accuracy. Review flagged events."
        else:
            impact_advisory = "Low quality batch. Significant quality issues detected. Manual review recommended before analysis."

        batch_id = f"batch_{self.source_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        report = DataQualityReport(
            batch_id=batch_id,
            source=self.source_name,
            total_records=total_records,
            valid_records=valid_records,
            rejected_records=rejected_records,
            completeness_score=round(completeness_score, 4),
            consistency_score=round(consistency_score, 4),
            timeliness_score=round(timeliness_score, 4),
            overall_quality_score=round(overall_quality_score, 4),
            quality_issues=quality_issues,
            impact_advisory=impact_advisory,
        )

        self.logger.info(
            "quality_report_generated",
            batch_id=batch_id,
            total_records=total_records,
            valid_records=valid_records,
            rejected_records=rejected_records,
            overall_quality_score=round(overall_quality_score, 4),
        )

        return report

    def _create_entity_id(self, entity_type: str, source_id: str) -> str:
        """
        Create a normalized entity ID in BRE namespace.

        Args:
            entity_type: Type of entity (customer, order, etc.)
            source_id: Source system entity ID

        Returns:
            Normalized entity ID in format: {entity_type}:{source}:{source_id}
        """
        return f"{entity_type}:{self.source_name}:{source_id}"
