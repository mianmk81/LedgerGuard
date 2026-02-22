"""
Telco Customer Churn Dataset Adapter.

This adapter transforms telecommunications customer churn datasets into
canonical events for churn analysis and prediction. Only processes customers
who have actually churned (Churn=Yes/1/True).

Dataset source: Typically from Kaggle Telco Customer Churn datasets.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import structlog

from api.models.enums import EntityType, EventType
from api.models.events import CanonicalEvent, DataQualityReport, QualityIssue

from .base_adapter import BaseAdapter

logger = structlog.get_logger()


class ChurnAdapter(BaseAdapter):
    """
    Adapts telco customer churn data into canonical events.

    Processes customer churn records and creates events only for customers
    who have churned. Captures tenure, contract details, and financial metrics
    to enable churn pattern analysis and prediction.

    Supported event types:
    - CUSTOMER_CHURNED: When customer terminates service
    """

    # Column name mappings for flexible dataset handling
    COLUMN_MAPPINGS = {
        "customer_id": ["customerID", "customer_id", "CustomerID", "ID", "customer"],
        "churn": ["Churn", "churn", "Churned", "churned", "Attrition"],
        "tenure": ["tenure", "Tenure", "tenure_months", "months_active"],
        "contract_type": [
            "Contract",
            "contract",
            "contract_type",
            "ContractType",
            "subscription_type",
        ],
        "monthly_charges": [
            "MonthlyCharges",
            "monthly_charges",
            "monthly_fee",
            "MonthlyFee",
        ],
        "total_charges": [
            "TotalCharges",
            "total_charges",
            "total_revenue",
            "TotalRevenue",
        ],
    }

    def __init__(self):
        """Initialize Churn adapter."""
        super().__init__(source_name="telco_churn")

    def _find_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        """
        Find the actual column name in DataFrame using flexible matching.

        Args:
            df: DataFrame to search
            field: Field type to find (from COLUMN_MAPPINGS)

        Returns:
            Actual column name if found, None otherwise
        """
        possible_names = self.COLUMN_MAPPINGS.get(field, [])
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        return None

    def _is_churned(self, value) -> bool:
        """
        Check if churn indicator represents a churned customer.

        Args:
            value: Churn field value (Yes/No, 1/0, True/False, etc.)

        Returns:
            True if customer has churned
        """
        if self._is_missing(value):
            return False

        value_str = str(value).strip().lower()
        return value_str in ["yes", "1", "true", "churned", "y"]

    def _estimate_churn_date(
        self, tenure_months: Optional[float], snapshot_date: Optional[datetime] = None
    ) -> datetime:
        """
        Estimate churn date based on tenure.

        Since churn datasets are typically snapshots, we estimate the churn
        event time by working backward from a snapshot date using tenure.

        Args:
            tenure_months: Customer tenure in months
            snapshot_date: Date of dataset snapshot (defaults to now)

        Returns:
            Estimated churn event datetime
        """
        if not snapshot_date:
            snapshot_date = datetime.utcnow()

        if tenure_months is not None and tenure_months > 0:
            # Approximate: churn occurred around the snapshot date
            # (This is a simplification; actual churn date may not be in dataset)
            return snapshot_date
        else:
            return snapshot_date

    def ingest(
        self,
        churn_df: pd.DataFrame,
        snapshot_date: Optional[datetime] = None,
        **kwargs,
    ) -> tuple[list[CanonicalEvent], DataQualityReport]:
        """
        Transform telco churn dataset into canonical events.

        Args:
            churn_df: DataFrame containing customer churn data
            snapshot_date: Date when the dataset snapshot was taken (defaults to now)
            **kwargs: Additional configuration options

        Returns:
            Tuple of (canonical events, data quality report)

        Expected columns (flexible matching):
        - customerID/customer_id: Unique customer identifier
        - Churn/churn: Churn indicator (Yes/No, 1/0, True/False)
        - tenure/Tenure: Customer tenure in months
        - Contract/contract: Contract type (Month-to-month, One year, Two year)
        - MonthlyCharges/monthly_charges: Monthly service charge
        - TotalCharges/total_charges: Total charges to date

        Note: Only customers with Churn=Yes/1/True will generate events.
        """
        self.logger.info(
            "churn_ingestion_started",
            customers_count=len(churn_df),
            snapshot_date=snapshot_date.isoformat() if snapshot_date else "now",
        )

        events = []
        quality_issues = []
        rejected_count = 0

        # Track quality metrics
        missing_customer_ids = 0
        missing_churn_flags = 0
        missing_tenure = 0
        missing_charges = 0
        invalid_charges = 0

        # Map column names
        col_customer_id = self._find_column(churn_df, "customer_id")
        col_churn = self._find_column(churn_df, "churn")
        col_tenure = self._find_column(churn_df, "tenure")
        col_contract = self._find_column(churn_df, "contract_type")
        col_monthly_charges = self._find_column(churn_df, "monthly_charges")
        col_total_charges = self._find_column(churn_df, "total_charges")

        if not col_customer_id:
            raise ValueError(
                "Cannot find customer_id column. Expected one of: "
                + ", ".join(self.COLUMN_MAPPINGS["customer_id"])
            )

        if not col_churn:
            raise ValueError(
                "Cannot find churn column. Expected one of: "
                + ", ".join(self.COLUMN_MAPPINGS["churn"])
            )

        # Use provided snapshot date or default to now
        if not snapshot_date:
            snapshot_date = datetime.utcnow()

        # Track non-churned customers (for reporting)
        non_churned_count = 0

        # Process each customer record
        for idx, row in churn_df.iterrows():
            customer_id = self._safe_str(row.get(col_customer_id))

            if not customer_id:
                missing_customer_ids += 1
                rejected_count += 1
                continue

            # Check churn status
            churn_value = row.get(col_churn)
            if self._is_missing(churn_value):
                missing_churn_flags += 1
                rejected_count += 1
                continue

            # Only process churned customers
            if not self._is_churned(churn_value):
                non_churned_count += 1
                continue

            # Extract customer attributes
            tenure = self._safe_float(row.get(col_tenure)) if col_tenure else None
            contract_type = (
                self._safe_str(row.get(col_contract)) if col_contract else None
            )
            monthly_charges = (
                self._safe_float(row.get(col_monthly_charges))
                if col_monthly_charges
                else None
            )
            total_charges = (
                self._safe_float(row.get(col_total_charges))
                if col_total_charges
                else None
            )

            # Track missing data
            event_flags = []
            if tenure is None:
                missing_tenure += 1
                event_flags.append("missing_tenure")

            if monthly_charges is None:
                missing_charges += 1
                event_flags.append("missing_monthly_charges")

            if total_charges is None:
                missing_charges += 1
                event_flags.append("missing_total_charges")

            # Validate charges
            if monthly_charges is not None and monthly_charges < 0:
                invalid_charges += 1
                event_flags.append("invalid_monthly_charges")
                monthly_charges = None

            if total_charges is not None and total_charges < 0:
                invalid_charges += 1
                event_flags.append("invalid_total_charges")
                total_charges = None

            # Estimate churn date
            churn_date = self._estimate_churn_date(tenure, snapshot_date)

            # Create CUSTOMER_CHURNED event
            churned_event = CanonicalEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.CUSTOMER_CHURNED,
                event_time=churn_date,
                source=self.source_name,
                source_entity_id=customer_id,
                entity_type=EntityType.CUSTOMER,
                entity_id=self._create_entity_id("customer", customer_id),
                related_entity_ids={},
                amount=total_charges,  # Use total charges as the lifetime value lost
                currency="USD",
                attributes={
                    "customer_id": customer_id,
                    "tenure": tenure,
                    "contract_type": contract_type,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "churn_label": str(churn_value),
                    "estimated_churn_date": churn_date.isoformat(),
                },
                data_quality_flags=event_flags,
                schema_version="canonical_v1",
            )
            events.append(churned_event)

        # Compile quality issues
        if missing_customer_ids > 0:
            quality_issues.append(
                QualityIssue(
                    field="customer_id",
                    issue_type="missing",
                    count=missing_customer_ids,
                    description=f"Missing customer ID in {missing_customer_ids} records",
                )
            )

        if missing_churn_flags > 0:
            quality_issues.append(
                QualityIssue(
                    field="churn",
                    issue_type="missing",
                    count=missing_churn_flags,
                    description=f"Missing churn indicator in {missing_churn_flags} records",
                )
            )

        if missing_tenure > 0:
            quality_issues.append(
                QualityIssue(
                    field="tenure",
                    issue_type="missing",
                    count=missing_tenure,
                    description=f"Missing tenure in {missing_tenure} churned customer records",
                )
            )

        if missing_charges > 0:
            quality_issues.append(
                QualityIssue(
                    field="charges",
                    issue_type="missing",
                    count=missing_charges,
                    description=f"Missing charge information in {missing_charges} field instances",
                )
            )

        if invalid_charges > 0:
            quality_issues.append(
                QualityIssue(
                    field="charges",
                    issue_type="invalid_value",
                    count=invalid_charges,
                    description=f"Invalid (negative) charges in {invalid_charges} field instances",
                )
            )

        # Generate quality report
        total_records = len(churn_df)
        quality_report = self._compute_quality_scores(
            events=events,
            total_records=total_records,
            rejected_records=rejected_count,
            quality_issues=quality_issues,
        )

        self.logger.info(
            "churn_ingestion_completed",
            total_events=len(events),
            total_records=total_records,
            churned_customers=len(events),
            non_churned_customers=non_churned_count,
            rejected_records=rejected_count,
            quality_score=quality_report.overall_quality_score,
        )

        return events, quality_report
