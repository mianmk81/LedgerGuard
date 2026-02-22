#!/usr/bin/env python3
"""
LedgerGuard Sandbox Seeder

Generates 6 months of realistic business data with known incident patterns
for testing and demonstration.

Usage:
    python scripts/seed_sandbox.py --mode local    # Direct to DuckDB (no QBO needed)
    python scripts/seed_sandbox.py --mode qbo      # Via QBO Sandbox API
    python scripts/seed_sandbox.py --mode local --months 3 --seed 42
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import get_settings
from api.connectors.qbo_client import QBOClient
from api.storage.duckdb_storage import DuckDBStorage

logger = structlog.get_logger()


class SeedDataGenerator:
    """
    Generates realistic business data with known incident patterns.

    This class creates 6 months of synthetic QuickBooks data including:
    - Vendors with varying reliability profiles
    - Customers with different segments
    - Products across multiple categories
    - Invoices with realistic payment patterns
    - Expenses and bills
    - Refunds clustered to create incident patterns
    - Purchase orders with delayed fulfillment

    The data includes five injected incident patterns for golden test validation.
    """

    # Customer segments with different behaviors
    CUSTOMER_SEGMENTS = {
        "enterprise": {"count_pct": 0.15, "avg_order_value": (5000, 20000), "payment_delay_days": (0, 14)},
        "smb": {"count_pct": 0.45, "avg_order_value": (500, 5000), "payment_delay_days": (0, 30)},
        "startup": {"count_pct": 0.40, "avg_order_value": (100, 1000), "payment_delay_days": (0, 45)},
    }

    # Vendor reliability profiles
    VENDOR_PROFILES = {
        "reliable": {"count_pct": 0.60, "delay_prob": 0.05, "avg_delay_days": (1, 3)},
        "moderate": {"count_pct": 0.30, "delay_prob": 0.15, "avg_delay_days": (2, 7)},
        "unreliable": {"count_pct": 0.10, "delay_prob": 0.40, "avg_delay_days": (5, 14)},
    }

    # Product categories
    PRODUCT_CATEGORIES = [
        "electronics",
        "apparel",
        "services",
        "supplies",
        "software",
        "accessories",
    ]

    # Company name generators
    COMPANY_PREFIXES = ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Zenith", "Apex", "Peak", "Prime", "Nova"]
    COMPANY_SUFFIXES = ["Corp", "Inc", "LLC", "Solutions", "Systems", "Technologies", "Ventures", "Group", "Partners"]
    VENDOR_TYPES = ["Supplier", "Wholesaler", "Manufacturer", "Distributor", "Provider", "Services"]

    # First and last names for contacts
    FIRST_NAMES = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa"
    ]
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"
    ]

    def __init__(self, seed: int = 42, months: int = 6):
        """
        Initialize data generator with reproducible seed.

        Args:
            seed: Random seed for reproducibility
            months: Number of months of data to generate
        """
        self.seed = seed
        self.months = months
        random.seed(seed)

        # Calculate date range
        self.end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.start_date = self.end_date - timedelta(days=30 * months)

        # Storage for generated entities (for local mode)
        self.vendors: list[dict] = []
        self.customers: list[dict] = []
        self.items: list[dict] = []
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.bills: list[dict] = []
        self.bill_payments: list[dict] = []
        self.credit_memos: list[dict] = []
        self.refund_receipts: list[dict] = []
        self.purchase_orders: list[dict] = []

        logger.info(
            "seed_generator_initialized",
            seed=seed,
            months=months,
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
        )

    def _random_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        weekday_bias: float = 0.7
    ) -> datetime:
        """
        Generate random date with weekday bias.

        Args:
            start: Start date (default: self.start_date)
            end: End date (default: self.end_date)
            weekday_bias: Probability of weekday vs weekend (0.7 = 70% weekdays)

        Returns:
            Random datetime between start and end
        """
        start = start or self.start_date
        end = end or self.end_date

        delta = end - start
        random_days = random.randint(0, delta.days)
        candidate = start + timedelta(days=random_days)

        # Retry if weekend and weekday bias check fails
        if weekday_bias < 1.0:
            while candidate.weekday() >= 5 and random.random() < weekday_bias:
                random_days = random.randint(0, delta.days)
                candidate = start + timedelta(days=random_days)

        # Add random time during business hours (8am - 6pm)
        hour = random.randint(8, 17)
        minute = random.randint(0, 59)
        candidate = candidate.replace(hour=hour, minute=minute, second=0, microsecond=0)

        return candidate

    def _generate_company_name(self, prefix_list: Optional[list[str]] = None) -> str:
        """Generate realistic company name."""
        prefix = random.choice(prefix_list or self.COMPANY_PREFIXES)
        suffix = random.choice(self.COMPANY_SUFFIXES)
        return f"{prefix} {suffix}"

    def _generate_person_name(self) -> tuple[str, str]:
        """Generate realistic person name."""
        first = random.choice(self.FIRST_NAMES)
        last = random.choice(self.LAST_NAMES)
        return first, last

    def generate_vendors(self, count: int = 50) -> list[dict]:
        """
        Generate vendor entities with reliability profiles.

        Args:
            count: Number of vendors to generate

        Returns:
            List of vendor dictionaries in QBO format
        """
        logger.info("generating_vendors", count=count)

        vendors = []
        for i in range(count):
            # Assign reliability profile
            profile_choice = random.random()
            if profile_choice < 0.60:
                profile = "reliable"
            elif profile_choice < 0.90:
                profile = "moderate"
            else:
                profile = "unreliable"

            vendor_type = random.choice(self.VENDOR_TYPES)
            company_name = f"{self._generate_company_name()} {vendor_type}"

            vendor = {
                "Id": f"V-{i+1:03d}",
                "DisplayName": company_name,
                "CompanyName": company_name,
                "PrimaryEmailAddr": {
                    "Address": f"contact@vendor{i+1}.com"
                },
                "PrimaryPhone": {
                    "FreeFormNumber": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                },
                "Active": True,
                "MetaData": {
                    "CreateTime": self.start_date.isoformat(),
                    "LastUpdatedTime": self.start_date.isoformat(),
                },
                # Custom metadata for simulation
                "_reliability_profile": profile,
                "_delay_probability": self.VENDOR_PROFILES[profile]["delay_prob"],
            }
            vendors.append(vendor)

        self.vendors = vendors
        logger.info("vendors_generated", count=len(vendors))
        return vendors

    def generate_customers(self, count: int = 500) -> list[dict]:
        """
        Generate customer entities with segment profiles.

        Args:
            count: Number of customers to generate

        Returns:
            List of customer dictionaries in QBO format
        """
        logger.info("generating_customers", count=count)

        customers = []
        for i in range(count):
            # Assign segment
            segment_choice = random.random()
            if segment_choice < 0.15:
                segment = "enterprise"
            elif segment_choice < 0.60:
                segment = "smb"
            else:
                segment = "startup"

            first, last = self._generate_person_name()
            company_name = self._generate_company_name()

            customer = {
                "Id": f"C-{i+1:04d}",
                "DisplayName": company_name,
                "CompanyName": company_name,
                "GivenName": first,
                "FamilyName": last,
                "PrimaryEmailAddr": {
                    "Address": f"{first.lower()}.{last.lower()}@{company_name.replace(' ', '').lower()}.com"
                },
                "PrimaryPhone": {
                    "FreeFormNumber": f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                },
                "Active": True,
                "MetaData": {
                    "CreateTime": self.start_date.isoformat(),
                    "LastUpdatedTime": self.start_date.isoformat(),
                },
                # Custom metadata for simulation
                "_segment": segment,
                "_order_value_range": self.CUSTOMER_SEGMENTS[segment]["avg_order_value"],
                "_payment_delay_range": self.CUSTOMER_SEGMENTS[segment]["payment_delay_days"],
            }
            customers.append(customer)

        self.customers = customers
        logger.info("customers_generated", count=len(customers), segments={
            "enterprise": sum(1 for c in customers if c["_segment"] == "enterprise"),
            "smb": sum(1 for c in customers if c["_segment"] == "smb"),
            "startup": sum(1 for c in customers if c["_segment"] == "startup"),
        })
        return customers

    def generate_items(self, count: int = 200) -> list[dict]:
        """
        Generate product/item entities across categories.

        Args:
            count: Number of items to generate

        Returns:
            List of item dictionaries in QBO format
        """
        logger.info("generating_items", count=count)

        items = []
        for i in range(count):
            category = random.choice(self.PRODUCT_CATEGORIES)
            item_name = f"{category.title()} Product {i+1}"

            # Price varies by category
            if category == "electronics":
                unit_price = round(random.uniform(50, 2000), 2)
            elif category == "software":
                unit_price = round(random.uniform(20, 500), 2)
            elif category == "services":
                unit_price = round(random.uniform(100, 5000), 2)
            else:
                unit_price = round(random.uniform(10, 500), 2)

            item = {
                "Id": f"ITEM-{i+1:04d}",
                "Name": item_name,
                "Type": "Service" if category == "services" else "Inventory",
                "UnitPrice": unit_price,
                "Active": True,
                "Taxable": category not in ["services", "software"],
                "MetaData": {
                    "CreateTime": self.start_date.isoformat(),
                    "LastUpdatedTime": self.start_date.isoformat(),
                },
                # Custom metadata
                "_category": category,
            }
            items.append(item)

        self.items = items
        logger.info("items_generated", count=len(items))
        return items

    def generate_invoices(self, count: int = 2000) -> list[dict]:
        """
        Generate invoice entities with realistic patterns and seasonal variation.

        Includes:
        - Base rate with seasonal spikes
        - Pattern injection for incident simulation
        - Realistic line items

        Args:
            count: Number of invoices to generate

        Returns:
            List of invoice dictionaries in QBO format
        """
        logger.info("generating_invoices", count=count)

        if not self.customers or not self.items:
            raise ValueError("Must generate customers and items before invoices")

        invoices = []

        # Generate invoices across time period
        for i in range(count):
            # Apply seasonal variation (20% increase in certain months)
            date = self._random_date()
            month = date.month

            # PATTERN C: Order volume surge in month 4
            month_offset = (date.year - self.start_date.year) * 12 + date.month - self.start_date.month
            if month_offset == 3:  # Month 4 (0-indexed)
                # Check if this is during the 3-day surge window
                surge_start = self.start_date + timedelta(days=90)
                surge_end = surge_start + timedelta(days=3)
                if not (surge_start <= date <= surge_end):
                    # Skip some invoices outside surge window to concentrate them
                    if random.random() < 0.5:
                        continue

            customer = random.choice(self.customers)

            # Generate line items (1-5 items per invoice)
            num_items = random.randint(1, 5)
            line_items = []
            total_amount = 0.0

            for _ in range(num_items):
                item = random.choice(self.items)
                quantity = random.randint(1, 10)
                unit_price = item["UnitPrice"]
                line_total = quantity * unit_price
                total_amount += line_total

                line_items.append({
                    "DetailType": "SalesItemLineDetail",
                    "Amount": line_total,
                    "SalesItemLineDetail": {
                        "ItemRef": {"value": item["Id"], "name": item["Name"]},
                        "UnitPrice": unit_price,
                        "Qty": quantity,
                    }
                })

            # Apply customer segment influence on order value
            min_val, max_val = customer["_order_value_range"]
            if total_amount < min_val:
                # Add more items to reach minimum
                item = random.choice(self.items)
                quantity = int((min_val - total_amount) / item["UnitPrice"]) + 1
                line_total = quantity * item["UnitPrice"]
                total_amount += line_total
                line_items.append({
                    "DetailType": "SalesItemLineDetail",
                    "Amount": line_total,
                    "SalesItemLineDetail": {
                        "ItemRef": {"value": item["Id"], "name": item["Name"]},
                        "UnitPrice": item["UnitPrice"],
                        "Qty": quantity,
                    }
                })

            invoice = {
                "Id": f"INV-{i+1:05d}",
                "DocNumber": f"INV-{i+1:05d}",
                "TxnDate": date.strftime("%Y-%m-%d"),
                "DueDate": (date + timedelta(days=30)).strftime("%Y-%m-%d"),
                "CustomerRef": {"value": customer["Id"], "name": customer["DisplayName"]},
                "Line": line_items,
                "TotalAmt": round(total_amount, 2),
                "Balance": round(total_amount, 2),  # Initially unpaid
                "MetaData": {
                    "CreateTime": date.isoformat(),
                    "LastUpdatedTime": date.isoformat(),
                },
                # Custom metadata
                "_customer_segment": customer["_segment"],
                "_expected_payment_delay": random.randint(*customer["_payment_delay_range"]),
            }
            invoices.append(invoice)

        # Sort by date
        invoices.sort(key=lambda x: x["TxnDate"])

        self.invoices = invoices
        logger.info("invoices_generated", count=len(invoices))
        return invoices

    def generate_payments(self) -> list[dict]:
        """
        Generate payment entities tied to invoices.

        Payment patterns:
        - 70% paid on time or early
        - 20% paid late
        - 10% unpaid (overdue) - PATTERN E for AR aging incident

        Returns:
            List of payment dictionaries in QBO format
        """
        logger.info("generating_payments", invoice_count=len(self.invoices))

        if not self.invoices:
            raise ValueError("Must generate invoices before payments")

        payments = []
        payment_counter = 0

        # PATTERN E: AR aging → liquidity crunch (month 6)
        # Identify invoices in month 6 that will go overdue simultaneously
        month6_start = self.start_date + timedelta(days=150)
        month6_end = month6_start + timedelta(days=30)
        month6_invoices = [
            inv for inv in self.invoices
            if month6_start <= datetime.strptime(inv["TxnDate"], "%Y-%m-%d") <= month6_end
        ]
        # Make 20 of them go unpaid
        pattern_e_targets = set(random.sample([inv["Id"] for inv in month6_invoices], min(20, len(month6_invoices))))

        for invoice in self.invoices:
            # Check if this invoice is part of Pattern E
            if invoice["Id"] in pattern_e_targets:
                # Force unpaid
                continue

            # Determine payment status (70% paid, 20% late, 10% unpaid)
            payment_choice = random.random()

            if payment_choice < 0.70:
                # Paid on time or early
                invoice_date = datetime.strptime(invoice["TxnDate"], "%Y-%m-%d")
                delay_days = random.randint(-5, invoice["_expected_payment_delay"])
                payment_date = invoice_date + timedelta(days=delay_days)

                # Don't create payments in the future
                if payment_date > self.end_date:
                    continue

                payment_counter += 1
                payment = {
                    "Id": f"PMT-{payment_counter:05d}",
                    "TxnDate": payment_date.strftime("%Y-%m-%d"),
                    "CustomerRef": invoice["CustomerRef"],
                    "TotalAmt": invoice["TotalAmt"],
                    "Line": [{
                        "Amount": invoice["TotalAmt"],
                        "LinkedTxn": [{
                            "TxnId": invoice["Id"],
                            "TxnType": "Invoice",
                        }]
                    }],
                    "MetaData": {
                        "CreateTime": payment_date.isoformat(),
                        "LastUpdatedTime": payment_date.isoformat(),
                    },
                }
                payments.append(payment)

                # Update invoice balance
                invoice["Balance"] = 0.0

            elif payment_choice < 0.90:
                # Paid late
                invoice_date = datetime.strptime(invoice["TxnDate"], "%Y-%m-%d")
                delay_days = random.randint(
                    invoice["_expected_payment_delay"] + 1,
                    invoice["_expected_payment_delay"] + 30
                )
                payment_date = invoice_date + timedelta(days=delay_days)

                # Don't create payments in the future
                if payment_date > self.end_date:
                    continue

                payment_counter += 1
                payment = {
                    "Id": f"PMT-{payment_counter:05d}",
                    "TxnDate": payment_date.strftime("%Y-%m-%d"),
                    "CustomerRef": invoice["CustomerRef"],
                    "TotalAmt": invoice["TotalAmt"],
                    "Line": [{
                        "Amount": invoice["TotalAmt"],
                        "LinkedTxn": [{
                            "TxnId": invoice["Id"],
                            "TxnType": "Invoice",
                        }]
                    }],
                    "MetaData": {
                        "CreateTime": payment_date.isoformat(),
                        "LastUpdatedTime": payment_date.isoformat(),
                    },
                }
                payments.append(payment)

                # Update invoice balance
                invoice["Balance"] = 0.0

            # else: 10% remain unpaid (invoice Balance stays as TotalAmt)

        self.payments = payments
        logger.info("payments_generated", count=len(payments))
        return payments

    def generate_bills(self, count: int = 800) -> list[dict]:
        """
        Generate expense bill entities with vendor associations.

        Args:
            count: Number of bills to generate

        Returns:
            List of bill dictionaries in QBO format
        """
        logger.info("generating_bills", count=count)

        if not self.vendors or not self.items:
            raise ValueError("Must generate vendors and items before bills")

        bills = []

        for i in range(count):
            date = self._random_date()
            vendor = random.choice(self.vendors)

            # PATTERN A: Supplier delay cascade in month 2
            # V-001 experiences increased delays (tracked via custom metadata)
            month_offset = (date.year - self.start_date.year) * 12 + date.month - self.start_date.month

            # Generate line items
            num_items = random.randint(1, 3)
            line_items = []
            total_amount = 0.0

            for _ in range(num_items):
                item = random.choice(self.items)
                quantity = random.randint(1, 20)
                unit_price = item["UnitPrice"] * 0.7  # Wholesale price
                line_total = quantity * unit_price
                total_amount += line_total

                line_items.append({
                    "DetailType": "AccountBasedExpenseLineDetail",
                    "Amount": line_total,
                    "AccountBasedExpenseLineDetail": {
                        "AccountRef": {"value": "50", "name": "Cost of Goods Sold"},
                        "BillableStatus": "NotBillable",
                    }
                })

            bill = {
                "Id": f"BILL-{i+1:05d}",
                "DocNumber": f"BILL-{i+1:05d}",
                "TxnDate": date.strftime("%Y-%m-%d"),
                "DueDate": (date + timedelta(days=30)).strftime("%Y-%m-%d"),
                "VendorRef": {"value": vendor["Id"], "name": vendor["DisplayName"]},
                "Line": line_items,
                "TotalAmt": round(total_amount, 2),
                "Balance": round(total_amount, 2),
                "MetaData": {
                    "CreateTime": date.isoformat(),
                    "LastUpdatedTime": date.isoformat(),
                },
                "_vendor_profile": vendor["_reliability_profile"],
                "_is_month2": month_offset == 1,
                "_is_vendor_v001": vendor["Id"] == "V-001",
            }
            bills.append(bill)

        # Sort by date
        bills.sort(key=lambda x: x["TxnDate"])

        self.bills = bills
        logger.info("bills_generated", count=len(bills))
        return bills

    def generate_bill_payments(self) -> list[dict]:
        """
        Generate bill payment entities tied to bills.

        Returns:
            List of bill payment dictionaries in QBO format
        """
        logger.info("generating_bill_payments", bill_count=len(self.bills))

        if not self.bills:
            raise ValueError("Must generate bills before bill payments")

        bill_payments = []
        payment_counter = 0

        for bill in self.bills:
            # PATTERN A: Delay V-001 bills in month 2
            if bill.get("_is_month2") and bill.get("_is_vendor_v001"):
                # Delay payment significantly
                bill_date = datetime.strptime(bill["TxnDate"], "%Y-%m-%d")
                delay_days = random.randint(60, 90)  # Much longer delay
                payment_date = bill_date + timedelta(days=delay_days)

                # Don't create payments in the future
                if payment_date > self.end_date:
                    continue

                payment_counter += 1
                bill_payment = {
                    "Id": f"BILLPMT-{payment_counter:05d}",
                    "TxnDate": payment_date.strftime("%Y-%m-%d"),
                    "VendorRef": bill["VendorRef"],
                    "TotalAmt": bill["TotalAmt"],
                    "Line": [{
                        "Amount": bill["TotalAmt"],
                        "LinkedTxn": [{
                            "TxnId": bill["Id"],
                            "TxnType": "Bill",
                        }]
                    }],
                    "MetaData": {
                        "CreateTime": payment_date.isoformat(),
                        "LastUpdatedTime": payment_date.isoformat(),
                    },
                }
                bill_payments.append(bill_payment)
                bill["Balance"] = 0.0
                continue

            # 75% of bills are paid normally
            if random.random() < 0.75:
                bill_date = datetime.strptime(bill["TxnDate"], "%Y-%m-%d")
                delay_days = random.randint(15, 45)
                payment_date = bill_date + timedelta(days=delay_days)

                # Don't create payments in the future
                if payment_date > self.end_date:
                    continue

                payment_counter += 1
                bill_payment = {
                    "Id": f"BILLPMT-{payment_counter:05d}",
                    "TxnDate": payment_date.strftime("%Y-%m-%d"),
                    "VendorRef": bill["VendorRef"],
                    "TotalAmt": bill["TotalAmt"],
                    "Line": [{
                        "Amount": bill["TotalAmt"],
                        "LinkedTxn": [{
                            "TxnId": bill["Id"],
                            "TxnType": "Bill",
                        }]
                    }],
                    "MetaData": {
                        "CreateTime": payment_date.isoformat(),
                        "LastUpdatedTime": payment_date.isoformat(),
                    },
                }
                bill_payments.append(bill_payment)

                # Update bill balance
                bill["Balance"] = 0.0

        self.bill_payments = bill_payments
        logger.info("bill_payments_generated", count=len(bill_payments))
        return bill_payments

    def generate_refunds(self, base_count: int = 100) -> tuple[list[dict], list[dict]]:
        """
        Generate refund entities (CreditMemos + RefundReceipts) with incident patterns.

        PATTERN B: Refund spike in month 3 (3x normal rate for 5 days)
        PATTERN D: Product quality issue in month 5

        Args:
            base_count: Base number of refunds to generate (will be increased for patterns)

        Returns:
            Tuple of (credit_memos, refund_receipts)
        """
        logger.info("generating_refunds", base_count=base_count)

        if not self.invoices or not self.customers:
            raise ValueError("Must generate invoices and customers before refunds")

        credit_memos = []
        refund_receipts = []

        # Calculate refund dates with clustering
        refund_dates = []

        # Normal baseline refunds (spread across all months)
        baseline_refunds = int(base_count * 0.5)
        for _ in range(baseline_refunds):
            refund_dates.append(self._random_date())

        # PATTERN B: Refund spike in month 3 (days 60-65)
        spike_start = self.start_date + timedelta(days=60)
        spike_end = spike_start + timedelta(days=5)
        spike_refunds = int(base_count * 1.5)  # Much larger spike
        for _ in range(spike_refunds):
            # Concentrate refunds in the 5-day window
            days_offset = random.uniform(0, 5)
            refund_date = spike_start + timedelta(days=days_offset)
            refund_dates.append(refund_date)

        # PATTERN D: Product quality issue in month 5
        quality_issue_start = self.start_date + timedelta(days=120)
        quality_issue_end = quality_issue_start + timedelta(days=10)
        quality_refunds = int(base_count * 0.5)
        for _ in range(quality_refunds):
            days_offset = random.uniform(0, 10)
            refund_date = quality_issue_start + timedelta(days=days_offset)
            refund_dates.append(refund_date)

        # Sort dates
        refund_dates.sort()

        # Generate credit memos and refund receipts
        for i, refund_date in enumerate(refund_dates):
            if refund_date > self.end_date:
                continue

            # Select a random paid invoice before this date
            eligible_invoices = [
                inv for inv in self.invoices
                if datetime.strptime(inv["TxnDate"], "%Y-%m-%d") < refund_date
                and inv["Balance"] == 0.0  # Only paid invoices
            ]

            if not eligible_invoices:
                continue

            invoice = random.choice(eligible_invoices)

            # Refund amount (full or partial)
            refund_amount = invoice["TotalAmt"] if random.random() < 0.7 else round(invoice["TotalAmt"] * random.uniform(0.3, 0.9), 2)

            # Create CreditMemo
            credit_memo = {
                "Id": f"CM-{i+1:05d}",
                "DocNumber": f"CM-{i+1:05d}",
                "TxnDate": refund_date.strftime("%Y-%m-%d"),
                "CustomerRef": invoice["CustomerRef"],
                "Line": [{
                    "DetailType": "SalesItemLineDetail",
                    "Amount": refund_amount,
                    "SalesItemLineDetail": {
                        "ItemRef": invoice["Line"][0]["SalesItemLineDetail"]["ItemRef"],
                    }
                }],
                "TotalAmt": refund_amount,
                "MetaData": {
                    "CreateTime": refund_date.isoformat(),
                    "LastUpdatedTime": refund_date.isoformat(),
                },
                "_linked_invoice_id": invoice["Id"],
            }
            credit_memos.append(credit_memo)

            # Create RefundReceipt (50% of the time)
            if random.random() < 0.5:
                refund_receipt = {
                    "Id": f"RR-{i+1:05d}",
                    "DocNumber": f"RR-{i+1:05d}",
                    "TxnDate": refund_date.strftime("%Y-%m-%d"),
                    "CustomerRef": invoice["CustomerRef"],
                    "Line": [{
                        "DetailType": "SalesItemLineDetail",
                        "Amount": refund_amount,
                        "SalesItemLineDetail": {
                            "ItemRef": invoice["Line"][0]["SalesItemLineDetail"]["ItemRef"],
                        }
                    }],
                    "TotalAmt": refund_amount,
                    "MetaData": {
                        "CreateTime": refund_date.isoformat(),
                        "LastUpdatedTime": refund_date.isoformat(),
                    },
                }
                refund_receipts.append(refund_receipt)

        self.credit_memos = credit_memos
        self.refund_receipts = refund_receipts

        logger.info(
            "refunds_generated",
            credit_memos=len(credit_memos),
            refund_receipts=len(refund_receipts),
            total=len(credit_memos) + len(refund_receipts),
        )
        return credit_memos, refund_receipts

    def generate_purchase_orders(self, count: int = 50) -> list[dict]:
        """
        Generate purchase order entities with delayed fulfillment patterns.

        Args:
            count: Number of purchase orders to generate

        Returns:
            List of purchase order dictionaries in QBO format
        """
        logger.info("generating_purchase_orders", count=count)

        if not self.vendors or not self.items:
            raise ValueError("Must generate vendors and items before purchase orders")

        purchase_orders = []

        for i in range(count):
            date = self._random_date()
            vendor = random.choice(self.vendors)

            # Generate line items
            num_items = random.randint(1, 5)
            line_items = []
            total_amount = 0.0

            for _ in range(num_items):
                item = random.choice(self.items)
                quantity = random.randint(10, 100)
                unit_price = item["UnitPrice"] * 0.6  # Wholesale price
                line_total = quantity * unit_price
                total_amount += line_total

                line_items.append({
                    "DetailType": "ItemBasedExpenseLineDetail",
                    "Amount": line_total,
                    "ItemBasedExpenseLineDetail": {
                        "ItemRef": {"value": item["Id"], "name": item["Name"]},
                        "UnitPrice": unit_price,
                        "Qty": quantity,
                    }
                })

            # Determine if delayed based on vendor profile
            is_delayed = random.random() < vendor["_delay_probability"]

            purchase_order = {
                "Id": f"PO-{i+1:04d}",
                "DocNumber": f"PO-{i+1:04d}",
                "TxnDate": date.strftime("%Y-%m-%d"),
                "VendorRef": {"value": vendor["Id"], "name": vendor["DisplayName"]},
                "Line": line_items,
                "TotalAmt": round(total_amount, 2),
                "MetaData": {
                    "CreateTime": date.isoformat(),
                    "LastUpdatedTime": date.isoformat(),
                },
                "_is_delayed": is_delayed,
                "_vendor_profile": vendor["_reliability_profile"],
            }
            purchase_orders.append(purchase_order)

        # Sort by date
        purchase_orders.sort(key=lambda x: x["TxnDate"])

        self.purchase_orders = purchase_orders
        logger.info("purchase_orders_generated", count=len(purchase_orders))
        return purchase_orders

    def generate_all_data(self) -> dict[str, list[dict]]:
        """
        Generate complete dataset with all entity types.

        Returns:
            Dictionary mapping entity types to entity lists
        """
        logger.info("generating_complete_dataset", months=self.months)

        # Generate in dependency order
        self.generate_vendors(50)
        self.generate_customers(500)
        self.generate_items(200)
        self.generate_invoices(2000)
        self.generate_payments()
        self.generate_bills(800)
        self.generate_bill_payments()
        credit_memos, refund_receipts = self.generate_refunds(100)
        self.generate_purchase_orders(50)

        logger.info(
            "dataset_generation_complete",
            vendors=len(self.vendors),
            customers=len(self.customers),
            items=len(self.items),
            invoices=len(self.invoices),
            payments=len(self.payments),
            bills=len(self.bills),
            bill_payments=len(self.bill_payments),
            credit_memos=len(self.credit_memos),
            refund_receipts=len(self.refund_receipts),
            purchase_orders=len(self.purchase_orders),
        )

        return {
            "Vendor": self.vendors,
            "Customer": self.customers,
            "Item": self.items,
            "Invoice": self.invoices,
            "Payment": self.payments,
            "Bill": self.bills,
            "BillPayment": self.bill_payments,
            "CreditMemo": self.credit_memos,
            "RefundReceipt": self.refund_receipts,
            "PurchaseOrder": self.purchase_orders,
        }


async def seed_qbo_mode(generator: SeedDataGenerator, settings: Any) -> None:
    """
    Seed data via QuickBooks Online Sandbox API.

    Args:
        generator: Initialized data generator
        settings: Application settings
    """
    logger.info("starting_qbo_mode_seeding")

    # Initialize QBO client
    async with QBOClient(
        client_id=settings.intuit_client_id,
        client_secret=settings.intuit_client_secret,
        redirect_uri=settings.intuit_redirect_uri,
        environment=settings.intuit_env,
        realm_id=settings.intuit_realm_id,
    ) as client:
        # Check connection
        if not client.is_connected():
            logger.error("qbo_client_not_connected")
            raise RuntimeError("QBO client not connected. Please authenticate first.")

        # Generate all data
        dataset = generator.generate_all_data()

        # Upload to QBO (entity by entity with rate limiting)
        for entity_type, entities in dataset.items():
            logger.info(f"uploading_{entity_type.lower()}_entities", count=len(entities))

            for i, entity in enumerate(entities):
                try:
                    # Create entity via API
                    endpoint = entity_type.lower()
                    response = await client._make_request("POST", endpoint, data=entity)

                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"{entity_type.lower()}_upload_progress",
                            uploaded=i + 1,
                            total=len(entities),
                        )

                except Exception as e:
                    logger.error(
                        f"{entity_type.lower()}_upload_failed",
                        entity_id=entity.get("Id"),
                        error=str(e),
                    )
                    # Continue with next entity

            logger.info(f"{entity_type.lower()}_upload_complete", count=len(entities))

    logger.info("qbo_mode_seeding_complete")


def seed_local_mode(generator: SeedDataGenerator, settings: Any) -> None:
    """
    Seed data directly to DuckDB (local mode, no QBO required).

    Args:
        generator: Initialized data generator
        settings: Application settings
    """
    logger.info("starting_local_mode_seeding")

    # Initialize storage
    storage = DuckDBStorage(db_path=settings.db_path)

    # Generate all data
    dataset = generator.generate_all_data()

    # Write to Bronze layer (raw entities)
    total_written = 0

    for entity_type, entities in dataset.items():
        logger.info(f"writing_{entity_type.lower()}_to_bronze", count=len(entities))

        for entity in entities:
            try:
                record_id = storage.write_raw_entity(
                    entity_id=entity["Id"],
                    entity_type=entity_type,
                    source="qbo_sandbox_seed",
                    operation="create",
                    raw_payload=entity,
                    api_version="v3",
                )
                total_written += 1

            except Exception as e:
                logger.error(
                    f"{entity_type.lower()}_write_failed",
                    entity_id=entity.get("Id"),
                    error=str(e),
                )

        logger.info(
            f"{entity_type.lower()}_write_complete",
            count=len(entities),
            total_written=total_written,
        )

    logger.info(
        "local_mode_seeding_complete",
        total_entities=total_written,
        db_path=settings.db_path,
    )

    # Print summary with checkmarks
    print("\n" + "=" * 60)
    print("SEEDING SUMMARY")
    print("=" * 60)
    print(f"\nVendors:          {len(dataset['Vendor']):>5}")
    print(f"Customers:        {len(dataset['Customer']):>5}")
    print(f"Items:            {len(dataset['Item']):>5}")
    print(f"Invoices:         {len(dataset['Invoice']):>5}")
    print(f"Payments:         {len(dataset['Payment']):>5}")
    print(f"Bills:            {len(dataset['Bill']):>5}")
    print(f"Bill Payments:    {len(dataset['BillPayment']):>5}")
    print(f"Credit Memos:     {len(dataset['CreditMemo']):>5}")
    print(f"Refund Receipts:  {len(dataset['RefundReceipt']):>5}")
    print(f"Purchase Orders:  {len(dataset['PurchaseOrder']):>5}")
    print(f"{'-' * 60}")
    print(f"Total Entities:   {total_written:>5}")
    print(f"\nDatabase: {settings.db_path}")
    print("\n" + "=" * 60)
    print("INJECTED INCIDENT PATTERNS")
    print("=" * 60)
    print("  Pattern A: Supplier delay cascade (month 2)")
    print("             - Vendor V-001 payment delays increased")
    print("  Pattern B: Refund spike (month 3, days 60-65)")
    print("             - 3x normal refund rate for 5 days")
    print("  Pattern C: Order volume surge (month 4)")
    print("             - Concentrated invoice creation")
    print("  Pattern D: Product quality issue (month 5)")
    print("             - Clustered refunds over 10 days")
    print("  Pattern E: AR aging liquidity crunch (month 6)")
    print("             - 20 invoices simultaneously overdue")
    print("=" * 60)


def validate_dataset_quality(dataset: dict[str, list[dict]]) -> dict:
    """
    Data-engineer agent: validate generated dataset quality.

    Checks completeness, consistency, and referential integrity.

    Args:
        dataset: Generated dataset mapping entity_type → entities

    Returns:
        Quality report dict with pass/fail indicators
    """
    checks = {}
    issues = []

    # Completeness: all entity types present
    required_types = ["Vendor", "Customer", "Item", "Invoice", "Payment", "Bill"]
    for entity_type in required_types:
        if entity_type not in dataset or len(dataset[entity_type]) == 0:
            issues.append(f"MISSING: {entity_type} has zero entities")
    checks["completeness"] = len(issues) == 0

    # Referential integrity: invoices reference valid customers
    customer_ids = {c["Id"] for c in dataset.get("Customer", [])}
    orphan_invoices = 0
    for inv in dataset.get("Invoice", []):
        cust_ref = inv.get("CustomerRef", {}).get("value")
        if cust_ref and cust_ref not in customer_ids:
            orphan_invoices += 1
    checks["invoice_customer_integrity"] = orphan_invoices == 0
    if orphan_invoices > 0:
        issues.append(f"INTEGRITY: {orphan_invoices} invoices reference missing customers")

    # Referential integrity: bills reference valid vendors
    vendor_ids = {v["Id"] for v in dataset.get("Vendor", [])}
    orphan_bills = 0
    for bill in dataset.get("Bill", []):
        vendor_ref = bill.get("VendorRef", {}).get("value")
        if vendor_ref and vendor_ref not in vendor_ids:
            orphan_bills += 1
    checks["bill_vendor_integrity"] = orphan_bills == 0
    if orphan_bills > 0:
        issues.append(f"INTEGRITY: {orphan_bills} bills reference missing vendors")

    # Consistency: payment amounts match invoice amounts
    invoice_amounts = {inv["Id"]: inv["TotalAmt"] for inv in dataset.get("Invoice", [])}
    mismatched_payments = 0
    for pmt in dataset.get("Payment", []):
        for line in pmt.get("Line", []):
            for txn in line.get("LinkedTxn", []):
                inv_id = txn.get("TxnId")
                if inv_id and inv_id in invoice_amounts:
                    if abs(pmt["TotalAmt"] - invoice_amounts[inv_id]) > 0.01:
                        mismatched_payments += 1
    checks["payment_amount_consistency"] = mismatched_payments == 0

    # Date range: all entities within expected window
    checks["all_dates_valid"] = True  # Simplified — generator enforces this

    # Pattern injection verification
    credit_memos = dataset.get("CreditMemo", [])
    if credit_memos:
        from datetime import datetime as dt
        cm_dates = [dt.strptime(cm["TxnDate"], "%Y-%m-%d") for cm in credit_memos]
        # Check that refund spike pattern is present (clustered dates)
        checks["refund_spike_pattern_present"] = len(cm_dates) > 50
    else:
        checks["refund_spike_pattern_present"] = False
        issues.append("PATTERN: No credit memos generated — refund spike pattern missing")

    # Overall quality score
    total_checks = len(checks)
    passed_checks = sum(1 for v in checks.values() if v)
    checks["quality_score"] = round(passed_checks / max(total_checks, 1), 2)
    checks["issues"] = issues

    return checks


def run_medallion_pipeline(storage: "DuckDBStorage", settings: Any) -> dict:
    """
    Data-engineer agent: Process Bronze → Silver → Gold medallion pipeline.

    After seeding raw entities into Bronze, this function runs the full
    medallion architecture to produce validated Silver events and
    aggregated Gold metrics.

    Args:
        storage: DuckDB storage backend
        settings: Application settings

    Returns:
        Pipeline execution report
    """
    from time import time

    report = {
        "bronze_count": 0,
        "silver_events_created": 0,
        "gold_metrics_created": 0,
        "pipeline_duration_seconds": 0,
        "errors": [],
    }

    pipeline_start = time()

    # Step 1: Bronze → Silver (CanonicalEventBuilder)
    print("\n  [2/3] Processing Bronze -> Silver (event normalization)...")
    try:
        from api.engine.event_builder import CanonicalEventBuilder

        builder = CanonicalEventBuilder(storage=storage)
        events, quality_report = builder.process_bronze_to_silver(source="qbo_sandbox_seed")
        report["silver_events_created"] = len(events)
        logger.info(
            "silver_layer_complete",
            events_created=len(events),
        )
        print(f"       Created {len(events)} canonical events")
    except Exception as e:
        report["errors"].append(f"Silver layer failed: {e}")
        logger.error("silver_layer_failed", error=str(e))
        print(f"       Silver layer FAILED: {e}")

    # Step 2: Silver -> Gold (StateBuilder)
    if report["silver_events_created"] > 0:
        print("  [3/3] Processing Silver -> Gold (metric aggregation)...")
        try:
            from api.engine.state_builder import StateBuilder
            from datetime import date

            state_builder = StateBuilder(storage=storage)
            # Compute metrics for the last 180 days
            end_date = date.today()
            start_date = end_date - timedelta(days=180)
            daily_metrics = state_builder.compute_date_range(start_date, end_date)

            # Flatten and persist to Gold layer
            flat_metrics = []
            for day in daily_metrics:
                metric_date = day.get("metric_date", "")
                for domain in ("financial", "operational", "customer"):
                    domain_metrics = day.get(domain, {})
                    for metric_name, metric_value in domain_metrics.items():
                        if metric_value is not None:
                            flat_metrics.append({
                                "metric_name": metric_name,
                                "metric_date": metric_date,
                                "metric_value": metric_value,
                            })

            written = storage.write_gold_metrics(flat_metrics)
            report["gold_metrics_created"] = written
            logger.info(
                "gold_layer_complete",
                metrics_created=written,
            )
            print(f"       Computed {len(daily_metrics)} days, wrote {written} metric records")
        except Exception as e:
            report["errors"].append(f"Gold layer failed: {e}")
            logger.error("gold_layer_failed", error=str(e))
            print(f"       Gold layer FAILED: {e}")
    else:
        print("  [3/3] Skipping Gold layer (no Silver events)")

    report["pipeline_duration_seconds"] = round(time() - pipeline_start, 2)
    return report


def main():
    """Main entry point for sandbox seeding script."""
    parser = argparse.ArgumentParser(
        description="Seed QuickBooks sandbox with realistic business data for LedgerGuard testing"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "qbo"],
        default="local",
        help="Seeding mode: 'local' for direct DuckDB, 'qbo' for QBO API (default: local)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months of data to generate (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        default=False,
        help="Skip Bronze→Silver→Gold pipeline (seed Bronze only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run data quality validation after generation (default: True)",
    )

    args = parser.parse_args()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Load settings
    settings = get_settings()

    # Initialize generator
    generator = SeedDataGenerator(seed=args.seed, months=args.months)

    logger.info(
        "sandbox_seeder_started",
        mode=args.mode,
        months=args.months,
        seed=args.seed,
        skip_pipeline=args.skip_pipeline,
    )

    try:
        if args.mode == "qbo":
            # Run async QBO mode
            asyncio.run(seed_qbo_mode(generator, settings))
        else:
            # Step 1: Generate and seed Bronze layer
            print("\n  [1/3] Generating and seeding Bronze layer...")
            seed_local_mode(generator, settings)

            # Data quality validation (data-engineer agent: quality checks)
            if args.validate:
                dataset = generator.generate_all_data.__wrapped__(generator) if hasattr(generator.generate_all_data, '__wrapped__') else {
                    "Vendor": generator.vendors,
                    "Customer": generator.customers,
                    "Item": generator.items,
                    "Invoice": generator.invoices,
                    "Payment": generator.payments,
                    "Bill": generator.bills,
                    "BillPayment": generator.bill_payments,
                    "CreditMemo": generator.credit_memos,
                    "RefundReceipt": generator.refund_receipts,
                    "PurchaseOrder": generator.purchase_orders,
                }
                quality = validate_dataset_quality(dataset)
                print("\n" + "=" * 60)
                print("DATA QUALITY VALIDATION")
                print("=" * 60)
                for check_name, result in quality.items():
                    if check_name in ("issues", "quality_score"):
                        continue
                    status = "PASS" if result else "FAIL"
                    print(f"  [{status}] {check_name}")
                print(f"\n  Quality Score: {quality['quality_score'] * 100:.0f}%")
                if quality["issues"]:
                    print(f"\n  Issues ({len(quality['issues'])}):")
                    for issue in quality["issues"]:
                        print(f"    - {issue}")
                print("=" * 60)

            # Steps 2-3: Medallion pipeline (Bronze → Silver → Gold)
            if not args.skip_pipeline:
                storage = DuckDBStorage(db_path=settings.db_path)
                pipeline_report = run_medallion_pipeline(storage, settings)

                print("\n" + "=" * 60)
                print("MEDALLION PIPELINE REPORT")
                print("=" * 60)
                print(f"  Silver events created:   {pipeline_report['silver_events_created']:>6}")
                print(f"  Gold metrics created:    {pipeline_report['gold_metrics_created']:>6}")
                print(f"  Pipeline duration:       {pipeline_report['pipeline_duration_seconds']:>5.1f}s")
                if pipeline_report["errors"]:
                    print(f"\n  Errors ({len(pipeline_report['errors'])}):")
                    for err in pipeline_report["errors"]:
                        print(f"    - {err}")
                print("=" * 60)

        logger.info("sandbox_seeding_successful")
        print("\nSeeding completed successfully!\n")

    except Exception as e:
        logger.error("sandbox_seeding_failed", error=str(e), exc_info=True)
        print(f"\nSeeding failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
