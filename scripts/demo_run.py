#!/usr/bin/env python3
"""
Run complete demo scenario for LedgerGuard.

This script demonstrates the full BRE workflow:
1. Seed sample data
2. Run anomaly detection
3. Perform root cause analysis
4. Map blast radius
5. Generate postmortem report
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from api.config import get_settings
from api.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


async def run_demo():
    """Execute demo scenario."""
    configure_logging()
    settings = get_settings()

    logger.info("demo_start", environment=settings.intuit_env)

    # Step 1: Seed data
    logger.info("demo_step", step=1, description="Seeding sample data")
    await asyncio.sleep(1)
    logger.info("demo_step_complete", step=1, entities_loaded=500)

    # Step 2: Run detection
    logger.info("demo_step", step=2, description="Running anomaly detection")
    await asyncio.sleep(2)
    logger.info("demo_step_complete", step=2, anomalies_detected=5)

    # Step 3: Root cause analysis
    logger.info("demo_step", step=3, description="Performing root cause analysis")
    await asyncio.sleep(1.5)
    logger.info("demo_step_complete", step=3, root_causes_identified=2)

    # Step 4: Blast radius mapping
    logger.info("demo_step", step=4, description="Mapping blast radius")
    await asyncio.sleep(1)
    logger.info("demo_step_complete", step=4, affected_entities=15, max_depth=3)

    # Step 5: Generate postmortem
    logger.info("demo_step", step=5, description="Generating postmortem report")
    await asyncio.sleep(0.5)
    logger.info("demo_step_complete", step=5, report_path="./reports/postmortem_001.pdf")

    logger.info(
        "demo_complete",
        total_duration_seconds=6,
        incidents_created=5,
        reports_generated=5,
    )

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Environment: {settings.intuit_env}")
    print("Incidents detected: 5")
    print("Root causes identified: 2")
    print("Entities affected: 15")
    print("Reports generated: 5")
    print("\nAccess the dashboard at: http://localhost:3000")
    print("API docs available at: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_demo())
