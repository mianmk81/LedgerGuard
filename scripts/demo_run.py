#!/usr/bin/env python3
"""
LedgerGuard Demo — Full project demo without QuickBooks credentials.

Runs the complete BRE workflow:
1. Seeds 6 months of sample data (Bronze → Silver → Gold)
2. Runs anomaly detection + RCA + blast radius + postmortems via API
3. Prints access instructions

Requires: API running (make dev or uvicorn api.main:app)
Usage:
    python scripts/demo_run.py              # Seed + run analysis
    python scripts/demo_run.py --no-seed   # Skip seed (data already exists)
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    httpx = None


def main():
    parser = argparse.ArgumentParser(description="Run LedgerGuard demo (no QuickBooks required)")
    parser.add_argument("--no-seed", action="store_true", help="Skip seeding; assume data exists")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LedgerGuard Demo")
    print("=" * 60)

    # Step 1: Seed data
    if not args.no_seed:
        print("\n[1/3] Seeding sample data (6 months)...")
        seed_script = Path(__file__).parent / "seed_sandbox.py"
        result = subprocess.run(
            [sys.executable, str(seed_script), "--mode", "local"],
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode != 0:
            print("\nSeeding failed. Fix errors above and retry.")
            sys.exit(1)
        print("  Seeding complete.\n")
    else:
        print("\n[1/3] Skipping seed (--no-seed).\n")

    # Step 2: Get demo token and run analysis
    if not httpx:
        print("[2/3] Install httpx: pip install httpx")
        print("      Then run analysis from the UI after clicking 'Try Demo'.")
        print_done()
        return

    print("[2/3] Getting demo token and running analysis...")
    base = args.api_url.rstrip("/")

    with httpx.Client(timeout=60) as client:
        # Get demo token
        try:
            r = client.post(f"{base}/api/v1/auth/demo-token")
            r.raise_for_status()
            token = r.json()["access_token"]
        except Exception as e:
            print(f"\n  Failed to get demo token: {e}")
            print("  Make sure the API is running: uvicorn api.main:app --reload")
            print("  And DEV_MODE=true in .env")
            sys.exit(1)

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Run analysis
        try:
            r = client.post(
                f"{base}/api/v1/analysis/run",
                headers=headers,
                json={"lookback_days": 90, "run_rca": True, "run_blast_radius": True, "run_postmortem": True},
            )
            r.raise_for_status()
            data = r.json().get("data", r.json())
            incidents = data.get("incidents_detected", 0)
            run_id = data.get("run_id", "?")
            print(f"  Analysis complete: {incidents} incidents detected (run_id={run_id})")
        except Exception as e:
            print(f"  Analysis request failed: {e}")
            print("  You can still run analysis from the UI after clicking 'Try Demo'.")

    print("\n[3/3] Done.")
    print_done()


def print_done():
    print("\n" + "=" * 60)
    print("DEMO READY")
    print("=" * 60)
    print()
    print("  1. Open the app:   http://localhost:3000  (or 5173 for Vite)")
    print("  2. Click 'Try Demo' (no QuickBooks needed)")
    print("  3. Explore: Dashboard, Incidents, Insights, Credit Pulse")
    print("  4. Run analysis: POST /api/v1/analysis/run from the UI or API")
    print()
    print("  API docs:  http://localhost:8000/docs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
