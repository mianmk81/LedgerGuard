# LedgerGuard Demo Checklist

Demo the full project **without QuickBooks or Intuit credentials**.

## Prerequisites

- Python 3.11+
- `DEV_MODE=true` in `.env` (default)
- Optional: `pip install httpx` for automated analysis run

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start backend + frontend (two terminals, or use make dev)
# Terminal 1:
uvicorn api.main:app --reload --port 8000

# Terminal 2:
cd frontend && npm run dev

# 3. Run demo (seeds data + runs analysis)
python scripts/demo_run.py

# 4. Open http://localhost:3000 (or 5173)
# 5. Click "Try Demo" → explore the dashboard
```

## Step-by-Step

### 1. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Or: `make dev-backend`

### 2. Start the Frontend

```bash
cd frontend && npm run dev
```

Or: `make dev-frontend`

### 3. Seed Sample Data

```bash
python scripts/seed_sandbox.py --mode local
```

This creates 6 months of synthetic business data and runs the Bronze→Silver→Gold pipeline.

### 4. Run Full Demo (optional)

```bash
python scripts/demo_run.py
```

This seeds (if not done) and triggers an analysis run via the API.

### 5. Open the App

1. Go to **http://localhost:3000** (or **http://localhost:5173** for Vite)
2. Click **Try Demo** (no QuickBooks login)
3. Browse:
   - **Dashboard** — Health overview
   - **Incidents** — Detected anomalies
   - **Credit Pulse** — Financial health score
   - **Insights** — Cash runway, sentiment, recommendations
   - **Run Analysis** — Trigger detection manually

## What You Can Show

| Feature | Where |
|--------|-------|
| Anomaly detection | Incidents list, Analysis run |
| Root cause analysis | Incident detail |
| Blast radius | Incident detail |
| Postmortem reports | Incident detail |
| ML-powered sentiment | Insights → Support ticket sentiment |
| Delivery/churn prediction | `POST /api/v1/prediction/delivery-risk` |
| Credit Pulse | Credit Pulse page |
| What-if simulation | Comparison page |

## Troubleshooting

**"Try Demo" button not showing**
- Ensure API is running and reachable
- Check `DEV_MODE=true` in `.env`
- Open DevTools → Network to verify `/api/v1/auth/demo-available` returns `{"available": true}`

**401 on API calls**
- Click "Try Demo" again to refresh the token
- Token expires after 24h (JWT_EXIRATION_MINUTES)

**No incidents after analysis**
- Ensure data was seeded: `python scripts/seed_sandbox.py --mode local`
- Gold metrics need 30+ days; seed creates 6 months

**Analysis fails**
- Check API logs for errors
- Ensure DuckDB exists: `./data/bre.duckdb` (created by seed)
