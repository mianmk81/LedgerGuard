# LedgerGuard - Business Reliability Engine

**Principal-engineer level implementation of SRE principles for financial operations.**

LedgerGuard applies Site Reliability Engineering (SRE) methodologies to business operations, specifically QuickBooks Online data. It detects anomalies, performs root cause analysis, maps blast radius, and generates actionable incident postmortems.

## Features

- **Automated Anomaly Detection**: Statistical + ML hybrid approach (Z-score, IQR, changepoint detection, LightGBM)
- **Root Cause Analysis**: Graph-based causal inference using NetworkX
- **Blast Radius Mapping**: Impact propagation through entity relationship graphs
- **Health Monitors**: SLO tracking and composite health scoring
- **Incident Postmortems**: Automated report generation (JSON, HTML, PDF)
- **What-If Simulation**: Monte Carlo and scenario analysis
- **Real-time Dashboard**: React 18 + Vite with Cytoscape graph visualization

## Architecture

### Backend (Python 3.11+)
- **Framework**: FastAPI (async-first, OpenAPI native)
- **Database**: DuckDB (embedded OLAP)
- **Cache/Queue**: Redis (Celery broker)
- **ML Stack**: scikit-learn, LightGBM, ruptures
- **Graph Engine**: NetworkX
- **Logging**: structlog (structured, request-scoped)

### Frontend (React 18 + Vite)
- **Framework**: React 18 (functional components, hooks)
- **Build**: Vite 5 (fast HMR)
- **Styling**: Tailwind CSS 3
- **Graphs**: Cytoscape.js + dagre
- **Charts**: Recharts

### Data Flow
```
QuickBooks API → Bronze (raw) → Silver (validated) → Gold (enriched)
                                                    ↓
                                        Detection Engine → Incidents
                                                    ↓
                                        RCA + Blast Radius → Postmortems
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Redis (or use Docker Compose)
- QuickBooks Online sandbox account

### Installation

```bash
# Clone repository (if applicable)
cd LedgerGuard

# Install dependencies
make install

# Copy environment file and configure
cp .env.example .env
# Edit .env with your QuickBooks credentials

# Start services with Docker
make docker-up

# OR start locally
make dev
```

### Access Points
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development

### Project Structure
```
ledgerguard/
├── api/                    # Backend (FastAPI)
│   ├── auth/              # JWT authentication
│   ├── routers/           # API endpoints
│   ├── services/          # Business logic
│   ├── engine/            # BRE core (detection, RCA, etc.)
│   ├── connectors/        # QuickBooks connector
│   ├── storage/           # Bronze/Silver/Gold layers
│   └── utils/             # Shared utilities
├── frontend/              # Frontend (React + Vite)
│   └── src/
│       ├── api/          # API client
│       ├── components/   # UI components
│       └── pages/        # Route pages
├── tests/                 # Test suite
├── scripts/              # Operational scripts
└── docs/                 # Documentation
```

### Common Commands

```bash
# Development
make dev              # Start dev servers (backend + frontend)
make dev-backend      # Backend only
make dev-frontend     # Frontend only

# Testing
make test            # Run all tests with coverage
make test-unit       # Unit tests only
make test-integration # Integration tests

# Code Quality
make lint            # Run linters (ruff + mypy)
make format          # Format code (black + ruff)

# Docker
make docker-up       # Start all services
make docker-down     # Stop all services
make docker-logs     # View logs

# Data & Demo
make seed            # Seed sandbox data
make demo            # Run demo scenario

# Utilities
make db-shell        # Open DuckDB shell
make redis-cli       # Open Redis CLI
```

## Configuration

All configuration via environment variables (see `.env.example`):

### Required
- `INTUIT_CLIENT_ID`: QuickBooks OAuth2 client ID
- `INTUIT_CLIENT_SECRET`: QuickBooks OAuth2 client secret
- `INTUIT_REDIRECT_URI`: OAuth2 callback URL
- `JWT_SECRET`: JWT signing secret (generate with `openssl rand -hex 32`)

### Optional
- `DB_PATH`: DuckDB file path (default: `./data/bre.duckdb`)
- `REDIS_URL`: Redis connection URL
- `ANOMALY_DETECTION_SENSITIVITY`: Detection sensitivity (0.0-1.0)
- `MIN_CONFIDENCE_THRESHOLD`: Minimum confidence for incidents (0.0-1.0)

## API Overview

### Authentication
```bash
# Initiate OAuth2 flow
GET /api/v1/auth/authorize

# Handle callback
GET /api/v1/auth/callback?code={code}&state={state}&realmId={realmId}
```

### Ingestion
```bash
# Start ingestion
POST /api/v1/ingestion/start
{
  "entity_types": ["invoice", "payment"],
  "full_refresh": false
}

# Check status
GET /api/v1/ingestion/status/{job_id}
```

### Analysis
```bash
# Run analysis
POST /api/v1/analysis/run
{
  "analysis_type": "full",
  "time_range_days": 30
}

# Get results
GET /api/v1/analysis/result/{analysis_id}
```

### Incidents
```bash
# List incidents
GET /api/v1/incidents/?severity=high&status=open

# Get detail
GET /api/v1/incidents/{incident_id}

# Get postmortem
GET /api/v1/incidents/{incident_id}/postmortem?format=json
```

## Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Golden Path Tests
```bash
pytest tests/golden/ -v
```

### Property-Based Testing
Uses Hypothesis for invariant checking:
```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0, max_value=1))
def test_confidence_bounds(confidence):
    assert 0 <= confidence <= 1
```

## Deployment

### Docker Compose (Recommended)
```bash
make docker-up
```

### Manual Deployment
```bash
# Backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Worker
celery -A api.tasks.celery_app worker --loglevel=info

# Frontend
cd frontend && npm run build && npm run preview
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### System Diagnostics
```bash
curl http://localhost:8000/api/v1/system/diagnostics
```

### Metrics Dashboard
Access comprehensive metrics at http://localhost:3000/dashboard

## Contributing

See [CLAUDE.md](./CLAUDE.md) for project conventions and architecture details.

### Code Style
- Python: Black + Ruff (100 char line length)
- JavaScript: Prettier (default config)
- Imports: Sorted with `ruff --select I`

### Commit Convention
```
feat: Add ML-based anomaly detector
fix: Correct blast radius traversal depth
docs: Update API documentation
test: Add integration tests for monitors
```

## License

Proprietary - All rights reserved

## Support

For issues and questions:
- Review [CLAUDE.md](./CLAUDE.md) for architecture details
- Check API docs at `/docs`
- Review test suite for usage examples

---

**Built with principal-engineer discipline. Production-ready. Scalable. Observable.**
