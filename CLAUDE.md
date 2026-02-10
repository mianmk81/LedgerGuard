# LedgerGuard - Project Conventions & Architecture

**Principal-Engineer Level Business Reliability Engine**

## Project Overview

LedgerGuard is a sophisticated Business Reliability Engineering (BRE) platform that applies SRE principles to financial operations. It ingests data from QuickBooks Online, detects anomalies, performs root cause analysis, maps blast radius, and generates actionable incident postmortems.

## Architecture Philosophy

### Core Principles

1. **Medallion Architecture**: Bronze (raw) → Silver (validated) → Gold (enriched)
2. **Event-Driven Design**: Async processing via Celery for long-running operations
3. **ML-First Approach**: Statistical + ML hybrid detection with MLflow tracking
4. **Graph-Based RCA**: NetworkX for entity relationship and cascade analysis
5. **Type Safety**: Pydantic v2 for runtime validation across all boundaries
6. **Observability**: Structured logging (structlog) with request tracing

## Technology Stack

### Backend (Python 3.11+)

- **Framework**: FastAPI 0.104+ (async-first, OpenAPI native)
- **Validation**: Pydantic v2 (strict type validation)
- **Database**: DuckDB (embedded OLAP, analytics workload)
- **Cache/Queue**: Redis (session, celery broker, results)
- **ML Stack**: scikit-learn, LightGBM, ruptures (changepoint detection)
- **Graph**: NetworkX (entity graphs, blast radius)
- **Async Tasks**: Celery 5.3+ (ingestion, analysis, training)
- **Logging**: structlog (structured, request-scoped)
- **ML Tracking**: MLflow (experiment tracking, model registry)
- **Optimization**: Optuna (hyperparameter tuning)
- **Testing**: pytest, hypothesis (property-based testing)

### Frontend (React 18 + Vite)

- **Framework**: React 18 (functional components, hooks)
- **Build Tool**: Vite 5 (fast HMR, modern ESM)
- **Styling**: Tailwind CSS 3 (utility-first)
- **Graphs**: Cytoscape.js + cytoscape-dagre (entity graphs)
- **Charts**: Recharts (time series, sparklines)
- **HTTP**: Axios (interceptors for JWT, request ID)
- **Routing**: React Router v6 (declarative routing)

### Infrastructure

- **Orchestration**: Docker Compose (local development)
- **Services**: API, Worker, Redis, Frontend
- **Development**: Hot reload enabled for all services

## Directory Structure

```
ledgerguard/
├── api/                    # Backend application
│   ├── adapters/          # External data adapters (supplemental datasets)
│   ├── auth/              # JWT authentication & authorization
│   ├── config.py          # Settings management (pydantic-settings)
│   ├── connectors/        # QuickBooks Online OAuth2 + API client
│   ├── engine/            # Core BRE engine (detection, RCA, monitors)
│   ├── main.py            # FastAPI application factory
│   ├── models/            # Pydantic schemas (request/response)
│   ├── routers/           # API endpoints (versioned routes)
│   ├── services/          # Business logic layer
│   ├── storage/           # Bronze/Silver/Gold data layers
│   └── utils/             # Shared utilities (logging, etc.)
├── frontend/              # React application
│   ├── src/
│   │   ├── api/          # API client (axios instance)
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Route-level pages
│   │   ├── App.jsx       # Root component + router
│   │   └── main.jsx      # Entry point
│   └── package.json
├── scripts/               # Operational scripts
│   ├── seed_sandbox.py   # QuickBooks sandbox data seeding
│   └── demo_run.py       # End-to-end demo scenario
├── tests/                 # Test suite
│   ├── unit/             # Unit tests (pure logic)
│   ├── integration/      # Integration tests (API + DB)
│   └── golden/           # Golden path scenarios
├── data/                  # DuckDB database files
├── docs/                  # Documentation
├── mlruns/               # MLflow tracking data
└── reports/              # Generated postmortem PDFs
```

## API Design Conventions

### Endpoint Structure

```
/api/v1/
  /auth              # Authentication (OAuth2, JWT)
  /connection        # QuickBooks connection management
  /ingestion         # Data ingestion triggers
  /analysis          # On-demand analysis
  /incidents         # Incident management
  /cascades          # Cascade/blast radius analysis
  /monitors          # Health monitors & SLOs
  /comparison        # What-if simulation & comparison
  /simulation        # Scenario simulation
  /metrics           # Business metrics aggregation
  /system            # System health & diagnostics
```

### Response Envelope

All API responses follow this structure:

```python
{
    "success": bool,
    "data": Any,
    "error": Optional[str],
    "metadata": {
        "request_id": str,
        "timestamp": str,
        "duration_ms": float
    }
}
```

### Error Handling

- **400**: Validation errors (Pydantic)
- **401**: Unauthorized (missing/invalid JWT)
- **403**: Forbidden (insufficient permissions)
- **404**: Resource not found
- **422**: Semantic validation errors
- **500**: Internal server error (logged with stack trace)
- **503**: Service unavailable (dependency failure)

## Data Flow

### Ingestion Pipeline

1. **Bronze Layer**: Raw QuickBooks API responses (JSON, timestamped)
2. **Silver Layer**: Validated, normalized entities (Pydantic validated)
3. **Gold Layer**: Enriched with metrics, relationships, anomaly scores

### Detection Flow

1. **Statistical Detection**: Z-score, IQR, seasonal decomposition
2. **Changepoint Detection**: Ruptures (PELT, BottomUp algorithms)
3. **ML Detection**: LightGBM classifier (trained on historical patterns)
4. **Ensemble Scoring**: Weighted confidence from all methods

### RCA Process

1. **Entity Graph Construction**: NetworkX directed graph
2. **Temporal Correlation**: Cross-correlation analysis (scipy)
3. **Causal Ranking**: PageRank + temporal ordering
4. **Human-Readable Explanation**: Natural language generation

## Authentication Flow

### QuickBooks OAuth2

1. User initiates connection → Frontend redirects to Intuit
2. User authorizes → Intuit redirects to `/api/v1/auth/callback`
3. Backend exchanges code for tokens → Stores in Redis
4. Returns JWT to frontend with realm_id embedded

### JWT Structure

```json
{
  "sub": "realm_id",
  "exp": 1234567890,
  "iat": 1234567890,
  "type": "access",
  "scopes": ["read:financials", "write:monitors"]
}
```

## Frontend Architecture

### Component Hierarchy

```
App (Router)
├── ConnectionSetup (OAuth initiation)
├── HealthDashboard (Overview + SLO tiles)
├── IncidentsList (Table view with filters)
├── IncidentDetail (Detail + timeline)
├── PostmortemView (Formatted report + PDF export)
├── MonitorsDashboard (Monitor config + status)
└── ComparisonSimulation (What-if scenarios)
```

### State Management

- **API State**: React Query (caching, refetching)
- **Auth State**: Context API (JWT + user info)
- **UI State**: Local component state (useState)

### API Client Pattern

```javascript
// All requests include:
// - Authorization: Bearer <jwt>
// - X-Request-ID: <uuid>
// - Content-Type: application/json
```

## Development Workflow

### Local Setup

```bash
# 1. Install dependencies
make install

# 2. Copy environment file
cp .env.example .env
# Edit .env with your Intuit credentials

# 3. Start services
make docker-up

# 4. Seed sandbox data (optional)
make seed

# 5. Run demo
make demo
```

### Testing Strategy

- **Unit Tests**: Pure business logic, no I/O
- **Integration Tests**: API endpoints + database
- **Golden Tests**: End-to-end scenarios with fixed datasets
- **Property Tests**: Hypothesis for invariant checking

### Code Quality

```bash
make format  # Black + Ruff autofix
make lint    # Ruff + Mypy
make test    # Pytest with coverage
make check   # Lint + Test
```

## Engine Components

### Anomaly Detection (`api/engine/detection/`)

- `statistical.py`: Z-score, IQR, MAD detectors
- `changepoint.py`: Ruptures integration (PELT, BottomUp)
- `ml_detector.py`: LightGBM supervised classifier
- `ensemble.py`: Weighted scoring + confidence

### Root Cause Analysis (`api/engine/rca/`)

- `graph_builder.py`: Entity relationship graph
- `temporal_correlation.py`: Cross-correlation analysis
- `causal_ranker.py`: PageRank-based ranking
- `explainer.py`: Natural language explanations

### Blast Radius (`api/engine/blast_radius/`)

- `mapper.py`: BFS/DFS graph traversal
- `impact_scorer.py`: Impact quantification
- `visualizer.py`: Cytoscape JSON generation

### Monitors (`api/engine/monitors/`)

- `slo_evaluator.py`: SLO compliance checking
- `alert_router.py`: Alert routing logic
- `health_scorer.py`: Composite health scoring

## Configuration Management

All configuration via environment variables (12-factor app):

```python
from api.config import get_settings

settings = get_settings()  # Cached singleton
print(settings.intuit_client_id)
print(settings.db_path)
```

## Logging Standards

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "ingestion_complete",
    realm_id=realm_id,
    entities_count=count,
    duration_ms=duration,
    layer="bronze"
)
```

## MLflow Integration

```python
import mlflow

with mlflow.start_run(run_name="anomaly_detection"):
    mlflow.log_params({"algorithm": "lightgbm", "threshold": 0.85})
    mlflow.log_metrics({"precision": 0.92, "recall": 0.88})
    mlflow.sklearn.log_model(model, "model")
```

## Deployment Considerations

- **Database**: DuckDB file in `/data` (persistent volume)
- **Redis**: Ephemeral cache + queue (can lose data)
- **Worker Scaling**: Horizontal (multiple celery workers)
- **API Scaling**: Horizontal (stateless, JWT-based)

## Security Best Practices

- **Secrets**: Never commit `.env` (use `.env.example` template)
- **JWT**: Short-lived tokens (24h default), refresh flow
- **QuickBooks Tokens**: Encrypted at rest in Redis
- **CORS**: Whitelist origins in production
- **Rate Limiting**: Applied per realm_id

## Performance Targets

- **API Latency**: p95 < 200ms (excluding async tasks)
- **Ingestion**: 1000 entities/sec (bronze layer)
- **Detection**: Real-time (< 5s for 10k records)
- **RCA**: < 30s for complex graphs (1000+ nodes)

## Future Enhancements

- [ ] Real-time streaming (Kafka/WebSockets)
- [ ] Multi-tenancy (organization-level isolation)
- [ ] Advanced ML (deep learning, transformers)
- [ ] Auto-remediation workflows
- [ ] Mobile app (React Native)
- [ ] Slack/Teams integrations

## Contributing

When adding new features:

1. Follow existing patterns (routers, services, models)
2. Add Pydantic schemas for all I/O
3. Write tests (unit + integration)
4. Update OpenAPI docs (FastAPI auto-generates)
5. Add structlog events for observability
6. Track experiments with MLflow

## References

- [FastAPI Best Practices](https://fastapi.tiangolo.com/async/)
- [Pydantic V2 Migration](https://docs.pydantic.dev/latest/migration/)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [Intuit OAuth2 Guide](https://developer.intuit.com/app/developer/qbo/docs/develop/authentication-and-authorization)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

**Last Updated**: 2026-02-10
**Maintained By**: Principal Engineering Team
