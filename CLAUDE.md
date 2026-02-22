# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This

LedgerGuard is a Business Reliability Engine — SRE principles applied to financial operations. It ingests business data (QuickBooks Online or seeded demo data), detects anomalies via a 3-layer ensemble, performs graph-based root cause analysis, maps blast radius, and generates incident postmortems. Built for Hacklytics 2025.

## Commands

```bash
# Backend
uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev

# Both (parallel via Makefile)
make dev

# Seed demo data (run in order, no QB needed)
python scripts/seed_sandbox.py --mode local
python scripts/seed_demo_incidents.py

# Tests
pytest tests/                                      # All tests with coverage
pytest tests/unit/ -v                              # Unit only
pytest tests/integration/ -v                       # Integration only
pytest tests/golden/ -v                            # Golden path only
pytest tests/unit/test_engine.py::test_name -v     # Single test

# Lint & format
ruff check api/ tests/                 # Lint
black api/ tests/ scripts/             # Format
mypy api/                              # Type check
make check                             # Lint + test (CI equivalent)

# ML training — industry models (preferred for demo)
python scripts/train_churn_model.py --model lgbm          # Telco LightGBM → models/churn_industry/
python scripts/train_late_delivery.py --model xgboost     # DataCo XGBoost → models/delivery_industry/
python scripts/train_sentiment.py --model linear_svc      # LinearSVC → models/sentiment_industry/
python scripts/train_trend_forecaster.py                  # LightGBM ×8 → models/trend/

# ML training — research/Olist models
python scripts/train_models.py --model all                # Olist anomaly + churn + delivery
python scripts/train_anomaly_detector.py --model autoencoder

# Generate explainability artifacts (run after training)
python scripts/generate_explainability.py --model all     # SHAP PNGs → reports/
python scripts/generate_explainability.py --model churn_telco
python scripts/generate_explainability.py --model delivery_dataco
python scripts/generate_explainability.py --model sentiment
python scripts/generate_explainability.py --model trend

# Generate model cards with CV metrics (run after training)
python scripts/generate_model_cards.py --model all        # JSON cards → reports/
python scripts/generate_model_cards.py --model churn_telco
python scripts/generate_model_cards.py --model delivery_dataco
python scripts/generate_model_cards.py --model sentiment_industry
python scripts/generate_model_cards.py --model trend
```

## Architecture

### Data Flow (Medallion Architecture)

```
QuickBooks/Olist → Bronze (raw entities) → Silver (canonical events) → Gold (daily metrics)
                                                                        ↓
                                                        Detection Engine (3-layer ensemble)
                                                                        ↓
                                                        Incidents → RCA → Blast Radius → Postmortems
```

`seed_sandbox.py` seeds Bronze, runs `CanonicalEventBuilder` (Bronze→Silver), then `StateBuilder` (Silver→Gold). `seed_demo_incidents.py` inserts 5 hardcoded incidents with causal chains.

### Backend (FastAPI + DuckDB)

- **Routers (`api/routers/`):** Thin HTTP layer. 16 modules, all prefixed `/api/v1/`. Auth via `get_current_realm_id` dependency (JWT). Response envelope: `{success, data, error, metadata}`.
- **Engine (`api/engine/`):** All business logic lives here, not in routers.
  - `detection/` — `statistical.py` (MAD Z-Score), `ml_detector.py` (4-model ensemble: IF + OCSVM + LOF + Autoencoder, 3/4 majority voting, trains at runtime on historical Gold metrics), `changepoint.py` (PELT via ruptures), `ensemble.py` (fusion + severity classification)
  - `rca/` — `graph_builder.py` (NetworkX dependency graph from `BUSINESS_DEPENDENCY_EDGES`), `causal_ranker.py` (PageRank + weighted scoring with bootstrap CIs)
  - `blast_radius/` — BFS/DFS traversal, outputs Cytoscape JSON format consumed by Graph3D
  - `monitors/` — `health_scorer.py` computes composite A-F grade across 27 metrics in 3 domains (financial, operational, customer) with configurable weights
  - `prediction/` — `churn_predictor.py` (lazy-loads from `models/churn_industry/` then `models/churn/`), `delivery_predictor.py` (priority: stacked ensemble → two-stage → DataCo XGBoost → Olist XGBoost), `cash_runway.py`, `future_score.py`
  - `sentiment/` — `ticket_sentiment.py` (lazy-loads LinearSVC from `models/sentiment_industry/`, falls back to keyword lexicon)
  - `state_builder.py` — Computes all 27 Gold metrics from Silver events for a date range
- **Storage (`api/storage/`):** `StorageBackend` abstract class in `base.py`, implemented by `duckdb_storage.py`. Thread-safe connection pooling. Schema auto-created. `DB_PATH=:memory:` for tests.
- **Config (`api/config.py`):** pydantic-settings, all from env vars. Access via `get_settings()` (cached).
- **Auth (`api/auth/`):** JWT-based. Demo mode: `POST /api/v1/auth/demo-token` returns a signed JWT with `sub="demo"`. The frontend "Try Demo" button calls this automatically.

### System Endpoints (model observability)

| Endpoint | Description |
|---|---|
| `GET /api/v1/system/models` | Availability status of all model artifacts (available/missing/loaded/error) |
| `GET /api/v1/system/model-cards` | Returns all `model_card_*.json` files + image inventory from `reports/` |
| `GET /api/v1/system/experiments` | MLflow run counts + best metrics for churn/anomaly/delivery experiments |
| `GET /api/v1/system/reports/{filename}` | Sanitized FileResponse serving `reports/*.png` and `reports/*.json` |

### Frontend (React 18 + Vite + Tailwind)

- **Graph3D (`components/graph/Graph3D.jsx`):** Shared 3D force-directed graph used on Dashboard, Credit Pulse, and Incident Detail. Converts Cytoscape format via `cytoscapeToForceGraph()`. Features: node search (Ctrl+F), connection side panel, keyboard shortcuts (Escape clears), loading/empty states.
- **IncidentDetail (`pages/IncidentDetail.jsx`):** Dual-layer graph (entity impact + causal chain). RCA section has clickable `CausalPathCard` components that expand to show mini Graph3D chains with breadcrumb trail. `buildPathGraph()` converts API causal paths to Cytoscape format.
- **HealthDashboard (`pages/HealthDashboard.jsx`):** Hero with reliability score, future score with WHY/WHEN drivers, 3D causal graph, incidents list.
- **ModelPerformance (`pages/ModelPerformance.jsx`):** 4 production model tabs (Churn/Telco | Delivery/DataCo | Sentiment | Trend). Each tab shows: SHAP plots (click-to-zoom lightbox via `ZoomableImage`/`ImageLightbox`), CV confidence intervals, confusion matrix, ROC curve. Hard-coded `PRODUCTION_MODELS` constant provides fallback metrics when API unavailable. Collapsible `PretrainedShowcase` section displays the 4-model anomaly ensemble voting diagram plus 9 research models. MLflow experiment stats in sidebar panel.
- **CreditPulse (`pages/CreditPulse.jsx`):** Financial health score, domain breakdown, contributing factors, 3D causal graph.
- **API client (`api/client.js`):** Axios with JWT interceptor. `system.reportImage(filename)` returns a direct URL string (not a Promise) for use in `<img src>`. Auth token stored in `localStorage` as `ledgerguard_token`.

### ML Models

Two tiers of models coexist:

**Production / Industry models** (used in live demo, trained on public industry datasets):

| Directory | Model | Dataset | Key Metric |
|---|---|---|---|
| `models/churn_industry/` | LightGBM | Telco (IBM) — 7,043 customers | AUC 0.851, F1 0.650 (CV) |
| `models/delivery_industry/` | XGBoost | DataCo Supply Chain — 180k orders | F1 0.779, AUC 0.845 (CV 0.856) |
| `models/sentiment_industry/` | LinearSVC (calibrated) | FinancialPhraseBank — 4.8k sentences | Macro F1 0.884, Acc 0.918 |
| `models/trend/` | LightGBM ×8 | Synthetic Gold metrics | refund_rate MAE 0.113 |

**Research / Pre-trained models** (Olist e-commerce dataset, shown in PretrainedShowcase):

| Directory | Models |
|---|---|
| `models/anomaly/` | IF + OCSVM + LOF + Autoencoder ensemble (pre-trained, NAB benchmark) |
| `models/anomaly_industry/` | isolation_forest_nab.joblib |
| `models/churn/` | LightGBM + RF + LR (Olist) |
| `models/delivery/` | XGBoost + RF + LR + stacked ensemble + two-stage (Olist) |
| `models/sentiment/` | TF-IDF + LR (Olist reviews) |

Prediction endpoints lazy-load artifacts and return `model_used: true/false`. Missing models return a fallback response with the training command to run.

### SHAP / Explainability

`scripts/generate_explainability.py` produces PNGs in `reports/`:

- `shap_summary_*.png`, `shap_bar_*.png`, `shap_waterfall_*.png` for each model
- LinearSVC uses coefficient importance (not SHAP TreeExplainer) — saved under SHAP filenames for UI consistency
- Trend forecasters use 200 synthetic samples (`np.random.default_rng(42)`) since they operate on Gold layer DuckDB data
- **XGBoost 2.x compat**: `shap_xgb_compat()` context manager monkey-patches `shap.explainers._tree.decode_ubjson_buffer` to handle the `[5E-1]` base_score format that breaks SHAP 0.49

`scripts/generate_model_cards.py` produces `model_card_*.json` and statistical artifact PNGs:

- `cv_boxplot_*.png`, `confusion_matrix_*.png`, `roc_curve_*.png`
- 5-fold stratified CV with 95% CI: `ci95_half_width = 1.96 * std / sqrt(n_folds)`
- `_prepare_telco_features()` and `_prepare_dataco_features()` helpers replicate training pipeline inline (needed because model cards run independently of training scripts)

## Key Patterns

- **Logging:** structlog everywhere. `logger.info("event_name", key=value)` style.
- **Testing:** `tests/conftest.py` sets `TESTING=true` and `DB_PATH=:memory:`. Has factory functions (`make_incident`, `make_causal_chain`, etc.) and `MockStorage` class. Integration tests override auth via `app.dependency_overrides[get_current_realm_id]`.
- **Seed data:** `seed_sandbox.py` uses `random.seed(42)` for reproducibility. Generates 6 months of synthetic QB data with 5 injected incident patterns (refund spike, supplier delay cascade, order surge, product quality, AR aging).

## Code Style

- Python: Black (line-length 100), Ruff (`E,F,I,N,W,UP` rules, ignores E501), target Python 3.11
- Frontend: Tailwind utility classes, functional components with hooks
- All tool configs in `pyproject.toml`

## Data Leakage Guards

These were audited and fixed — do not regress:
1. Late delivery prediction does NOT use `review_score` (reviews written after delivery)
2. Anomaly detection aggregates delivery metrics by delivery date, not purchase date
3. `avg_review_score` NaN filled with `.ffill().fillna(0)`, not global mean
4. Churn pipeline filters reviews by `review_creation_date <= observation_end`
5. All time-series splits are temporal (no future data in training)
