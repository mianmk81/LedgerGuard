# LedgerGuard — Task List

## Completed Tasks

- [x] **1. All models plugged in (no random/dummy data)** — Models wired: Churn (LightGBM), Delivery (XGBoost/stacked), Sentiment (TF-IDF+LR), Anomaly (IF+OCSVM+LOF+AE). All predictors have `model_used` flag and graceful fallbacks.
- [x] **2. Data accuracy & model traceability** — All prediction endpoints return `model_used`, `model_name`, `model_version`. Model load events logged via structlog.
- [x] **3. Wire ML models into Analysis & Simulation page** — `WhatIfSimulator` now runs Churn model + Health Score impact after propagation. Scenario response includes `ml_insights` and `models_used`. Narrative includes model-powered projections.
- [x] **4. Auto-run RCA on incidents** — `GET /incidents/{id}` auto-computes RCA + blast radius if missing. `GET /incidents/{id}/postmortem` auto-computes everything. No more "RCA not yet complete" errors.
- [x] **5. Plain-English explanations everywhere** — Health scores, domain scores, severity, confidence, z-scores, and metric cards all have plain-English explanations. Backend `HealthScorer` generates `explanation` field. Metrics router returns `explanation` per metric.
- [x] **6. One-click "Analyze Root Cause" button** — New `POST /incidents/{id}/analyze` endpoint runs RCA→blast radius→postmortem. Frontend "Analyze Root Cause" button with progress indicator, auto-refresh on completion.
- [x] **7. Use models and sensible defaults so metrics are never blank/zero** — Metrics router uses forward-fill from last non-zero value + industry baselines when no data. Each metric has a baseline (e.g., refund_rate: 0.03, review_score_avg: 4.1, order_volume: 85).
- [x] **Root Cause Analysis — click cause to show chain & graph** — `CausalPathCard` + `buildPathGraph` in `IncidentDetail.jsx`
- [x] **Model usage documentation** — `docs/MODELS_AND_DATA.md` created
- [x] **RCA UI on Incident Detail** — Expandable causal paths with mini Graph3D chains
- [x] **Keyboard shortcuts for graph** — Escape clears selection, Ctrl/Cmd+F opens node search
- [x] **Frontend loading states** — `LoadingFallback` / `LoadingState` with `role="status"`, `aria-live="polite"` across all pages

---

## Model Usage Matrix (Current)

| Feature | Model(s) | Location | Data Source | When |
|---------|----------|----------|-------------|------|
| **Ticket sentiment** | TF-IDF + Logistic Regression | `models/sentiment_industry/tfidf_lr_financial_sentiment.joblib` | Support tickets (adapter/canonical events) | On sentiment analysis request |
| **Delivery risk** | XGBoost | `models/delivery_industry/xgboost_dataco_delivery.joblib` | Order features (adapters/QBO) | POST `/prediction/delivery-risk` |
| **Churn risk** | LightGBM / RF / LogReg | `models/churn/*.pkl` | Customer features (RFM, etc.) | POST `/prediction/churn-risk` |
| **Anomaly (statistical)** | Z-score (MAD) | `api/engine/detection/statistical.py` | Gold daily metrics | During detection run |
| **Anomaly (ML)** | IF + OCSVM + LOF + Autoencoder | Trained in-memory on historical | Gold daily metrics (flat) | During detection (EnsembleDetector) |
| **Anomaly (changepoint)** | Ruptures PELT | `api/engine/detection/changepoint.py` | Gold daily metrics (time series) | During detection run |
| **Incident creation** | Ensemble fusion | `api/engine/detection/ensemble.py` | Outputs of stat + ML + changepoint | Full detection pipeline |
| **Health score** | Composite (rule-based) | `api/engine/monitors/health_scorer.py` | Gold metrics | Dashboard, Credit Pulse |
| **Future score** | TrendDetector + WarningService | `api/engine/prediction/` | Gold metrics | GET `/dashboard/future-score` |
| **RCA** | BRE-RCA (graph + correlation) | `api/engine/rca/` | Gold metrics + dependency graph | During analysis / incident detail |
| **Blast radius** | BFS traversal + ImpactScorer | `api/engine/blast_radius/` | Canonical events | Cascade/blast-radius API |
| **Postmortem** | Template-based NLG | `api/engine/postmortem_generator.py` | Incident + CausalChain + BlastRadius | GET postmortem |
| **Simulation** | Churn + HealthScorer (ML-enhanced) | `api/engine/simulation.py` | Simulated metrics → ML models | POST `/comparison/whatIf` |

---

## Remaining / Stretch Tasks

### 8. **Model availability dashboard**
- [ ] Add a `/api/v1/system/models` endpoint listing loaded models and status
- [ ] Show model status (loaded / missing / failed) in System or Admin UI

### 9. **Demo data consistency**
- [ ] `scripts/demo_run.py` and `scripts/seed_sandbox.py` use consistent seeds
- [ ] Demo incidents align with seeded metrics (no impossible z-scores)
- [ ] Causal chains for demo incidents are coherent with dependency graph

### 10. **End-to-end model integration test**
- [ ] Script or test that: seeds data → runs analysis → runs predictions → checks no "model not found" in responses
- [ ] Document minimal steps to train all models and run full pipeline
