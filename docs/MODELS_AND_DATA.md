# LedgerGuard — Models and Data Usage

This document maps each feature to the model(s) it uses, the data sources, and how to train or update them.

---

## 1. Ticket Sentiment

| Item | Detail |
|------|--------|
| **Model** | TF-IDF + Logistic Regression (FinancialPhraseBank) |
| **File** | `models/sentiment_industry/tfidf_lr_financial_sentiment.joblib` |
| **Code** | `api/engine/sentiment/ticket_sentiment.py` → `TicketSentimentAnalyzer` |
| **Data** | Support ticket text from `support_tickets_adapter` or canonical events |
| **When** | When analyzing support ticket sentiment for escalation risk |
| **Fallback** | Lexicon-based if model not found |
| **Train** | `python scripts/train_industry_sentiment.py` |

---

## 2. Delivery Risk

| Item | Detail |
|------|--------|
| **Model** | XGBoost (DataCo-trained) |
| **File** | `models/delivery_industry/xgboost_dataco_delivery.joblib` |
| **Code** | `api/engine/prediction/delivery_predictor.py` → `predict_delivery_risk` |
| **Data** | Order features (distance, weight, carrier, etc.) from adapters or request body |
| **When** | `POST /api/v1/prediction/delivery-risk` |
| **Fallback** | Returns `model_used: false` if model not found |
| **Train** | `python scripts/train_industry_delivery.py --model xgboost` |

---

## 3. Churn Risk

| Item | Detail |
|------|--------|
| **Model** | LightGBM (preferred) or Random Forest or Logistic Regression |
| **File** | `models/churn/lightgbm_churn_model.pkl` (or `random_forest_*`, `logistic_regression_*`) |
| **Code** | `api/engine/prediction/churn_predictor.py` → `predict_churn_risk` |
| **Data** | Customer RFM, review score, delivery experience (Olist schema) |
| **When** | `POST /api/v1/prediction/churn-risk` |
| **Fallback** | Returns `model_used: false` if no model found |
| **Train** | `python scripts/train_churn_model.py --model lgbm` |

---

## 4. Anomaly Detection (Statistical)

| Item | Detail |
|------|--------|
| **Model** | MAD-based Z-score (no ML model file) |
| **File** | N/A — pure computation |
| **Code** | `api/engine/detection/statistical.py` → `StatisticalDetector` |
| **Data** | Gold daily metrics from `state_builder` |
| **When** | Every detection run as Layer 1 |
| **Train** | N/A |

---

## 5. Anomaly Detection (ML — 4-model ensemble)

| Item | Detail |
|------|--------|
| **Model** | Isolation Forest + One-Class SVM + LOF + Autoencoder (strict 4/4 voting) |
| **File** | Trained in-memory; `models/anomaly/autoencoder_threshold.txt` for AE threshold |
| **Code** | `api/engine/detection/ml_detector.py` → `MLDetector` |
| **Data** | Flat Gold metrics (current snapshot) — trained on historical flat metrics |
| **When** | During detection run as Layer 2 (EnsembleDetector) |
| **Train** | `MLDetector.train(historical_flat_metrics)` at runtime; or `scripts/train_anomaly_strict.py` for offline |

---

## 6. Anomaly Detection (Changepoint)

| Item | Detail |
|------|--------|
| **Model** | Ruptures PELT algorithm |
| **File** | N/A — library (ruptures) |
| **Code** | `api/engine/detection/changepoint.py` → `ChangepointDetector` |
| **Data** | Gold daily metrics as time series |
| **When** | During detection run as Layer 3 |
| **Train** | N/A |

---

## 7. Incident Creation (Ensemble)

| Item | Detail |
|------|--------|
| **Model** | Fusion rules (statistical + ML + changepoint outputs) |
| **File** | N/A — rule-based |
| **Code** | `api/engine/detection/ensemble.py` → `EnsembleDetector` |
| **Data** | Outputs of StatisticalDetector, MLDetector, ChangepointDetector + canonical events |
| **When** | Full detection pipeline |
| **Incident Types** | CASH_CRISIS, MARGIN_EROSION, FULFILLMENT_SLA_DEGRADATION, CHURN_ACCELERATION, etc. |

---

## 8. Health Score

| Item | Detail |
|------|--------|
| **Model** | Composite rule-based scoring |
| **File** | N/A |
| **Code** | `api/engine/monitors/health_scorer.py` → `HealthScorer` |
| **Data** | Gold daily metrics (domain scores, weak/strong metrics) |
| **When** | Dashboard, Credit Pulse, reports |

---

## 9. Future Score

| Item | Detail |
|------|--------|
| **Model** | `TrendDetector` + `WarningService` (LightGBM forecaster when trained, else linear regression) |
| **File** | `models/trend/forecaster_*.joblib` (optional, per-metric) |
| **Code** | `api/engine/prediction/future_score.py` → `FutureScorePredictor` |
| **Data** | Gold metrics over lookback window |
| **When** | `GET /api/v1/dashboard/future-score` |
| **Train** | `python scripts/train_trend_forecaster.py` (run seed first) |

---

## 10. Root Cause Analysis (RCA)

| Item | Detail |
|------|--------|
| **Model** | BRE-RCA: dependency graph + temporal correlation + causal ranker |
| **File** | `api/engine/rca/dependency_graph.py` (static DAG), no learned weights |
| **Code** | `api/engine/rca/analyzer.py` → RCA pipeline |
| **Data** | Gold metrics + dependency graph + canonical events |
| **When** | During analysis; incident detail / postmortem |
| **Output** | `CausalChain` with ranked `CausalPath` and `CausalNode` |

---

## 11. Blast Radius

| Item | Detail |
|------|--------|
| **Model** | BFS traversal + ImpactScorer (rule-based) |
| **File** | N/A |
| **Code** | `api/engine/blast_radius/mapper.py`, `impact_scorer.py` |
| **Data** | Canonical events, incident evidence |
| **When** | `GET /api/v1/cascades/{incident_id}` |

---

## 12. Postmortem

| Item | Detail |
|------|--------|
| **Model** | Template-based NLG (no ML) |
| **File** | N/A |
| **Code** | `api/engine/postmortem_generator.py` → `PostmortemGenerator` |
| **Data** | Incident + CausalChain + BlastRadius |
| **When** | `GET /api/v1/incidents/{id}/postmortem` |

---

## Data Flow

```
Bronze (QBO raw, adapters)
    → Event Builder (Silver: canonical events)
    → State Builder (Gold: daily metrics)
    → Detection (Statistical + ML + Changepoint → Ensemble)
    → Incidents
    → RCA, Blast Radius, Postmortem

Predictions (separate flows):
  - Support tickets → Sentiment analyzer
  - Orders → Delivery predictor
  - Customers → Churn predictor
```

---

## Training Order (if starting from scratch)

1. `scripts/download_financial_phrasebank.py` (for sentiment)
2. `scripts/download_datasets.py` (Olist, DataCo, etc.)
3. `scripts/train_industry_sentiment.py`
4. `scripts/train_industry_delivery.py --model xgboost`
5. `scripts/train_churn_model.py --model lgbm`
6. `scripts/train_anomaly_strict.py` (optional; ML detector trains at runtime)
7. `scripts/seed_sandbox.py` or `scripts/demo_run.py` for demo data
8. `scripts/train_trend_forecaster.py` (optional; improves Future Score projection — run after seed)
