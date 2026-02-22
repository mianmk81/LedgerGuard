# Industry-Level Performance — Targets & Roadmap

## Current Best vs Industry Benchmarks

| Task | Our Best | Industry | Gap |
|------|----------|----------|-----|
| **Churn** | AUC 0.8509 | 0.82-0.85 | ✅ **HIT** |
| **Delivery** | F1 0.7537 | 0.78+ | ⚠️ Close |
| **Anomaly** | F1 0.36 (XGBoost) / 0.20 (4-ens) | 0.55-0.69 | ❌ Below |
| **Sentiment** | Macro F1 0.81 | F1 0.83+ | ⚠️ Close |

## Reaching Industry Level

### Anomaly (NAB) — Target F1 0.55-0.69

**Current options:**

1. **Supervised XGBoost** (F1 0.36) — `scripts/train_anomaly_supervised.py --model xgboost`
   - Uses rich features, class weights, threshold tuning
   - Best per-series: machine_temperature F1 0.90, exchange-4_cpm F1 1.0

2. **4-Model Ensemble** (F1 0.20) — Production engine default
   - IF + OCSVM + LOF + Autoencoder, strict 4/4 voting
   - `api/engine/detection/ml_detector.py`

3. **Forecast-then-detect** (MLP F1 0.18, LSTM F1 0.69 in literature)
   - `scripts/train_anomaly_lstm.py` — MLP works; LSTM needs `numpy<2` + `torch`

**To reach F1 0.55+:**
- Use **LSTM forecasting** (Karami et al. 2025): `pip install "numpy<2" torch` then `python scripts/train_anomaly_lstm.py --backend lstm`
- Or **Google Cloud**: Vertex AI Timeseries Insights API, BigQuery `ML.DETECT_ANOMALIES`
- **Pooled model**: Train one XGBoost on all series (add series_id feature)

### Delivery — Target F1 0.78+

- Current F1 0.75 (DataCo). Tune: more trees, learning rate, feature selection.
- `scripts/train_industry_delivery.py`

### Sentiment — Target Macro F1 0.83+ ⚠️ Close

- **Fixed**: Download without `datasets`: `python scripts/download_financial_phrasebank.py`
- **Result**: TF-IDF+LR Macro F1 0.81 (target 0.83). Tune ngram_range, C, max_features to push higher.
- `scripts/train_industry_sentiment.py`

## Google Cloud Resources

- **Vertex AI Timeseries Insights API**: Real-time forecasting + anomaly detection
- **BigQuery ML**: `ARIMA_PLUS_XREG` + `ML.DETECT_ANOMALIES` for multivariate
- **TimesFM**: Google's time series foundation model on Vertex AI

See: https://cloud.google.com/vertex-ai/docs/timeseries-insights

## Data Wiring (Churn / Delivery)

- **ChurnClassifier** (`api/engine/churn_classifier.py`): Call `train(df, target_col="Churn")` with your DataFrame (e.g. from Olist or IBM Telco). Use `load_model(path)` to load industry-trained model from `models/churn_industry/`.
- **Delivery**: Scripts train on DataCo. No engine integration yet — delivery metrics come from Olist adapter → StateBuilder → `delivery_delay_rate`. To use a delivery classifier, wire `scripts/train_industry_delivery.py` output into a new service.
- **Anomaly**: Production MLDetector trains on LedgerGuard Gold metrics at runtime (your data).

## Quick Commands

```bash
# Anomaly — supervised (best current)
python scripts/train_anomaly_supervised.py --model xgboost

# Anomaly — forecast-based (MLP, no torch)
python scripts/train_anomaly_lstm.py --backend mlp

# Churn, delivery, all
python scripts/train_industry_best.py --task all
```
