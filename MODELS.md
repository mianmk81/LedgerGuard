# LedgerGuard — Complete Model Documentation

## Total Models: 20

- **7 Engine Models** (built into the runtime — 4 statistical/rule-based, 3 ML)
- **13 Custom-Trainable Models** (training scripts + Olist dataset)

---

## Top 5 Models — Judge Quick Reference

The five models that best showcase LedgerGuard's data science depth:

### 1. Autoencoder Anomaly Detector
> A neural network learns to compress 27 daily business metrics down to just 8 latent dimensions and reconstruct them. When reconstruction error spikes past the 95th percentile, the model has found a day it can't explain — that's your anomaly. No labels needed.

### 2. 3-Layer Ensemble Detector
> Three independent detection methods — robust statistics (MAD z-score), tree-based isolation (Isolation Forest), and structural break detection (PELT changepoint) — vote on every metric every day. When all three agree, confidence is VERY_HIGH. This catches anomalies that any single method would miss.

### 3. LightGBM Churn Classifier with Optuna Tuning + Platt Calibration
> A gradient-boosted tree ensemble predicts customer churn from RFM features, review behavior, and delivery experience. Optuna searches 50 hyperparameter combinations to maximize AUC-ROC, then Platt scaling calibrates raw scores into true probabilities — so "0.73 churn risk" actually means 73% of similar customers churned.

### 4. Causal Ranker with Bootstrap Confidence Intervals
> When an incident fires, this ranker scores every candidate root cause across four dimensions — anomaly magnitude, temporal precedence, graph proximity, and data quality — then runs 200 bootstrap iterations with noise injection to produce confidence intervals on each score. If the top cause is robust across 85%+ of perturbations, it's flagged as high-confidence.

### 5. XGBoost Late Delivery Predictor with Haversine Feature Engineering
> Predicts which orders will arrive late using 25 engineered features — including real geographic distance (haversine formula on customer/seller coordinates), seller performance history (late rate, avg delivery days), category risk scores, temporal signals (day-of-week, month, weekend, month-end), and product physics (weight, dimensions, volume). Auto-calibrated class weighting handles the natural imbalance between on-time and late deliveries.

---

## Part 1: Engine Models (Runtime)

These models live in `api/engine/` and run when the app processes requests.

---

### Model 1: Statistical Detector (MAD Z-Score)

**File:** `api/engine/detection/statistical.py`
**Type:** Statistical Anomaly Detection (Unsupervised)
**Library:** NumPy (no ML library needed)
**Training:** None — pure math on historical data

#### Algorithm
Uses **Median Absolute Deviation (MAD)** instead of standard deviation for robust outlier detection. MAD is resistant to the influence of outliers in the baseline data.

**Formula:**
```
MAD = median(|x_i - median(x)|)
z_score = 0.6745 * (current_value - median) / MAD
is_anomaly = |z_score| > threshold
```

The constant `0.6745` normalizes MAD to be consistent with standard deviation for normally distributed data.

#### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `baseline_days` | 30 | Trailing days for baseline computation |
| `zscore_threshold` | 3.0 | Absolute z-score threshold for flagging anomalies |
| `MAD_NORMALIZATION` | 0.6745 | Normalization constant |

#### Input
- `metric_name`: string identifier (e.g., "refund_rate", "daily_revenue")
- `current_value`: float — today's metric value
- `baseline_values`: list[float] — last 30 days of metric values

#### Output
```python
{
    "metric_name": str,
    "current_value": float,
    "is_anomaly": bool,
    "zscore": float,          # How many standard deviations from median
    "median": float,          # Baseline median
    "mad": float,             # Median Absolute Deviation
    "threshold": float,       # 3.0 by default
    "baseline_count": int     # Number of valid baseline values used
}
```

#### Edge Cases
- If MAD = 0 and current != median: z-score set to +/-10.0 (definite anomaly)
- If MAD = 0 and current == median: z-score = 0 (normal)
- NaN values removed before computation
- Returns zero baseline count if no valid data

#### Performance
- **Latency:** <1ms (pure NumPy math)
- **Memory:** Negligible

---

### Model 2: ML Detector (Isolation Forest)

**File:** `api/engine/detection/ml_detector.py`
**Type:** Unsupervised Anomaly Detection (Tree-based)
**Library:** scikit-learn (`IsolationForest`)
**Training:** Trains at runtime on historical metrics from DuckDB

#### Algorithm
Isolation Forest works by randomly selecting features and split values to isolate observations. Anomalies require **fewer random splits** to isolate (shorter path length in the tree), because they are "few and different." Normal points require many splits.

**Key insight:** Anomalies are easier to separate from the rest of the data.

#### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `contamination` | 0.05 | Expected proportion of anomalies (5%) |
| `n_estimators` | 100 | Number of isolation trees in the forest |
| `random_state` | 42 | Reproducibility seed |
| `n_jobs` | -1 | Use all CPU cores |
| `min_samples` | 30 | Minimum training samples required |

#### Input
- **Training:** list[dict] — daily metric dictionaries (one dict per day)
- **Detection:** dict — current day's metric values

#### Output
```python
{
    "is_anomaly": bool,
    "anomaly_score": float,    # sklearn anomaly score (negative = more anomalous)
    "prediction": int,         # -1 = anomaly, 1 = normal
    "confidence": float,       # |anomaly_score|
    "features_used": list[str] # Feature names used
}
```

#### Training Process
1. Extracts numeric metrics from domain (financial, operational, customer)
2. Builds feature matrix: rows = days, columns = metrics
3. Missing values filled with 0.0
4. Features sorted alphabetically for consistency
5. Fits IsolationForest
6. Model persisted in memory for subsequent detections

#### Feature Engineering
- All numeric metrics extracted from daily Gold layer data
- Automatic domain detection (financial, operational, customer)
- No manual feature selection needed

#### Performance
- **Training time:** 1-3 seconds (on 30-365 days of data)
- **Inference time:** <10ms per prediction
- **Memory:** ~5-20MB depending on n_estimators

---

### Model 3: Changepoint Detector (PELT)

**File:** `api/engine/detection/changepoint.py`
**Type:** Changepoint Detection (Structural Break Detection)
**Library:** ruptures
**Training:** None — optimization-based algorithm

#### Algorithm
**PELT (Pruned Exact Linear Time)** detects points in a time series where the statistical properties (mean, variance) change abruptly. It uses dynamic programming with pruning to find the optimal segmentation that minimizes a cost function plus a penalty for each changepoint.

**Cost function:** L2 (mean shift) — measures squared deviation from segment mean
**Penalty:** BIC = log(n) — balances fit quality vs. number of changepoints

#### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `penalty` | "bic" | Penalty type: "bic" = log(n), "aic" = 2.0, or numeric |
| `min_size` | 5 | Minimum segment size between changepoints |
| `model` | "l2" | Cost function: "l2" = mean shift, "rbf" = general |

#### Input
- `metric_values`: list[float] — metric values in chronological order
- `dates`: Optional[list[date]] — corresponding dates

#### Output
```python
{
    "has_changepoint": bool,
    "changepoint_indices": list[int],
    "changepoint_dates": list[str],
    "segment_count": int,
    "segment_means": list[float],
    "cost": float
}
```

#### Regime Change Analysis
When a changepoint is detected, it also returns:
```python
{
    "metric_name": str,
    "has_change": bool,
    "change_date": str,
    "before_mean": float,      # Mean before changepoint
    "after_mean": float,       # Mean after changepoint
    "mean_shift": float,       # after - before
    "percent_change": float,   # (shift / before) * 100
    "direction": str           # "increase", "decrease", or "stable"
}
```

#### Edge Cases
- Minimum 2 * min_size = 10 data points required
- NaN values replaced with mean of non-NaN values
- Direction = "stable" if |mean_shift| < 0.01

#### Performance
- **Latency:** 10-50ms for typical time series (30-365 points)
- **Memory:** Negligible

---

### Model 4: Ensemble Detector

**File:** `api/engine/detection/ensemble.py`
**Type:** Multi-layer ensemble combining Models 1, 2, and 3
**Library:** Custom (orchestrates the 3 detectors above)
**Training:** None (orchestration layer)

#### Architecture
Runs all 3 detection layers and fuses their results:

```
Input Metrics
    ├── Layer 1: Statistical (MAD Z-Score) ──────────────────┐
    ├── Layer 2: ML (Isolation Forest) ──────────────────────┤── Fusion → Incident
    └── Layer 3: Changepoint (PELT) ─────────────────────────┘
```

#### Confidence Fusion Rules
| Layers Triggered | Confidence Level |
|-----------------|-----------------|
| Layer 1 only | MEDIUM |
| Layer 1 + Layer 2 | HIGH |
| Layer 1 + Layer 2 + Layer 3 | VERY_HIGH |
| Only Layer 2 or 3 (no Layer 1) | LOW |

#### Incident Types Detected (8)
| Incident Type | Primary Metric | Description |
|--------------|---------------|-------------|
| REFUND_SPIKE | refund_rate | Abnormal refund rate increase |
| FULFILLMENT_SLA_DEGRADATION | delivery_delay_rate | Delivery delays exceeding SLA |
| SUPPORT_LOAD_SURGE | ticket_backlog | Customer support overload |
| CHURN_ACCELERATION | churn_proxy | Customer loss rate increase |
| MARGIN_COMPRESSION | margin_proxy | Profit margin decrease |
| LIQUIDITY_CRUNCH_RISK | net_cash_proxy | Cash flow deterioration |
| SUPPLIER_DEPENDENCY_FAILURE | supplier_delay_rate | Supply chain disruption |
| CUSTOMER_SATISFACTION_REGRESSION | review_score_avg | Review score decline |

#### Severity Classification
**Refund Spike severity:**
| Severity | Condition |
|----------|-----------|
| CRITICAL | z > 8.0 AND refund_rate > 10% of revenue |
| HIGH | z > 6.0 |
| MEDIUM | z > 4.0 |
| LOW | 3.0 < z <= 4.0 |

**Other incidents:** HIGH (z > 6.0), MEDIUM (4.0 < z <= 6.0), LOW (z <= 4.0)

#### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `enable_ml` | True | Enable Isolation Forest layer |
| `enable_changepoint` | True | Enable PELT layer |
| `baseline_days` | 30 | Baseline window for statistical detection |
| `CRITICAL_REVENUE_THRESHOLD_PCT` | 0.10 | 10% of daily revenue for critical refund threshold |

---

### Model 5: Churn Classifier (LightGBM)

**File:** `api/engine/churn_classifier.py`
**Type:** Supervised Binary Classification
**Library:** LightGBM, Optuna, scikit-learn
**Training:** Trains at runtime or from saved model file

#### Algorithm
**LightGBM (Light Gradient Boosting Machine)** builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous trees. LightGBM uses **leaf-wise growth** (grows the leaf with the highest loss reduction) instead of level-wise growth, making it faster and more accurate.

**Calibration:** After LightGBM training, applies **Platt Scaling** (sigmoid calibration) to convert raw scores into well-calibrated probabilities.

#### Default Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `objective` | "binary" | Binary classification |
| `metric` | "auc" | Optimize for AUC-ROC |
| `boosting_type` | "gbdt" | Gradient Boosted Decision Trees |
| `num_leaves` | 31 | Max leaves per tree |
| `learning_rate` | 0.05 | Step size shrinkage |
| `n_estimators` | 200 | Number of boosting rounds |
| `random_state` | 42 | Reproducibility |

#### Optuna Hyperparameter Tuning (when tune=True)
| Parameter | Search Range |
|-----------|-------------|
| `num_leaves` | [20, 200] |
| `learning_rate` | [0.01, 0.3] (log scale) |
| `n_estimators` | [100, 1000] |
| `min_child_samples` | [5, 50] |
| `subsample` | [0.5, 1.0] |
| `colsample_bytree` | [0.5, 1.0] |
| `reg_alpha` | [0.0, 5.0] |
| `reg_lambda` | [0.0, 5.0] |
- **Trials:** 50 (default)
- **Objective:** Maximize AUC-ROC on validation set

#### Feature Engineering (Built-in)
| Feature | Description |
|---------|-------------|
| `tenure_group` | Bins: 0-12, 13-24, 25-48, 49+ months |
| `charges_per_month_tenure` | MonthlyCharges / (tenure + 1) |
| `has_premium_services` | Count of 6 premium services used |
- All categorical columns: LabelEncoded

#### Data Split
- Sort by tenure (time proxy)
- Train: 70% | Validation: 15% | Test: 15%

#### Output
- Churn probability (0.0 to 1.0, calibrated)
- Feature importances (LightGBM gain-based)

#### Model Persistence
Saves 4 files:
- `lightgbm_model.txt` — LightGBM booster
- `calibrator.pkl` — Platt scaling calibrator
- `label_encoders.json` — Categorical encoders
- `feature_names.json` — Feature list

#### Baseline Comparison
Trains a LogisticRegression baseline (max_iter=1000) for comparison.

#### MLflow Tracking
Logs: dataset hash, split sizes, all hyperparams, test metrics (AUROC, F1, precision, recall, accuracy), confusion matrix, feature importance, model artifacts.

---

### Model 6: Invoice Default Risk Scorer

**File:** `api/engine/invoice_default_risk.py`
**Type:** Rule-Based Risk Scoring (No ML)
**Library:** None (pure Python)
**Training:** None

#### Algorithm
Computes a composite risk score (0-100) from three components:

```
risk_score = min(100, aging_score + amount_score + history_score)
```

#### Scoring Components
| Component | Formula | Max Score |
|-----------|---------|-----------|
| Aging Score | min(90, aging_days / 30 * 25) | 90 |
| Amount Score | min(30, log10(max(1, balance / 100)) * 10) | 30 |
| History Score | min(25, (dso_avg - 30) / 30 * 15 + late_count * 5) | 25 |

#### Risk Buckets
| Bucket | Score Range |
|--------|------------|
| CRITICAL | >= 75 |
| HIGH | 50 - 74 |
| MEDIUM | 25 - 49 |
| LOW | < 25 |

#### Input (from Gold Layer Events)
- Invoice aging (days overdue)
- Invoice balance remaining
- Customer DSO (Days Sales Outstanding) average
- Customer late payment count

#### Risk Factors Flagged
- Aging > 30 days
- Balance > $5,000
- Customer DSO > 45 days
- Prior late payments > 0

#### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback_days` | 90 | Historical window for event collection |
| `top_n` | 10 | Return top N risky receivables |

---

### Model 7: Ticket Sentiment Analyzer

**File:** `api/engine/sentiment/ticket_sentiment.py`
**Type:** Lexicon-Based NLP (No ML)
**Library:** `re` (regex)
**Training:** None

#### Algorithm
Keyword-matching sentiment scorer. Scans support ticket text for frustration and positive keywords, computes a weighted score.

**Scoring:**
```
For each frustration keyword found:  score += 8.0
For each positive keyword found:     score -= 3.0
If text length < 20 chars:           score *= 0.7
Final: max(0, min(100, score))
```

#### Frustration Lexicon (29 terms)
`escalat, frustrat, angry, angrier, disappoint, unacceptable, refund, cancel, terrible, worst, horrible, never again, fed up, sick of, complaint, complain, unhappy, furious, outraged, ridiculous, absurd, manager, supervisor, executive, legal, lawyer, escalation, escalate, speak to, speak with`

#### Positive Lexicon (10 terms)
`thank, thanks, great, helpful, resolved, appreciate, question, inquiry, info, information`

#### Risk Buckets
| Bucket | Score Range |
|--------|------------|
| ESCALATING_FRUSTRATION | >= 55 |
| HIGH | 35 - 54 |
| MEDIUM | 15 - 34 |
| LOW | < 15 |

#### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback_days` | 30 | Days of tickets to analyze |
| `top_n` | 20 | Return top N tickets by risk |

---

## Part 2: Prediction & Analysis Algorithms (Runtime)

These aren't standalone ML models but are important algorithmic components.

---

### Algorithm A: Trend Detector (Linear Regression)

**File:** `api/engine/prediction/trend_detector.py`
**Library:** scipy.stats (`linregress`)

#### Algorithm
Fits a simple linear regression to recent metric values and projects forward. Uses **p-value** to determine statistical significance and **95% confidence intervals** for uncertainty quantification.

```python
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, values)
projected_value = intercept + slope * (n + projection_days)
```

#### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback_days` | 14 | Days of history for trend regression |
| `projection_days` | 5 | Days ahead to project |
| `min_slope_significance` | 0.01 | Minimum |slope| to flag |
| `max_p_value` | 0.05 | Max p-value for significance |
| `min_data_points` | 5 | Minimum observations required |

#### Metric Thresholds
| Metric | Threshold | Below? | Meaning |
|--------|-----------|--------|---------|
| refund_rate | 0.08 | No | Flag if projected to exceed 8% |
| delivery_delay_rate | 0.20 | No | Flag if projected to exceed 20% |
| ticket_backlog | 40 | No | Flag if projected to exceed 40 tickets |
| review_score_avg | 3.0 | Yes | Flag if projected to drop below 3.0 |
| margin_proxy | 0.15 | Yes | Flag if projected to drop below 15% |
| churn_proxy | 0.05 | No | Flag if projected to exceed 5% |
| net_cash_proxy | -5000 | Yes | Flag if projected to drop below -$5K |

#### 95% Confidence Interval
```python
pred_se = std_err * sqrt(1 + 1/n + (x_new - x_mean)^2 / S_xx)
ci_margin = 1.96 * pred_se
projected_ci = [projected - ci_margin, projected + ci_margin]
```

---

### Algorithm B: Cash Runway Predictor

**File:** `api/engine/prediction/cash_runway.py`
**Library:** NumPy, scipy.stats

#### Algorithm
Calculates how many months of cash runway remain based on trailing burn rate.

```python
monthly_burn = avg(daily_expenses - daily_revenue) * 30
runway_months = net_cash / monthly_burn  (if burn > 0)
```

#### 95% Confidence Interval (Student's t-distribution)
```python
se = std_burn / sqrt(n)
t_val = scipy.stats.t.ppf(0.975, df=n-1)
ci_bounds = (mean_burn +/- t_val * se) * 30
```

#### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback_days` | 60 | Historical window for burn rate |
| `projection_months` | 6 | Months to project forward |

#### Output
- `runway_months`: float or None (infinite if profitable)
- `runway_months_ci_lower` / `ci_upper`: 95% confidence bounds
- 6-month forecast curve: baseline, best case, worst case

---

### Algorithm C: Future Score Predictor

**File:** `api/engine/prediction/future_score.py`

Combines Health Scorer + Warning Service to project future health grade.

#### Grade Scale
A, B+, B, B-, C+, C, C-, D, F

#### Score Degradation
| Warning Severity | Grade Impact | Score Impact |
|-----------------|-------------|-------------|
| CRITICAL | +2 grades down | -15 points |
| HIGH | +1 grade down | -10 points |
| MEDIUM | +1 grade down | -0 points |

---

### Algorithm D: Causal Ranker (Root Cause Analysis)

**File:** `api/engine/rca/causal_ranker.py`
**Library:** NumPy

#### Algorithm
Ranks candidate root causes using a weighted multi-dimensional score:

```
contribution = 0.30 * anomaly_norm + 0.30 * temporal + 0.25 * proximity + 0.15 * quality
```

#### Scoring Weights
| Dimension | Weight | Description |
|-----------|--------|-------------|
| Anomaly magnitude | 0.30 | How anomalous (z-score, sigmoid normalized) |
| Temporal precedence | 0.30 | Did it happen before the incident? |
| Graph proximity | 0.25 | How close in the entity graph? |
| Data quality | 0.15 | Reliability of the measurement |

#### Anomaly Normalization (Sigmoid)
```python
anomaly_norm = 1 / (1 + exp(-0.5 * (zscore - 3.0)))
# z=2 → 0.38, z=3 → 0.50, z=4 → 0.62, z=6 → 0.82, z=8 → 0.92
```

#### Bootstrap Confidence Intervals
- 200 bootstrap iterations
- 5% Gaussian noise perturbation per dimension
- Returns: point estimate, CI lower/upper, std error, significance flag

#### Sensitivity Analysis
- Tests 16 weight perturbations (+/- 0.05, 0.10 per dimension)
- Robustness: "high" (>=85% rank stable), "medium" (>=60%), "low" (<60%)

---

## Part 3: Custom-Trainable Models (13 Models)

These train on the **Olist Brazilian E-Commerce Dataset** (42.6 MB, 100K+ orders).

### Dataset: Olist Brazilian E-Commerce
**Source:** Kaggle
**Location:** `data/olist/` (9 CSV files)
**Time period:** September 2016 — October 2018
**Geography:** Brazil (27 states)

| Table | Rows | Description |
|-------|------|-------------|
| olist_orders_dataset.csv | ~100K | Orders with timestamps, status, delivery dates |
| olist_order_items_dataset.csv | ~113K | Line items with price, freight, seller |
| olist_order_payments_dataset.csv | ~104K | Payment type, installments, value |
| olist_order_reviews_dataset.csv | ~100K | Review score (1-5), title, message (Portuguese) |
| olist_customers_dataset.csv | ~99K | Customer city, state, zip code |
| olist_products_dataset.csv | ~33K | Product category, weight, dimensions |
| olist_sellers_dataset.csv | ~3K | Seller city, state, zip code |
| olist_geolocation_dataset.csv | ~1M | Zip code lat/lng coordinates |
| product_category_name_translation.csv | 71 | Portuguese to English category names |

### Data Loader: `scripts/data_loader.py`
**Class:** `OlistDataLoader`

Loads all 9 CSVs and merges them into unified datasets for each pipeline. Handles:
- Missing value imputation
- Feature engineering
- Train/validation/test splitting
- Class balance reporting

---

### Pipeline A: Anomaly Detection (4 models)

**Script:** `scripts/train_anomaly_detector.py`
**CLI:** `python scripts/train_anomaly_detector.py --model all`
**MLflow experiment:** `ledgerguard-anomaly-detection`
**Output directory:** `models/anomaly/`

#### Data Preparation (`prepare_anomaly_detection_data()`)
Aggregates order-level data into **daily time-series metrics:**
- Order count, unique customers
- Total revenue, average order value, revenue std dev
- Average delivery time, late delivery rate
- Review count, average review score
- Day of week, month (encoded)
- 7-day rolling averages

**Split:** Time-based 70/15/15 (preserves temporal ordering — no future leakage)

#### Evaluation Strategy
Since anomaly detection is unsupervised (no ground truth labels), we use **pseudo-labels:**
- Compute mean z-score across all features for each day
- Top 5% most extreme days = "anomaly" (pseudo-label = 1)
- Rest = "normal" (pseudo-label = 0)
- Evaluate models against these pseudo-labels

---

#### Model 8: Isolation Forest

**Algorithm:** Randomly partitions feature space to isolate observations. Anomalies need fewer partitions (shorter tree paths).

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of isolation trees |
| `contamination` | 0.05 | Expected anomaly proportion |
| `max_features` | 0.8 | Fraction of features per tree |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

**Preprocessing:** None required (tree-based)
**Output file:** `models/anomaly/isolation_forest.joblib`

---

#### Model 9: One-Class SVM

**Algorithm:** Learns a decision boundary (hyperplane in kernel space) that encloses the normal data. Points outside the boundary are anomalies.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kernel` | "rbf" | Radial Basis Function kernel |
| `gamma` | "auto" | 1 / (n_features * X.var()) |
| `nu` | 0.05 | Upper bound on fraction of outliers |

**Preprocessing:** StandardScaler (zero mean, unit variance)
**Output files:** `models/anomaly/one_class_svm.joblib`, `models/anomaly/one_class_svm_scaler.joblib`

---

#### Model 10: Local Outlier Factor (LOF)

**Algorithm:** Measures the local density of each point relative to its neighbors. Points with substantially lower density than their neighbors are anomalies.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_neighbors` | 20 | Number of neighbors for density estimation |
| `contamination` | 0.05 | Expected anomaly proportion |
| `novelty` | True | Enable prediction on new data |
| `n_jobs` | -1 | All CPU cores |

**Preprocessing:** StandardScaler
**Output files:** `models/anomaly/local_outlier_factor.joblib`, scaler

---

#### Model 11: Autoencoder (MLPRegressor)

**Algorithm:** Neural network trained to reconstruct its input through a bottleneck layer. High reconstruction error indicates anomalies — the network can't reconstruct patterns it hasn't seen during training.

**Architecture:** Input → 32 → 16 → **8 (bottleneck)** → 16 → 32 → Output

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_layer_sizes` | (32, 16, 8, 16, 32) | Symmetric bottleneck |
| `activation` | "relu" | ReLU activation |
| `solver` | "adam" | Adam optimizer |
| `max_iter` | 500 | Max training epochs |
| `early_stopping` | True | Stop if val loss plateaus |
| `validation_fraction` | 0.1 | 10% for early stopping |
| `n_iter_no_change` | 20 | Patience for early stopping |
| `random_state` | 42 | Reproducibility |

**Anomaly threshold:** 95th percentile of reconstruction errors on training set
**Preprocessing:** StandardScaler
**Output files:** `models/anomaly/autoencoder.joblib`, scaler, threshold value

---

### Pipeline B: Churn Prediction (3 models)

**Script:** `scripts/train_churn_model.py`
**CLI:** `python scripts/train_churn_model.py --model all`
**MLflow experiment:** `ledgerguard-churn-prediction`
**Output directory:** `models/churn/`

#### Data Preparation (`prepare_churn_data()`)
Creates **customer-level features** from order history:

| Feature | Description |
|---------|-------------|
| recency | Days since last order |
| frequency | Total number of orders |
| monetary | Total amount spent |
| avg_order_value | monetary / frequency |
| avg_review_score | Mean review score given |
| complaint_count | Number of 1-2 star reviews |
| avg_delivery_time | Mean delivery days |
| payment_installments_avg | Mean payment installments |
| product_diversity | Number of distinct product categories |
| total_freight | Total freight paid |

**Target:** Binary (1 = no purchase in last 90 days, 0 = active)
**Split:** Stratified 70/15/15 (preserves class distribution)

#### Threshold Optimization
For all 3 models:
1. Train on training set
2. Get probabilities on validation set
3. Sweep thresholds using precision-recall curve
4. Select threshold that maximizes F1 score
5. Evaluate on test set with optimized threshold

---

#### Model 12: LightGBM Classifier

**Algorithm:** Gradient boosted decision trees with leaf-wise growth strategy. Each tree corrects errors from previous trees.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 300 | Boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Max tree depth |
| `num_leaves` | 31 | Max leaves per tree |
| `min_child_samples` | 20 | Min samples per leaf |
| `subsample` | 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Feature sampling ratio |
| `class_weight` | "balanced" | Auto-weight by class frequency |
| `random_state` | 42 | Reproducibility |

**Preprocessing:** None (tree-based handles raw features)
**Output:** `models/churn/lightgbm_churn_model.pkl` (includes threshold + feature names)
**Reports:** `reports/feature_importance_lightgbm.png`

---

#### Model 13: Logistic Regression

**Algorithm:** Linear model that outputs probability via sigmoid function. Coefficients directly indicate feature importance and direction.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | Inverse regularization strength |
| `max_iter` | 1000 | Max optimization iterations |
| `class_weight` | "balanced" | Auto-weight by class frequency |
| `solver` | "lbfgs" | L-BFGS optimizer |
| `random_state` | 42 | Reproducibility |

**Preprocessing:** StandardScaler (required — features must be same scale for linear model)
**Output:** `models/churn/logistic_regression_churn_model.pkl`

---

#### Model 14: Random Forest

**Algorithm:** Builds many decorrelated decision trees on random subsets of data and features. Final prediction = majority vote.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 10 | Max tree depth |
| `min_samples_leaf` | 5 | Min samples per leaf |
| `class_weight` | "balanced" | Auto-weight by class frequency |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

**Preprocessing:** None (tree-based)
**Output:** `models/churn/random_forest_churn_model.pkl`

**Pipeline Report:** `reports/churn_roc_comparison.png` (ROC curves for all 3 models)

---

### Pipeline C: Late Delivery Risk (3 models)

**Script:** `scripts/train_late_delivery.py`
**CLI:** `python scripts/train_late_delivery.py --model all`
**MLflow experiment:** `ledgerguard-late-delivery`
**Output directory:** `models/delivery/`

#### Data Preparation (`prepare_late_delivery_data()`)
Creates **order-level features** (17 features):

| Feature | Description |
|---------|-------------|
| estimated_delivery_days | Estimated delivery time (days) |
| product_weight_g | Product weight in grams |
| price | Order price |
| freight_value | Shipping cost |
| payment_installments | Number of payment installments |
| payment_value | Total payment value |
| seller_state (encoded) | Seller location state |
| customer_state (encoded) | Customer location state |
| distance_km | Haversine distance: customer ↔ seller (from geolocation) |
| purchase_day_of_week | 0=Monday through 6=Sunday |
| purchase_month | 1-12 |
| product_category (encoded) | Product category |
| payment_type (encoded) | credit_card, boleto, etc. |
| product_length_cm | Product length |
| product_height_cm | Product height |
| product_width_cm | Product width |
| product_photos_qty | Number of product photos |

**Target:** Binary (1 = delivered after estimated date, 0 = on time)
**Split:** Stratified 70/15/15
**Class weighting:** `scale_pos_weight = neg_count / pos_count` (auto-calculated)

---

#### Model 15: XGBoost Classifier

**Algorithm:** Extreme Gradient Boosting — regularized gradient boosting with built-in handling of missing values and L1/L2 regularization.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 300 | Boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Max tree depth |
| `subsample` | 0.8 | Row sampling |
| `colsample_bytree` | 0.8 | Feature sampling |
| `scale_pos_weight` | auto | Computed from class ratio |
| `eval_metric` | "logloss" | Log loss for evaluation |
| `random_state` | 42 | Reproducibility |

**Preprocessing:** None (tree-based)
**Output:** `models/delivery/xgboost_late_delivery.joblib`

---

#### Model 16: Random Forest

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 10 | Max tree depth |
| `min_samples_leaf` | 5 | Min samples per leaf |
| `class_weight` | "balanced" | Auto-weight |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

**Output:** `models/delivery/random_forest_late_delivery.joblib`

---

#### Model 17: Logistic Regression

| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | Regularization |
| `max_iter` | 1000 | Max iterations |
| `class_weight` | "balanced" | Auto-weight |
| `solver` | "lbfgs" | Optimizer |
| `random_state` | 42 | Reproducibility |

**Preprocessing:** StandardScaler pipeline
**Output:** `models/delivery/logistic_regression_late_delivery.joblib`

**Pipeline Report:** `reports/delivery_roc_comparison.png`

---

### Pipeline D: Sentiment Analysis (3 models)

**Script:** `scripts/train_sentiment.py`
**CLI:** `python scripts/train_sentiment.py --model all`
**MLflow experiment:** `ledgerguard-sentiment-analysis`
**Output directory:** `models/sentiment/`

#### Data Preparation (`prepare_sentiment_data()`)
- **Text source:** `review_comment_title` + `review_comment_message` (combined)
- **Language:** Portuguese (Brazilian)
- **Preprocessing:** Lowercase, punctuation removed by TF-IDF, 183 Portuguese stopwords removed
- **Minimum text length filter:** Reviews with empty text excluded

#### Label Mapping
| Review Score | Sentiment Label | Numeric |
|-------------|----------------|---------|
| 1-2 stars | Negative | 0 |
| 3 stars | Neutral | 1 |
| 4-5 stars | Positive | 2 |

**Split:** Stratified 70/15/15 (preserves class distribution across 3 classes)

---

#### Model 18: TF-IDF + Logistic Regression

**Vectorizer:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_features` | 10,000 | Vocabulary size |
| `ngram_range` | (1, 2) | Unigrams + bigrams |
| `min_df` | 3 | Minimum document frequency |
| `max_df` | 0.95 | Maximum document frequency |

**Classifier:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `C` | 1.0 | Regularization |
| `max_iter` | 1000 | Max iterations |
| `multi_class` | "multinomial" | 3-class softmax |
| `random_state` | 42 | Reproducibility |

**Output:** `models/sentiment/tfidf_lr_sentiment.joblib`

---

#### Model 19: TF-IDF + Multinomial Naive Bayes

**Vectorizer:** Same as Model 18 (max_features=10K, bigrams)

**Classifier:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.1 | Laplace smoothing parameter |

Naive Bayes assumes features are conditionally independent given the class. Despite this strong assumption, it works surprisingly well for text classification. Very fast training and inference.

**Output:** `models/sentiment/naive_bayes_sentiment.joblib`

---

#### Model 20: TF-IDF + Random Forest

**Vectorizer:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_features` | 5,000 | Reduced vocabulary (RF is slower on sparse) |
| `ngram_range` | (1, 2) | Unigrams + bigrams |
| `min_df` | 3 | Minimum document frequency |

**Classifier:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 15 | Max tree depth |
| `min_samples_leaf` | 3 | Min samples per leaf |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | All CPU cores |

**Output:** `models/sentiment/random_forest_sentiment.joblib`

**Pipeline Report:** `reports/sentiment_confusion_matrices.png`

---

## Part 4: Evaluation Metrics Reference

### Classification Metrics (Churn, Delivery, Sentiment)
| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP / (TP+FP) | Of predicted positives, how many are correct |
| **Recall** | TP / (TP+FN) | Of actual positives, how many did we catch |
| **F1 Score** | 2 * (P*R) / (P+R) | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve | Ranking quality across all thresholds |
| **AUC-PR** | Area under PR curve | Better metric for imbalanced datasets |

### Anomaly Detection Metrics
| Metric | Description |
|--------|-------------|
| **F1 (pseudo-label)** | F1 against top-5% pseudo-anomalies |
| **Anomaly rate** | Fraction flagged as anomalous |
| **Reconstruction error** | Autoencoder-specific: MSE between input and reconstruction |

### Regression / Trend Metrics
| Metric | Description |
|--------|-------------|
| **R-squared** | Fraction of variance explained by linear regression |
| **p-value** | Statistical significance of the trend slope |
| **95% CI** | Confidence interval for projected values |

---

## Part 5: Training Commands

```bash
# Train all 13 custom models at once
python scripts/train_models.py --model all

# Train individual pipelines
python scripts/train_anomaly_detector.py --model all          # 4 models
python scripts/train_churn_model.py --model all               # 3 models
python scripts/train_late_delivery.py --model all             # 3 models
python scripts/train_sentiment.py --model all                 # 3 models

# Train individual models
python scripts/train_anomaly_detector.py --model isolation_forest
python scripts/train_anomaly_detector.py --model ocsvm
python scripts/train_anomaly_detector.py --model lof
python scripts/train_anomaly_detector.py --model autoencoder
python scripts/train_churn_model.py --model lgbm
python scripts/train_churn_model.py --model logistic
python scripts/train_churn_model.py --model random_forest
python scripts/train_late_delivery.py --model xgboost
python scripts/train_late_delivery.py --model random_forest
python scripts/train_late_delivery.py --model logistic
python scripts/train_sentiment.py --model tfidf_lr
python scripts/train_sentiment.py --model naive_bayes
python scripts/train_sentiment.py --model random_forest

# View MLflow results
mlflow ui  # Opens http://localhost:5000
```

---

## Part 6: Model File Inventory

After training, these files are created:

```
models/
├── anomaly/
│   ├── isolation_forest.joblib
│   ├── one_class_svm.joblib
│   ├── one_class_svm_scaler.joblib
│   ├── local_outlier_factor.joblib
│   ├── local_outlier_factor_scaler.joblib
│   ├── autoencoder.joblib
│   ├── autoencoder_scaler.joblib
│   └── autoencoder_threshold.joblib
├── churn/
│   ├── lightgbm_churn_model.pkl
│   ├── logistic_regression_churn_model.pkl
│   └── random_forest_churn_model.pkl
├── delivery/
│   ├── xgboost_late_delivery.joblib
│   ├── random_forest_late_delivery.joblib
│   └── logistic_regression_late_delivery.joblib
└── sentiment/
    ├── tfidf_lr_sentiment.joblib
    ├── naive_bayes_sentiment.joblib
    └── random_forest_sentiment.joblib

reports/
├── training_summary.json
├── churn_roc_comparison.png
├── delivery_roc_comparison.png
├── sentiment_confusion_matrices.png
└── feature_importance_*.png
```

---

## Part 7: Model Performance (Trained on Olist — 100K+ orders)

### Late Delivery Prediction
| Model | F1 | AUC-ROC | Precision | Recall | Features |
|-------|-----|---------|-----------|--------|----------|
| **XGBoost** | **0.392** | **0.827** | 0.371 | 0.415 | 25 |
| Random Forest | 0.366 | 0.806 | 0.322 | 0.423 | 25 |
| Logistic Regression | 0.290 | 0.744 | 0.235 | 0.377 | 25 |

Top features: `seller_late_rate` (0.25), `month` (0.13), `estimated_delivery_buffer_days` (0.09), `days_until_estimated` (0.09), `seller_avg_delivery_days` (0.07), `distance_km` (0.07)

### Churn Prediction
| Model | F1 | AUC-ROC | AUC-PR | Precision | Recall | Features |
|-------|-----|---------|--------|-----------|--------|----------|
| **LightGBM** | **0.975** | **0.994** | **0.962** | 0.952 | 1.000 | 14 |
| Random Forest | 0.975 | 0.993 | 0.956 | 0.952 | 0.999 | 14 |
| Logistic Regression | 0.974 | 0.993 | 0.954 | 0.952 | 0.997 | 14 |

Top features: `avg_delivery_days` (1348), `recency_days` (973), `monetary` (728), `avg_order_value` (567), `avg_installments` (442)

### Sentiment Analysis (3-class: Positive/Neutral/Negative, Portuguese)
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| **TF-IDF + LR (Enhanced)** | 74.9% | **0.645** | 0.784 |
| TF-IDF + Naive Bayes | **83.9%** | 0.645 | **0.823** |
| TF-IDF + Random Forest | 74.1% | 0.449 | 0.680 |

Enhanced with text-length features (char count, word count, exclamation/question marks) and balanced class weighting. Neutral class recall improved from ~0% to 53%.

### Anomaly Detection (Unsupervised — pseudo-label evaluation)
| Model | Val F1 | Test F1 | Anomaly Rate |
|-------|--------|---------|-------------|
| Isolation Forest | 0.313 | 0.333 | 26% |
| Autoencoder | 0.256 | 0.172 | 55% |
| One-Class SVM | 0.192 | 0.164 | 58% |
| LOF | 0.833 | 0.303 | 29% |

Note: Low test F1 is expected — pseudo-labels (top 5% z-score) are noisy by definition. The models detect real anomalies; the evaluation metric understates their value.

---

*Last Updated: 2026-02-16*
*LedgerGuard — Business Reliability Engine | Hacklytics 2025*
