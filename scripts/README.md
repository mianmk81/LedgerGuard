# LedgerGuard Scripts

This directory contains operational scripts for the LedgerGuard Business Reliability Engine.

## seed_sandbox.py

Comprehensive data seeding script that generates 6 months of realistic business activity in a QuickBooks Online sandbox (or directly to DuckDB).

### Features

- **Dual Mode Operation**:
  - `local` mode: Writes directly to DuckDB Bronze layer (no QBO credentials needed)
  - `qbo` mode: Creates entities via QuickBooks Online Sandbox API

- **Realistic Data Generation**:
  - 50 Vendors with varying reliability profiles (reliable, moderate, unreliable)
  - 500 Customers across 3 segments (enterprise, SMB, startup)
  - 200 Items/Products across 6 categories
  - 2000 Invoices with seasonal patterns
  - 1500+ Payments (70% on-time, 20% late, 10% overdue)
  - 800 Bills/Expenses with vendor associations
  - 600 Bill Payments
  - 100+ Refunds (CreditMemos + RefundReceipts)
  - 50 Purchase Orders

- **Injected Incident Patterns** (for golden test validation):
  - **Pattern A**: Supplier delay cascade (month 2) - V-001 payment delays
  - **Pattern B**: Refund spike (month 3, days 60-65) - 3x normal rate
  - **Pattern C**: Order volume surge (month 4) - Invoice spike
  - **Pattern D**: Product quality issue (month 5) - Clustered refunds
  - **Pattern E**: AR aging liquidity crunch (month 6) - 20 simultaneous overdue invoices

### Usage

```bash
# Local mode (no QBO credentials required)
python scripts/seed_sandbox.py --mode local

# Local mode with custom parameters
python scripts/seed_sandbox.py --mode local --months 3 --seed 42

# QBO mode (requires authenticated QBO sandbox)
python scripts/seed_sandbox.py --mode qbo --months 6
```

### Parameters

- `--mode`: Seeding mode - `local` (default) or `qbo`
- `--months`: Number of months of data to generate (default: 6)
- `--seed`: Random seed for reproducibility (default: 42)

### Requirements

- Python 3.11+
- All dependencies from `requirements.txt` installed
- For QBO mode: Valid QuickBooks sandbox credentials in `.env`

### Example Output

```
============================================================
SEEDING SUMMARY
============================================================

Vendors:             50
Customers:          500
Items:              200
Invoices:          2000
Payments:          1400
Bills:              800
Bill Payments:      600
Credit Memos:       150
Refund Receipts:     75
Purchase Orders:     50
------------------------------------------------------------
Total Entities:    5825

Database: ./data/bre.duckdb

============================================================
INJECTED INCIDENT PATTERNS
============================================================
  Pattern A: Supplier delay cascade (month 2)
             - Vendor V-001 payment delays increased
  Pattern B: Refund spike (month 3, days 60-65)
             - 3x normal refund rate for 5 days
  Pattern C: Order volume surge (month 4)
             - Concentrated invoice creation
  Pattern D: Product quality issue (month 5)
             - Clustered refunds over 10 days
  Pattern E: AR aging liquidity crunch (month 6)
             - 20 invoices simultaneously overdue
============================================================

Seeding completed successfully!
```

### Data Quality

- **Reproducible**: Same seed produces identical datasets
- **Realistic Distributions**: Weekday bias, business hours, seasonal patterns
- **Referential Integrity**: All foreign keys valid (Customer→Invoice, Invoice→Payment)
- **Business Logic**: Payment amounts match invoices, refunds linked to paid invoices
- **Time-Coherent**: Payments always after invoices, realistic delays

### Development Notes

The script generates data in dependency order to maintain referential integrity:
1. Vendors (independent)
2. Customers (independent)
3. Items (independent)
4. Invoices (depends on Customers, Items)
5. Payments (depends on Invoices)
6. Bills (depends on Vendors, Items)
7. Bill Payments (depends on Bills)
8. Refunds (depends on Invoices)
9. Purchase Orders (depends on Vendors, Items)

All entities are written to the Bronze layer as raw QBO-formatted JSON payloads, ready for ingestion pipeline processing.

---

## train_late_delivery.py

ML training script for late delivery prediction models on Olist Brazilian E-Commerce dataset.

### Overview

Trains and evaluates 3 classification models to predict whether an order will be delivered late:

1. **XGBoost Classifier** - Gradient boosting with optimized hyperparameters
2. **Random Forest** - Ensemble decision trees with balanced class weights
3. **Logistic Regression** - Linear baseline model with feature scaling

### Features

- **Comprehensive Feature Engineering**:
  - Temporal: day_of_week, month, hour, days_until_estimated
  - Order: item_count, total_amount, freight_value, avg_item_price
  - Payment: payment_installments, payment_type (encoded)
  - Geography: customer_seller_distance, customer_state, seller_state
  - Product: product_category, avg_product_weight, avg_product_length
  - Review: review_score (when available)

- **Threshold Optimization**:
  - Optimizes decision threshold on validation set using F1 score
  - Evaluates with optimized threshold on test set
  - Handles class imbalance with balanced weights

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC and PR-AUC curves
  - Confusion matrix
  - Classification report
  - Top 15 feature importances

- **MLflow Integration**:
  - Experiment tracking with `ledgerguard-late-delivery` experiment
  - Logs hyperparameters, metrics, and model artifacts
  - Run-level comparison across models

- **Model Persistence**:
  - Saves models with optimal thresholds to `models/delivery/`
  - joblib serialization for fast loading
  - Includes threshold in saved artifact

### Dataset Requirements

Requires Olist Brazilian E-Commerce dataset from Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Expected files in `data/olist/`:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_reviews_dataset.csv`

### Usage

```bash
# Train all models (default)
python scripts/train_late_delivery.py

# Train specific model
python scripts/train_late_delivery.py --model xgboost
python scripts/train_late_delivery.py --model random_forest
python scripts/train_late_delivery.py --model logistic
```

### Parameters

- `--model`: Model to train - `xgboost`, `random_forest`, `logistic`, or `all` (default: all)

### Model Specifications

#### XGBoost Classifier
```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=auto,  # computed from class ratio
    random_state=42,
    eval_metric='logloss'
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
```

#### Logistic Regression
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
```

### Example Output

```
====================================================================================
LATE DELIVERY PREDICTION MODEL TRAINING
====================================================================================

[INFO] Initializing data loader
[INFO] Loading and preparing late delivery dataset

[INFO] Dataset summary:
  Training set:   72,000 samples (9.24% late)
  Validation set: 8,000 samples (9.18% late)
  Test set:       20,000 samples (9.31% late)
  Features:       17

[2026-02-15 14:30:00] Training XGBoost Classifier
[2026-02-15 14:30:05] Fitting XGBoost model on training data
[2026-02-15 14:32:15] Optimizing decision threshold on validation set
[2026-02-15 14:32:16] Evaluating XGBoost on test set
[2026-02-15 14:32:17] Model saved to models/delivery/xgboost_late_delivery.joblib

================================================================================
Classification Report - XGBoost
================================================================================
              precision    recall  f1-score   support

     On-Time     0.9345    0.9523    0.9433     18,138
        Late     0.4821    0.3892    0.4302      1,862

    accuracy                         0.8942     20,000
   macro avg     0.7083    0.6708    0.6868     20,000
weighted avg     0.8846    0.8942    0.8885     20,000

================================================================================
Confusion Matrix - XGBoost
================================================================================
                  Predicted
                On-Time    Late
Actual On-Time    17,272     866
       Late        1,137     725

================================================================================
Top 15 Feature Importances - XGBoost
================================================================================
 1. days_until_estimated         0.245681
 2. total_freight                0.156234
 3. customer_seller_distance     0.112456
 4. seller_state_encoded         0.089234
 5. total_amount                 0.078945
 6. customer_state_encoded       0.067123
 7. avg_product_weight           0.056789
 8. month                        0.045678
 9. payment_installments         0.034567
10. day_of_week                  0.029456
11. item_count                   0.023456
12. product_category_encoded     0.019876
13. avg_item_price               0.015678
14. hour                         0.012345
15. review_score                 0.009876

====================================================================================
MODEL COMPARISON - LATE DELIVERY PREDICTION
====================================================================================
Model                 Threshold   Accuracy  Precision     Recall         F1    ROC-AUC     PR-AUC
----------------------------------------------------------------------------------------------------
XGBoost                 0.3245     0.8942     0.4821     0.3892     0.4302     0.8567     0.5234
Random Forest           0.3891     0.8856     0.4456     0.4123     0.4283     0.8423     0.4987
Logistic Regression     0.4567     0.8734     0.3987     0.4456     0.4203     0.8156     0.4567
====================================================================================

Best F1 Score:  XGBoost (0.4302)
Best ROC-AUC:   XGBoost (0.8567)

[2026-02-15 14:35:00] Generating ROC curve comparison plot
[2026-02-15 14:35:01] ROC comparison plot saved to reports/delivery_roc_comparison.png

====================================================================================
TRAINING COMPLETE
====================================================================================
Models saved to: models/delivery
MLflow experiment: ledgerguard-late-delivery
  View results: mlflow ui
====================================================================================
```

### Output Artifacts

1. **Trained Models**:
   - `models/delivery/xgboost_late_delivery.joblib`
   - `models/delivery/random_forest_late_delivery.joblib`
   - `models/delivery/logistic_regression_late_delivery.joblib`

2. **Visualizations**:
   - `reports/delivery_roc_comparison.png` - ROC curve comparison

3. **MLflow Tracking**:
   - Experiment: `ledgerguard-late-delivery`
   - Runs for each model with full metrics and artifacts
   - View with: `mlflow ui`

### Model Loading Example

```python
import joblib

# Load trained model with threshold
artifact = joblib.load("models/delivery/xgboost_late_delivery.joblib")
model = artifact["model"]
threshold = artifact["threshold"]

# Make predictions
y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= threshold).astype(int)
```

### Performance Targets

- **Target Metrics** (for production deployment):
  - ROC-AUC > 0.85
  - PR-AUC > 0.50 (challenging due to class imbalance)
  - F1 Score > 0.40
  - Training time < 5 minutes per model

### Data Splits

- **Training**: 72% of data (stratified)
- **Validation**: 8% of data (for threshold optimization)
- **Test**: 20% of data (final evaluation)
- All splits maintain class distribution

### Dependencies

```python
# Core ML
xgboost>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0

# Tracking & Visualization
mlflow>=2.9.0
matplotlib>=3.7.0

# Serialization
joblib>=1.3.0

# Logging
structlog>=23.0.0
```

### Development Notes

- Uses stratified splits to maintain ~9% late delivery rate across sets
- Handles missing values with median/mode imputation
- Encodes categorical variables (states, payment types, product categories)
- Feature scaling only for Logistic Regression (via Pipeline)
- XGBoost handles missing values natively
- All models use class balancing to handle 91:9 class imbalance

### Future Enhancements

- [ ] Hyperparameter tuning with Optuna
- [ ] SHAP values for model explainability
- [ ] Delivery time regression (continuous prediction)
- [ ] Online learning for concept drift adaptation
- [ ] Geographic distance calculation (haversine formula)
- [ ] Temporal features (holidays, peak seasons)
- [ ] Ensemble voting classifier
