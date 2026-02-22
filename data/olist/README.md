# Olist Brazilian E-Commerce Dataset

This directory should contain the Olist Brazilian E-Commerce Public Dataset used for training LedgerGuard's ML models.

## Download Instructions

1. **Download from Kaggle**:
   - Visit: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
   - Click "Download" (requires Kaggle account)
   - Extract the ZIP file

2. **Required Files**:
   Place these CSV files in this directory (`data/olist/`):
   - `olist_orders_dataset.csv` (99,441 orders from 2016-2018)
   - `olist_order_items_dataset.csv` (112,650 order items)
   - `olist_order_reviews_dataset.csv` (99,224 reviews)
   - `olist_order_payments_dataset.csv` (103,886 payment records)
   - `olist_customers_dataset.csv` (99,441 customers)
   - `olist_products_dataset.csv` (32,951 products)

3. **Alternative: Kaggle API**:
   ```bash
   # Install kaggle CLI
   pip install kaggle

   # Configure credentials (~/.kaggle/kaggle.json)
   # Download dataset
   kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist/
   unzip data/olist/brazilian-ecommerce.zip -d data/olist/
   ```

4. **Verify Installation**:
   ```bash
   python scripts/train_models.py
   ```
   The script will validate all required files are present.

## Dataset Overview

The Olist dataset contains real commercial data from 100k orders (2016-2018) from multiple marketplaces in Brazil. It includes:

- **Orders**: Order status, timestamps, customer info
- **Order Items**: Products, sellers, prices, freight
- **Reviews**: Customer satisfaction scores and comments
- **Payments**: Payment methods, installments, values
- **Customers**: Geographic location
- **Products**: Categories, dimensions, weights

## Models Trained

LedgerGuard uses this data to train 4 production models:

1. **Anomaly Detection**: Daily revenue/order pattern anomalies
2. **Churn Prediction**: Customer churn risk scoring
3. **Late Delivery Risk**: Delivery delay probability
4. **Review Sentiment**: Negative sentiment detection

## Data Privacy

This is a public anonymized dataset. All personally identifiable information has been removed by Olist.

## Citation

```
@misc{olist_brazilian_ecommerce,
  title={Brazilian E-Commerce Public Dataset by Olist},
  author={Olist},
  year={2018},
  url={https://www.kaggle.com/olistbr/brazilian-ecommerce}
}
```

## License

This dataset is released under CC BY-NC-SA 4.0 license.
