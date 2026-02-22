"""
Olist Data Loader for LedgerGuard ML Training

Loads and prepares Olist Brazilian E-Commerce dataset for multiple ML tasks:
- Anomaly Detection (time-series business metrics)
- Customer Churn Prediction
- Late Delivery Prediction
- Sentiment Classification

Requirements:
    pandas>=2.0.0
    numpy>=1.24.0
    scikit-learn>=1.3.0
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

logger = structlog.get_logger()


class OlistDataLoader:
    """
    Loads and prepares Olist e-commerce data for anomaly detection.

    Downloads data from Kaggle if not present, aggregates to daily metrics,
    and splits into train/val/test sets for time-series anomaly detection.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing Olist CSV files.
                     Defaults to ../data/olist/
        """
        if data_dir is None:
            # Default to data/olist relative to project root
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data" / "olist"

        self.data_dir = Path(data_dir)
        self.logger = logger.bind(component="olist_data_loader")

    def _ensure_data_exists(self) -> None:
        """
        Ensure Olist dataset exists, download if necessary.

        Expects the following CSV files:
        - olist_orders_dataset.csv
        - olist_order_items_dataset.csv
        - olist_order_reviews_dataset.csv
        - olist_customers_dataset.csv
        - olist_products_dataset.csv
        - olist_sellers_dataset.csv
        """
        required_files = [
            "olist_orders_dataset.csv",
            "olist_order_items_dataset.csv",
        ]

        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(
                "data_directory_created",
                path=str(self.data_dir),
                message="Please download Olist dataset from Kaggle and place CSV files here"
            )

        missing_files = [f for f in required_files if not (self.data_dir / f).exists()]

        if missing_files:
            self.logger.warning(
                "missing_data_files",
                missing_files=missing_files,
                data_dir=str(self.data_dir),
                message="Please download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            )

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw Olist datasets.

        Returns:
            Tuple of (orders_df, order_items_df, reviews_df)
        """
        self._ensure_data_exists()

        self.logger.info("loading_raw_data", data_dir=str(self.data_dir))

        # Load orders
        orders_path = self.data_dir / "olist_orders_dataset.csv"
        if not orders_path.exists():
            raise FileNotFoundError(
                f"Orders dataset not found at {orders_path}. "
                "Please download from Kaggle."
            )

        orders_df = pd.read_csv(orders_path)

        # Load order items
        items_path = self.data_dir / "olist_order_items_dataset.csv"
        order_items_df = pd.read_csv(items_path) if items_path.exists() else None

        # Load reviews
        reviews_path = self.data_dir / "olist_order_reviews_dataset.csv"
        reviews_df = pd.read_csv(reviews_path) if reviews_path.exists() else None

        self.logger.info(
            "raw_data_loaded",
            orders_count=len(orders_df),
            items_count=len(order_items_df) if order_items_df is not None else 0,
            reviews_count=len(reviews_df) if reviews_df is not None else 0,
        )

        return orders_df, order_items_df, reviews_df

    def aggregate_daily_metrics(
        self,
        orders_df: pd.DataFrame,
        order_items_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate order data to daily time-series metrics.

        Creates features suitable for anomaly detection:
        - Order volume metrics
        - Revenue metrics
        - Delivery performance metrics
        - Review sentiment metrics

        Args:
            orders_df: Orders dataframe
            order_items_df: Order items dataframe (optional)
            reviews_df: Reviews dataframe (optional)

        Returns:
            DataFrame with daily aggregated metrics, indexed by date
        """
        self.logger.info("aggregating_daily_metrics")

        # Parse timestamps
        orders_df = orders_df.copy()
        orders_df['order_purchase_timestamp'] = pd.to_datetime(
            orders_df['order_purchase_timestamp']
        )
        orders_df['order_delivered_customer_date'] = pd.to_datetime(
            orders_df['order_delivered_customer_date']
        )
        orders_df['order_estimated_delivery_date'] = pd.to_datetime(
            orders_df['order_estimated_delivery_date']
        )

        # Extract date
        orders_df['order_date'] = orders_df['order_purchase_timestamp'].dt.date

        # Merge with order items to get pricing
        if order_items_df is not None:
            order_totals = order_items_df.groupby('order_id').agg({
                'price': 'sum',
                'freight_value': 'sum',
            }).reset_index()
            order_totals['total_amount'] = order_totals['price'] + order_totals['freight_value']
            orders_df = orders_df.merge(order_totals, on='order_id', how='left')
        else:
            orders_df['total_amount'] = 0
            orders_df['price'] = 0
            orders_df['freight_value'] = 0

        # --- Order-level metrics (known at purchase time) ---
        # Daily aggregation by PURCHASE date — only features available on that day
        daily_metrics = orders_df.groupby('order_date').agg({
            'order_id': 'count',  # Order volume
            'customer_id': 'nunique',  # Unique customers
            'total_amount': ['sum', 'mean', 'std'],  # Revenue metrics
            'price': ['sum', 'mean'],
            'freight_value': ['sum', 'mean'],
        })

        # Flatten multi-level columns
        daily_metrics.columns = ['_'.join(col).strip('_') for col in daily_metrics.columns]
        daily_metrics = daily_metrics.rename(columns={
            'order_id_count': 'order_volume',
            'customer_id_nunique': 'unique_customers',
            'total_amount_sum': 'total_revenue',
            'total_amount_mean': 'avg_order_value',
            'total_amount_std': 'order_value_std',
            'price_sum': 'total_product_revenue',
            'price_mean': 'avg_product_price',
            'freight_value_sum': 'total_freight',
            'freight_value_mean': 'avg_freight',
        })

        # --- Delivery metrics aggregated by DELIVERY date (no leakage) ---
        # Delivery outcome is only known when the package arrives, so we
        # aggregate by delivery date, not purchase date.
        delivered = orders_df.dropna(subset=['order_delivered_customer_date']).copy()
        delivered['delivery_date'] = delivered['order_delivered_customer_date'].dt.date
        delivered['delivery_time_days'] = (
            delivered['order_delivered_customer_date'] -
            delivered['order_purchase_timestamp']
        ).dt.total_seconds() / 86400
        delivered['is_late'] = (
            delivered['order_delivered_customer_date'] >
            delivered['order_estimated_delivery_date']
        ).astype(int)

        delivery_daily = delivered.groupby('delivery_date').agg({
            'delivery_time_days': ['mean', 'std', 'max'],
            'is_late': ['sum', 'mean'],
        })
        delivery_daily.columns = ['_'.join(col).strip('_') for col in delivery_daily.columns]
        delivery_daily = delivery_daily.rename(columns={
            'delivery_time_days_mean': 'avg_delivery_time',
            'delivery_time_days_std': 'delivery_time_std',
            'delivery_time_days_max': 'max_delivery_time',
            'is_late_sum': 'late_deliveries_count',
            'is_late_mean': 'late_delivery_rate',
        })

        # Join delivery metrics (by delivery date) to the daily index
        daily_metrics = daily_metrics.join(delivery_daily, how='left')
        daily_metrics[['avg_delivery_time', 'delivery_time_std', 'max_delivery_time',
                        'late_deliveries_count', 'late_delivery_rate']] = \
            daily_metrics[['avg_delivery_time', 'delivery_time_std', 'max_delivery_time',
                            'late_deliveries_count', 'late_delivery_rate']].fillna(0)

        # Add review metrics if available
        if reviews_df is not None:
            reviews_df = reviews_df.copy()
            reviews_df['review_creation_date'] = pd.to_datetime(
                reviews_df['review_creation_date']
            )
            reviews_df['review_date'] = reviews_df['review_creation_date'].dt.date

            review_metrics = reviews_df.groupby('review_date').agg({
                'review_score': ['count', 'mean', 'std'],
            })
            review_metrics.columns = ['_'.join(col).strip('_') for col in review_metrics.columns]
            review_metrics = review_metrics.rename(columns={
                'review_score_count': 'review_count',
                'review_score_mean': 'avg_review_score',
                'review_score_std': 'review_score_std',
            })

            daily_metrics = daily_metrics.join(review_metrics, how='left')
            daily_metrics['review_count'] = daily_metrics['review_count'].fillna(0)
            # Forward-fill review score for days with no reviews (use last known
            # score, not global mean which would leak future information)
            daily_metrics['avg_review_score'] = daily_metrics['avg_review_score'].ffill().fillna(0)
            daily_metrics['review_score_std'] = daily_metrics['review_score_std'].fillna(0)

        # Fill missing values
        daily_metrics = daily_metrics.fillna(0)

        # Add time-based features
        daily_metrics['day_of_week'] = pd.to_datetime(daily_metrics.index).dayofweek
        daily_metrics['day_of_month'] = pd.to_datetime(daily_metrics.index).day
        daily_metrics['month'] = pd.to_datetime(daily_metrics.index).month

        # Add rolling statistics (7-day, 14-day, 30-day windows)
        for w in [7, 14, 30]:
            daily_metrics[f'order_volume_rolling_{w}d'] = daily_metrics['order_volume'].rolling(
                window=w, min_periods=1
            ).mean()
            daily_metrics[f'revenue_rolling_{w}d'] = daily_metrics['total_revenue'].rolling(
                window=w, min_periods=1
            ).mean()
            daily_metrics[f'order_volume_std_{w}d'] = daily_metrics['order_volume'].rolling(
                window=w, min_periods=1
            ).std()
            daily_metrics[f'revenue_std_{w}d'] = daily_metrics['total_revenue'].rolling(
                window=w, min_periods=1
            ).std()

        # Lag features (1d, 7d) — capture sudden changes vs recent past
        daily_metrics['order_volume_lag1'] = daily_metrics['order_volume'].shift(1)
        daily_metrics['order_volume_lag7'] = daily_metrics['order_volume'].shift(7)
        daily_metrics['revenue_lag1'] = daily_metrics['total_revenue'].shift(1)
        daily_metrics['revenue_lag7'] = daily_metrics['total_revenue'].shift(7)

        # Fill any remaining NaN values (lags create NaN for first rows)
        daily_metrics = daily_metrics.fillna(0)

        self.logger.info(
            "daily_metrics_aggregated",
            date_range=f"{daily_metrics.index.min()} to {daily_metrics.index.max()}",
            total_days=len(daily_metrics),
            features_count=len(daily_metrics.columns),
        )

        return daily_metrics

    def prepare_anomaly_detection_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Prepare data for anomaly detection model training.

        Loads raw data, aggregates to daily metrics, and splits into
        train/validation/test sets using temporal ordering.

        Args:
            train_ratio: Fraction of data for training (default 0.7)
            val_ratio: Fraction of data for validation (default 0.15)
            test_ratio: Fraction of data for testing (default 0.15)

        Returns:
            Tuple of (X_train, X_val, X_test, dates_train, dates_val, dates_test)
            where X_* are numpy arrays of features and dates_* are date indices
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.logger.info(
            "preparing_anomaly_detection_data",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # Load and aggregate data
        orders_df, order_items_df, reviews_df = self.load_raw_data()
        daily_metrics = self.aggregate_daily_metrics(orders_df, order_items_df, reviews_df)

        # Sort by date (temporal ordering critical for time series)
        daily_metrics = daily_metrics.sort_index()

        # Split data
        n = len(daily_metrics)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = daily_metrics.iloc[:train_end]
        val_df = daily_metrics.iloc[train_end:val_end]
        test_df = daily_metrics.iloc[val_end:]

        # Extract features and dates
        X_train = train_df.values
        X_val = val_df.values
        X_test = test_df.values

        dates_train = pd.to_datetime(train_df.index)
        dates_val = pd.to_datetime(val_df.index)
        dates_test = pd.to_datetime(test_df.index)

        self.logger.info(
            "data_prepared",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            features=X_train.shape[1],
            train_date_range=f"{dates_train.min()} to {dates_train.max()}",
            val_date_range=f"{dates_val.min()} to {dates_val.max()}",
            test_date_range=f"{dates_test.min()} to {dates_test.max()}",
        )

        return X_train, X_val, X_test, dates_train, dates_val, dates_test

    def prepare_late_delivery_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        return_orders: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare late delivery prediction dataset.

        Creates order-level features with binary late delivery label (1 if late, 0 if on-time).
        Only includes delivered orders with valid delivery dates.

        Features include:
        - Temporal: day_of_week, month, hour, days_until_estimated, is_weekend, is_month_end
        - Order: item_count, total_amount, freight_value, avg_item_price, freight_ratio
        - Payment: payment_installments, payment_type (encoded)
        - Geography: distance_km (haversine), customer_state, seller_state
        - Product: product_category (encoded), product_dimensions, product_volume_cm3
        - Seller history: seller_avg_delivery_days, seller_late_rate, seller_order_count
        - Category history: category_late_rate
        - Delivery planning: estimated_delivery_buffer_days
        - Review: review_score EXCLUDED (data leakage guard)

        Args:
            test_size: Proportion of data for test set (0.2 = 20%)
            val_size: Proportion of training data for validation (0.1 = 10%)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        def haversine_km(lat1, lon1, lat2, lon2):
            """Calculate haversine distance in km between two lat/lng points."""
            R = 6371  # Earth radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            return R * 2 * np.arcsin(np.sqrt(a))

        self.logger.info(
            "preparing_late_delivery_data",
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        # Load raw data
        orders_df, order_items_df, reviews_df = self.load_raw_data()

        # Load additional datasets
        customers_path = self.data_dir / "olist_customers_dataset.csv"
        customers_df = pd.read_csv(customers_path) if customers_path.exists() else None

        sellers_path = self.data_dir / "olist_sellers_dataset.csv"
        sellers_df = pd.read_csv(sellers_path) if sellers_path.exists() else None

        products_path = self.data_dir / "olist_products_dataset.csv"
        products_df = pd.read_csv(products_path) if products_path.exists() else None

        payments_path = self.data_dir / "olist_order_payments_dataset.csv"
        payments_df = pd.read_csv(payments_path) if payments_path.exists() else None

        # Load geolocation data
        geolocation_path = self.data_dir / "olist_geolocation_dataset.csv"
        geolocation_df = pd.read_csv(geolocation_path) if geolocation_path.exists() else None

        # Start with orders that are delivered
        orders = orders_df.copy()
        orders = orders[orders["order_status"] == "delivered"].copy()

        # Convert timestamps
        orders["order_purchase_timestamp"] = pd.to_datetime(
            orders["order_purchase_timestamp"]
        )
        orders["order_delivered_customer_date"] = pd.to_datetime(
            orders["order_delivered_customer_date"]
        )
        orders["order_estimated_delivery_date"] = pd.to_datetime(
            orders["order_estimated_delivery_date"]
        )

        # Remove orders with missing delivery dates
        orders = orders.dropna(
            subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
        )

        # Create target: 1 if late, 0 if on-time
        orders["is_late"] = (
            orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
        ).astype(int)

        # Calculate actual delivery time for buffer calculation later
        orders["actual_delivery_days"] = (
            orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        # Extract temporal features
        orders["day_of_week"] = orders["order_purchase_timestamp"].dt.dayofweek
        orders["month"] = orders["order_purchase_timestamp"].dt.month
        orders["hour"] = orders["order_purchase_timestamp"].dt.hour
        orders["days_until_estimated"] = (
            orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400

        # New temporal features
        orders["is_weekend"] = (orders["day_of_week"] >= 5).astype(int)
        orders["day_of_month"] = orders["order_purchase_timestamp"].dt.day
        orders["days_in_month"] = orders["order_purchase_timestamp"].dt.days_in_month
        orders["is_month_end"] = (orders["days_in_month"] - orders["day_of_month"] <= 2).astype(int)

        # Aggregate order items
        if order_items_df is not None:
            item_aggs = order_items_df.groupby("order_id").agg(
                item_count=("order_item_id", "count"),
                total_price=("price", "sum"),
                total_freight=("freight_value", "sum"),
                avg_item_price=("price", "mean"),
            ).reset_index()

            orders = orders.merge(item_aggs, on="order_id", how="left")
            orders["total_amount"] = orders["total_price"] + orders["total_freight"]
        else:
            orders["item_count"] = 1
            orders["total_amount"] = 0
            orders["total_freight"] = 0
            orders["avg_item_price"] = 0
            orders["total_price"] = 0

        # Freight-to-price ratio
        orders["freight_ratio"] = orders["total_freight"] / (orders["total_price"] + 0.01)

        # Aggregate payments
        if payments_df is not None:
            payment_aggs = payments_df.groupby("order_id").agg(
                payment_installments=("payment_installments", "max"),
                payment_type=("payment_type", "first"),
            ).reset_index()

            orders = orders.merge(payment_aggs, on="order_id", how="left")
        else:
            orders["payment_installments"] = 1
            orders["payment_type"] = "unknown"

        # Join customers for geography
        if customers_df is not None:
            customers = customers_df[["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]].copy()
            orders = orders.merge(customers, on="customer_id", how="left")
        else:
            orders["customer_state"] = "unknown"
            orders["customer_zip_code_prefix"] = 0

        # Join sellers via order_items
        if sellers_df is not None and order_items_df is not None:
            seller_info = order_items_df.merge(
                sellers_df[["seller_id", "seller_state", "seller_city", "seller_zip_code_prefix"]],
                on="seller_id",
                how="left"
            ).groupby("order_id").agg(
                seller_id=("seller_id", "first"),
                seller_state=("seller_state", "first"),
                seller_city=("seller_city", "first"),
                seller_zip_code_prefix=("seller_zip_code_prefix", "first"),
            ).reset_index()

            orders = orders.merge(seller_info, on="order_id", how="left")
        else:
            orders["seller_id"] = "unknown"
            orders["seller_state"] = "unknown"
            orders["seller_zip_code_prefix"] = 0

        # Join product category and dimensions
        if products_df is not None and order_items_df is not None:
            # Load additional product dimensions
            product_cols = ["product_id", "product_category_name", "product_weight_g",
                           "product_length_cm", "product_height_cm", "product_width_cm"]
            available_cols = [col for col in product_cols if col in products_df.columns]

            product_info = order_items_df.merge(
                products_df[available_cols],
                on="product_id",
                how="left"
            ).groupby("order_id").agg(
                product_category=("product_category_name", "first"),
                avg_product_weight=("product_weight_g", "mean"),
                avg_product_length=("product_length_cm", "mean"),
                avg_product_height=("product_height_cm", "mean") if "product_height_cm" in available_cols else ("product_length_cm", "mean"),
                avg_product_width=("product_width_cm", "mean") if "product_width_cm" in available_cols else ("product_length_cm", "mean"),
            ).reset_index()

            orders = orders.merge(product_info, on="order_id", how="left")

            # Calculate product volume
            orders["product_volume_cm3"] = (
                orders["avg_product_length"].fillna(0) *
                orders["avg_product_height"].fillna(0) *
                orders["avg_product_width"].fillna(0)
            )
        else:
            orders["product_category"] = "unknown"
            orders["avg_product_weight"] = 0
            orders["avg_product_length"] = 0
            orders["avg_product_height"] = 0
            orders["avg_product_width"] = 0
            orders["product_volume_cm3"] = 0

        # Join review scores
        if reviews_df is not None:
            reviews = reviews_df[["order_id", "review_score"]].copy()
            orders = orders.merge(reviews, on="order_id", how="left")
        else:
            orders["review_score"] = 3.0

        # --- NEW FEATURE: Real haversine distance ---
        if geolocation_df is not None:
            self.logger.info("calculating_haversine_distances")

            # Average lat/lng per zip_code_prefix (many duplicates in geo data)
            geo_agg = geolocation_df.groupby("geolocation_zip_code_prefix").agg({
                "geolocation_lat": "mean",
                "geolocation_lng": "mean",
            }).reset_index()

            # Join customer coordinates
            orders = orders.merge(
                geo_agg.rename(columns={
                    "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                    "geolocation_lat": "customer_lat",
                    "geolocation_lng": "customer_lng",
                }),
                on="customer_zip_code_prefix",
                how="left"
            )

            # Join seller coordinates
            orders = orders.merge(
                geo_agg.rename(columns={
                    "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                    "geolocation_lat": "seller_lat",
                    "geolocation_lng": "seller_lng",
                }),
                on="seller_zip_code_prefix",
                how="left"
            )

            # Calculate haversine distance
            mask = (
                orders["customer_lat"].notna() &
                orders["customer_lng"].notna() &
                orders["seller_lat"].notna() &
                orders["seller_lng"].notna()
            )

            orders.loc[mask, "distance_km"] = orders.loc[mask].apply(
                lambda row: haversine_km(
                    row["customer_lat"], row["customer_lng"],
                    row["seller_lat"], row["seller_lng"]
                ),
                axis=1
            )

            # Fill missing distances with median
            orders["distance_km"] = orders["distance_km"].fillna(orders["distance_km"].median())
        else:
            # Fallback: binary same/different state
            orders["distance_km"] = (
                orders["customer_state"] != orders["seller_state"]
            ).astype(int) * 1000  # Rough approximation: different state = 1000km

        # --- NEW FEATURES: Seller performance history ---
        # For hackathon, compute from all data (acceptable simplification)
        if "seller_id" in orders.columns:
            seller_history = orders.groupby("seller_id").agg(
                seller_avg_delivery_days=("actual_delivery_days", "mean"),
                seller_late_rate=("is_late", "mean"),
                seller_order_count=("order_id", "count"),
            ).reset_index()

            orders = orders.merge(seller_history, on="seller_id", how="left")
            orders["seller_avg_delivery_days"] = orders["seller_avg_delivery_days"].fillna(
                orders["actual_delivery_days"].median()
            )
            orders["seller_late_rate"] = orders["seller_late_rate"].fillna(orders["is_late"].mean())
            orders["seller_order_count"] = orders["seller_order_count"].fillna(1)
        else:
            orders["seller_avg_delivery_days"] = orders["actual_delivery_days"].median()
            orders["seller_late_rate"] = orders["is_late"].mean()
            orders["seller_order_count"] = 1

        # --- NEW FEATURE: Product category late rate ---
        if "product_category" in orders.columns:
            category_history = orders.groupby("product_category").agg(
                category_late_rate=("is_late", "mean"),
            ).reset_index()

            orders = orders.merge(category_history, on="product_category", how="left")
            orders["category_late_rate"] = orders["category_late_rate"].fillna(orders["is_late"].mean())
        else:
            orders["category_late_rate"] = orders["is_late"].mean()

        # --- NEW FEATURE: Estimated delivery buffer days ---
        # Buffer = days_until_estimated - typical delivery time
        typical_delivery = orders["actual_delivery_days"].median()
        orders["estimated_delivery_buffer_days"] = orders["days_until_estimated"] - typical_delivery

        # --- NEW FEATURE: Daily order count ---
        # Number of orders placed on the same calendar day (platform-wide load signal)
        daily_counts = orders.groupby(
            orders["order_purchase_timestamp"].dt.date
        )["order_id"].transform("count")
        orders["daily_order_count"] = daily_counts.values

        # --- NEW FEATURE: Seller daily order count ---
        # Number of orders from the same seller on the same calendar day
        if "seller_id" in orders.columns:
            seller_daily = orders.groupby([
                orders["order_purchase_timestamp"].dt.date, "seller_id"
            ])["order_id"].transform("count")
            orders["seller_daily_order_count"] = seller_daily.values
        else:
            orders["seller_daily_order_count"] = 1

        # --- NEW FEATURE: Price per gram ---
        # Price-to-weight ratio; clip weight at 1g to avoid division by zero
        orders["price_per_gram"] = orders["total_amount"] / orders["avg_product_weight"].clip(lower=1)

        # Encode categorical variables
        le_payment = LabelEncoder()
        le_customer_state = LabelEncoder()
        le_seller_state = LabelEncoder()
        le_category = LabelEncoder()

        orders["payment_type_encoded"] = le_payment.fit_transform(
            orders["payment_type"].fillna("unknown")
        )
        orders["customer_state_encoded"] = le_customer_state.fit_transform(
            orders["customer_state"].fillna("unknown")
        )
        orders["seller_state_encoded"] = le_seller_state.fit_transform(
            orders["seller_state"].fillna("unknown")
        )
        orders["product_category_encoded"] = le_category.fit_transform(
            orders["product_category"].fillna("unknown")
        )

        # Same-state flag: orders within same state often have faster delivery
        orders["same_state"] = (
            orders["customer_state"].fillna("") == orders["seller_state"].fillna("")
        ).astype(int)

        # Select feature columns
        # NOTE: review_score excluded — it is written AFTER delivery, so using it
        # to predict late delivery would be data leakage (post-outcome information).
        feature_cols = [
            # Original temporal features
            "day_of_week",
            "month",
            "hour",
            "days_until_estimated",
            # New temporal features
            "is_weekend",
            "is_month_end",
            # Original order features
            "item_count",
            "total_amount",
            "total_freight",
            "avg_item_price",
            # New order features
            "freight_ratio",
            # Payment features
            "payment_installments",
            "payment_type_encoded",
            # Geography features
            "customer_state_encoded",
            "seller_state_encoded",
            "distance_km",  # REPLACES customer_seller_distance
            "same_state",  # Same customer/seller state → often faster
            # Product features
            "product_category_encoded",
            "avg_product_weight",
            "avg_product_length",
            "product_volume_cm3",  # NEW
            # Seller history features
            "seller_avg_delivery_days",  # NEW
            "seller_late_rate",  # NEW
            "seller_order_count",  # NEW
            # Category history
            "category_late_rate",  # NEW
            # Delivery planning
            "estimated_delivery_buffer_days",  # NEW
            # Load / congestion signals
            "daily_order_count",  # NEW
            "seller_daily_order_count",  # NEW
            # Weight-adjusted price
            "price_per_gram",  # NEW
        ]

        # Fill missing values with median for numeric columns
        for col in feature_cols:
            if col in orders.columns:
                if orders[col].dtype in ["float64", "int64"]:
                    median_val = orders[col].median()
                    orders[col] = orders[col].fillna(median_val)
                else:
                    orders[col] = orders[col].fillna(0)

        # Extract features and target
        X = orders[feature_cols].copy()
        y = orders["is_late"].copy()

        # Split into train+val and test BEFORE fitting encoders
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            random_state=random_state,
            stratify=y_trainval,
        )

        self.logger.info(
            "late_delivery_data_prepared",
            total_samples=len(X),
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            late_rate=f"{y.mean():.2%}",
            num_features=len(feature_cols),
        )

        # Log class distribution
        self.logger.info(
            "class_distribution",
            train_late=f"{y_train.mean():.2%}",
            val_late=f"{y_val.mean():.2%}",
            test_late=f"{y_test.mean():.2%}",
        )

        if return_orders:
            return X_train, X_val, X_test, y_train, y_val, y_test, orders
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_late_delivery_data_with_duration(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
    ]:
        """
        Extended version that also returns actual delivery duration and
        scheduled shipping days for two-stage modeling.

        Calls ``prepare_late_delivery_data()`` internally to guarantee that the
        feature matrix and the duration arrays are derived from the identical
        row population and use the same train/val/test indices.

        The duration arrays are extracted from the full ``orders`` DataFrame
        using the index positions produced by the stratified splits inside
        ``prepare_late_delivery_data()``.  Because both DataFrames share the
        same original ``orders`` index (preserved through ``train_test_split``
        with ``shuffle=True``), we can align on that shared index.

        Args:
            test_size: Proportion of data for the test set (default 0.2).
            val_size: Proportion of the training pool for validation (default 0.1).
            random_state: Random seed forwarded to both internal splits so that
                results are fully reproducible.

        Returns:
            Tuple of 12 arrays:
            ``(X_train, X_val, X_test,
              y_train, y_val, y_test,
              dur_train, dur_val, dur_test,
              sched_train, sched_val, sched_test)``

            where:
            - ``X_*`` — feature DataFrames (same as ``prepare_late_delivery_data``)
            - ``y_*`` — binary late-delivery labels (same as above)
            - ``dur_*`` — numpy float arrays of actual delivery duration in days
              (``actual_delivery_days``)
            - ``sched_*`` — numpy float arrays of scheduled shipping window in days
              (``days_until_estimated``)
        """
        self.logger.info(
            "preparing_late_delivery_data_with_duration",
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        # ------------------------------------------------------------------
        # Step 1: get the feature/label splits and the orders DataFrame from
        # the canonical method. X_train/val/test retain the original orders
        # index, so we can extract real actual_delivery_days by index.
        # ------------------------------------------------------------------
        X_train, X_val, X_test, y_train, y_val, y_test, orders = self.prepare_late_delivery_data(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            return_orders=True,
        )

        # ------------------------------------------------------------------
        # Step 2: Extract real actual_delivery_days and days_until_estimated
        # using the shared index (X splits came from orders[feature_cols]).
        # ------------------------------------------------------------------
        dur_train = orders.loc[X_train.index, "actual_delivery_days"].to_numpy(dtype=float)
        dur_val = orders.loc[X_val.index, "actual_delivery_days"].to_numpy(dtype=float)
        dur_test = orders.loc[X_test.index, "actual_delivery_days"].to_numpy(dtype=float)

        sched_train = orders.loc[X_train.index, "days_until_estimated"].to_numpy(dtype=float)
        sched_val = orders.loc[X_val.index, "days_until_estimated"].to_numpy(dtype=float)
        sched_test = orders.loc[X_test.index, "days_until_estimated"].to_numpy(dtype=float)

        self.logger.info(
            "late_delivery_with_duration_prepared",
            dur_train_mean=float(np.nanmean(dur_train)),
            dur_test_mean=float(np.nanmean(dur_test)),
            sched_train_mean=float(np.nanmean(sched_train)),
            sched_test_mean=float(np.nanmean(sched_test)),
            train_samples=len(dur_train),
            val_samples=len(dur_val),
            test_samples=len(dur_test),
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            dur_train, dur_val, dur_test,
            sched_train, sched_val, sched_test,
        )

    def prepare_churn_data(
        self,
        churn_window_days: int = 90,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        repeat_customers_only: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare customer-level features for churn prediction.

        Features:
        - RFM (Recency, Frequency, Monetary)
        - Average review score
        - Complaint/cancellation count
        - Average delivery delay
        - Product category preferences

        Label: Binary churn. For repeat customers (2+ orders): churned if no
        purchase in second half of their activity window. Single-order customers
        are excluded when repeat_customers_only=True for meaningful churn.

        Args:
            churn_window_days: Days of inactivity to classify as churned
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            random_state: Random seed for reproducibility
            repeat_customers_only: If True, only include customers with 2+
                orders (produces ~30-60% churn). If False, includes
                single-order customers with arbitrary labels (~50% churn mix).

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info(
            "preparing_churn_data",
            churn_window_days=churn_window_days,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        # Load raw data
        orders_df, order_items_df, reviews_df = self.load_raw_data()

        # Load additional datasets
        customers_path = self.data_dir / "olist_customers_dataset.csv"
        customers_df = pd.read_csv(customers_path) if customers_path.exists() else None

        payments_path = self.data_dir / "olist_order_payments_dataset.csv"
        payments_df = pd.read_csv(payments_path) if payments_path.exists() else None

        products_path = self.data_dir / "olist_products_dataset.csv"
        products_df = pd.read_csv(products_path) if products_path.exists() else None

        # Only consider delivered orders
        orders = orders_df[orders_df['order_status'] == 'delivered'].copy()

        # Convert timestamps
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        # Merge with customers to get unique customer ID
        if customers_df is not None:
            orders = orders.merge(
                customers_df[['customer_id', 'customer_unique_id']],
                on='customer_id',
                how='left'
            )
        else:
            orders['customer_unique_id'] = orders['customer_id']

        # Merge with payments (value + installments + type)
        if payments_df is not None:
            payment_aggs = payments_df.groupby('order_id').agg(
                payment_value=('payment_value', 'sum'),
                payment_installments=('payment_installments', 'max'),
                payment_type=('payment_type', 'first'),
            ).reset_index()
            orders = orders.merge(payment_aggs, on='order_id', how='left')
        else:
            orders['payment_value'] = 0
            orders['payment_installments'] = 1
            orders['payment_type'] = 'unknown'

        # Compute delivery time per order (for delivery experience features)
        orders['order_delivered_customer_date'] = pd.to_datetime(
            orders['order_delivered_customer_date']
        )
        orders['order_estimated_delivery_date'] = pd.to_datetime(
            orders['order_estimated_delivery_date']
        )
        orders['delivery_time_days'] = (
            orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
        ).dt.total_seconds() / 86400
        orders['is_late'] = (
            orders['order_delivered_customer_date'] > orders['order_estimated_delivery_date']
        ).astype(float)
        # Fill NaN for orders missing delivery dates
        orders['delivery_time_days'] = orders['delivery_time_days'].fillna(orders['delivery_time_days'].median())
        orders['is_late'] = orders['is_late'].fillna(0)

        # Merge product diversity (distinct categories per order)
        if products_df is not None and order_items_df is not None:
            order_products = order_items_df.merge(
                products_df[['product_id', 'product_category_name']],
                on='product_id', how='left'
            )
            product_diversity_per_order = order_products.groupby('order_id').agg(
                n_distinct_categories=('product_category_name', 'nunique'),
                n_items=('order_item_id', 'count'),
            ).reset_index()
            orders = orders.merge(product_diversity_per_order, on='order_id', how='left')
        else:
            orders['n_distinct_categories'] = 1
            orders['n_items'] = 1

        # Define observation window BEFORE merging reviews
        # (reviews written after observation_end would leak future info)
        max_date = orders['order_purchase_timestamp'].max()
        observation_end = max_date - timedelta(days=churn_window_days)

        # Merge with reviews — only reviews created BEFORE observation_end
        if reviews_df is not None:
            reviews_filtered = reviews_df.copy()
            reviews_filtered['review_creation_date'] = pd.to_datetime(
                reviews_filtered['review_creation_date']
            )
            reviews_filtered = reviews_filtered[
                reviews_filtered['review_creation_date'] <= observation_end
            ]
            review_aggs = reviews_filtered.groupby('order_id').agg({
                'review_score': 'mean'
            }).reset_index()
            orders = orders.merge(review_aggs, on='order_id', how='left')
        else:
            orders['review_score'] = 3.0

        # ---------------------------------------------------------------
        # Churn definition for Olist (non-subscription e-commerce):
        # Most Olist customers buy only once; naive "no purchase in 90 days"
        # yields 99%+ churn. For meaningful churn prediction we use:
        # - Repeat customers (2+ orders): features from first half of their
        #   order history; churned if they did NOT return in second half.
        # - Single-order customers (optional): labeled by dataset midpoint
        #   for balance; often excluded (repeat_customers_only=True).
        # ---------------------------------------------------------------

        # Split customers by order count
        customer_order_counts = orders.groupby('customer_unique_id')['order_id'].nunique()
        repeat_customers = set(customer_order_counts[customer_order_counts >= 2].index)
        single_customers = set(customer_order_counts[customer_order_counts == 1].index)

        self.logger.info(
            "customer_segments",
            repeat_customers=len(repeat_customers),
            single_customers=len(single_customers),
            repeat_customers_only=repeat_customers_only,
        )

        # --- Repeat customers: use temporal split ---
        repeat_orders = orders[orders['customer_unique_id'].isin(repeat_customers)].copy()

        # Per-customer observation cutoff: midpoint of their order history
        customer_dates = repeat_orders.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max'])
        customer_dates['cutoff'] = customer_dates['min'] + (customer_dates['max'] - customer_dates['min']) / 2

        repeat_hist_rows = []
        repeat_future_rows = []
        for cid, row in customer_dates.iterrows():
            cust_orders = repeat_orders[repeat_orders['customer_unique_id'] == cid]
            repeat_hist_rows.append(cust_orders[cust_orders['order_purchase_timestamp'] <= row['cutoff']])
            repeat_future_rows.append(cust_orders[cust_orders['order_purchase_timestamp'] > row['cutoff']])

        hist_repeat = pd.concat(repeat_hist_rows, ignore_index=True) if repeat_hist_rows else pd.DataFrame()
        future_repeat = pd.concat(repeat_future_rows, ignore_index=True) if repeat_future_rows else pd.DataFrame()
        future_repeat_custs = set(future_repeat['customer_unique_id'].unique()) if len(future_repeat) > 0 else set()

        # --- Single-order customers: include only if repeat_customers_only=False ---
        min_date = orders['order_purchase_timestamp'].min()
        dataset_midpoint = min_date + (max_date - min_date) / 2
        single_orders = (
            orders[orders['customer_unique_id'].isin(single_customers)].copy()
            if not repeat_customers_only else pd.DataFrame()
        )

        # Build customer-level features
        all_hist = pd.concat([hist_repeat, single_orders], ignore_index=True)

        if len(all_hist) == 0:
            raise ValueError(
                "No customers in dataset. With repeat_customers_only=True, "
                "need at least some customers with 2+ orders. Try repeat_customers_only=False."
            )

        customers_agg = all_hist.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': ['min', 'max', 'count'],
            'payment_value': ['sum', 'mean', 'std'],
            'review_score': 'mean',
            'order_id': 'nunique',
            'delivery_time_days': 'mean',
            'is_late': ['sum', 'mean'],
            'payment_installments': 'mean',
            'n_distinct_categories': 'sum',
            'n_items': 'sum',
        })
        customers_agg.columns = ['_'.join(col).strip('_') for col in customers_agg.columns]
        customers_agg = customers_agg.reset_index()

        # Recency (days since last purchase relative to observation_end)
        customers_agg['recency_days'] = (
            observation_end - customers_agg['order_purchase_timestamp_max']
        ).dt.days

        # Frequency (number of orders)
        customers_agg['frequency'] = customers_agg['order_id_nunique']

        # Monetary (total spend)
        customers_agg['monetary'] = customers_agg['payment_value_sum']

        # Average order value
        customers_agg['avg_order_value'] = customers_agg['payment_value_mean']

        # Lifespan (days between first and last purchase)
        customers_agg['lifespan_days'] = (
            customers_agg['order_purchase_timestamp_max'] -
            customers_agg['order_purchase_timestamp_min']
        ).dt.days
        customers_agg['lifespan_days'] = customers_agg['lifespan_days'].replace(0, 1)

        # Purchase rate (orders per day)
        customers_agg['purchase_rate'] = customers_agg['frequency'] / customers_agg['lifespan_days']

        # Average review score
        customers_agg['avg_review_score'] = customers_agg['review_score_mean'].fillna(3.0)

        # Delivery experience features
        customers_agg['avg_delivery_days'] = customers_agg['delivery_time_days_mean']
        customers_agg['late_order_count'] = customers_agg['is_late_sum']
        customers_agg['late_order_rate'] = customers_agg['is_late_mean']

        # Payment behavior
        customers_agg['avg_installments'] = customers_agg['payment_installments_mean']

        # Product diversity
        customers_agg['product_diversity'] = customers_agg['n_distinct_categories_sum']
        customers_agg['total_items'] = customers_agg['n_items_sum']

        # Complaint count (1-2 star reviews)
        if reviews_df is not None:
            reviews_for_complaints = reviews_df.copy()
            reviews_for_complaints['review_creation_date'] = pd.to_datetime(
                reviews_for_complaints['review_creation_date']
            )
            reviews_for_complaints = reviews_for_complaints[
                reviews_for_complaints['review_creation_date'] <= observation_end
            ]
            complaint_orders = reviews_for_complaints[
                reviews_for_complaints['review_score'] <= 2
            ]['order_id'].values
            # Map complaint orders to customers
            complaint_customer_counts = all_hist[
                all_hist['order_id'].isin(complaint_orders)
            ].groupby('customer_unique_id').size().reset_index(name='complaint_count')
            customers_agg = customers_agg.merge(complaint_customer_counts, on='customer_unique_id', how='left')
            customers_agg['complaint_count'] = customers_agg['complaint_count'].fillna(0)
        else:
            customers_agg['complaint_count'] = 0

        # Churn label
        def label_churn(row):
            cid = row['customer_unique_id']
            if cid in repeat_customers:
                # Repeat customer: churned if they didn't return in 2nd half
                return 0 if cid in future_repeat_custs else 1
            else:
                # Single-order customer: churned if purchase was before midpoint
                last_purchase = row['order_purchase_timestamp_max']
                return 1 if last_purchase < dataset_midpoint else 0

        customers_agg['churned'] = customers_agg.apply(label_churn, axis=1)

        # Engineered features for better churn signal
        customers_agg['log_monetary'] = np.log1p(customers_agg['monetary'])
        customers_agg['recency_squared'] = customers_agg['recency_days'] ** 2
        customers_agg['recency_frequency'] = (
            customers_agg['recency_days'] * customers_agg['frequency']
        )

        # Select features
        feature_cols = [
            'customer_unique_id',
            'recency_days',
            'frequency',
            'monetary',
            'log_monetary',
            'recency_squared',
            'recency_frequency',
            'avg_order_value',
            'lifespan_days',
            'purchase_rate',
            'avg_review_score',
            'avg_delivery_days',
            'late_order_count',
            'late_order_rate',
            'avg_installments',
            'product_diversity',
            'total_items',
            'complaint_count',
            'churned'
        ]

        customers_agg = customers_agg[feature_cols].copy()
        customers_agg = customers_agg.fillna(0)

        # Stratified split
        train_val, test = train_test_split(
            customers_agg,
            test_size=1 - (train_ratio + val_ratio),
            stratify=customers_agg['churned'],
            random_state=random_state
        )

        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (train_ratio + val_ratio),
            stratify=train_val['churned'],
            random_state=random_state
        )

        churn_rate = customers_agg['churned'].mean()

        # Split into X (features) and y (target) for each set
        non_feature_cols = ['customer_unique_id', 'churned']
        X_train = train.drop(columns=non_feature_cols)
        y_train = train['churned']
        X_val = val.drop(columns=non_feature_cols)
        y_val = val['churned']
        X_test = test.drop(columns=non_feature_cols)
        y_test = test['churned']

        self.logger.info(
            "churn_data_prepared",
            total_customers=len(customers_agg),
            churn_rate=f"{churn_rate:.2%}",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test),
            num_features=X_train.shape[1],
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_sentiment_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare review data for sentiment classification.

        Features:
        - review_text (Portuguese text from review_comment_message)
        - review_score (1-5)

        Label: Sentiment (negative: 1-2, neutral: 3, positive: 4-5)

        Args:
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df, test_df) with stratified split
        """
        self.logger.info(
            "preparing_sentiment_data",
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        # Load reviews
        reviews_path = self.data_dir / "olist_order_reviews_dataset.csv"
        if not reviews_path.exists():
            raise FileNotFoundError(f"Reviews dataset not found at {reviews_path}")

        reviews = pd.read_csv(reviews_path)

        # Only reviews with text
        reviews = reviews[
            reviews['review_comment_message'].notna() &
            (reviews['review_comment_message'].str.len() > 10)
        ].copy()

        # Sentiment mapping
        def score_to_sentiment(score):
            if score <= 2:
                return 'negative'
            elif score == 3:
                return 'neutral'
            else:
                return 'positive'

        reviews['sentiment'] = reviews['review_score'].apply(score_to_sentiment)

        # Select columns
        reviews = reviews[['review_comment_message', 'review_score', 'sentiment']].copy()
        reviews = reviews.rename(columns={'review_comment_message': 'review_text'})

        # Stratified split
        train_val, test = train_test_split(
            reviews,
            test_size=1 - (train_ratio + val_ratio),
            stratify=reviews['sentiment'],
            random_state=random_state
        )

        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (train_ratio + val_ratio),
            stratify=train_val['sentiment'],
            random_state=random_state
        )

        sentiment_dist = reviews['sentiment'].value_counts(normalize=True)
        self.logger.info(
            "sentiment_data_prepared",
            total_reviews=len(reviews),
            sentiment_distribution={k: f"{v:.2%}" for k, v in sentiment_dist.items()},
            train_samples=len(train),
            val_samples=len(val),
            test_samples=len(test)
        )

        return train, val, test

    def get_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        self.logger.info("computing_dataset_statistics")

        orders_df, order_items_df, reviews_df = self.load_raw_data()

        customers_path = self.data_dir / "olist_customers_dataset.csv"
        customers_df = pd.read_csv(customers_path) if customers_path.exists() else None

        sellers_path = self.data_dir / "olist_sellers_dataset.csv"
        sellers_df = pd.read_csv(sellers_path) if sellers_path.exists() else None

        products_path = self.data_dir / "olist_products_dataset.csv"
        products_df = pd.read_csv(products_path) if products_path.exists() else None

        payments_path = self.data_dir / "olist_order_payments_dataset.csv"
        payments_df = pd.read_csv(payments_path) if payments_path.exists() else None

        orders_df['order_purchase_timestamp'] = pd.to_datetime(
            orders_df['order_purchase_timestamp']
        )

        stats = {
            'total_orders': len(orders_df),
            'total_customers': customers_df['customer_unique_id'].nunique() if customers_df is not None else 0,
            'total_sellers': len(sellers_df) if sellers_df is not None else 0,
            'total_products': len(products_df) if products_df is not None else 0,
            'total_reviews': len(reviews_df) if reviews_df is not None else 0,
            'date_range': {
                'start': str(orders_df['order_purchase_timestamp'].min()),
                'end': str(orders_df['order_purchase_timestamp'].max())
            },
            'order_status_distribution': orders_df['order_status'].value_counts().to_dict(),
            'total_revenue': float(payments_df['payment_value'].sum()) if payments_df is not None else 0,
            'avg_order_value': float(
                payments_df.groupby('order_id')['payment_value'].sum().mean()
            ) if payments_df is not None else 0,
            'avg_review_score': float(reviews_df['review_score'].mean()) if reviews_df is not None else 0,
            'payment_types': payments_df['payment_type'].value_counts().to_dict() if payments_df is not None else {}
        }

        return stats
