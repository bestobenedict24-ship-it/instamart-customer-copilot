import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib

def run_lapse_modeling():
    print("=== STARTING PART B: LAPSE PROBLEM & MODELING ===")
    
    # 1. Load clean data
    orders_df = pd.read_csv("data/clean/orders.csv")
    customers_df = pd.read_csv("data/clean/customers.csv")
    
    # Check if products table exists for category, else fallback
    try:
        products_df = pd.read_csv("data/clean/products.csv")
    except FileNotFoundError:
        products_df = None

    # Standardize datetime columns
    orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
    customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
    
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    
    # Merge product category if available
    if products_df is not None and 'category' in products_df.columns:
        orders_df = orders_df.merge(products_df[['product_id', 'category']], on='product_id', how='left')
    else:
        orders_df['category'] = 'Unknown'

    # 2. Define Cohort (Customers with at least one completed order BEFORE cutoff)
    pre_cutoff_orders = orders_df[orders_df['order_ts'] < cutoff]
    completed_pre = pre_cutoff_orders[pre_cutoff_orders['status'] == 'completed']
    
    cohort_customer_ids = completed_pre['customer_id'].unique()
    print(f"Cohort size (customers with >=1 completed order before cutoff): {len(cohort_customer_ids)}")
    
    # 3. Define Label (Lapsed = 1 if NO completed order between 2025-09-01 and 2025-09-30)
    sept_start = pd.to_datetime("2025-09-01 00:00:00")
    sept_end = pd.to_datetime("2025-09-30 23:59:59")
    
    sept_completed = orders_df[
        (orders_df['order_ts'] >= sept_start) & 
        (orders_df['order_ts'] <= sept_end) & 
        (orders_df['status'] == 'completed')
    ]['customer_id'].unique()
    
    labels = []
    for cid in cohort_customer_ids:
        lapsed = 0 if cid in sept_completed else 1
        labels.append({'customer_id': cid, 'lapsed': lapsed})
        
    labels_df = pd.DataFrame(labels)
    
    # 4. Feature Engineering (Strictly using pre-cutoff data)
    # Recency: days from cutoff to last completed order before cutoff
    recency = completed_pre.groupby('customer_id')['order_ts'].max().reset_index()
    recency['recency'] = (cutoff - recency['order_ts']).dt.days
    recency = recency[['customer_id', 'recency']]
    
    # Frequency & Monetary & Avg Discount & Avg Delivery Minutes
    order_metrics = completed_pre.groupby('customer_id').agg(
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum'),
        avg_discount=('discount_pct', 'mean'),
        avg_delivery=('delivery_minute', 'mean') if 'delivery_minute' in completed_pre.columns else ('delivery_minutes', 'mean')
    ).reset_index()
    
    # Rename delivery column safely
    if 'delivery_minute' in order_metrics.columns:
        order_metrics = order_metrics.rename(columns={'delivery_minute': 'avg_delivery'})
    elif 'delivery_minutes' in order_metrics.columns:
        order_metrics = order_metrics.rename(columns={'delivery_minutes': 'avg_delivery'})
    else:
        order_metrics['avg_delivery'] = 0.0

    # Cancellation rate: cancelled orders / all orders with final status before cutoff
    all_pre = pre_cutoff_orders[pre_cutoff_orders['status'].isin(['completed', 'cancelled'])].copy()
    all_pre['is_cancelled'] = (all_pre['status'] == 'cancelled').astype(int)
    cancel_metrics = all_pre.groupby('customer_id').agg(
        total_final_orders=('order_id', 'count'),
        cancelled_count=('is_cancelled', 'sum')
    ).reset_index()
    cancel_metrics['cancellation_rate'] = cancel_metrics['cancelled_count'] / cancel_metrics['total_final_orders']
    
    # Most frequent product category
    cat_mode = pre_cutoff_orders.groupby('customer_id')['category'].agg(
        lambda x: x.mode()[0] if not x.empty else 'Unknown'
    ).reset_index().rename(columns={'category': 'favourite_category'})
    
    # Customer demographic features & tenure
    cust_feat = customers_df[customers_df['customer_id'].isin(cohort_customer_ids)].copy()
    cust_feat['tenure_days'] = (cutoff - cust_feat['signup_date']).dt.days
    cust_feat = cust_feat[['customer_id', 'city', 'gender', 'age', 'tenure_days']]
    
    # Merge all features together
    features_df = pd.DataFrame({'customer_id': cohort_customer_ids})
    for df in [recency, order_metrics, cancel_metrics[['customer_id', 'cancellation_rate']], cat_mode, cust_feat]:
        features_df = features_df.merge(df, on='customer_id', how='left')
        
    dataset = features_df.merge(labels_df, on='customer_id')
    dataset = dataset.fillna({'avg_discount': 0, 'avg_delivery': dataset['avg_delivery'].mean(), 'cancellation_rate': 0})
    
    print(f"Total cohort dataset shape: {dataset.shape}")
    print(f"Lapsed counts:\n{dataset['lapsed'].value_counts(normalize=True)}")
    
    # 5. Train-Test Split (stratified)
    feature_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancellation_rate', 
                    'avg_delivery', 'favourite_category', 'city', 'gender', 'age', 'tenure_days']
    X = dataset[feature_cols]
    y = dataset['lapsed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
    
    # Baseline Metrics
    baseline_pred = np.zeros_like(y_test)
    base_acc = accuracy_score(y_test, baseline_pred)
    base_f1 = f1_score(y_test, baseline_pred, pos_label=1, zero_division=0)
    print(f"Baseline (Always Not Lapsed) -> Accuracy: {base_acc:.3f}, Lapsed F1: {base_f1:.3f}")
    
    # 6. Preprocessing Pipelines
    numeric_features = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancellation_rate', 'avg_delivery', 'age', 'tenure_days']
    categorical_features = ['favourite_category', 'city', 'gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Check frequency mean difference as required
    scaler_check = StandardScaler()
    scaler_check.fit(X_train[['frequency']])
    print(f"Learned mean of frequency (train set): {scaler_check.mean_[0]:.2f}")
    print(f"Full cohort mean of frequency: {X['frequency'].mean():.2f}")
    
    # Confirm no missing values
    assert X_train.isnull().sum().sum() == 0, "Missing values found in X_train!"
    assert X_test.isnull().sum().sum() == 0, "Missing values found in X_test!"
    print("Confirmed: Zero missing values in training and test sets.")
    
    # 7. Model Tuning: Logistic Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
    
    lr_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    lr_gcv = GridSearchCV(lr_pipeline, lr_grid, cv=5, scoring='f1')
    lr_gcv.fit(X_train, y_train)
    print(f"\nBest Logistic Regression Params: {lr_gcv.best_params_}, CV F1: {lr_gcv.best_score_:.3f}")
    
    # 8. Model Tuning: Random Forest
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])
    
    rf_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [3, 5, 8],
        'classifier__min_samples_leaf': [5, 10, 20]
    }
    rf_gcv = GridSearchCV(rf_pipeline, rf_grid, cv=5, scoring='f1')
    rf_gcv.fit(X_train, y_train)
    print(f"Best Random Forest Params: {rf_gcv.best_params_}, CV F1: {rf_gcv.best_score_:.3f}")
    
    # 9. Evaluation on Test Set
    print("\n--- Test Set Evaluation ---")
    for name, model in [("Logistic Regression", lr_gcv), ("Random Forest", rf_gcv)]:
        preds = model.predict(X_test)
        f1_lapsed = f1_score(y_test, preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        print(f"\n{name}:")
        print(f"Test Lapsed F1: {f1_lapsed:.3f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(y_test, preds, zero_division=0))

if __name__ == "__main__":
    run_lapse_modeling()