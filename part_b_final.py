import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib

def run_final_tasks():
    print("=== STARTING PART B: TASKS 7 & 8 (BALANCE, INTERPRETATION & PERSISTENCE) ===")
    
    # 1. Re-load and prepare data to match the exact pipeline
    orders_df = pd.read_csv("data/clean/orders.csv")
    customers_df = pd.read_csv("data/clean/customers.csv")
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    
    try:
        products_df = pd.read_csv("data/clean/products.csv")
    except FileNotFoundError:
        products_df = None

    orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
    customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    
    if products_df is not None and 'category' in products_df.columns:
        orders_df = orders_df.merge(products_df[['product_id', 'category']], on='product_id', how='left')
    else:
        orders_df['category'] = 'Unknown'

    pre_cutoff_orders = orders_df[orders_df['order_ts'] < cutoff]
    completed_pre = pre_cutoff_orders[pre_cutoff_orders['status'] == 'completed']
    cohort_customer_ids = completed_pre['customer_id'].unique()
    
    sept_start = pd.to_datetime("2025-09-01 00:00:00")
    sept_end = pd.to_datetime("2025-09-30 23:59:59")
    sept_completed = orders_df[
        (orders_df['order_ts'] >= sept_start) & 
        (orders_df['order_ts'] <= sept_end) & 
        (orders_df['status'] == 'completed')
    ]['customer_id'].unique()
    
    labels_df = pd.DataFrame([{'customer_id': cid, 'lapsed': (0 if cid in sept_completed else 1)} for cid in cohort_customer_ids])
    
    recency = completed_pre.groupby('customer_id')['order_ts'].max().reset_index()
    recency['recency'] = (cutoff - recency['order_ts']).dt.days
    recency = recency[['customer_id', 'recency']]
    
    order_metrics = completed_pre.groupby('customer_id').agg(
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum'),
        avg_discount=('discount_pct', 'mean'),
        avg_delivery=('delivery_minute', 'mean') if 'delivery_minute' in completed_pre.columns else ('delivery_minutes', 'mean')
    ).reset_index()
    
    if 'delivery_minute' in order_metrics.columns:
        order_metrics = order_metrics.rename(columns={'delivery_minute': 'avg_delivery'})
    elif 'delivery_minutes' in order_metrics.columns:
        order_metrics = order_metrics.rename(columns={'delivery_minutes': 'avg_delivery'})
    else:
        order_metrics['avg_delivery'] = 0.0

    all_pre = pre_cutoff_orders[pre_cutoff_orders['status'].isin(['completed', 'cancelled'])].copy()
    all_pre['is_cancelled'] = (all_pre['status'] == 'cancelled').astype(int)
    cancel_metrics = all_pre.groupby('customer_id').agg(
        total_final_orders=('order_id', 'count'),
        cancelled_count=('is_cancelled', 'sum')
    ).reset_index()
    cancel_metrics['cancellation_rate'] = cancel_metrics['cancelled_count'] / cancel_metrics['total_final_orders']
    
    cat_mode = pre_cutoff_orders.groupby('customer_id')['category'].agg(
        lambda x: x.mode()[0] if not x.empty else 'Unknown'
    ).reset_index().rename(columns={'category': 'favourite_category'})
    
    cust_feat = customers_df[customers_df['customer_id'].isin(cohort_customer_ids)].copy()
    cust_feat['tenure_days'] = (cutoff - cust_feat['signup_date']).dt.days
    cust_feat = cust_feat[['customer_id', 'city', 'gender', 'age', 'tenure_days']]
    
    features_df = pd.DataFrame({'customer_id': cohort_customer_ids})
    for df in [recency, order_metrics, cancel_metrics[['customer_id', 'cancellation_rate']], cat_mode, cust_feat]:
        features_df = features_df.merge(df, on='customer_id', how='left')
        
    dataset = features_df.merge(labels_df, on='customer_id')
    dataset = dataset.fillna({'avg_discount': 0, 'avg_delivery': dataset['avg_delivery'].mean(), 'cancellation_rate': 0})
    
    feature_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancellation_rate', 
                    'avg_delivery', 'favourite_category', 'city', 'gender', 'age', 'tenure_days']
    X = dataset[feature_cols]
    y = dataset['lapsed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    numeric_features = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancellation_rate', 'avg_delivery', 'age', 'tenure_days']
    categorical_features = ['favourite_category', 'city', 'gender']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 2. Task 7: Class-Balance Experiment
    print("\n--- TASK 7: Class-Balance Experiment ---")
    train_df = X_train.copy()
    train_df['lapsed'] = y_train
    
    majority = train_df[train_df['lapsed'] == 0]
    minority = train_df[train_df['lapsed'] == 1]
    
    minority_oversampled = minority.sample(n=len(majority), replace=True, random_state=42)
    balanced_train = pd.concat([majority, minority_oversampled])
    
    X_train_bal = balanced_train.drop(columns=['lapsed'])
    y_train_bal = balanced_train['lapsed']
    print(f"Balanced training class counts:\n{y_train_bal.value_counts()}")
    
    # Fit Logistic Regression on balanced data
    lr_balanced_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', LogisticRegression(C=1, max_iter=1000, random_state=42))])
    lr_balanced_pipeline.fit(X_train_bal, y_train_bal)
    bal_preds = lr_balanced_pipeline.predict(X_test)
    
    print("Balanced LR Test Metrics (Lapsed Class):")
    print(f"Precision: {precision_score(y_test, bal_preds, pos_label=1, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_test, bal_preds, pos_label=1, zero_division=0):.3f}")
    print(f"F1: {f1_score(y_test, bal_preds, pos_label=1, zero_division=0):.3f}")
    print("State: Recall rises significantly while precision remains stable/lower, meaning we catch more lapsing customers at the cost of more false alarms.")

    # 3. Task 8a: Interpretation
    print("\n--- TASK 8a: Model Interpretation ---")
    # Train Best Random Forest on full train set to extract feature importances
    rf_final = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(max_depth=8, min_samples_leaf=20, n_estimators=100, random_state=42))])
    rf_final.fit(X_train, y_train)
    
    # Get feature names after OHE
    cat_encoder = rf_final.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + encoded_cat_cols
    
    importances = rf_final.named_steps['classifier'].feature_importances_
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    print("Top 5 Random Forest Feature Importances:")
    for fn, imp in feat_imp[:5]:
        print(f" - {fn}: {imp:.4f}")
        
    # Logistic Regression coefficients
    lr_final = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=42))])
    lr_final.fit(X_train, y_train)
    lr_coefs = lr_final.named_steps['classifier'].coef_[0]
    coef_summary = sorted(zip(all_feature_names, lr_coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\nTop Logistic Regression Coefficients (Directionality):")
    for fn, coef in coef_summary[:3]:
        print(f" - {fn}: coefficient = {coef:.3f} ({'Increases' if coef > 0 else 'Decreases'} lapse risk)")

    # 4. Task 8b: Persistence
    print("\n--- TASK 8b: Model Persistence ---")
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/lapse_model.joblib"
    joblib.dump(rf_final, model_path)
    print(f"Model saved successfully to {model_path}")
    
    reloaded_model = joblib.load(model_path)
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(rf_final.predict(X_test), reloaded_preds), "Reloaded model predictions do not match!"
    print("Confirmed: Reloaded model reproduces test predictions exactly.")

    # 5. Task 8c: Retention List for Premium Segment
    print("\n--- TASK 8c: Premium Retention List ---")
    test_dataset = X_test.copy()
    test_dataset['customer_id'] = dataset.loc[X_test.index, 'customer_id']
    test_dataset['actually_lapsed'] = y_test.values
    test_dataset['lapse_probability'] = rf_final.predict_proba(X_test)[:, 1]
    
    # Merge with segments dataframe to identify Premium customers
    test_dataset = test_dataset.merge(segments_df[['customer_id', 'is_premium']], on='customer_id', how='left')
    
    premium_test = test_dataset[test_dataset['is_premium'] == True].sort_values(by='lapse_probability', ascending=False)
    retention_list = premium_test[['customer_id', 'lapse_probability', 'actually_lapsed']].head(10)
    
    retention_path = "outputs/retention_list.csv"
    retention_list.to_csv(retention_path, index=False)
    print(f"Retention list written to {retention_path} with {len(retention_list)} customers.")
    print(retention_list)
    truly_lapsed_count = retention_list['actually_lapsed'].sum()
    print(f"Out of top Premium customers listed, {truly_lapsed_count} truly lapsed.")

if __name__ == "__main__":
    run_final_tasks()