import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_rfm_and_clustering():
    print("=== STARTING PART B: RFM & CLUSTERING ===")
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Load clean orders and customers
    orders_df = pd.read_csv("data/clean/orders.csv")
    
    # Convert timestamp to datetime using the correct column name 'order_ts'
    orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
    
    # Filter for completed orders only using 'status'
    completed_orders = orders_df[orders_df['status'] == 'completed'].copy()
    
    # Snapshot moment for RFM
    snapshot = pd.to_datetime("2025-10-01 00:00:00")
    
    # 2. Compute RFM Table
    rfm = completed_orders.groupby('customer_id').agg(
        last_order_ts=('order_ts', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum')
    ).reset_index()
    
    rfm['recency'] = (snapshot - rfm['last_order_ts']).dt.days
    rfm = rfm[['customer_id', 'recency', 'frequency', 'monetary']]
    
    print(f"RFM computed for {len(rfm)} customers.")
    print(f"Recency median: {rfm['recency'].median()}, min: {rfm['recency'].min()}, max: {rfm['recency'].max()}")
    print(f"Frequency mean: {rfm['frequency'].mean():.2f}, max: {rfm['frequency'].max()}")
    print(f"Monetary mean: {rfm['monetary'].mean():.2f}, max: {rfm['monetary'].max()}")
    
    # 3. K-Means Clustering (K from 2 to 8)
    X = rfm[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- K-Means Tuning ---")
    inertia_scores = {}
    silhouette_scores = {}
    
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)
        inertia_scores[k] = inertia
        silhouette_scores[k] = sil
        print(f"K={k}: Inertia = {inertia:.1f} | Silhouette = {sil:.3f}")
        
    # Choose K = 4 based on silhouette and business interpretability
    chosen_k = 4
    print(f"\nChoosing K = {chosen_k} for final segments.")
    
    final_kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm['segment_id'] = final_kmeans.fit_predict(X_scaled)
    
    # 4. Profile and Name Segments
    profile = rfm.groupby('segment_id').agg(
        size=('customer_id', 'count'),
        recency=('recency', 'mean'),
        frequency=('frequency', 'mean'),
        monetary=('monetary', 'mean')
    ).reset_index()
    print("\n--- Segment Profiles ---")
    print(profile)
    
    # Identify Premium Segment (highest mean monetary value)
    premium_cluster_id = profile.loc[profile['monetary'].idxmax(), 'segment_id']
    
    # Assign business names and premium flag
    def assign_names_and_premium(row):
        s_id = row['segment_id']
        is_prem = (s_id == premium_cluster_id)
        if s_id == premium_cluster_id:
            name = "VIP High-Value"
        elif row['frequency'] > 15:
            name = "Loyal Regulars"
        elif row['recency'] > 60:
            name = "At-Risk Churners"
        else:
            name = "Recent / Casual"
        return pd.Series([name, is_prem])
    
    rfm[['segment_name', 'is_premium']] = rfm.apply(assign_names_and_premium, axis=1)
    
    # 5. PCA Analysis
    pca = PCA(n_components=2, random_state=42)
    pca.fit_transform(X_scaled)
    print(f"\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_} (Total: {pca.explained_variance_ratio_.sum():.3f})")
    
    # 6. Save Outputs
    output_path = "outputs/customer_segments.csv"
    rfm.to_csv(output_path, index=False)
    print(f"\nSaved customer segments successfully to {output_path}!")

if __name__ == "__main__":
    run_rfm_and_clustering()