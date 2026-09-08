import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run():
    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Select features for clustering
    features = ['tenure', 'monthly_charges', 'total_charges']
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find Optimal K
    inertias = []
    sil_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
        
    # Plot Elbow & Silhouette
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_range, inertias, marker='o', color='b', label='Inertia (Elbow)')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(k_range, sil_scores, marker='s', color='r', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.title('Optimal K Evaluation (Elbow & Silhouette)')
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "segmentation_evaluation.png"))
    plt.close()
    
    # Select Best K based on Silhouette
    optimal_k = k_range[np.argmax(sil_scores)]
    print(f"--- OPTIMAL K FOUND: {optimal_k} ---")
    for k, s in zip(k_range, sil_scores):
        print(f"K={k}: Silhouette={s:.4f}, Inertia={inertias[k-2]:.0f}")
    
    # Fit Final KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Cluster statistics
    cluster_stats = df.groupby('cluster').agg(
        tenure=('tenure', 'mean'),
        monthly_charges=('monthly_charges', 'mean'),
        total_charges=('total_charges', 'mean'),
        count=('tenure', 'count'),
        churn_rate=('churn', lambda x: (x == 'Yes').mean() * 100)
    ).round(2)
    
    cat_features = ['contract', 'internet_service', 'payment_method']
    for cat in cat_features:
        cluster_stats[f'top_{cat}'] = df.groupby('cluster')[cat].agg(lambda x: x.mode()[0])
        
    print("\n--- CLUSTER PROFILES ---")
    print(cluster_stats)
    
    # Scatter plot of segments
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tenure', y='monthly_charges', hue='cluster', palette='tab10', alpha=0.6, edgecolor=None)
    plt.title(f'Customer Segments (K={optimal_k}): Tenure vs Monthly Charges')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "optimal_segments_scatter.png"))
    plt.close()

if __name__ == '__main__':
    run()
