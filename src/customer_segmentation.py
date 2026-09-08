import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
    
    # Fit KMeans (4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Cluster statistics
    cluster_stats = df.groupby('cluster').agg(
        tenure=('tenure', 'mean'),
        monthly_charges=('monthly_charges', 'mean'),
        total_charges=('total_charges', 'mean'),
        count=('tenure', 'count'),
        churn_rate=('churn', lambda x: (x == 'Yes').mean() * 100)
    ).round(2)
    
    # Add most frequent categorical values
    cat_features = ['contract', 'internet_service', 'payment_method']
    for cat in cat_features:
        cluster_stats[f'top_{cat}'] = df.groupby('cluster')[cat].agg(lambda x: x.mode()[0])
        
    print("--- CLUSTER ANALYSIS ---")
    print(cluster_stats)
    
    # Visualization: Scatter plot of segments
    plt.figure(figsize=(10, 6))
    
    # Rename clusters for legend mapping
    df['Segment'] = df['cluster'].map({
        0: 'Segment 0',
        1: 'Segment 1',
        2: 'Segment 2',
        3: 'Segment 3'
    })
    
    sns.scatterplot(data=df, x='tenure', y='monthly_charges', hue='Segment', palette='Set1', alpha=0.6, edgecolor=None)
    plt.title('Customer Segmentation: Tenure vs Monthly Charges')
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Monthly Charges ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "customer_segments_scatter.png"))
    plt.close()

if __name__ == '__main__':
    run()
