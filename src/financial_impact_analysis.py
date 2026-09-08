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
    
    features = ['tenure', 'monthly_charges', 'total_charges']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    # Segment 0 is Transient (low tenure), Segment 1 is Premium (high tenure)
    # Let's verify by checking mean tenure
    mean_t_0 = df[df['segment']==0]['tenure'].mean()
    mean_t_1 = df[df['segment']==1]['tenure'].mean()
    if mean_t_0 < mean_t_1:
        df['segment_name'] = df['segment'].map({0: 'Transient Majority', 1: 'Premium Veterans'})
    else:
        df['segment_name'] = df['segment'].map({1: 'Transient Majority', 0: 'Premium Veterans'})
    
    # 1. Overall Financials
    total_cumulative_charges = df['total_charges'].sum()
    avg_monthly = df['monthly_charges'].mean()
    avg_total = df['total_charges'].mean()
    
    print(f"Total Cumulative Charges (All Time): ${total_cumulative_charges:,.2f}")
    print(f"Average Monthly Charge: ${avg_monthly:,.2f}")
    print(f"Average Cumulative Total Charge: ${avg_total:,.2f}")
    
    # 2. Churn Financials
    churned_df = df[df['churn'] == 'Yes']
    churned_cumulative_charges = churned_df['total_charges'].sum()
    avg_churned_total = churned_df['total_charges'].mean()
    monthly_churn_exposure = churned_df['monthly_charges'].sum()
    
    print(f"Cumulative Charges of Churned Customers: ${churned_cumulative_charges:,.2f}")
    print(f"Average Total Charges of Churned Customers: ${avg_churned_total:,.2f}")
    print(f"Monthly MRR Exposure (lost to churn): ${monthly_churn_exposure:,.2f}")
    
    # 3. By Contract
    contract_financials = df.groupby('contract').agg(
        total_cumulative_charges=('total_charges', 'sum'),
        avg_monthly=('monthly_charges', 'mean'),
        churned_monthly_exposure=('monthly_charges', lambda x: df.loc[x.index, 'monthly_charges'][df['churn']=='Yes'].sum())
    )
    print("\n--- FINANCIALS BY CONTRACT ---")
    print(contract_financials)
    
    # 4. By Segment
    segment_financials = df.groupby('segment_name').agg(
        total_cumulative_charges=('total_charges', 'sum'),
        avg_monthly=('monthly_charges', 'mean'),
        churned_monthly_exposure=('monthly_charges', lambda x: df.loc[x.index, 'monthly_charges'][df['churn']=='Yes'].sum())
    )
    print("\n--- FINANCIALS BY SEGMENT ---")
    print(segment_financials)
    
    # Visualizations
    retained_cumulative = total_cumulative_charges - churned_cumulative_charges
    plt.figure(figsize=(6, 6))
    plt.pie([retained_cumulative, churned_cumulative_charges], labels=['Retained Value', 'Lost Value (Churned)'], 
            colors=['#66b3ff', '#ff9999'], autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Cumulative Charges: Retained vs Lost')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "financial_cumulative_pie.png"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=segment_financials.index, y=segment_financials['churned_monthly_exposure'], palette='Reds_d')
    plt.title('Monthly Revenue Exposure by Segment (Lost to Churn)')
    plt.ylabel('Monthly Charges Exposure ($)')
    plt.xlabel('Customer Segment')
    for p in ax.patches:
        ax.annotate(f"${p.get_height():,.0f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "financial_exposure_by_segment.png"))
    plt.close()

if __name__ == '__main__':
    run()
