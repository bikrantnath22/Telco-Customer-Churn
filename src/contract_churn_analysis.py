import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run():
    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    total_customers = len(df)
    
    grouped = df.groupby('contract')
    
    counts = grouped.size()
    pct = (counts / total_customers) * 100
    churn_counts = grouped['churn'].apply(lambda x: (x == 'Yes').sum())
    churn_rate = (churn_counts / counts) * 100
    
    avg_tenure = grouped['tenure'].mean()
    avg_monthly = grouped['monthly_charges'].mean()
    avg_total = grouped['total_charges'].mean()
    
    stats_df = pd.DataFrame({
        'Total Customers': counts,
        '% of Customers': pct,
        'Churned': churn_counts,
        'Churn Rate (%)': churn_rate,
        'Avg Tenure': avg_tenure,
        'Avg Monthly ($)': avg_monthly,
        'Avg Total ($)': avg_total
    })
    
    print("--- CONTRACT STATS ---")
    print(stats_df)
    
    # 1. Churn Rate Barplot
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=stats_df.reset_index(), x='contract', y='Churn Rate (%)', hue='contract', palette='Set1', legend=False)
    plt.title('Churn Rate by Contract Type')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel('Contract Type')
    plt.ylim(0, 55)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format(height, '.1f') + '%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "churn_by_contract.png"))
    plt.close()

    # 2. Tenure Boxplot
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='contract', y='tenure', hue='contract', palette='Set1', order=['Month-to-month', 'One year', 'Two year'], legend=False)
    plt.title('Tenure Distribution by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Tenure (Months)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "tenure_by_contract.png"))
    plt.close()

if __name__ == '__main__':
    run()
