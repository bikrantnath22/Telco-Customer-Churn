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
    
    bins = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ['0-6 months', '7-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61-72 months']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    
    grouped = df.groupby('tenure_group', observed=True)
    counts = grouped.size()
    churn_counts = grouped['churn'].apply(lambda x: (x == 'Yes').sum())
    churn_rate = (churn_counts / counts) * 100
    avg_monthly = grouped['monthly_charges'].mean()
    avg_total = grouped['total_charges'].mean()
    
    stats_df = pd.DataFrame({
        'Total Customers': counts,
        'Churned': churn_counts,
        'Churn Rate (%)': churn_rate,
        'Avg Monthly ($)': avg_monthly,
        'Avg Total ($)': avg_total
    })
    
    print("--- TENURE GROUPS ---")
    print(stats_df)
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=stats_df.reset_index(), x='tenure_group', y='Churn Rate (%)', hue='tenure_group', palette='viridis', legend=False)
    plt.title('Churn Rate by Tenure Group')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel('Tenure Group')
    plt.ylim(0, 60)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format(height, '.1f') + '%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "churn_by_tenure_group.png"))
    plt.close()

if __name__ == '__main__':
    run()
