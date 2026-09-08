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
    
    cols = ['paperless_billing', 'payment_method']
    
    for col in cols:
        grouped = df.groupby(col)
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
        print(f"--- {col.upper()} ---")
        print(stats_df)
        
        plt.figure(figsize=(9, 5))
        ax = sns.barplot(data=stats_df.reset_index(), x=col, y='Churn Rate (%)', hue=col, palette='muted', legend=False)
        plt.title(f'Churn Rate by {col.replace("_", " ").title()}')
        plt.ylabel('Churn Rate (%)')
        plt.ylim(0, 55)
        if col == 'payment_method':
            plt.xticks(rotation=15)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(format(height, '.1f') + '%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"churn_by_{col}.png"))
        plt.close()
        
    # Bins
    bins = [0, 30, 50, 70, 90, 120]
    labels = ['< $30', '$30-50', '$50-70', '$70-90', '>$90']
    df['monthly_charge_bin'] = pd.cut(df['monthly_charges'], bins=bins, labels=labels)
    
    grouped = df.groupby('monthly_charge_bin', observed=True)
    counts = grouped.size()
    churn_counts = grouped['churn'].apply(lambda x: (x == 'Yes').sum())
    churn_rate = (churn_counts / counts) * 100
    
    stats_df_bins = pd.DataFrame({
        'Total Customers': counts,
        'Churned': churn_counts,
        'Churn Rate (%)': churn_rate
    })
    print("\n--- MONTHLY CHARGES BINS ---")
    print(stats_df_bins)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=stats_df_bins.reset_index(), x='monthly_charge_bin', y='Churn Rate (%)', hue='monthly_charge_bin', palette='coolwarm', legend=False)
    plt.title('Churn Rate by Monthly Charge Ranges')
    plt.ylabel('Churn Rate (%)')
    plt.xlabel('Monthly Charge Range')
    plt.ylim(0, 55)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format(height, '.1f') + '%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"churn_by_monthly_bins.png"))
    plt.close()

if __name__ == '__main__':
    run()
