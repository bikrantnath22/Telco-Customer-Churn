import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', None)

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run():
    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    total = len(df)
    retained = len(df[df['churn'] == 'No'])
    overall_retention = (retained / total) * 100
    print(f"Overall Retention Rate: {overall_retention:.2f}%")
    
    bins = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ['0-6m', '7-12m', '13-24m', '25-36m', '37-48m', '49-60m', '61-72m']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    
    tenure_ret = df.groupby('tenure_group', observed=True)['churn'].apply(lambda x: (x=='No').mean() * 100)
    print("\nRetention by Tenure:\n", tenure_ret)
    
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(x=tenure_ret.index, y=tenure_ret.values, marker='o', linewidth=2.5, color='green')
    plt.title('Customer Retention Curve by Tenure Group')
    plt.ylabel('Retention Rate (%)')
    plt.xlabel('Tenure Bucket')
    plt.ylim(0, 100)
    for x, y in zip(range(len(tenure_ret)), tenure_ret.values):
        plt.text(x, y - 5, f"{y:.1f}%", ha='center', color='darkgreen', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "retention_by_tenure.png"))
    plt.close()
    
    contract_ret = df.groupby('contract')['churn'].apply(lambda x: (x=='No').mean() * 100).sort_values()
    print("\nRetention by Contract:\n", contract_ret)
    
    payment_ret = df.groupby('payment_method')['churn'].apply(lambda x: (x=='No').mean() * 100).sort_values()
    print("\nRetention by Payment Method:\n", payment_ret)
    
    security_ret = df.groupby('online_security')['churn'].apply(lambda x: (x=='No').mean() * 100)
    print("\nRetention by Online Security:\n", security_ret)

if __name__ == "__main__":
    run()
