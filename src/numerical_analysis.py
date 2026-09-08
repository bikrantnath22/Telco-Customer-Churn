import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run_numerical_analysis():
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    num_cols = ['tenure', 'monthly_charges', 'total_charges']

    print("--- Numerical Statistics ---")
    stats_df = df[num_cols].describe()
    stats_df.loc['IQR'] = stats_df.loc['75%'] - stats_df.loc['25%']
    print(stats_df)

    # 2. Distributions & Outliers
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'{col} Distribution')
        
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'{col} Boxplot (Outlier Check)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{col}_overall.png"))
        plt.close()

    # 3. Compare by Churn
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.boxplot(data=df, x='churn', y=col, ax=axes[0], palette={'Yes':'red', 'No':'blue'})
        axes[0].set_title(f'{col} by Churn')
        
        sns.kdeplot(data=df, x=col, hue='churn', common_norm=False, fill=True, ax=axes[1], palette={'Yes':'red', 'No':'blue'})
        axes[1].set_title(f'{col} Density by Churn')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{col}_by_churn.png"))
        plt.close()
        
    print("\n--- Means by Churn ---")
    print(df.groupby('churn')[num_cols].mean())
    print("\n--- Medians by Churn ---")
    print(df.groupby('churn')[num_cols].median())

if __name__ == "__main__":
    run_numerical_analysis()
