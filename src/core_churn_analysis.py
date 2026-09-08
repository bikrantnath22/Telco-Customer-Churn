import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import os

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run():
    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    
    total = len(df)
    churned = len(df[df['churn'] == 'Yes'])
    non_churned = len(df[df['churn'] == 'No'])
    churn_rate = (churned / total) * 100
    retention_rate = (non_churned / total) * 100
    
    print(f"Total: {total}")
    print(f"Churned: {churned}")
    print(f"Non-Churned: {non_churned}")
    print(f"Churn Rate: {churn_rate:.2f}%")
    print(f"Retention Rate: {retention_rate:.2f}%")
    
    # Pie chart visualization
    plt.figure(figsize=(6, 6))
    labels = ['Retained (No)', 'Churned (Yes)']
    sizes = [non_churned, churned]
    colors = ['#66b3ff', '#ff9999']
    explode = (0, 0.1)  
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=90, textprops={'fontsize': 14})
    plt.title('Overall Customer Churn Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "core_churn_pie.png"))
    plt.close()

if __name__ == '__main__':
    run()
