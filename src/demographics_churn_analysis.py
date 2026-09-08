import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run():
    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    demographics = ['gender', 'senior_citizen', 'partner', 'dependents']
    
    # Map senior citizen to Yes/No just for display if it's 1/0
    if df['senior_citizen'].dtype == 'int64':
        df['senior_citizen'] = df['senior_citizen'].map({1: 'Yes', 0: 'No'})
    
    for col in demographics:
        # Calculate stats
        stats = df.groupby(col)['churn'].value_counts().unstack().fillna(0)
        if 'Yes' not in stats.columns:
            stats['Yes'] = 0
        if 'No' not in stats.columns:
            stats['No'] = 0
            
        stats['Total'] = stats['Yes'] + stats['No']
        stats['Churn Rate (%)'] = (stats['Yes'] / stats['Total']) * 100
        
        print(f"--- {col.upper()} ---")
        print(stats[['Total', 'Yes', 'Churn Rate (%)']])
        
        # Plot
        plt.figure(figsize=(7, 5))
        
        # Barplot of churn rate
        # Using specific hue instead of passing palette directly to x to avoid warnings
        ax = sns.barplot(data=stats.reset_index(), x=col, y='Churn Rate (%)', hue=col, palette='Blues_d', legend=False)
        plt.title(f'Churn Rate by {col.replace("_", " ").title()}')
        plt.ylabel('Churn Rate (%)')
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylim(0, 50) # Set to 50 for better visibility since max is around ~45
        
        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(format(height, '.2f') + '%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')
                        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"churn_by_{col}.png"))
        plt.close()

if __name__ == '__main__':
    run()
