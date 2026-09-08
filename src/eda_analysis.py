import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import CLEAN_DATA_PATH, FIGURES_DIR

def run_eda():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (8, 5)})

    df = pd.read_csv(CLEAN_DATA_PATH)
    figures_dir = str(FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Demographics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.countplot(data=df, x='gender', ax=axes[0,0], palette="Blues")
    axes[0,0].set_title('Gender Distribution')
    sns.countplot(data=df, x='senior_citizen', ax=axes[0,1], palette="Blues")
    axes[0,1].set_title('Senior Citizen Distribution')
    sns.countplot(data=df, x='partner', ax=axes[1,0], palette="Blues")
    axes[1,0].set_title('Partner Distribution')
    sns.countplot(data=df, x='dependents', ax=axes[1,1], palette="Blues")
    axes[1,1].set_title('Dependents Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "demographics.png"))
    plt.close()

    # 3. Tenure
    plt.figure()
    sns.histplot(data=df, x='tenure', bins=30, kde=True, color='skyblue')
    plt.title('Tenure Distribution (Months)')
    plt.xlabel('Tenure')
    plt.ylabel('Count')
    plt.savefig(os.path.join(figures_dir, "tenure_dist.png"))
    plt.close()

    # 4. Contract
    plt.figure()
    sns.countplot(data=df, x='contract', palette='Set2')
    plt.title('Contract Type Distribution')
    plt.savefig(os.path.join(figures_dir, "contract_dist.png"))
    plt.close()

    # 5. Internet
    plt.figure()
    sns.countplot(data=df, x='internet_service', palette='Set2')
    plt.title('Internet Service Distribution')
    plt.savefig(os.path.join(figures_dir, "internet_dist.png"))
    plt.close()

    # 6. Payment
    plt.figure(figsize=(10,5))
    sns.countplot(data=df, y='payment_method', palette='Set2')
    plt.title('Payment Method Distribution')
    plt.savefig(os.path.join(figures_dir, "payment_dist.png"))
    plt.tight_layout()
    plt.close()

    # 7. Monthly
    plt.figure()
    sns.histplot(data=df, x='monthly_charges', bins=30, kde=True, color='coral')
    plt.title('Monthly Charges Distribution')
    plt.savefig(os.path.join(figures_dir, "monthly_charges_dist.png"))
    plt.close()

    # 8. Total
    plt.figure()
    sns.histplot(data=df, x='total_charges', bins=30, kde=True, color='coral')
    plt.title('Total Charges Distribution')
    plt.savefig(os.path.join(figures_dir, "total_charges_dist.png"))
    plt.close()

    # 9. Services
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    yes_counts = [df[col].value_counts().get('Yes', 0) for col in services]
    plt.figure(figsize=(10,5))
    sns.barplot(x=services, y=yes_counts, palette='viridis')
    plt.title('Service Adoption ("Yes" Counts)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "service_adoption.png"))
    plt.close()

    # 10. Churn
    plt.figure()
    sns.countplot(data=df, x='churn', palette='Reds')
    plt.title('Overall Churn Distribution')
    plt.savefig(os.path.join(figures_dir, "churn_dist.png"))
    plt.close()

    print("Total customers:", len(df))
    print("\n--- Demographics ---")
    print(df['gender'].value_counts(normalize=True))
    print(df['senior_citizen'].value_counts(normalize=True))
    print(df['partner'].value_counts(normalize=True))
    print(df['dependents'].value_counts(normalize=True))
    print("\n--- Tenure ---")
    print(df['tenure'].describe())
    print("\n--- Contract ---")
    print(df['contract'].value_counts(normalize=True))
    print("\n--- Internet ---")
    print(df['internet_service'].value_counts(normalize=True))
    print("\n--- Payment ---")
    print(df['payment_method'].value_counts(normalize=True))
    print("\n--- Monthly ---")
    print(df['monthly_charges'].describe())
    print("\n--- Total ---")
    print(df['total_charges'].describe())
    print("\n--- Services ---")
    for col in services:
        print(f"{col}: {df[col].value_counts(normalize=True).get('Yes', 0):.2f}")
    print("\n--- Churn ---")
    print(df['churn'].value_counts(normalize=True))

if __name__ == "__main__":
    run_eda()
