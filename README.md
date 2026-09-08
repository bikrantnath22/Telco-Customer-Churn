# 📊 Telco Customer Churn Analysis

> A comprehensive end-to-end data analytics project using **Python**, **SQL**, and **Power BI** to analyze, understand, and reduce customer churn for a telecommunications company — based on the IBM Telco Customer Churn dataset.

---

## 🧠 Project Overview

Customer churn is one of the most critical challenges in the telecom industry. Losing a customer means losing their future revenue — and acquiring new ones is significantly more expensive than retaining existing ones.

This project performs a **full analytical pipeline** from raw data cleaning to visual business intelligence dashboards, including:

- Data cleaning and quality reporting
- Exploratory Data Analysis (EDA)
- Churn segmentation across demographics, services, billing, and contracts
- Customer segmentation using K-Means clustering
- Financial impact and revenue exposure analysis
- Retention curve analysis
- SQL database creation and BI-ready queries
- Predictive churn model with best-fit model selection (Random Forest, AUC = 0.846)

---

## 📂 Project Structure

```
Churn/
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv                  # Original IBM Telco dataset (~7,043 rows)
│   └── processed/
│       └── cleaned_dataset.csv          # Cleaned & standardized dataset
│
├── src/                                 # Python analysis scripts
│   ├── data_preprocessing.py            # Data cleaning pipeline
│   ├── eda_analysis.py                  # Exploratory Data Analysis
│   ├── core_churn_analysis.py           # Overall churn rate & pie chart
│   ├── demographics_churn_analysis.py   # Churn by gender, age, partner, dependents
│   ├── services_churn_analysis.py       # Churn by internet & phone services
│   ├── contract_churn_analysis.py       # Churn by contract type
│   ├── billing_churn_analysis.py        # Churn by billing method & monthly charges
│   ├── tenure_churn_analysis.py         # Churn by customer tenure
│   ├── numerical_analysis.py            # Numerical distribution analysis
│   ├── customer_segmentation.py         # K-Means segmentation (4 clusters)
│   ├── optimal_segmentation.py          # Elbow + Silhouette for optimal K
│   ├── financial_impact_analysis.py     # Revenue exposure & financial analysis
│   └── retention_analysis.py           # Retention curve by tenure & contract
│
├── sql/
│   ├── create_database.sql              # MySQL schema, LOAD DATA, validation queries
│   └── retention_analysis.sql           # SQL-based retention rate queries
│
├── notebooks/
│   └── telco_churn_analysis.ipynb       # Full narrative analysis notebook
│
├── models/
│   └── best_model.pkl                   # Saved best predictive model (Random Forest)
│
├── reports/
│   └── figures/                         # 45 auto-generated charts & visualizations
│
├── docs/                                # (Planned) Documentation
│
├── config.py                            # Central path configuration (no hardcoded paths)
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Source** | IBM Telco Customer Churn (public dataset) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target Variable** | `Churn` (Yes / No) |

### Key Features

| Category | Columns |
|----------|---------|
| **Demographics** | `gender`, `senior_citizen`, `partner`, `dependents` |
| **Account Info** | `customer_id`, `tenure`, `contract`, `paperless_billing`, `payment_method` |
| **Financial** | `monthly_charges`, `total_charges` |
| **Services** | `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies` |
| **Label** | `churn` |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core scripting & analysis |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |
| **Scikit-learn** | K-Means clustering, Random Forest, StandardScaler, Silhouette Score |
| **Joblib** | Model serialization (save/load best model) |
| **MySQL** | Relational database for BI pipelines |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telco-churn-analysis.git
cd telco-churn-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Data Cleaning Pipeline

This is the **first step** — always run this before any analysis script.

```bash
python src/data_preprocessing.py
```

This will:
- Load `data/raw/Dataset.csv`
- Standardize column names to snake_case
- Fix the `total_charges` data type (string → numeric)
- Handle 11 missing values for new customers (tenure = 0)
- Validate logical relationships
- Save the cleaned output to `data/processed/cleaned_dataset.csv`

### 4. Run Analysis Scripts

Run any script individually from the `src/` directory:

```bash
python src/eda_analysis.py
python src/core_churn_analysis.py
python src/demographics_churn_analysis.py
python src/services_churn_analysis.py
python src/contract_churn_analysis.py
python src/billing_churn_analysis.py
python src/tenure_churn_analysis.py
python src/numerical_analysis.py
python src/customer_segmentation.py
python src/optimal_segmentation.py
python src/financial_impact_analysis.py
python src/retention_analysis.py
python src/predictive_model.py
```

All outputs (charts) are automatically saved to `reports/figures/`.

---

## 🗄️ SQL Database Setup

### 1. Create Database and Load Data

```sql
SOURCE sql/create_database.sql;
```

This creates the `telco_churn_analytics` database, the `customer_churn` table with proper types and indexes, and loads the cleaned CSV into it.

> **Important:** Update the file path in `create_database.sql` to match your local path before running.

### 2. Run Retention Queries

```sql
SOURCE sql/retention_analysis.sql;
```

Provides retention rate breakdowns by overall rate, contract type, tenure group, and payment method.

---

## 📈 Analysis Modules

### 1. Data Preprocessing (`data_preprocessing.py`)
Professional cleaning pipeline with a full data quality report — column standardization, type fixing, missing value imputation, duplicate checks, and logical validation.

### 2. Exploratory Data Analysis (`eda_analysis.py`)
Statistical summaries and distribution charts for all 20 features.

### 3. Core Churn Analysis (`core_churn_analysis.py`)
Overall churn rate, retention rate, and pie chart visualization.

### 4. Demographics Churn Analysis (`demographics_churn_analysis.py`)
Churn rates by gender, senior citizen status, partner, and dependents.

### 5. Services Churn Analysis (`services_churn_analysis.py`)
Churn rates for all 9 telecom add-on services.

### 6. Contract Churn Analysis (`contract_churn_analysis.py`)
Churn by Month-to-Month, One Year, and Two Year contracts.

### 7. Billing Churn Analysis (`billing_churn_analysis.py`)
Churn by paperless billing, payment method, and monthly charge ranges.

### 8. Tenure Churn Analysis (`tenure_churn_analysis.py`)
Churn rates and box plots across customer tenure groups.

### 9. Customer Segmentation (`customer_segmentation.py`)
K-Means clustering (K=4) on tenure, monthly charges, and total charges.

### 10. Optimal Segmentation (`optimal_segmentation.py`)
Elbow Method + Silhouette Score to find the statistically best K.

### 11. Financial Impact Analysis (`financial_impact_analysis.py`)
Total revenue, MRR exposure lost to churn, breakdown by contract and segment.

### 12. Retention Analysis (`retention_analysis.py`)
Retention curve by tenure bucket (0–72 months), contract, and payment method.

### 13. Predictive Model (`predictive_model.py`)
Best-fit model selection pipeline — trains 6 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, KNN), cross-validates all with 5-fold CV, auto-selects the best by ROC-AUC, and outputs confusion matrix, ROC curve, feature importance, and performance summary. **Best model: Random Forest (AUC = 0.846, Recall = 77.5%)**.

---

## 📊 Generated Visualizations (41 charts)

All charts saved to `reports/figures/`:

| Chart | Description |
|-------|-------------|
| `core_churn_pie.png` | Overall churn vs retention |
| `churn_dist.png` | Churn count distribution |
| `demographics.png` | 2x2 demographics panel |
| `churn_by_gender.png` | Churn rate by gender |
| `churn_by_senior_citizen.png` | Churn rate by senior status |
| `churn_by_partner.png` | Churn rate by partner status |
| `churn_by_dependents.png` | Churn rate by dependents |
| `churn_by_internet_service.png` | Churn rate by internet type |
| `churn_by_online_security.png` | Churn rate by online security |
| `churn_by_online_backup.png` | Churn rate by online backup |
| `churn_by_device_protection.png` | Churn rate by device protection |
| `churn_by_tech_support.png` | Churn rate by tech support |
| `churn_by_streaming_tv.png` | Churn rate by streaming TV |
| `churn_by_streaming_movies.png` | Churn rate by streaming movies |
| `churn_by_multiple_lines.png` | Churn rate by multiple lines |
| `churn_by_phone_service.png` | Churn rate by phone service |
| `churn_by_contract.png` | Churn rate by contract type |
| `churn_by_paperless_billing.png` | Churn rate by paperless billing |
| `churn_by_payment_method.png` | Churn rate by payment method |
| `churn_by_monthly_bins.png` | Churn rate by charge range |
| `churn_by_tenure_group.png` | Churn rate by tenure group |
| `tenure_by_churn.png` | Tenure distribution vs churn |
| `tenure_by_contract.png` | Tenure by contract type |
| `tenure_dist.png` | Overall tenure distribution |
| `tenure_overall.png` | Tenure summary |
| `monthly_charges_dist.png` | Monthly charges distribution |
| `monthly_charges_by_churn.png` | Monthly charges: churned vs retained |
| `monthly_charges_overall.png` | Monthly charges summary |
| `total_charges_dist.png` | Total charges distribution |
| `total_charges_by_churn.png` | Total charges: churned vs retained |
| `total_charges_overall.png` | Total charges summary |
| `contract_dist.png` | Contract type distribution |
| `internet_dist.png` | Internet service distribution |
| `payment_dist.png` | Payment method distribution |
| `service_adoption.png` | Add-on service adoption counts |
| `customer_segments_scatter.png` | K-Means 4-cluster scatter plot |
| `optimal_segments_scatter.png` | Optimal K cluster scatter plot |
| `segmentation_evaluation.png` | Elbow + Silhouette evaluation |
| `retention_by_tenure.png` | Retention curve by tenure group |
| `financial_cumulative_pie.png` | Revenue: retained vs churned |
| `financial_exposure_by_segment.png` | Monthly MRR exposure by segment |

---

## 🔑 Key Findings

- 📉 **Overall churn rate is ~26.5%** — roughly 1 in 4 customers leaves
- 📄 **Month-to-month contracts** have drastically higher churn than annual/biennial contracts
- 💳 **Electronic check users** churn at the highest rate of all payment methods
- 🌐 **Fiber optic internet** customers churn more than DSL or no-internet customers
- 🔒 Customers **without online security or tech support** are far more likely to churn
- 👴 **Senior citizens** churn at ~2x the rate of non-seniors
- 📅 **New customers (0–6 months tenure)** have the highest churn risk
- 💰 **Higher monthly charges** (>$70/month) correlate with significantly higher churn
- 🏆 Long-term customers (two-year contracts, 60+ months tenure) show near-zero churn

---

## 🗺️ Roadmap

- [x] Data cleaning & preprocessing pipeline
- [x] Exploratory Data Analysis (EDA)
- [x] Churn analysis by all categorical features
- [x] Numerical distribution analysis
- [x] Customer segmentation (K-Means)
- [x] Optimal cluster selection (Elbow + Silhouette)
- [x] Financial impact & MRR exposure analysis
- [x] Retention curve analysis
- [x] SQL database schema + BI queries
- [x] Jupyter notebook with full narrative (11 sections)
- [x] Predictive churn model — Random Forest (AUC = 0.846, Recall = 77.5%)
- [ ] Power BI dashboard
- [ ] Retention strategy recommendations report

---


---

## 👤 Author

**Bikrant** 

*Dataset Source: IBM Telco Customer Churn (publicly available on Kaggle)*
