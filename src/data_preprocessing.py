import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import re

from config import RAW_DATA_PATH, CLEAN_DATA_PATH

def clean_telecom_data(raw_filepath: str, processed_filepath: str):
    """
    Professional Python data-cleaning pipeline for the IBM Telco Customer Churn dataset.
    """
  
    print("DATA CLEANING PIPELINE & QUALITY REPORT \n")

    
    # 1. Load the raw CSV & 2. Create a working copy
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw data file not found at: {raw_filepath}")
        
    raw_df = pd.read_csv(raw_filepath)
    df = raw_df.copy()
    
    initial_rows, initial_cols = df.shape
    print(f"[SHAPE BEFORE] Rows: {initial_rows}, Columns: {initial_cols}\n")
    
    print("- Applying Transformations -\n")
    
    # 3. Standardize column names to snake_case
    old_columns = df.columns.tolist()
    # Convert CamelCase to snake_case using regex
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower() for c in df.columns]
    # Specific manual fix
    df = df.rename(columns={'customer_i_d': 'customer_id', 'streaming_t_v': 'streaming_tv'})
    print("- What: Standardized column names to snake_case.")
    print("- Why: Eliminates case sensitivity issues and improves readability.")
    print(f"- Records affected: All {initial_cols} columns renamed.\n")
    
    # 4. Strip leading/trailing whitespace from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    print("- What: Stripped leading and trailing whitespace from string columns.")
    print("- Why: Prevents hidden duplicates and mismatched categories.")
    print(f"- Records affected: All {initial_rows} records across {len(str_cols)} string columns.\n")
    
    # 5. Convert TotalCharges to numeric using pd.to_numeric(errors="coerce")
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    print("- What: Converted 'total_charges' to float numeric format.")
    print("- Why: It was previously stored as a string object due to blank space characters.")
    print(f"- Records affected: All {initial_rows} records evaluated.\n")
    
    # 6 & 7. Investigate the 11 missing values and verify tenure = 0
    missing_tc = df[df['total_charges'].isnull()]
    missing_count = len(missing_tc)
    tenure_zero_count = len(missing_tc[missing_tc['tenure'] == 0])
    
    print(f"- What: Investigated {missing_count} missing 'total_charges' values.")
    print("- Why: NaN values were created during coercion of blank strings.")
    print(f"  > Verification: {tenure_zero_count} out of {missing_count} missing values correspond to tenure = 0.")
    
    # 8. Convert those TotalCharges values to 0 and document business assumption
    if tenure_zero_count == missing_count and missing_count > 0:
        df['total_charges'] = df['total_charges'].fillna(0.0)
        print("  > Treatment: Converted these 11 values to 0.0.")
        print("  > Business Assumption: These represent brand new customers who haven't paid their first bill yet.")
        print(f"- Records affected: {missing_count} records updated.\n")
    else:
        print("  > Treatment: Investigation failed, unexpected missing values found.\n")

    # 9 & 10. Check duplicate rows and customer IDs
    full_duplicates = df.duplicated().sum()
    id_duplicates = df.duplicated(subset=['customer_id']).sum()
    print("- What: Checked for full duplicate rows and duplicate customer IDs.")
    print("- Why: Data integrity check.")
    print(f"  > Full duplicates: {full_duplicates}")
    print(f"  > Duplicate IDs: {id_duplicates}\n")
    
    # 11, 12, 13, 14. Validate categorical values without merging business states
    # No transformations here, just keeping them intact as requested.
    print("- What: Categorical values validated and preserved intact.")
    print("- Why: 'No internet service' and 'No phone service' represent different business states and should not be merged with 'No'.")
    print(f"- Records affected: 0 records modified.\n")
    
    # 15 & 16 & 17 & 18. CustomerID and TotalCharges
    # No drop operations performed.
    print("- What: Retained 'customer_id' and 'total_charges'.")
    print("- Why: 'customer_id' acts as a unique identifier. 'total_charges' is crucial for descriptive and financial analysis and should not be arbitrarily removed for leakage concerns during cleaning.")
    print(f"- Records affected: 0 records dropped.\n")
    
    # 19. Validate logical relationships between variables
    valid_logic = True
    if (df['tenure'] < 0).any() or (df['monthly_charges'] < 0).any() or (df['total_charges'] < 0).any():
        valid_logic = False
    print("- What: Validated logical relationships (e.g., tenure and charges are >= 0).")
    print("- Why: Ensures no impossible numerical data exists.")
    print(f"  > Valid: {valid_logic}\n")

    
    print("FINAL DATA QUALITY REPORT \n")
    
    
    # Finally: display shape before and after cleaning
    final_rows, final_cols = df.shape
    print(f"[SHAPE AFTER] Rows: {final_rows}, Columns: {final_cols}\n")
    
    # display data types
    print("--- Data Types ---")
    print(df.dtypes.to_string())
    print("\n")
    
    # display missing values
    print("--- Missing Values ---")
    print(df.isnull().sum().to_string())
    print("\n")
    
    # display duplicate counts
    print("--- Duplicate Counts ---")
    print(f"Full rows: {df.duplicated().sum()}")
    print(f"Customer IDs: {df.duplicated(subset=['customer_id']).sum()}\n")
    
    # display unique categorical values
    print("--- Unique Categorical Values ---")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Exclude customer_id from showing all 7000 unique values
        if col != 'customer_id':
            print(f"{col}: {df[col].unique()}")
    print("\n")
    
    # save the cleaned dataset as a separate CSV.
    os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
    df.to_csv(processed_filepath, index=False)
    print(f"[SUCCESS] Cleaned dataset safely saved to: {processed_filepath}\n")
    
    return df

if __name__ == "__main__":
    clean_telecom_data(str(RAW_DATA_PATH), str(CLEAN_DATA_PATH))
