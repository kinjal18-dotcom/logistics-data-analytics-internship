"""
Logistics Analytics Pipeline - Task 2: Data Preprocessing & Cleaning
Reference Dataset: DataCo Smart Supply Chain Dataset (Kaggle)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_inspect_dataset(file_path):
    """
    Simulates data ingestion and checks dataset dimensions.
    """
    df = pd.read_csv(file_path, encoding='latin1')
    print(f"Raw Dataset Loaded. Shape: {df.shape}")
    return df

def clean_missing_values(df):
    """
    Handles missing data using threshold dropping and median/label imputation.
    """
    # Drop columns exceeding 50% missing values
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.50].index
    df = df.drop(columns=cols_to_drop)
    
    # Impute numeric missing values using column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Impute categorical missing values with placeholder label
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            
    return df

def filter_iqr_outliers(df, target_columns):
    """
    Filters operational outliers using the Interquartile Range (IQR) method.
    """
    for col in target_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep rows within statistical boundaries
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

def normalize_logistics_features(df, feature_list):
    """
    Applies Min-Max Scaling to normalize features into a 0 to 1 range.
    """
    scaler = MinMaxScaler()
    df[feature_list] = scaler.fit_transform(df[feature_list])
    return df, scaler

if __name__ == "__main__":
    print("Task 2 Data Preprocessing Pipeline Module Ready.")
