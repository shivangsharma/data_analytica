"""
Data Cleaning: Handling Missing Values
Problem: Handle missing values in a dataset using various strategies
"""

import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample dataset with missing values"""
    data = {
        'name': ['John', 'Anna', 'Peter', None, 'Linda'],
        'age': [28, None, 35, 29, None],
        'salary': [50000, 60000, None, 55000, 65000],
        'department': ['IT', 'HR', 'IT', None, 'Finance']
    }
    return pd.DataFrame(data)


def handle_missing_values(df):
    """
    Various strategies to handle missing values:
    1. Drop rows with any missing values
    2. Fill with mean/median/mode
    3. Forward fill
    4. Backward fill
    """
    print("Original DataFrame:")
    print(df)
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    # Strategy 1: Drop rows with missing values
    df_dropped = df.dropna()
    print("\n1. After dropping rows with missing values:")
    print(df_dropped)
    
    # Strategy 2: Fill numeric columns with mean
    df_filled_mean = df.copy()
    df_filled_mean['age'] = df_filled_mean['age'].fillna(df_filled_mean['age'].mean())
    df_filled_mean['salary'] = df_filled_mean['salary'].fillna(df_filled_mean['salary'].mean())
    print("\n2. After filling numeric columns with mean:")
    print(df_filled_mean)
    
    # Strategy 3: Fill categorical columns with mode
    df_filled_mode = df.copy()
    df_filled_mode['name'] = df_filled_mode['name'].fillna(df_filled_mode['name'].mode()[0] if not df_filled_mode['name'].mode().empty else 'Unknown')
    df_filled_mode['department'] = df_filled_mode['department'].fillna(df_filled_mode['department'].mode()[0] if not df_filled_mode['department'].mode().empty else 'Unknown')
    print("\n3. After filling categorical columns with mode:")
    print(df_filled_mode)
    
    # Strategy 4: Forward fill
    df_ffill = df.ffill()
    print("\n4. After forward fill:")
    print(df_ffill)
    
    return df_filled_mean, df_filled_mode


if __name__ == "__main__":
    df = create_sample_data()
    handle_missing_values(df)
