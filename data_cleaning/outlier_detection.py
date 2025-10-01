"""
Data Cleaning: Outlier Detection
Problem: Detect and handle outliers using various methods
"""

import pandas as pd
import numpy as np


def create_sample_data_with_outliers():
    """Create sample dataset with outliers"""
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),  # Normal values
            [300, 350, -50, 400, 500]  # Outliers
        ])
    }
    return pd.DataFrame(data)


def detect_outliers_iqr(df, column):
    """Detect outliers using IQR (Interquartile Range) method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"IQR Method:")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {outliers[column].values}")
    
    return outliers


def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    
    outliers = df[z_scores > threshold]
    
    print(f"\nZ-Score Method (threshold={threshold}):")
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {outliers[column].values}")
    
    return outliers


def handle_outliers(df, column, method='remove'):
    """
    Handle outliers using different strategies:
    - remove: Remove outlier rows
    - cap: Cap outliers to bounds
    - median: Replace with median
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if method == 'remove':
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"\nAfter removing outliers: {len(df)} -> {len(df_clean)} rows")
        return df_clean
    
    elif method == 'cap':
        df_clean = df.copy()
        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
        print(f"\nAfter capping outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
        return df_clean
    
    elif method == 'median':
        df_clean = df.copy()
        median = df_clean[column].median()
        df_clean.loc[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound), column] = median
        print(f"\nAfter replacing outliers with median ({median:.2f})")
        return df_clean


if __name__ == "__main__":
    df = create_sample_data_with_outliers()
    
    print("Original Data Statistics:")
    print(df['values'].describe())
    
    # Detect outliers using different methods
    outliers_iqr = detect_outliers_iqr(df, 'values')
    outliers_zscore = detect_outliers_zscore(df, 'values')
    
    # Handle outliers
    df_removed = handle_outliers(df, 'values', method='remove')
    df_capped = handle_outliers(df, 'values', method='cap')
    df_median = handle_outliers(df, 'values', method='median')
