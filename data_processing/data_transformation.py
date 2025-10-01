"""
Data Processing: Data Transformation
Problem: Transform and prepare data for analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def create_sample_data():
    """Create sample dataset for transformation"""
    np.random.seed(42)
    data = {
        'id': range(1, 11),
        'name': ['John', 'Anna', 'Peter', 'Linda', 'Mark', 'Emma', 'David', 'Sarah', 'Tom', 'Jane'],
        'age': [25, 32, 28, 45, 35, 29, 41, 33, 27, 38],
        'salary': [50000, 75000, 60000, 90000, 68000, 55000, 85000, 72000, 58000, 80000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'HR', 'IT', 'Finance'],
        'performance': ['Good', 'Excellent', 'Good', 'Excellent', 'Average', 'Good', 'Excellent', 'Average', 'Good', 'Excellent']
    }
    return pd.DataFrame(data)


def normalize_data(df):
    """Normalize numerical columns (0-1 scale)"""
    print("=" * 60)
    print("DATA NORMALIZATION (Min-Max Scaling)")
    print("=" * 60)
    
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    
    numerical_cols = ['age', 'salary']
    df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("\nOriginal data:")
    print(df[numerical_cols].describe())
    
    print("\nNormalized data (0-1 scale):")
    print(df_normalized[numerical_cols].describe())
    
    return df_normalized


def standardize_data(df):
    """Standardize numerical columns (mean=0, std=1)"""
    print("\n\n" + "=" * 60)
    print("DATA STANDARDIZATION (Z-Score)")
    print("=" * 60)
    
    df_standardized = df.copy()
    scaler = StandardScaler()
    
    numerical_cols = ['age', 'salary']
    df_standardized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("\nOriginal data:")
    print(df[numerical_cols].describe())
    
    print("\nStandardized data (mean≈0, std≈1):")
    print(df_standardized[numerical_cols].describe())
    
    return df_standardized


def encode_categorical_data(df):
    """Encode categorical variables"""
    print("\n\n" + "=" * 60)
    print("CATEGORICAL DATA ENCODING")
    print("=" * 60)
    
    df_encoded = df.copy()
    
    # Label Encoding
    print("\n1. Label Encoding (department):")
    le = LabelEncoder()
    df_encoded['department_encoded'] = le.fit_transform(df['department'])
    print(f"Original values: {df['department'].unique()}")
    print(f"Encoded mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # One-Hot Encoding
    print("\n2. One-Hot Encoding (department):")
    df_onehot = pd.get_dummies(df['department'], prefix='dept')
    print(df_onehot.head())
    
    # Ordinal Encoding
    print("\n3. Ordinal Encoding (performance):")
    performance_mapping = {'Average': 1, 'Good': 2, 'Excellent': 3}
    df_encoded['performance_encoded'] = df['performance'].map(performance_mapping)
    print(f"Mapping: {performance_mapping}")
    print(df_encoded[['performance', 'performance_encoded']].head())
    
    return df_encoded


def create_bins(df):
    """Create bins for continuous variables"""
    print("\n\n" + "=" * 60)
    print("BINNING (DISCRETIZATION)")
    print("=" * 60)
    
    df_binned = df.copy()
    
    # Age bins
    age_bins = [0, 30, 40, 100]
    age_labels = ['Young', 'Middle', 'Senior']
    df_binned['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    print("\nAge binning:")
    print(f"Bins: {age_bins}")
    print(f"Labels: {age_labels}")
    print(df_binned[['age', 'age_group']].head())
    
    # Salary bins
    salary_bins = [0, 60000, 75000, 100000]
    salary_labels = ['Low', 'Medium', 'High']
    df_binned['salary_range'] = pd.cut(df['salary'], bins=salary_bins, labels=salary_labels)
    
    print("\nSalary binning:")
    print(f"Bins: {salary_bins}")
    print(f"Labels: {salary_labels}")
    print(df_binned[['salary', 'salary_range']].head())
    
    # Distribution of bins
    print("\nAge group distribution:")
    print(df_binned['age_group'].value_counts())
    
    print("\nSalary range distribution:")
    print(df_binned['salary_range'].value_counts())
    
    return df_binned


def create_derived_features(df):
    """Create new features from existing ones"""
    print("\n\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df_featured = df.copy()
    
    # Age to salary ratio
    df_featured['age_salary_ratio'] = df['age'] / (df['salary'] / 1000)
    print("\n1. Age to Salary Ratio (age per $1000):")
    print(df_featured[['age', 'salary', 'age_salary_ratio']].head())
    
    # Log transformation
    df_featured['log_salary'] = np.log(df['salary'])
    print("\n2. Log transformation of salary:")
    print(df_featured[['salary', 'log_salary']].head())
    
    # Square root transformation
    df_featured['sqrt_age'] = np.sqrt(df['age'])
    print("\n3. Square root transformation of age:")
    print(df_featured[['age', 'sqrt_age']].head())
    
    return df_featured


if __name__ == "__main__":
    # Create sample data
    df = create_sample_data()
    
    print("Original Dataset:")
    print(df)
    
    # Apply transformations
    df_normalized = normalize_data(df)
    df_standardized = standardize_data(df)
    df_encoded = encode_categorical_data(df)
    df_binned = create_bins(df)
    df_featured = create_derived_features(df)
    
    print("\n\nData transformation completed!")
