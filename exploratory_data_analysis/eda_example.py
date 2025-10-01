"""
Exploratory Data Analysis (EDA): Complete Example
Problem: Perform comprehensive EDA on a dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_sample_dataset():
    """Create a realistic sample dataset for EDA"""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'purchase_amount': np.random.randint(50, 5000, n_samples),
        'visits': np.random.randint(1, 50, n_samples),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation: higher income tends to have higher purchase amounts
    df['purchase_amount'] = df['purchase_amount'] + (df['income'] / 100).astype(int)
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[missing_indices, 'satisfaction'] = np.nan
    
    return df


def perform_eda(df):
    """Perform comprehensive EDA"""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMemory usage:")
    print(df.memory_usage(deep=True))
    
    # 2. Missing Values
    print("\n\n2. MISSING VALUES ANALYSIS")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_table = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    print(missing_table[missing_table['Missing Count'] > 0])
    
    # 3. Numerical Variables Analysis
    print("\n\n3. NUMERICAL VARIABLES ANALYSIS")
    print("-" * 80)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical columns: {numerical_cols}")
    print("\nDescriptive statistics:")
    print(df[numerical_cols].describe())
    
    # 4. Categorical Variables Analysis
    print("\n\n4. CATEGORICAL VARIABLES ANALYSIS")
    print("-" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        print(f"\n{col} - Value counts:")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")
    
    # 5. Correlation Analysis
    print("\n\n5. CORRELATION ANALYSIS")
    print("-" * 80)
    correlation_matrix = df[numerical_cols].corr()
    print("Correlation matrix:")
    print(correlation_matrix)
    
    # Find strong correlations
    print("\nStrong correlations (|r| > 0.5):")
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.5:
                strong_corr.append({
                    'Var1': correlation_matrix.columns[i],
                    'Var2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
    
    if strong_corr:
        print(pd.DataFrame(strong_corr))
    else:
        print("No strong correlations found.")
    
    # 6. Outlier Detection
    print("\n\n6. OUTLIER DETECTION")
    print("-" * 80)
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"\n{col}: {len(outliers)} outliers detected")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 7. Data Distribution
    print("\n\n7. DATA DISTRIBUTION ANALYSIS")
    print("-" * 80)
    for col in numerical_cols:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        print(f"\n{col}:")
        print(f"  Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})")
        print(f"  Kurtosis: {kurtosis:.4f} ({'Heavy-tailed' if kurtosis > 0 else 'Light-tailed'})")
    
    # 8. Summary Statistics by Category
    print("\n\n8. SUMMARY BY CATEGORICAL VARIABLES")
    print("-" * 80)
    if 'category' in df.columns:
        print("\nMean values by category:")
        print(df.groupby('category')[numerical_cols].mean())
    
    return df


def create_eda_visualizations(df):
    """Create visualizations for EDA"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns for visualization
    numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]
    
    if len(numerical_cols) >= 2:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Correlation heatmap
        correlation = df[numerical_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 2. Distribution of first numerical variable
        if len(numerical_cols) > 0:
            axes[0, 1].hist(df[numerical_cols[0]].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title(f'Distribution of {numerical_cols[0]}', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel(numerical_cols[0])
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Box plot for numerical variables
        if len(numerical_cols) > 0:
            df[numerical_cols[:4]].boxplot(ax=axes[1, 0])
            axes[1, 0].set_title('Box Plots of Numerical Variables', fontsize=14, fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Scatter plot
        if len(numerical_cols) >= 2:
            axes[1, 1].scatter(df[numerical_cols[0]], df[numerical_cols[1]], alpha=0.5)
            axes[1, 1].set_title(f'{numerical_cols[0]} vs {numerical_cols[1]}', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel(numerical_cols[0])
            axes[1, 1].set_ylabel(numerical_cols[1])
        
        plt.tight_layout()
        plt.savefig('/tmp/eda_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n\nEDA visualizations created successfully!")
        print("Saved to: /tmp/eda_visualization.png")


if __name__ == "__main__":
    # Create and analyze dataset
    df = create_sample_dataset()
    
    # Perform EDA
    df_analyzed = perform_eda(df)
    
    # Create visualizations
    create_eda_visualizations(df_analyzed)
    
    print("\n\nEDA completed successfully!")
