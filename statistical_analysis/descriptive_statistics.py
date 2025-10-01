"""
Statistical Analysis: Descriptive Statistics
Problem: Calculate and interpret descriptive statistics
"""

import pandas as pd
import numpy as np


def create_sample_dataset():
    """Create sample dataset for analysis"""
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 60, 100),
        'salary': np.random.randint(30000, 120000, 100),
        'years_experience': np.random.randint(0, 20, 100),
        'satisfaction_score': np.random.randint(1, 11, 100)
    }
    return pd.DataFrame(data)


def calculate_descriptive_statistics(df):
    """Calculate various descriptive statistics"""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("\n1. Basic Statistics (describe()):")
    print(df.describe())
    
    # Measures of central tendency
    print("\n2. Measures of Central Tendency:")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Mode: {df[col].mode().values[0] if not df[col].mode().empty else 'N/A'}")
    
    # Measures of dispersion
    print("\n3. Measures of Dispersion:")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Variance: {df[col].var():.2f}")
        print(f"  Standard Deviation: {df[col].std():.2f}")
        print(f"  Range: {df[col].max() - df[col].min():.2f}")
        print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    
    # Measures of shape
    print("\n4. Measures of Shape:")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Skewness: {df[col].skew():.4f}")
        print(f"  Kurtosis: {df[col].kurtosis():.4f}")
    
    # Correlation matrix
    print("\n5. Correlation Matrix:")
    print(df.corr())
    
    # Quantiles
    print("\n6. Quantiles:")
    print(df.quantile([0.25, 0.5, 0.75]))
    
    return df.describe()


def analyze_distribution(df, column):
    """Analyze the distribution of a specific column"""
    print(f"\n\nDetailed Distribution Analysis for '{column}':")
    print("=" * 60)
    
    data = df[column]
    
    print(f"Count: {data.count()}")
    print(f"Missing values: {data.isnull().sum()}")
    print(f"Unique values: {data.nunique()}")
    
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th percentile: {data.quantile(p/100):.2f}")
    
    # Interpret skewness
    skewness = data.skew()
    print(f"\nSkewness: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("  Interpretation: Fairly symmetric distribution")
    elif skewness > 0:
        print("  Interpretation: Right-skewed (positive skew)")
    else:
        print("  Interpretation: Left-skewed (negative skew)")
    
    # Interpret kurtosis
    kurtosis = data.kurtosis()
    print(f"\nKurtosis: {kurtosis:.4f}")
    if abs(kurtosis) < 0.5:
        print("  Interpretation: Normal distribution (mesokurtic)")
    elif kurtosis > 0:
        print("  Interpretation: Heavy-tailed distribution (leptokurtic)")
    else:
        print("  Interpretation: Light-tailed distribution (platykurtic)")


if __name__ == "__main__":
    df = create_sample_dataset()
    
    print("Sample Data (first 10 rows):")
    print(df.head(10))
    
    calculate_descriptive_statistics(df)
    analyze_distribution(df, 'salary')
    analyze_distribution(df, 'age')
