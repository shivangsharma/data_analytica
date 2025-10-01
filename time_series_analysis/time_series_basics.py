"""
Time Series Analysis: Basic Time Series Operations
Problem: Analyze and forecast time series data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def create_time_series_data():
    """Create sample time series data with trend, seasonality, and noise"""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # Create components
    trend = np.linspace(100, 200, 365)
    seasonal = 20 * np.sin(np.arange(365) * 2 * np.pi / 365)
    noise = np.random.randn(365) * 5
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    df.set_index('date', inplace=True)
    
    return df


def analyze_time_series(df):
    """Perform basic time series analysis"""
    print("=" * 80)
    print("TIME SERIES ANALYSIS")
    print("=" * 80)
    
    print("\n1. Basic Statistics:")
    print(df.describe())
    
    print("\n2. First and Last Values:")
    print(f"First date: {df.index[0]}")
    print(f"Last date: {df.index[-1]}")
    print(f"Total observations: {len(df)}")
    
    # Rolling statistics
    print("\n3. Rolling Statistics (30-day window):")
    rolling_mean = df['value'].rolling(window=30).mean()
    rolling_std = df['value'].rolling(window=30).std()
    
    print(f"Mean of rolling means: {rolling_mean.mean():.2f}")
    print(f"Mean of rolling std: {rolling_std.mean():.2f}")
    
    return rolling_mean, rolling_std


def decompose_time_series(df):
    """Decompose time series into trend, seasonal, and residual components"""
    print("\n\n" + "=" * 80)
    print("TIME SERIES DECOMPOSITION")
    print("=" * 80)
    
    # Perform decomposition
    decomposition = seasonal_decompose(df['value'], model='additive', period=30)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    print("\nDecomposition completed!")
    print(f"Trend component shape: {trend.dropna().shape}")
    print(f"Seasonal component shape: {seasonal.shape}")
    print(f"Residual component shape: {residual.dropna().shape}")
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original
    df['value'].plot(ax=axes[0], title='Original Time Series')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    trend.plot(ax=axes[1], title='Trend Component', color='red')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    seasonal.plot(ax=axes[2], title='Seasonal Component', color='green')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    residual.plot(ax=axes[3], title='Residual Component', color='orange')
    axes[3].set_ylabel('Residual')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Decomposition visualization saved to: /tmp/time_series_decomposition.png")
    
    return decomposition


def calculate_moving_averages(df):
    """Calculate moving averages"""
    print("\n\n" + "=" * 80)
    print("MOVING AVERAGES")
    print("=" * 80)
    
    df_ma = df.copy()
    
    # Simple Moving Averages
    df_ma['MA_7'] = df['value'].rolling(window=7).mean()
    df_ma['MA_30'] = df['value'].rolling(window=30).mean()
    df_ma['MA_90'] = df['value'].rolling(window=90).mean()
    
    print("\nMoving averages calculated:")
    print(df_ma[['value', 'MA_7', 'MA_30', 'MA_90']].tail(10))
    
    # Exponential Moving Average
    df_ma['EMA_30'] = df['value'].ewm(span=30, adjust=False).mean()
    
    print("\nExponential Moving Average (30-day):")
    print(df_ma[['value', 'MA_30', 'EMA_30']].tail(10))
    
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(df_ma.index, df_ma['value'], label='Original', alpha=0.5)
    plt.plot(df_ma.index, df_ma['MA_7'], label='7-day MA', linewidth=2)
    plt.plot(df_ma.index, df_ma['MA_30'], label='30-day MA', linewidth=2)
    plt.plot(df_ma.index, df_ma['MA_90'], label='90-day MA', linewidth=2)
    
    plt.title('Time Series with Moving Averages', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/moving_averages.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Moving averages visualization saved to: /tmp/moving_averages.png")
    
    return df_ma


def calculate_percentage_change(df):
    """Calculate percentage changes"""
    print("\n\n" + "=" * 80)
    print("PERCENTAGE CHANGES")
    print("=" * 80)
    
    df_pct = df.copy()
    
    # Daily percentage change
    df_pct['pct_change'] = df['value'].pct_change() * 100
    
    # 7-day percentage change
    df_pct['pct_change_7d'] = df['value'].pct_change(periods=7) * 100
    
    # 30-day percentage change
    df_pct['pct_change_30d'] = df['value'].pct_change(periods=30) * 100
    
    print("\nPercentage changes:")
    print(df_pct[['value', 'pct_change', 'pct_change_7d', 'pct_change_30d']].tail(10))
    
    print("\nSummary statistics of daily percentage changes:")
    print(df_pct['pct_change'].describe())
    
    return df_pct


def detect_outliers_in_series(df):
    """Detect outliers in time series"""
    print("\n\n" + "=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)
    
    # Calculate rolling statistics
    rolling_mean = df['value'].rolling(window=30, center=True).mean()
    rolling_std = df['value'].rolling(window=30, center=True).std()
    
    # Define outliers as values beyond 3 standard deviations
    lower_bound = rolling_mean - 3 * rolling_std
    upper_bound = rolling_mean + 3 * rolling_std
    
    outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    
    print(f"Number of outliers detected: {len(outliers)}")
    if len(outliers) > 0:
        print("\nOutlier dates and values:")
        print(outliers)
    
    return outliers


if __name__ == "__main__":
    # Create time series data
    df = create_time_series_data()
    
    print("Time Series Data (first 10 rows):")
    print(df.head(10))
    
    # Perform analysis
    rolling_mean, rolling_std = analyze_time_series(df)
    decomposition = decompose_time_series(df)
    df_ma = calculate_moving_averages(df)
    df_pct = calculate_percentage_change(df)
    outliers = detect_outliers_in_series(df)
    
    print("\n\nTime series analysis completed!")
