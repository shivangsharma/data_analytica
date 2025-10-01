"""
Data Visualization: Basic Plots
Problem: Create various types of plots for data visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample datasets for visualization"""
    np.random.seed(42)
    
    # Time series data
    dates = pd.date_range('2023-01-01', periods=100)
    time_series = pd.DataFrame({
        'date': dates,
        'sales': np.cumsum(np.random.randn(100)) + 100
    })
    
    # Categorical data
    categories = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'],
        'values': [23, 45, 56, 78, 32]
    })
    
    # Distribution data
    distribution = pd.DataFrame({
        'group': ['Group 1'] * 50 + ['Group 2'] * 50,
        'values': np.concatenate([
            np.random.normal(100, 15, 50),
            np.random.normal(120, 20, 50)
        ])
    })
    
    # Scatter data
    scatter = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 2 + 5
    })
    
    return time_series, categories, distribution, scatter


def create_line_plot(data, save_path=None):
    """Create a line plot for time series data"""
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['sales'], marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.title('Sales Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('/tmp/line_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Line plot created successfully!")


def create_bar_chart(data, save_path=None):
    """Create a bar chart for categorical data"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data['category'], data['values'], color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Values by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('/tmp/bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Bar chart created successfully!")


def create_histogram(data, save_path=None):
    """Create a histogram with distribution curves"""
    plt.figure(figsize=(10, 6))
    
    for group in data['group'].unique():
        group_data = data[data['group'] == group]['values']
        plt.hist(group_data, alpha=0.6, bins=20, label=group, edgecolor='black')
    
    plt.title('Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('/tmp/histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Histogram created successfully!")


def create_scatter_plot(data, save_path=None):
    """Create a scatter plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(data['x'], data['y'], alpha=0.6, s=50, c='coral', edgecolors='red')
    
    # Add trend line
    z = np.polyfit(data['x'], data['y'], 1)
    p = np.poly1d(z)
    plt.plot(data['x'].sort_values(), p(data['x'].sort_values()), 
             "r--", alpha=0.8, linewidth=2, label='Trend line')
    
    plt.title('Scatter Plot with Trend Line', fontsize=16, fontweight='bold')
    plt.xlabel('X Variable', fontsize=12)
    plt.ylabel('Y Variable', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('/tmp/scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Scatter plot created successfully!")


def create_box_plot(data, save_path=None):
    """Create a box plot to show distribution"""
    plt.figure(figsize=(10, 6))
    data.boxplot(column='values', by='group', patch_artist=True)
    plt.title('Box Plot Comparison', fontsize=16, fontweight='bold')
    plt.suptitle('')  # Remove the automatic title
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('/tmp/box_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Box plot created successfully!")


if __name__ == "__main__":
    # Create sample data
    time_series, categories, distribution, scatter = create_sample_data()
    
    # Create various plots
    print("Creating visualizations...")
    create_line_plot(time_series)
    create_bar_chart(categories)
    create_histogram(distribution)
    create_scatter_plot(scatter)
    create_box_plot(distribution)
    
    print("\nAll visualizations created successfully!")
    print("Plots saved in /tmp/ directory")
