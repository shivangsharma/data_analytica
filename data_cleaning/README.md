# Data Cleaning

This directory contains solutions for common data cleaning problems.

## Files

### 1. missing_values.py
Demonstrates various strategies for handling missing data:
- Drop rows with missing values
- Fill with mean/median for numeric columns
- Fill with mode for categorical columns
- Forward fill and backward fill

**Usage:**
```python
python missing_values.py
```

### 2. outlier_detection.py
Shows different methods for detecting and handling outliers:
- IQR (Interquartile Range) method
- Z-score method
- Handling strategies: remove, cap, or replace with median

**Usage:**
```python
python outlier_detection.py
```

## Key Concepts

- **Missing Values**: Can occur due to data entry errors, system failures, or intentional non-response
- **Outliers**: Extreme values that differ significantly from other observations
- **Data Quality**: Clean data is essential for accurate analysis and modeling
