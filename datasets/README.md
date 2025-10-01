# Sample Datasets

This directory contains sample datasets used throughout the repository examples.

## Available Datasets

### 1. employee_data.csv
Sample employee dataset with the following columns:
- employee_id: Unique identifier
- name: Employee name
- age: Employee age
- department: Department name
- salary: Annual salary
- years_experience: Years of work experience
- performance_rating: Performance score (1-5)

### 2. sales_data.csv
Sample sales dataset with columns:
- order_id: Unique order identifier
- order_date: Date of order
- customer_id: Customer identifier
- product: Product name
- quantity: Number of items
- amount: Total sale amount
- region: Sales region

## Creating Your Own Datasets

All example scripts in this repository generate their own sample data, so these CSV files are optional. However, they can be useful for:
- Practicing data loading
- Testing your own analysis code
- Learning data formats

## Data Format

All datasets are in CSV (Comma-Separated Values) format, which can be easily loaded using:

```python
import pandas as pd

df = pd.read_csv('datasets/employee_data.csv')
```
