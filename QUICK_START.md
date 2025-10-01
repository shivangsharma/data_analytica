# Quick Start Guide

Get up and running with Data Analytica in 5 minutes!

## 1. Installation (1 minute)

```bash
# Clone the repository
git clone https://github.com/shivangsharma/data_analytica.git
cd data_analytica

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Your First Example (30 seconds)

```bash
# Handle missing values
python data_cleaning/missing_values.py

# Or run hypothesis tests
python statistical_analysis/hypothesis_testing.py

# Or try machine learning classification
python machine_learning/classification.py
```

## 3. Explore Sample Datasets (30 seconds)

```python
import pandas as pd

# Load employee data
df = pd.read_csv('datasets/employee_data.csv')
print(df.head())

# Load sales data
sales = pd.read_csv('datasets/sales_data.csv')
print(sales.head())
```

## 4. Try the Jupyter Notebook (2 minutes)

```bash
# Start Jupyter
jupyter notebook

# Open: examples/getting_started.ipynb
# Run all cells to see complete analysis workflow
```

## 5. Common Use Cases

### Data Cleaning
```bash
# Handle missing values
python data_cleaning/missing_values.py

# Detect and handle outliers
python data_cleaning/outlier_detection.py
```

### Statistical Analysis
```bash
# Descriptive statistics
python statistical_analysis/descriptive_statistics.py

# Hypothesis testing
python statistical_analysis/hypothesis_testing.py
```

### Machine Learning
```bash
# Linear regression
python machine_learning/linear_regression.py

# Classification
python machine_learning/classification.py
```

### Data Visualization
```bash
# Create various plots
python data_visualization/basic_plots.py
# Plots saved to /tmp/ directory
```

## 6. Customize for Your Data

Each script has a `create_sample_data()` function. Replace it with your own data:

```python
# Instead of:
df = create_sample_data()

# Use:
df = pd.read_csv('your_data.csv')
```

## Quick Reference

| Task | Script | Time |
|------|--------|------|
| Clean missing data | `data_cleaning/missing_values.py` | < 1s |
| Detect outliers | `data_cleaning/outlier_detection.py` | < 1s |
| Statistical tests | `statistical_analysis/hypothesis_testing.py` | < 1s |
| Create visualizations | `data_visualization/basic_plots.py` | < 2s |
| Complete EDA | `exploratory_data_analysis/eda_example.py` | < 2s |
| Train ML model | `machine_learning/classification.py` | < 5s |
| Time series analysis | `time_series_analysis/time_series_basics.py` | < 2s |

## Next Steps

1. **Read the full [README.md](README.md)** for comprehensive documentation
2. **Explore individual modules** to understand specific techniques
3. **Check [CONTRIBUTING.md](CONTRIBUTING.md)** if you want to add your own solutions
4. **Adapt the code** for your specific data analytics projects

## Need Help?

- 📖 Check the README for detailed documentation
- 💡 Look at example scripts for reference implementations
- 🐛 Open an issue for bugs or questions
- 🤝 See CONTRIBUTING.md to add your own solutions

---

**Happy Analyzing!** 📊✨
