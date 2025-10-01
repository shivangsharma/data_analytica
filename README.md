# Data Analytica

**End-to-end analytics: from business question → dashboard → decisions**

A comprehensive repository containing solutions to data analytics problems, covering everything from data cleaning to machine learning. This repository serves as a practical guide and reference for data analysts, data scientists, and anyone working with data.

## 📚 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Topics Covered](#topics-covered)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository provides hands-on examples and solutions for common data analytics problems. Each module includes:
- **Problem statements** explaining the business context
- **Complete working code** with explanations
- **Sample datasets** for practice
- **Visualizations** to understand results
- **Best practices** for data analysis

## 📁 Repository Structure

```
data_analytica/
├── data_cleaning/              # Data cleaning techniques
│   ├── missing_values.py       # Handle missing data
│   └── outlier_detection.py    # Detect and handle outliers
│
├── statistical_analysis/       # Statistical methods
│   ├── descriptive_statistics.py    # Basic statistics
│   └── hypothesis_testing.py        # Statistical tests
│
├── data_visualization/         # Visualization techniques
│   └── basic_plots.py         # Create various plots
│
├── exploratory_data_analysis/  # EDA techniques
│   └── eda_example.py         # Complete EDA workflow
│
├── machine_learning/           # ML algorithms
│   ├── linear_regression.py   # Regression models
│   └── classification.py      # Classification models
│
├── data_processing/            # Data transformation
│   └── data_transformation.py # Normalization, encoding, etc.
│
├── time_series_analysis/       # Time series methods
│   └── time_series_basics.py  # Time series analysis
│
├── sql_queries/                # SQL examples
│   └── basic_queries.sql      # SQL patterns for analytics
│
├── datasets/                   # Sample datasets
│
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shivangsharma/data_analytica.git
cd data_analytica
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run any example:
```bash
python data_cleaning/missing_values.py
python statistical_analysis/hypothesis_testing.py
```

## 📖 Topics Covered

### 1. Data Cleaning
- **Missing Values**: Multiple strategies for handling missing data (drop, fill, forward/backward fill)
- **Outlier Detection**: IQR method, Z-score method, and various handling strategies
- **Data Validation**: Ensure data quality and consistency

### 2. Statistical Analysis
- **Descriptive Statistics**: Mean, median, mode, variance, standard deviation, skewness, kurtosis
- **Hypothesis Testing**: T-tests, Chi-square tests, ANOVA, correlation tests
- **Probability Distributions**: Normal, binomial, Poisson distributions
- **Confidence Intervals**: Statistical inference

### 3. Data Visualization
- **Line Plots**: Time series and trends
- **Bar Charts**: Categorical comparisons
- **Histograms**: Distribution analysis
- **Scatter Plots**: Relationship exploration
- **Box Plots**: Distribution and outlier visualization
- **Heatmaps**: Correlation matrices

### 4. Exploratory Data Analysis (EDA)
- Dataset overview and structure analysis
- Missing values analysis
- Numerical and categorical variable analysis
- Correlation analysis
- Outlier detection
- Distribution analysis
- Summary statistics by groups

### 5. Machine Learning
- **Linear Regression**: Predict continuous values
  - Model training and evaluation
  - Feature importance
  - Residual analysis
- **Classification**: Categorize data
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Model comparison and evaluation
  - Confusion matrices and metrics

### 6. Data Processing
- **Normalization**: Min-Max scaling (0-1 range)
- **Standardization**: Z-score scaling (mean=0, std=1)
- **Encoding**: Label encoding, one-hot encoding, ordinal encoding
- **Binning**: Discretization of continuous variables
- **Feature Engineering**: Create derived features

### 7. Time Series Analysis
- Time series decomposition (trend, seasonal, residual)
- Moving averages (simple and exponential)
- Percentage changes
- Outlier detection in time series
- Forecasting basics

### 8. SQL for Analytics
- Basic queries (SELECT, WHERE, GROUP BY)
- Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- Joins (INNER, LEFT, RIGHT, FULL)
- Subqueries and CTEs
- Window functions (RANK, ROW_NUMBER, LAG, LEAD)
- Date and string functions
- Complex analytics queries

## 💡 Usage Examples

### Example 1: Handling Missing Values

```python
from data_cleaning.missing_values import create_sample_data, handle_missing_values

# Create sample dataset
df = create_sample_data()

# Handle missing values using different strategies
df_filled_mean, df_filled_mode = handle_missing_values(df)
```

### Example 2: Hypothesis Testing

```python
from statistical_analysis.hypothesis_testing import ttest_example, chi_square_test

# Perform t-test
t_statistic, p_value = ttest_example()

# Perform chi-square test
chi2, p_value = chi_square_test()
```

### Example 3: Machine Learning Classification

```python
from machine_learning.classification import create_sample_data, train_and_evaluate_models

# Create dataset
df = create_sample_data()

# Train and compare multiple models
X = df[['feature1', 'feature2']].values
y = df['target'].values
results, X_test, y_test, scaler = train_and_evaluate_models(X, y)
```

### Example 4: Exploratory Data Analysis

```python
from exploratory_data_analysis.eda_example import create_sample_dataset, perform_eda

# Create dataset
df = create_sample_dataset()

# Perform comprehensive EDA
df_analyzed = perform_eda(df)
```

## 📊 Key Features

- ✅ **Comprehensive Coverage**: From basic statistics to machine learning
- ✅ **Practical Examples**: Real-world problem scenarios
- ✅ **Well-Documented**: Clear explanations and comments
- ✅ **Visualization**: Visual outputs for better understanding
- ✅ **Best Practices**: Industry-standard approaches
- ✅ **Modular Code**: Easy to understand and modify
- ✅ **Production-Ready**: Can be adapted for real projects

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning
- **SciPy**: Scientific computing
- **Statsmodels**: Statistical models

## 📈 Common Use Cases

1. **Business Analytics**: Sales analysis, customer segmentation, trend analysis
2. **Data Science Projects**: Feature engineering, model building, performance evaluation
3. **Research**: Statistical hypothesis testing, correlation analysis
4. **Reporting**: Creating dashboards and visualizations
5. **Data Cleaning**: Preparing data for analysis
6. **Predictive Analytics**: Building forecasting models

## 🎓 Learning Path

For beginners, we recommend following this order:

1. Start with **Data Cleaning** to understand data preparation
2. Move to **Statistical Analysis** for foundational concepts
3. Learn **Data Visualization** to communicate findings
4. Practice **Exploratory Data Analysis** to combine skills
5. Explore **Data Processing** for advanced transformations
6. Study **Machine Learning** for predictive modeling
7. Master **Time Series Analysis** for temporal data
8. Practice **SQL** for database analytics

## 🤝 Contributing

Contributions are welcome! If you have:
- New data analytics problems and solutions
- Improvements to existing code
- Additional examples or use cases
- Bug fixes or optimizations

Please feel free to submit a pull request or open an issue.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions, suggestions, or feedback:
- **Author**: Shivang Sharma
- **Repository**: [shivangsharma/data_analytica](https://github.com/shivangsharma/data_analytica)

## 🌟 Acknowledgments

This repository is designed to help data professionals at all levels. Whether you're just starting your data analytics journey or looking for quick reference implementations, we hope you find this resource valuable.

---

**Happy Analyzing! 📊🔍**
