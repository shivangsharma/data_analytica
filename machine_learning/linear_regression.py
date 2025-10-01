"""
Machine Learning: Linear Regression
Problem: Predict continuous values using linear regression
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def create_sample_data():
    """Create sample dataset for regression"""
    np.random.seed(42)
    n_samples = 200
    
    # Features
    X = np.random.rand(n_samples, 3) * 100
    
    # Target with some noise (linear relationship)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 50 + np.random.randn(n_samples) * 10
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y
    
    return df


def train_linear_regression(X, y):
    """Train a linear regression model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate model
    print("=" * 60)
    print("LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    print("\nModel Coefficients:")
    for i, coef in enumerate(model.coef_):
        print(f"  Feature {i+1}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    print("\nTraining Set Performance:")
    print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    
    print("\nTest Set Performance:")
    print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted (Training)
    axes[0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0].plot([y_train.min(), y_train.max()], 
                 [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Training Set: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Test)
    axes[1].scatter(y_test, y_test_pred, alpha=0.5, color='green')
    axes[1].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('Test Set: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/linear_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to: /tmp/linear_regression.png")
    
    return model, X_test, y_test, y_test_pred


def analyze_residuals(y_true, y_pred):
    """Analyze residuals to check model assumptions"""
    residuals = y_true - y_pred
    
    print("\n\nRESIDUAL ANALYSIS")
    print("=" * 60)
    print(f"Mean of residuals: {residuals.mean():.4f} (should be close to 0)")
    print(f"Std of residuals: {residuals.std():.4f}")
    
    # Create residual plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/tmp/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Residual analysis visualization saved to: /tmp/residual_analysis.png")


def make_prediction(model, new_data):
    """Make predictions on new data"""
    predictions = model.predict(new_data)
    
    print("\n\nNEW PREDICTIONS")
    print("=" * 60)
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: {pred:.2f}")
    
    return predictions


if __name__ == "__main__":
    # Create dataset
    df = create_sample_data()
    
    print("Dataset Overview:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    
    # Prepare features and target
    X = df[['feature1', 'feature2', 'feature3']].values
    y = df['target'].values
    
    # Train model
    model, X_test, y_test, y_test_pred = train_linear_regression(X, y)
    
    # Analyze residuals
    analyze_residuals(y_test, y_test_pred)
    
    # Make new predictions
    new_samples = np.array([[50, 60, 40], [80, 20, 90], [30, 30, 30]])
    make_prediction(model, new_samples)
    
    print("\n\nLinear Regression analysis completed!")
