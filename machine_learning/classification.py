"""
Machine Learning: Classification
Problem: Classify data into categories using various algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


def create_sample_data():
    """Create sample dataset for classification"""
    np.random.seed(42)
    n_samples = 300
    
    # Create features
    class_0 = np.random.randn(n_samples // 3, 2) + np.array([0, 0])
    class_1 = np.random.randn(n_samples // 3, 2) + np.array([3, 3])
    class_2 = np.random.randn(n_samples // 3, 2) + np.array([0, 3])
    
    X = np.vstack([class_0, class_1, class_2])
    y = np.hstack([
        np.zeros(n_samples // 3),
        np.ones(n_samples // 3),
        np.full(n_samples // 3, 2)
    ])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y.astype(int)
    
    return df


def train_and_evaluate_models(X, y):
    """Train and evaluate multiple classification models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    print("=" * 80)
    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}")
        print("-" * 80)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
    
    return results, X_test, y_test, scaler


def compare_models(results):
    """Compare model performances"""
    print("\n\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1 Score': [results[m]['f1'] for m in results]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # Find best model
    best_model = comparison.loc[comparison['F1 Score'].idxmax(), 'Model']
    print(f"\nBest Model: {best_model} (based on F1 Score)")
    
    return comparison


def make_predictions(model, scaler, new_data):
    """Make predictions on new data"""
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled) if hasattr(model, 'predict_proba') else None
    
    print("\n\nNEW PREDICTIONS")
    print("=" * 80)
    
    for i, (pred, data) in enumerate(zip(predictions, new_data)):
        print(f"\nSample {i+1}: {data}")
        print(f"  Predicted Class: {int(pred)}")
        if probabilities is not None:
            print(f"  Class Probabilities: {probabilities[i]}")
    
    return predictions


if __name__ == "__main__":
    # Create dataset
    df = create_sample_data()
    
    print("Dataset Overview:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts().sort_index()}")
    
    # Prepare features and target
    X = df[['feature1', 'feature2']].values
    y = df['target'].values
    
    # Train and evaluate models
    results, X_test, y_test, scaler = train_and_evaluate_models(X, y)
    
    # Compare models
    comparison = compare_models(results)
    
    # Make predictions with best model
    best_model_name = comparison.loc[comparison['F1 Score'].idxmax(), 'Model']
    best_model = results[best_model_name]['model']
    
    new_samples = np.array([
        [0.5, 0.5],
        [3.5, 3.5],
        [0.5, 3.5]
    ])
    
    print(f"\n\nUsing {best_model_name} for new predictions:")
    make_predictions(best_model, scaler, new_samples)
    
    print("\n\nClassification analysis completed!")
