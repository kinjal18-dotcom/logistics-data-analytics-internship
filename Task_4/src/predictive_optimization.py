"""
Logistics Analytics Pipeline - Task 4: Predictive Modeling & Optimization
Uses Pandas, NumPy, Matplotlib, and Scikit-Learn.
Reference Dataset: DataCo Smart Supply Chain Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_pipeline(df):
    """
    Splits data, performs cross-validation, tunes hyperparameters, and evaluates models.
    """
    # Select features and target column
    feature_cols = [
        'Days for shipment (scheduled)', 
        'Sales per customer', 
        'Order Item Quantity', 
        'Order Profit Per Order'
    ]
    
    X = df[feature_cols]
    y = df['Days for shipping (real)']
    
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # 1. Linear Regression Baseline
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    # 2. Decision Tree with Hyperparameter Tuning via GridSearchCV
    param_grid = {'max_depth': [3, 5, 7, 10]}
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42), 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error'
    )
    dt_grid.fit(X_train, y_train)
    
    best_dt = dt_grid.best_estimator_
    dt_preds = best_dt.predict(X_test)
    
    # Evaluate Decision Tree
    mae = mean_absolute_error(y_test, dt_preds)
    rmse = np.sqrt(mean_squared_error(y_test, dt_preds))
    r2 = r2_score(y_test, dt_preds)
    
    print("Decision Tree Model Performance:")
    print(f"Optimal Tree Depth: {dt_grid.best_params_['max_depth']}")
    print(f"MAE: {mae:.2f} Days | RMSE: {rmse:.2f} Days | R2 Score: {r2:.2f}")
    
    return best_dt, X_test, y_test, dt_preds

def plot_prediction_scatter(y_test, y_pred):
    """
    Generates a scatter plot comparing actual vs predicted delivery days.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test[:100], y_pred[:100], alpha=0.7, color='teal')
    plt.plot([0, 6], [0, 6], color='red', linestyle='--')
    plt.title('Actual vs Predicted Delivery Duration (Sample of 100 Shipments)')
    plt.xlabel('Actual Shipping Days')
    plt.ylabel('Predicted Shipping Days')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Task 4 Predictive Modeling Pipeline Executed.")
