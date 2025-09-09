#!/usr/bin/env python3
"""
Model Training Script for Virus-DO Prediction

This script trains the best performing model and saves it for future predictions.

Author: Generated Script
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def load_data_and_features(data_dir, features_file=None):
    """
    Load processed data and selected features.
    
    Args:
        data_dir (str): Directory containing processed data
        features_file (str): Path to selected features file
        
    Returns:
        tuple: (X, y, feature_names)
    """
    try:
        # Load full dataset
        X = pd.read_csv(os.path.join(data_dir, 'features.csv'), index_col=0)
        y = pd.read_csv(os.path.join(data_dir, 'targets.csv'), index_col=0).squeeze()
        
        # Load selected features if provided
        if features_file and os.path.exists(features_file):
            with open(features_file, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            
            # Filter to selected features
            available_features = [f for f in selected_features if f in X.columns]
            X = X[available_features]
            print(f"Using {len(available_features)} selected features")
        else:
            print(f"Using all {X.shape[1]} features")
        
        print(f"Loaded data - Features: {X.shape}, Targets: {y.shape}")
        return X, y, list(X.columns)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_model_and_params(model_name):
    """
    Get model instance and hyperparameter grid for tuning.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, param_grid)
    """
    if model_name.lower() == 'linear regression':
        return LinearRegression(), {}
    
    elif model_name.lower() == 'ridge regression':
        return Ridge(), {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    
    elif model_name.lower() == 'lasso regression':
        return Lasso(), {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        }
    
    elif model_name.lower() == 'elastic net':
        return ElasticNet(), {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }
    
    elif model_name.lower() == 'random forest':
        return RandomForestRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    
    elif model_name.lower() == 'gradient boosting':
        return GradientBoostingRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    elif model_name.lower() == 'svr (rbf)':
        return SVR(kernel='rbf'), {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
    
    elif model_name.lower() == 'svr (linear)':
        return SVR(kernel='linear'), {
            'C': [0.1, 1.0, 10.0]
        }
    
    elif model_name.lower() == 'k-neighbors':
        return KNeighborsRegressor(), {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    
    elif model_name.lower() == 'xgboost' and XGBOOST_AVAILABLE:
        return XGBRegressor(random_state=42), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_and_tune_model(X, y, model_name, cv_folds=5, random_state=42, tune_hyperparams=True):
    """
    Train and optionally tune hyperparameters of a model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        model_name (str): Name of the model to train
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        tune_hyperparams (bool): Whether to tune hyperparameters
        
    Returns:
        tuple: (best_model, best_params, cv_score, test_metrics)
    """
    print(f"Training {model_name}...")
    
    # Get model and parameter grid
    model, param_grid = get_model_and_params(model_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Hyperparameter tuning
    if tune_hyperparams and param_grid:
        print(f"Tuning hyperparameters with {cv_folds}-fold CV...")
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        print(f"Best parameters: {best_params}")
    else:
        print("Training with default parameters...")
        model.fit(X_train, y_train)
        best_model = model
        best_params = {}
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        cv_score = cv_scores.mean()
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"Cross-validation R²: {cv_score:.4f}")
    print(f"Test set R²: {test_metrics['r2']:.4f}")
    
    return best_model, best_params, cv_score, test_metrics, X_test, y_test, y_pred

def plot_predictions(y_true, y_pred, model_name, output_dir):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual DO')
    ax1.set_ylabel('Predicted DO')
    ax1.set_title(f'{model_name} - Predicted vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Residuals plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted DO')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title(f'{model_name} - Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_predictions.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved prediction plot to {output_dir}/{filename}")

def plot_feature_importance(model, feature_names, model_name, output_dir, top_n=20):
    """
    Plot feature importance if available.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model
        output_dir (str): Output directory for plots
        top_n (int): Number of top features to plot
    """
    importance = None
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        print(f"Feature importance not available for {model_name}")
        return
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
    plt.yticks(range(len(top_features)), 
               [f"{feat[:30]}..." if len(feat) > 30 else feat for feat in top_features['feature']])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
        plt.text(imp + max(top_features['importance']) * 0.01, i, f'{imp:.3f}', 
                va='center', ha='left', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.pdf"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save importance data
    importance_df.to_csv(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"), index=False)
    
    print(f"Saved feature importance plot to {output_dir}/{filename}")

def save_model_and_results(model, model_name, best_params, cv_score, test_metrics, 
                          feature_names, scaler, output_dir):
    """
    Save trained model and results.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        best_params (dict): Best hyperparameters
        cv_score (float): Cross-validation score
        test_metrics (dict): Test set metrics
        feature_names (list): List of feature names
        scaler: Feature scaler (if used)
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, os.path.join(output_dir, model_filename))
    
    # Save scaler if used
    if scaler is not None:
        scaler_filename = f"{model_name.lower().replace(' ', '_')}_scaler.joblib"
        joblib.dump(scaler, os.path.join(output_dir, scaler_filename))
    
    # Save model info
    info_filename = f"{model_name.lower().replace(' ', '_')}_info.txt"
    with open(os.path.join(output_dir, info_filename), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Cross-validation R²: {cv_score:.4f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.4f}\n")
        f.write(f"Test MSE: {test_metrics['mse']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"Test RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Feature scaling: {'Yes' if scaler is not None else 'No'}\n")
    
    # Save feature names
    features_filename = f"{model_name.lower().replace(' ', '_')}_features.txt"
    with open(os.path.join(output_dir, features_filename), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Saved model and results to {output_dir}")
    print(f"Model file: {model_filename}")
    if scaler is not None:
        print(f"Scaler file: {scaler_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Train and save the best model for virus-DO prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py -i processed/ -m "Random Forest" -o model/
  python train_model.py -i processed/ -f results/selected_features.txt -m "XGBoost" -o model/ --scale
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing processed data')
    parser.add_argument('-f', '--features', default=None,
                       help='Path to selected features file')
    parser.add_argument('-m', '--model', required=True,
                       help='Model name to train')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for model and results')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--scale', action='store_true',
                       help='Apply feature scaling')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--plot-top-features', type=int, default=20,
                       help='Number of top features to plot (default: 20)')
    
    args = parser.parse_args()
    
    print("Starting model training...")
    print(f"Input directory: {args.input}")
    print(f"Features file: {args.features}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print(f"Feature scaling: {args.scale}")
    print(f"Hyperparameter tuning: {not args.no_tune}")
    
    # Load data
    X, y, feature_names = load_data_and_features(args.input, args.features)
    
    # Scale features if requested
    scaler = None
    if args.scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), 
                               index=X.index, columns=X.columns)
        X = X_scaled
        print("Applied feature scaling")
    
    # Train model
    model, best_params, cv_score, test_metrics, X_test, y_test, y_pred = train_and_tune_model(
        X, y, args.model, args.cv_folds, args.random_state, not args.no_tune
    )
    
    # Create plots
    plot_predictions(y_test, y_pred, args.model, args.output)
    plot_feature_importance(model, feature_names, args.model, args.output, args.plot_top_features)
    
    # Save model and results
    save_model_and_results(model, args.model, best_params, cv_score, test_metrics, 
                          feature_names, scaler, args.output)
    
    print("\nModel training completed successfully!")
    print(f"Model: {args.model}")
    print(f"Cross-validation R²: {cv_score:.4f}")
    print(f"Test set R²: {test_metrics['r2']:.4f}")
    print(f"Test set RMSE: {test_metrics['rmse']:.4f}")
    print(f"Number of features: {len(feature_names)}")

if __name__ == '__main__':
    main()