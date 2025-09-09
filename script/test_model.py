#!/usr/bin/env python3
"""
Model Testing Script for Virus-DO Prediction

This script loads a trained model and tests it on new data.

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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler(model_dir, model_name):
    """
    Load trained model and scaler.
    
    Args:
        model_dir (str): Directory containing model files
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, scaler, feature_names)
    """
    try:
        # Load model
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        model_path = os.path.join(model_dir, model_filename)
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load scaler if exists
        scaler_filename = f"{model_name.lower().replace(' ', '_')}_scaler.joblib"
        scaler_path = os.path.join(model_dir, scaler_filename)
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        
        # Load feature names
        features_filename = f"{model_name.lower().replace(' ', '_')}_features.txt"
        features_path = os.path.join(model_dir, features_filename)
        feature_names = []
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(feature_names)} feature names")
        
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_test_data(test_file, feature_names=None):
    """
    Load test data.
    
    Args:
        test_file (str): Path to test data file
        feature_names (list): List of required feature names
        
    Returns:
        pd.DataFrame: Test features
    """
    try:
        test_data = pd.read_csv(test_file, index_col=0)
        print(f"Loaded test data: {test_data.shape}")
        
        # Filter to required features if specified
        if feature_names:
            available_features = [f for f in feature_names if f in test_data.columns]
            missing_features = [f for f in feature_names if f not in test_data.columns]
            
            if missing_features:
                print(f"Warning: {len(missing_features)} features missing from test data")
                print(f"Missing features: {missing_features[:5]}..." if len(missing_features) > 5 else f"Missing features: {missing_features}")
            
            test_data = test_data[available_features]
            print(f"Using {len(available_features)} features for prediction")
        
        return test_data
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

def load_test_targets(targets_file, sample_names):
    """
    Load test targets if available.
    
    Args:
        targets_file (str): Path to targets file
        sample_names (list): List of sample names
        
    Returns:
        pd.Series or None: Test targets
    """
    if not targets_file or not os.path.exists(targets_file):
        return None
    
    try:
        targets = pd.read_csv(targets_file, index_col=0).squeeze()
        
        # Filter to test samples
        common_samples = set(targets.index).intersection(set(sample_names))
        if len(common_samples) > 0:
            targets = targets.loc[list(common_samples)]
            print(f"Loaded {len(targets)} test targets")
            return targets
        else:
            print("No matching samples found in targets file")
            return None
            
    except Exception as e:
        print(f"Error loading test targets: {e}")
        return None

def make_predictions(model, X_test, scaler=None):
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        scaler: Feature scaler (if used)
        
    Returns:
        np.array: Predictions
    """
    try:
        # Apply scaling if scaler is provided
        if scaler is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
            X_test = X_test_scaled
            print("Applied feature scaling to test data")
        
        # Make predictions
        predictions = model.predict(X_test)
        print(f"Generated {len(predictions)} predictions")
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate prediction performance.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Evaluation metrics
    """
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    return metrics

def plot_test_results(y_true, y_pred, model_name, output_dir):
    """
    Plot test results.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        model_name (str): Name of the model
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual DO')
    ax1.set_ylabel('Predicted DO')
    ax1.set_title(f'{model_name} - Test Set Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display R²
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
    
    # Distribution of residuals
    ax3 = axes[2]
    ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{model_name} - Residuals Distribution')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_test_results.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved test results plot to {output_dir}/{filename}")

def save_predictions(predictions, sample_names, output_file, y_true=None):
    """
    Save predictions to file.
    
    Args:
        predictions (array): Predicted values
        sample_names (list): Sample names
        output_file (str): Output file path
        y_true (array): True values (optional)
    """
    try:
        # Create DataFrame
        results_df = pd.DataFrame({
            'Sample': sample_names,
            'Predicted_DO': predictions
        })
        
        if y_true is not None:
            results_df['Actual_DO'] = y_true
            results_df['Residual'] = y_true - predictions
            results_df['Absolute_Error'] = np.abs(y_true - predictions)
        
        # Save to file
        results_df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")
        
    except Exception as e:
        print(f"Error saving predictions: {e}")

def save_test_metrics(metrics, model_name, output_dir):
    """
    Save test metrics to file.
    
    Args:
        metrics (dict): Evaluation metrics
        model_name (str): Name of the model
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{model_name.lower().replace(' ', '_')}_test_metrics.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Test Set Performance - {model_name}\n")
        f.write("=" * 40 + "\n")
        f.write(f"R² Score: {metrics['r2']:.4f}\n")
        f.write(f"Mean Squared Error: {metrics['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error: {metrics['rmse']:.4f}\n")
        f.write(f"Mean Absolute Error: {metrics['mae']:.4f}\n")
    
    print(f"Saved test metrics to {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description='Test trained model on new data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py -m model/ -n "Random Forest" -t test/virus.csv -o results/
  python test_model.py -m model/ -n "XGBoost" -t test/virus.csv -y test/targets.csv -o results/
        """
    )
    
    parser.add_argument('-m', '--model-dir', required=True,
                       help='Directory containing trained model files')
    parser.add_argument('-n', '--model-name', required=True,
                       help='Name of the model to load')
    parser.add_argument('-t', '--test-data', required=True,
                       help='Path to test data CSV file')
    parser.add_argument('-y', '--test-targets', default=None,
                       help='Path to test targets CSV file (optional)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--predictions-file', default='predictions.csv',
                       help='Name of predictions output file (default: predictions.csv)')
    
    args = parser.parse_args()
    
    print("Starting model testing...")
    print(f"Model directory: {args.model_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Test data: {args.test_data}")
    print(f"Test targets: {args.test_targets}")
    print(f"Output directory: {args.output}")
    
    # Load model and scaler
    model, scaler, feature_names = load_model_and_scaler(args.model_dir, args.model_name)
    
    # Load test data
    X_test = load_test_data(args.test_data, feature_names)
    
    # Load test targets if available
    y_test = load_test_targets(args.test_targets, X_test.index.tolist())
    
    # Make predictions
    predictions = make_predictions(model, X_test, scaler)
    
    # Evaluate if targets are available
    if y_test is not None:
        # Align predictions with targets
        common_samples = set(X_test.index).intersection(set(y_test.index))
        if len(common_samples) > 0:
            X_test_aligned = X_test.loc[list(common_samples)]
            y_test_aligned = y_test.loc[list(common_samples)]
            predictions_aligned = model.predict(scaler.transform(X_test_aligned) if scaler else X_test_aligned)
            
            # Evaluate performance
            metrics = evaluate_predictions(y_test_aligned, predictions_aligned)
            
            print("\nTest Set Performance:")
            print(f"R² Score: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            
            # Create plots
            plot_test_results(y_test_aligned, predictions_aligned, args.model_name, args.output)
            
            # Save metrics
            save_test_metrics(metrics, args.model_name, args.output)
            
            # Save predictions with targets
            predictions_file = os.path.join(args.output, args.predictions_file)
            save_predictions(predictions_aligned, list(common_samples), predictions_file, y_test_aligned.values.flatten())
        else:
            print("No common samples between test data and targets")
    else:
        print("No test targets provided - saving predictions only")
        
        # Save predictions without targets
        predictions_file = os.path.join(args.output, args.predictions_file)
        save_predictions(predictions, X_test.index.tolist(), predictions_file)
    
    print("\nModel testing completed successfully!")
    print(f"Predictions saved to {os.path.join(args.output, args.predictions_file)}")
    if y_test is not None:
        print(f"Performance metrics saved to {args.output}")

if __name__ == '__main__':
    main()