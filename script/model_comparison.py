#!/usr/bin/env python3
"""
Machine Learning Model Comparison Script for Virus-DO Prediction

This script compares different machine learning algorithms for predicting
dissolved oxygen from virus abundance data.

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
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
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
    print("Warning: XGBoost not available. Install with: pip install xgboost")

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

def get_models(task_type='regression'):
    """
    Define machine learning models to compare.
    
    Args:
        task_type (str): 'regression' or 'classification'
    
    Returns:
        dict: Dictionary of model name to model instance
    """
    if task_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR (Linear)': SVR(kernel='linear', C=1.0),
            'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
    
    elif task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVC (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
            'SVC (Linear)': SVC(kernel='linear', C=1.0, probability=True, random_state=42),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        if XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42)
    
    return models

def evaluate_model(model, X, y, cv_folds=5, random_state=42, task_type='regression'):
    """
    Evaluate a model using cross-validation.
    
    Args:
        model: Scikit-learn model instance
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        task_type (str): 'regression' or 'classification'
        
    Returns:
        dict: Evaluation metrics
    """
    # Cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    if task_type == 'regression':
        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        # MSE scores (negative, so we negate them)
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        
        # MAE scores (negative, so we negate them)
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        
        # Train-test split for additional metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mse_mean': mse_scores.mean(),
            'mse_std': mse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'test_r2': r2_score(y_test, y_pred),
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'cv_scores': r2_scores
        }
    
    elif task_type == 'classification':
        # Convert continuous target to binary classification
        # Use median as threshold
        y_median = y.median()
        y_binary = (y > y_median).astype(int)
        
        # AUC scores
        auc_scores = cross_val_score(model, X, y_binary, cv=kf, scoring='roc_auc')
        
        # Train-test split for additional metrics
        X_train, X_test, y_train_binary, y_test_binary = train_test_split(
            X, y_binary, test_size=0.2, random_state=random_state
        )
        
        # Fit model and predict probabilities
        model.fit(X_train, y_train_binary)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = model.predict(X_test)
        
        # Calculate AUC
        test_auc = roc_auc_score(y_test_binary, y_pred_proba)
        
        return {
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
            'test_auc': test_auc,
            'cv_scores': auc_scores,
            'threshold': y_median
        }

def compare_models(X, y, cv_folds=5, random_state=42, task_type='regression'):
    """
    Compare multiple machine learning models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        task_type (str): 'regression' or 'classification'
        
    Returns:
        pd.DataFrame: Comparison results
    """
    models = get_models(task_type)
    results = []
    
    print(f"Comparing {len(models)} models with {cv_folds}-fold cross-validation...")
    print(f"Task type: {task_type}")
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        try:
            metrics = evaluate_model(model, X, y, cv_folds, random_state, task_type)
            metrics['model'] = name
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if task_type == 'regression':
        results_df = results_df.sort_values('r2_mean', ascending=False)
    elif task_type == 'classification':
        results_df = results_df.sort_values('auc_mean', ascending=False)
    
    return results_df

def plot_model_comparison(results_df, output_dir, task_type='regression'):
    """
    Create visualization of model comparison results.
    
    Args:
        results_df (pd.DataFrame): Model comparison results
        output_dir (str): Output directory for plots
        task_type (str): 'regression' or 'classification'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    
    if task_type == 'regression':
        # Create subplots for regression
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(results_df)), results_df['r2_mean'], 
                       yerr=results_df['r2_std'], capsize=5, alpha=0.7)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Comparison - R² Score (Cross-Validation)')
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, (bar, score) in enumerate(zip(bars1, results_df['r2_mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # MSE comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(results_df)), results_df['mse_mean'], 
                       yerr=results_df['mse_std'], capsize=5, alpha=0.7, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('Model Comparison - MSE (Cross-Validation)')
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # MAE comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(results_df)), results_df['mae_mean'], 
                       yerr=results_df['mae_std'], capsize=5, alpha=0.7, color='green')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.set_title('Model Comparison - MAE (Cross-Validation)')
        ax3.set_xticks(range(len(results_df)))
        ax3.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Test set performance
        ax4 = axes[1, 1]
        x_pos = np.arange(len(results_df))
        width = 0.35
        
        bars4a = ax4.bar(x_pos - width/2, results_df['r2_mean'], width, 
                        label='CV R²', alpha=0.7)
        bars4b = ax4.bar(x_pos + width/2, results_df['test_r2'], width, 
                        label='Test R²', alpha=0.7)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R² Score')
        ax4.set_title('Cross-Validation vs Test Set Performance')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    elif task_type == 'classification':
        # Create subplots for classification
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(results_df)), results_df['auc_mean'], 
                       yerr=results_df['auc_std'], capsize=5, alpha=0.7, color='purple')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Model Comparison - AUC Score (Cross-Validation)')
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add values on bars
        for i, (bar, score) in enumerate(zip(bars1, results_df['auc_mean'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Test set performance comparison
        ax2 = axes[1]
        x_pos = np.arange(len(results_df))
        width = 0.35
        
        bars2a = ax2.bar(x_pos - width/2, results_df['auc_mean'], width, 
                        label='CV AUC', alpha=0.7, color='purple')
        bars2b = ax2.bar(x_pos + width/2, results_df['test_auc'], width, 
                        label='Test AUC', alpha=0.7, color='darkviolet')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Cross-Validation vs Test Set AUC Performance')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison plot to {output_dir}/model_comparison.pdf")

def plot_cv_scores_distribution(results_df, output_dir, task_type='regression'):
    """
    Plot distribution of cross-validation scores.
    
    Args:
        results_df (pd.DataFrame): Model comparison results
        output_dir (str): Output directory for plots
        task_type (str): 'regression' or 'classification'
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    cv_data = []
    labels = []
    
    for _, row in results_df.iterrows():
        cv_data.append(row['cv_scores'])
        labels.append(row['model'])
    
    # Create box plot
    box_plot = plt.boxplot(cv_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Models')
    
    if task_type == 'regression':
        plt.ylabel('R² Score')
        plt.title('Distribution of Cross-Validation R² Scores')
    elif task_type == 'classification':
        plt.ylabel('AUC Score')
        plt.title('Distribution of Cross-Validation AUC Scores')
        plt.ylim(0, 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_scores_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CV scores distribution plot to {output_dir}/cv_scores_distribution.pdf")

def save_results(results_df, output_dir, task_type='regression'):
    """
    Save model comparison results.
    
    Args:
        results_df (pd.DataFrame): Model comparison results
        output_dir (str): Output directory
        task_type (str): 'regression' or 'classification'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    results_save = results_df.drop('cv_scores', axis=1)  # Remove array column for CSV
    results_save.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
    
    # Save best model info
    best_model = results_df.iloc[0]
    with open(os.path.join(output_dir, 'best_model.txt'), 'w') as f:
        f.write(f"Best Model: {best_model['model']}\n")
        
        if task_type == 'regression':
            f.write(f"R² Score (CV): {best_model['r2_mean']:.4f} ± {best_model['r2_std']:.4f}\n")
            f.write(f"R² Score (Test): {best_model['test_r2']:.4f}\n")
            f.write(f"MSE (CV): {best_model['mse_mean']:.4f} ± {best_model['mse_std']:.4f}\n")
            f.write(f"MAE (CV): {best_model['mae_mean']:.4f} ± {best_model['mae_std']:.4f}\n")
        
        elif task_type == 'classification':
            f.write(f"AUC Score (CV): {best_model['auc_mean']:.4f} ± {best_model['auc_std']:.4f}\n")
            f.write(f"AUC Score (Test): {best_model['test_auc']:.4f}\n")
            f.write(f"Classification Threshold (median DO): {best_model['threshold']:.4f}\n")
    
    print(f"Saved model comparison results to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare machine learning models for virus-DO prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_comparison.py -i processed/ -o results/
  python model_comparison.py -i processed/ -f results/selected_features.txt -o results/ --cv-folds 10
  python model_comparison.py -i processed/ -o results/ --task-type classification
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing processed data')
    parser.add_argument('-f', '--features', default=None,
                       help='Path to selected features file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--scale-features', action='store_true',
                       help='Apply feature scaling')
    parser.add_argument('--task-type', choices=['regression', 'classification'], default='regression',
                       help='Task type: regression or classification (default: regression)')
    
    args = parser.parse_args()
    
    print("Starting model comparison...")
    print(f"Input directory: {args.input}")
    print(f"Features file: {args.features}")
    print(f"Output directory: {args.output}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Task type: {args.task_type}")
    
    # Load data
    X, y, feature_names = load_data_and_features(args.input, args.features)
    
    # Scale features if requested
    if args.scale_features:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), 
                               index=X.index, columns=X.columns)
        X = X_scaled
        print("Applied feature scaling")
    
    # Compare models
    results_df = compare_models(X, y, args.cv_folds, args.random_state, args.task_type)
    
    # Create plots
    plot_model_comparison(results_df, args.output, args.task_type)
    plot_cv_scores_distribution(results_df, args.output, args.task_type)
    
    # Save results
    save_results(results_df, args.output, args.task_type)
    
    print("\nModel comparison completed successfully!")
    
    if args.task_type == 'regression':
        print(f"\nTop 3 models by R² score:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            print(f"  {i+1}. {row['model']}: R² = {row['r2_mean']:.4f} ± {row['r2_std']:.4f}")
        
        best_model = results_df.iloc[0]
        print(f"\nBest model: {best_model['model']}")
        print(f"Cross-validation R²: {best_model['r2_mean']:.4f} ± {best_model['r2_std']:.4f}")
        print(f"Test set R²: {best_model['test_r2']:.4f}")
    
    elif args.task_type == 'classification':
        print(f"\nTop 3 models by AUC score:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            print(f"  {i+1}. {row['model']}: AUC = {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
        
        best_model = results_df.iloc[0]
        print(f"\nBest model: {best_model['model']}")
        print(f"Cross-validation AUC: {best_model['auc_mean']:.4f} ± {best_model['auc_std']:.4f}")
        print(f"Test set AUC: {best_model['test_auc']:.4f}")
        print(f"Classification threshold (median DO): {best_model['threshold']:.4f}")

if __name__ == '__main__':
    main()