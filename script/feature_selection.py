#!/usr/bin/env python3
"""
Feature Selection Script for Virus-DO Prediction Model

This script performs correlation analysis between virus abundance and dissolved oxygen,
selects key features based on correlation strength and statistical significance.

Author: Generated Script
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_processed_data(data_dir):
    """
    Load processed features and targets.
    
    Args:
        data_dir (str): Directory containing processed data
        
    Returns:
        tuple: (X, y) features and targets
    """
    try:
        X = pd.read_csv(os.path.join(data_dir, 'features.csv'), index_col=0)
        y = pd.read_csv(os.path.join(data_dir, 'targets.csv'), index_col=0).squeeze()
        print(f"Loaded processed data - Features: {X.shape}, Targets: {y.shape}")
        return X, y
    except Exception as e:
        print(f"Error loading processed data: {e}")
        sys.exit(1)

def calculate_correlations(X, y, method='pearson'):
    """
    Calculate correlations between each feature and target.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        method (str): Correlation method ('pearson' or 'spearman')
        
    Returns:
        pd.DataFrame: Correlation results with p-values
    """
    correlations = []
    
    print(f"Calculating {method} correlations for {X.shape[1]} features...")
    
    for feature in X.columns:
        feature_values = X[feature]
        
        # Remove samples with missing values
        mask = ~(pd.isna(feature_values) | pd.isna(y))
        if mask.sum() < 3:  # Need at least 3 samples
            continue
            
        feature_clean = feature_values[mask]
        y_clean = y[mask]
        
        # Calculate correlation
        if method == 'pearson':
            corr, p_value = pearsonr(feature_clean, y_clean)
        elif method == 'spearman':
            corr, p_value = spearmanr(feature_clean, y_clean)
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
        
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'abs_correlation': abs(corr),
            'n_samples': mask.sum()
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    print(f"Calculated correlations for {len(corr_df)} features")
    return corr_df

def select_features(corr_df, min_correlation=0.1, max_p_value=0.05, top_n=None):
    """
    Select features based on correlation criteria.
    
    Args:
        corr_df (pd.DataFrame): Correlation results
        min_correlation (float): Minimum absolute correlation threshold
        max_p_value (float): Maximum p-value threshold
        top_n (int): Maximum number of features to select
        
    Returns:
        list: Selected feature names
    """
    # Apply filters
    filtered_df = corr_df[
        (corr_df['abs_correlation'] >= min_correlation) & 
        (corr_df['p_value'] <= max_p_value)
    ]
    
    # Select top N if specified
    if top_n is not None:
        filtered_df = filtered_df.head(top_n)
    
    selected_features = filtered_df['feature'].tolist()
    
    print(f"Selected {len(selected_features)} features:")
    print(f"  - Min correlation: {min_correlation}")
    print(f"  - Max p-value: {max_p_value}")
    if top_n:
        print(f"  - Top N: {top_n}")
    
    return selected_features, filtered_df

def plot_correlation_distribution(corr_df, output_dir):
    """
    Plot correlation distribution and save figure.
    
    Args:
        corr_df (pd.DataFrame): Correlation results
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Correlation histogram
    axes[0, 0].hist(corr_df['correlation'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Correlations')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Absolute correlation histogram
    axes[0, 1].hist(corr_df['abs_correlation'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Absolute Correlation Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Absolute Correlations')
    
    # P-value histogram
    axes[1, 0].hist(corr_df['p_value'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('P-value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of P-values')
    axes[1, 0].axvline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1, 0].legend()
    
    # Correlation vs P-value scatter
    scatter = axes[1, 1].scatter(corr_df['abs_correlation'], -np.log10(corr_df['p_value']), 
                                alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Absolute Correlation')
    axes[1, 1].set_ylabel('-log10(P-value)')
    axes[1, 1].set_title('Correlation vs Significance')
    axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation analysis plot to {output_dir}/correlation_analysis.pdf")

def plot_top_features(corr_df, top_n=20, output_dir='.'):
    """
    Plot top correlated features.
    
    Args:
        corr_df (pd.DataFrame): Correlation results
        top_n (int): Number of top features to plot
        output_dir (str): Output directory for plots
    """
    top_features = corr_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_features['correlation']]
    
    bars = plt.barh(range(len(top_features)), top_features['correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), 
               [f"{feat[:30]}..." if len(feat) > 30 else feat for feat in top_features['feature']])
    plt.xlabel('Correlation with DO')
    plt.title(f'Top {top_n} Features by Absolute Correlation with Dissolved Oxygen')
    plt.grid(axis='x', alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, top_features['correlation'])):
        plt.text(corr + (0.01 if corr > 0 else -0.01), i, f'{corr:.3f}', 
                va='center', ha='left' if corr > 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved top features plot to {output_dir}/top_features.pdf")

def save_feature_results(selected_features, corr_df, output_dir):
    """
    Save feature selection results.
    
    Args:
        selected_features (list): Selected feature names
        corr_df (pd.DataFrame): Full correlation results
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save selected features list
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    # Save full correlation results
    corr_df.to_csv(os.path.join(output_dir, 'correlation_results.csv'), index=False)
    
    # Save selected features correlation results
    selected_corr = corr_df[corr_df['feature'].isin(selected_features)]
    selected_corr.to_csv(os.path.join(output_dir, 'selected_features_correlations.csv'), index=False)
    
    print(f"Saved feature selection results to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Perform feature selection based on correlation with dissolved oxygen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_selection.py -i processed/ -o results/ --min-corr 0.1 --max-pval 0.05
  python feature_selection.py -i processed/ -o results/ --top-n 50 --method spearman
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing processed data')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--min-corr', type=float, default=0.1,
                       help='Minimum absolute correlation threshold (default: 0.1)')
    parser.add_argument('--max-pval', type=float, default=0.05,
                       help='Maximum p-value threshold (default: 0.05)')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Maximum number of features to select')
    parser.add_argument('--method', choices=['pearson', 'spearman'], default='pearson',
                       help='Correlation method (default: pearson)')
    parser.add_argument('--plot-top', type=int, default=20,
                       help='Number of top features to plot (default: 20)')
    
    args = parser.parse_args()
    
    print("Starting feature selection analysis...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Correlation method: {args.method}")
    print(f"Min correlation: {args.min_corr}")
    print(f"Max p-value: {args.max_pval}")
    
    # Load data
    X, y = load_processed_data(args.input)
    
    # Calculate correlations
    corr_df = calculate_correlations(X, y, method=args.method)
    
    # Select features
    selected_features, selected_corr_df = select_features(
        corr_df, 
        min_correlation=args.min_corr,
        max_p_value=args.max_pval,
        top_n=args.top_n
    )
    
    # Create plots
    plot_correlation_distribution(corr_df, args.output)
    plot_top_features(corr_df, top_n=args.plot_top, output_dir=args.output)
    
    # Save results
    save_feature_results(selected_features, corr_df, args.output)
    
    print("\nFeature selection completed successfully!")
    print(f"Total features analyzed: {len(corr_df)}")
    print(f"Features selected: {len(selected_features)}")
    print(f"Selection rate: {len(selected_features)/len(corr_df)*100:.1f}%")
    
    if len(selected_features) > 0:
        print(f"\nTop 5 selected features:")
        for i, (_, row) in enumerate(selected_corr_df.head(5).iterrows()):
            print(f"  {i+1}. {row['feature'][:50]}... (r={row['correlation']:.3f}, p={row['p_value']:.3e})")

if __name__ == '__main__':
    main()