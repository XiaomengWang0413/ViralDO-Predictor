#!/usr/bin/env python3
"""
Data Preprocessing Script for Virus-DO Prediction Model

This script processes virus abundance data and environmental parameters,
matching samples and preparing data for machine learning analysis.

Author: Generated Script
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def load_abundance_data(abundance_file):
    """
    Load virus abundance data from CSV file.
    
    Args:
        abundance_file (str): Path to abundance.csv file
        
    Returns:
        pd.DataFrame: Abundance data with genes as rows and samples as columns
    """
    try:
        abundance_df = pd.read_csv(abundance_file, index_col=0)
        print(f"Loaded abundance data: {abundance_df.shape[0]} genes, {abundance_df.shape[1]} samples")
        return abundance_df
    except Exception as e:
        print(f"Error loading abundance data: {e}")
        sys.exit(1)

def load_environmental_data(env_file):
    """
    Load environmental parameters from CSV file.
    
    Args:
        env_file (str): Path to Env1.csv file
        
    Returns:
        pd.DataFrame: Environmental data with samples and DO values
    """
    try:
        env_df = pd.read_csv(env_file)
        print(f"Loaded environmental data: {len(env_df)} samples")
        return env_df
    except Exception as e:
        print(f"Error loading environmental data: {e}")
        sys.exit(1)

def match_samples(abundance_df, env_df):
    """
    Match samples between abundance and environmental data.
    
    Args:
        abundance_df (pd.DataFrame): Virus abundance data
        env_df (pd.DataFrame): Environmental data
        
    Returns:
        tuple: (matched_abundance, matched_env, common_samples)
    """
    # Get sample names from both datasets
    abundance_samples = set(abundance_df.columns)
    env_samples = set(env_df['Sample'])
    
    # Find common samples
    common_samples = abundance_samples.intersection(env_samples)
    print(f"Found {len(common_samples)} common samples")
    
    if len(common_samples) == 0:
        print("No common samples found between datasets!")
        sys.exit(1)
    
    # Filter data to common samples
    matched_abundance = abundance_df[list(common_samples)]
    matched_env = env_df[env_df['Sample'].isin(common_samples)].copy()
    matched_env = matched_env.set_index('Sample').loc[list(common_samples)]
    
    print(f"Matched data shape - Abundance: {matched_abundance.shape}, Environment: {matched_env.shape}")
    
    return matched_abundance, matched_env, list(common_samples)

def prepare_ml_data(abundance_df, env_df):
    """
    Prepare data for machine learning by transposing and combining.
    
    Args:
        abundance_df (pd.DataFrame): Matched abundance data
        env_df (pd.DataFrame): Matched environmental data
        
    Returns:
        tuple: (X, y) where X is features and y is target (DO)
    """
    # Transpose abundance data so samples are rows and genes are columns
    X = abundance_df.T
    
    # Get DO values as target
    y = env_df['DO']
    
    # Ensure same order
    X = X.loc[y.index]
    
    print(f"Prepared ML data - Features: {X.shape}, Target: {y.shape}")
    print(f"DO range: {y.min():.2f} - {y.max():.2f}")
    
    return X, y

def save_processed_data(X, y, output_dir):
    """
    Save processed data to files.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features and targets
    X.to_csv(os.path.join(output_dir, 'features.csv'))
    y.to_csv(os.path.join(output_dir, 'targets.csv'))
    
    print(f"Saved processed data to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess virus abundance and environmental data for ML analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_preprocessing.py -a data/abundance.csv -e data/Env1.csv -o processed/
  python data_preprocessing.py --abundance data/abundance.csv --env data/Env1.csv --output processed/
        """
    )
    
    parser.add_argument('-a', '--abundance', required=True,
                       help='Path to virus abundance CSV file')
    parser.add_argument('-e', '--env', required=True,
                       help='Path to environmental parameters CSV file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for processed data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("Starting data preprocessing...")
        print(f"Abundance file: {args.abundance}")
        print(f"Environment file: {args.env}")
        print(f"Output directory: {args.output}")
    
    # Load data
    abundance_df = load_abundance_data(args.abundance)
    env_df = load_environmental_data(args.env)
    
    # Match samples
    matched_abundance, matched_env, common_samples = match_samples(abundance_df, env_df)
    
    # Prepare ML data
    X, y = prepare_ml_data(matched_abundance, matched_env)
    
    # Save processed data
    save_processed_data(X, y, args.output)
    
    print("Data preprocessing completed successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of virus genes: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

if __name__ == '__main__':
    main()