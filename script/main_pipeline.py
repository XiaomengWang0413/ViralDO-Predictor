#!/usr/bin/env python3
"""
Main Pipeline Script for Virus-DO Prediction Model

This script orchestrates the complete machine learning pipeline:
1. Data preprocessing
2. Feature selection
3. Model comparison
4. Model training
5. Model testing

Author: Generated Script
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """
    Run a command and handle errors.
    
    Args:
        command (list): Command to run
        description (str): Description of the command
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_file_exists(filepath, description):
    """
    Check if a file exists and print status.
    
    Args:
        filepath (str): Path to file
        description (str): Description of the file
        
    Returns:
        bool: True if file exists
    """
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} not found: {filepath}")
        return False

def find_best_model(results_file):
    """
    Find the best model from comparison results.
    
    Args:
        results_file (str): Path to model comparison results
        
    Returns:
        str: Name of the best model
    """
    try:
        results_df = pd.read_csv(results_file)
        best_model = results_df.iloc[0]['model']
        print(f"Best model identified: {best_model}")
        return best_model
    except Exception as e:
        print(f"Error reading results file: {e}")
        return None

def create_test_dataset(abundance_file, selected_features_file, output_file):
    """
    Create test dataset from non-selected features.
    
    Args:
        abundance_file (str): Path to abundance data
        selected_features_file (str): Path to selected features
        output_file (str): Path to output test file
    """
    try:
        # Load abundance data
        abundance_df = pd.read_csv(abundance_file, index_col=0)
        
        # Load selected features
        with open(selected_features_file, 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        
        # Get non-selected features
        all_features = set(abundance_df.index)
        selected_features_set = set(selected_features)
        non_selected_features = list(all_features - selected_features_set)
        
        # Create test dataset with non-selected features
        test_df = abundance_df.loc[non_selected_features].T
        
        # Save test dataset
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        test_df.to_csv(output_file)
        
        print(f"Created test dataset with {len(non_selected_features)} features")
        print(f"Saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating test dataset: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline for virus-DO prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Data preprocessing: Match virus abundance with environmental data
  2. Feature selection: Identify key virus features correlated with DO
  3. Model comparison: Compare different ML algorithms
  4. Model training: Train the best performing model
  5. Model testing: Test model on non-selected features

Examples:
  python main_pipeline.py --abundance data/abundance.csv --env data/Env1.csv --output-dir results/
  python main_pipeline.py --abundance data/abundance.csv --env data/Env1.csv --output-dir results/ --min-corr 0.15 --max-pval 0.01
        """
    )
    
    # Input files
    parser.add_argument('--abundance', required=True,
                       help='Path to virus abundance CSV file')
    parser.add_argument('--env', required=True,
                       help='Path to environmental parameters CSV file')
    
    # Output directory
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for all results')
    
    # Feature selection parameters
    parser.add_argument('--min-corr', type=float, default=0.1,
                       help='Minimum absolute correlation threshold (default: 0.1)')
    parser.add_argument('--max-pval', type=float, default=0.05,
                       help='Maximum p-value threshold (default: 0.05)')
    parser.add_argument('--top-features', type=int, default=None,
                       help='Maximum number of features to select')
    parser.add_argument('--corr-method', choices=['pearson', 'spearman'], default='pearson',
                       help='Correlation method (default: pearson)')
    
    # Model parameters
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--scale-features', action='store_true',
                       help='Apply feature scaling')
    parser.add_argument('--no-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    
    # Pipeline control
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-feature-selection', action='store_true',
                       help='Skip feature selection step')
    parser.add_argument('--skip-model-comparison', action='store_true',
                       help='Skip model comparison step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--skip-testing', action='store_true',
                       help='Skip model testing step')
    parser.add_argument('--model-name', default=None,
                       help='Specific model to train (skip comparison)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    print("Virus-DO Prediction Model Pipeline")
    print("=" * 50)
    print(f"Abundance file: {args.abundance}")
    print(f"Environment file: {args.env}")
    print(f"Output directory: {output_dir}")
    print(f"Feature selection: min_corr={args.min_corr}, max_pval={args.max_pval}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Feature scaling: {args.scale_features}")
    
    success = True
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)
        
        preprocessing_cmd = [
            'python', os.path.join(script_dir, 'data_preprocessing.py'),
            '-a', args.abundance,
            '-e', args.env,
            '-o', os.path.join(output_dir, 'processed'),
            '--verbose'
        ]
        
        if not run_command(preprocessing_cmd, "Data Preprocessing"):
            success = False
    
    # Check preprocessing outputs
    features_file = os.path.join(output_dir, 'processed', 'features.csv')
    targets_file = os.path.join(output_dir, 'processed', 'targets.csv')
    
    if not (check_file_exists(features_file, "Features file") and 
            check_file_exists(targets_file, "Targets file")):
        print("Preprocessing outputs not found. Exiting.")
        sys.exit(1)
    
    # Step 2: Feature Selection
    if not args.skip_feature_selection:
        print("\n" + "="*60)
        print("STEP 2: FEATURE SELECTION")
        print("="*60)
        
        feature_selection_cmd = [
            'python', os.path.join(script_dir, 'feature_selection.py'),
            '-i', os.path.join(output_dir, 'processed'),
            '-o', os.path.join(output_dir, 'features'),
            '--min-corr', str(args.min_corr),
            '--max-pval', str(args.max_pval),
            '--method', args.corr_method
        ]
        
        if args.top_features:
            feature_selection_cmd.extend(['--top-n', str(args.top_features)])
        
        if not run_command(feature_selection_cmd, "Feature Selection"):
            success = False
    
    # Check feature selection outputs
    selected_features_file = os.path.join(output_dir, 'features', 'selected_features.txt')
    if not check_file_exists(selected_features_file, "Selected features file"):
        print("Feature selection outputs not found. Exiting.")
        sys.exit(1)
    
    # Step 3: Model Comparison
    best_model = args.model_name
    if not args.skip_model_comparison and not best_model:
        print("\n" + "="*60)
        print("STEP 3: MODEL COMPARISON")
        print("="*60)
        
        model_comparison_cmd = [
            'python', os.path.join(script_dir, 'model_comparison.py'),
            '-i', os.path.join(output_dir, 'processed'),
            '-f', selected_features_file,
            '-o', os.path.join(output_dir, 'comparison'),
            '--cv-folds', str(args.cv_folds)
        ]
        
        if args.scale_features:
            model_comparison_cmd.append('--scale-features')
        
        if not run_command(model_comparison_cmd, "Model Comparison"):
            success = False
        
        # Find best model
        results_file = os.path.join(output_dir, 'comparison', 'model_comparison_results.csv')
        if check_file_exists(results_file, "Model comparison results"):
            best_model = find_best_model(results_file)
        
        if not best_model:
            print("Could not determine best model. Exiting.")
            sys.exit(1)
    
    elif not best_model:
        print("No model specified and comparison skipped. Please provide --model-name.")
        sys.exit(1)
    
    # Step 4: Model Training
    if not args.skip_training:
        print("\n" + "="*60)
        print(f"STEP 4: MODEL TRAINING ({best_model})")
        print("="*60)
        
        training_cmd = [
            'python', os.path.join(script_dir, 'train_model.py'),
            '-i', os.path.join(output_dir, 'processed'),
            '-f', selected_features_file,
            '-m', best_model,
            '-o', os.path.join(output_dir, 'model'),
            '--cv-folds', str(args.cv_folds)
        ]
        
        if args.scale_features:
            training_cmd.append('--scale')
        
        if args.no_tune:
            training_cmd.append('--no-tune')
        
        if not run_command(training_cmd, f"Model Training ({best_model})"):
            success = False
    
    # Step 5: Create Test Dataset and Test Model
    if not args.skip_testing:
        print("\n" + "="*60)
        print("STEP 5: MODEL TESTING")
        print("="*60)
        
        # Create test dataset from non-selected features
        test_data_file = os.path.join(output_dir, 'test', 'virus.csv')
        print("Creating test dataset from non-selected features...")
        create_test_dataset(args.abundance, selected_features_file, test_data_file)
        
        if check_file_exists(test_data_file, "Test dataset"):
            # Test the model
            testing_cmd = [
                'python', os.path.join(script_dir, 'test_model.py'),
                '-m', os.path.join(output_dir, 'model'),
                '-n', best_model,
                '-t', test_data_file,
                '-o', os.path.join(output_dir, 'results')
            ]
            
            if not run_command(testing_cmd, "Model Testing"):
                success = False
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    if success:
        print("✓ Pipeline completed successfully!")
        
        # Print key results
        print(f"\nKey Results:")
        print(f"- Best model: {best_model}")
        
        # Check for training results
        model_info_file = os.path.join(output_dir, 'model', f"{best_model.lower().replace(' ', '_')}_info.txt")
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Cross-validation R²' in line or 'Test R²' in line:
                        print(f"- {line.strip()}")
        
        print(f"\nOutput Structure:")
        print(f"- Processed data: {os.path.join(output_dir, 'processed')}")
        print(f"- Selected features: {os.path.join(output_dir, 'features')}")
        print(f"- Model comparison: {os.path.join(output_dir, 'comparison')}")
        print(f"- Trained model: {os.path.join(output_dir, 'model')}")
        print(f"- Test results: {os.path.join(output_dir, 'results')}")
        print(f"- Test dataset: {os.path.join(output_dir, 'test')}")
        
    else:
        print("✗ Pipeline completed with errors. Check the logs above.")
        sys.exit(1)

if __name__ == '__main__':
    main()