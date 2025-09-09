# ViralDO-Predictor

A machine learning toolkit for predicting dissolved oxygen levels based on viral abundance data in aquatic environments.

## Overview

ViralDO-Predictor is a comprehensive machine learning pipeline designed to predict dissolved oxygen (DO) concentrations in aquatic ecosystems using viral abundance data. The toolkit supports both regression and classification tasks, offering multiple algorithms and comprehensive model evaluation capabilities.

## Features

- **Data Preprocessing**: Automated data loading, cleaning, and feature engineering
- **Feature Selection**: Correlation-based feature selection with statistical analysis
- **Multiple ML Algorithms**: Support for 10+ machine learning algorithms including:
  - Linear/Ridge/Lasso/Elastic Net Regression
  - Random Forest (Regressor/Classifier)
  - Gradient Boosting (Regressor/Classifier)
  - Support Vector Machines (SVR/SVC)
  - K-Nearest Neighbors
  - XGBoost (Regressor/Classifier)
  - Logistic Regression
- **Model Comparison**: Comprehensive model evaluation with cross-validation
- **Performance Metrics**: 
  - Regression: R², MSE, MAE
  - Classification: AUC scores, precision, recall
- **Visualization**: Automated generation of comparison plots and performance charts
- **Model Persistence**: Save and load trained models for future predictions

## Project Structure

```
ViralDO-Predictor/
├── data/                   # Raw data files
│   ├── abundance.csv       # Viral abundance data
│   └── Env1.csv           # Environmental data with DO measurements
├── script/                 # Main scripts
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── feature_selection.py     # Feature selection and correlation analysis
│   ├── model_comparison.py      # Model comparison and evaluation
│   ├── train_model.py           # Model training
│   ├── test_model.py            # Model testing
│   └── main_pipeline.py         # Complete pipeline execution
├── Training/               # Training data
│   ├── features.csv
│   └── targets.csv
├── test/                   # Test data
│   ├── virus.csv
│   └── targets.csv
├── model/                  # Saved models
├── Result/                 # Output results and visualizations
└── README.md
```

## Installation

### Prerequisites

- Python 3.7+
- Conda or pip package manager

### Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

Or using conda:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

## Quick Start

### 1. Complete Pipeline Execution

Run the entire pipeline from data preprocessing to model evaluation:

```bash
python script/main_pipeline.py
```

### 2. Individual Components

#### Data Preprocessing
```bash
python script/data_preprocessing.py -a data/abundance.csv -e data/Env1.csv -o Training
```

#### Feature Selection
```bash
python script/feature_selection.py -i Training -o Result -k 100
```

#### Model Comparison (Regression)
```bash
python script/model_comparison.py -i Training -f Result/selected_features.txt -o Result --task-type regression
```

#### Model Comparison (Classification)
```bash
python script/model_comparison.py -i Training -f Result/selected_features.txt -o Result --task-type classification
```

#### Train Best Model
```bash
python script/train_model.py -i Training -f Result/selected_features.txt -m model -n "Random Forest"
```

#### Test Model
```bash
python script/test_model.py -m model -n "Random Forest" -t test/virus.csv -y test/targets.csv -o Result
```

## Usage Examples

### Regression Task
Predict continuous dissolved oxygen values:

```bash
# Compare regression models
python script/model_comparison.py -i Training -f Result/selected_features.txt -o Result --task-type regression

# Train the best regression model
python script/train_model.py -i Training -f Result/selected_features.txt -m model -n "Random Forest"
```

### Classification Task
Classify dissolved oxygen levels (high/low based on median threshold):

```bash
# Compare classification models with AUC scores
python script/model_comparison.py -i Training -f Result/selected_features.txt -o Result --task-type classification

# Results will show AUC scores for each model
```

## Command Line Arguments

### model_comparison.py
- `-i, --input`: Input directory containing training data
- `-f, --features`: Path to selected features file
- `-o, --output`: Output directory for results
- `--task-type`: Task type ('regression' or 'classification')
- `--cv-folds`: Number of cross-validation folds (default: 5)

### train_model.py
- `-i, --input`: Input directory containing training data
- `-f, --features`: Path to selected features file
- `-m, --model-dir`: Directory to save the trained model
- `-n, --model-name`: Name of the model to train

### test_model.py
- `-m, --model-dir`: Directory containing the trained model
- `-n, --model-name`: Name of the model to test
- `-t, --test-features`: Path to test features file
- `-y, --test-targets`: Path to test targets file
- `-o, --output`: Output directory for test results

## Output Files

### Results Directory
- `model_comparison_results.csv`: Detailed model performance metrics
- `best_model.txt`: Information about the best performing model
- `model_comparison.png`: Model performance comparison chart
- `cv_scores_distribution.png`: Cross-validation scores distribution
- `correlation_analysis.png`: Feature correlation heatmap
- `selected_features.txt`: List of selected features

### Model Directory
- `{model_name}_model.joblib`: Serialized trained model
- `{model_name}_info.txt`: Model training information
- `{model_name}_features.txt`: Features used by the model
- `{model_name}_feature_importance.csv`: Feature importance scores

## Performance Metrics

### Regression Metrics
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error

### Classification Metrics
- **AUC Score**: Area Under the ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Example Results

### Regression Task Results
```
Top 3 models by R² score:
  1. Random Forest: R² = 0.6609 ± 0.1270
  2. K-Neighbors: R² = 0.6398 ± 0.1398
  3. XGBoost: R² = 0.6190 ± 0.2007
```

### Classification Task Results
```
Top 3 models by AUC score:
  1. Random Forest: AUC = 0.9076 ± 0.0344
  2. Gradient Boosting: AUC = 0.9042 ± 0.0390
  3. K-Neighbors: AUC = 0.8948 ± 0.0500
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
 

```

## Contact

For questions and support, please open an issue on GitHub.
