#!/usr/bin/env python3
"""
Script: Baseline Logistic Regression Pipeline for Crash Prediction

Purpose:
This script builds an interpretable baseline logistic regression model that predicts 
whether a crash occurred near a road cluster using only structured road attributes 
(no images). The model computes residuals (actual - predicted probability) that 
capture visual risk unexplained by structured data, which will serve as labels for 
CNN regression to explain visual risk.

Functionality:
- Loads cluster dataset with train/val/test splits
- Selects only infrastructure/road attributes (excludes coordinates, density, etc.)
- Handles missing data appropriately (categorical → "missing", numeric → median)
- Builds preprocessing pipeline with one-hot encoding for categoricals
- Trains logistic regression model on train split only
- Evaluates on all splits with test set as primary metrics (precision, recall, F1, ROC AUC)
- Computes residuals for train/val/test splits
- Saves enriched datasets with residuals
- Saves trained model pipeline and metadata
- Generates coefficient interpretation report

How to run:
    python src/modeling/baseline_logistic_regression.py

Prerequisites:
    - Run create_cluster_dataset.py first to generate clusters_with_crashes.csv
    - Ensure data/processed/clusters_with_crashes.csv exists

Output:
    - data/processed/clusters_{train,val,test}_with_residuals.csv
    - models/baseline_logistic_regression.pkl
    - models/baseline_logistic_regression_metadata.json
    - reports/baseline_logistic_regression_coefficients.csv
    - reports/Regression_beforeCNN/ (comparison visualizations and metrics)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report, confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.style.use('default')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = project_root / 'data' / 'processed'
CLUSTERS_DATA_PATH = DATA_DIR / 'clusters_with_crashes.csv'

# Output data paths
CLUSTERS_TRAIN_RESIDUALS_PATH = DATA_DIR / 'clusters_train_with_residuals.csv'
CLUSTERS_VAL_RESIDUALS_PATH = DATA_DIR / 'clusters_val_with_residuals.csv'
CLUSTERS_TEST_RESIDUALS_PATH = DATA_DIR / 'clusters_test_with_residuals.csv'

# Model paths
MODELS_DIR = project_root / 'models'
MODEL_PATH = MODELS_DIR / 'baseline_logistic_regression.pkl'
METADATA_PATH = MODELS_DIR / 'baseline_logistic_regression_metadata.json'

# Report paths
REPORTS_DIR = project_root / 'reports'
COEFFICIENTS_PATH = REPORTS_DIR / 'baseline_logistic_regression_coefficients.csv'
VIZ_DIR = REPORTS_DIR / 'Regression_beforeCNN'


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_split_data(data_path):
    """
    Load the cluster dataset and split into train/val/test DataFrames.
    
    Parameters:
        data_path: Path to clusters_with_crashes.csv
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split
    """
    print("=" * 80)
    print("LOADING AND SPLITTING DATA")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total rows")
    
    # Split by existing split column
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Check target distribution
    print(f"\nTarget distribution (match_label):")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pos_count = split_df['match_label'].sum()
        pos_pct = (pos_count / len(split_df) * 100) if len(split_df) > 0 else 0
        print(f"  {split_name}: {pos_count} positive ({pos_pct:.1f}%), {len(split_df) - pos_count} negative")
    
    return train_df, val_df, test_df


def define_features():
    """
    Define which features to include in the model.
    
    Returns:
        tuple: (categorical_features, boolean_features, numeric_features, excluded_features)
    """
    # Categorical features (object type, will be one-hot encoded)
    categorical_features = [
        'highway', 'surface', 'lit', 'cycleway', 'sidewalk', 'oneway',
        'bridge', 'tunnel', 'junction', 'access', 'smoothness'
    ]
    
    # Boolean features (treated as categorical for one-hot encoding)
    boolean_features = ['is_intersection', 'near_traffic_signal']
    
    # Numeric features (float64 type)
    # Note: intersection_degree excluded as it's constant (always 0)
    numeric_features = [
        'road_segment_length_m', 'maxspeed', 'lanes', 'width'
    ]
    
    # Excluded features (to avoid leakage)
    excluded_features = [
        'lon', 'lat', 'amount_of_matches', 'crash_years', 'captured_at_years',
        'amount_of_images', 'list_of_thumb_1024_url', 'list_of_point_ids',
        'cluster_id', 'osm_way_id', 'split', 'match_label', 'intersection_degree'
    ]
    
    return categorical_features, boolean_features, numeric_features, excluded_features


def remove_constant_features(df, feature_cols):
    """
    Remove features that are constant (zero variance).
    
    Parameters:
        df: DataFrame to analyze
        feature_cols: List of feature column names
        
    Returns:
        list: List of features to keep (non-constant)
    """
    constant_features = []
    
    for col in feature_cols:
        if col in df.columns:
            unique_values = df[col].nunique()
            if unique_values <= 1:
                constant_features.append(col)
                print(f"  Removed constant feature: {col} (unique values: {unique_values})")
            else:
                # Check for practical constant (all values same except one)
                value_counts = df[col].value_counts()
                if len(value_counts) == 1:
                    constant_features.append(col)
                    print(f"  Removed constant feature: {col}")
    
    features_to_keep = [f for f in feature_cols if f not in constant_features]
    
    return features_to_keep, constant_features


def detect_separation(df, feature_cols, target_col='match_label'):
    """
    Detect features that exhibit complete separation.
    A feature has complete separation if one of its values perfectly predicts the target.
    
    Parameters:
        df: DataFrame to analyze
        feature_cols: List of feature column names
        target_col: Name of target column
        
    Returns:
        dict: Dictionary with separation information
    """
    separation_issues = {}
    
    for col in feature_cols:
        if col in df.columns:
            # Group by feature value and check target distribution
            groups = df.groupby(col)[target_col].agg(['count', 'sum', 'nunique'])
            
            # Check for complete separation
            for val, row in groups.iterrows():
                # If all samples have same target value
                if row['nunique'] == 1 and row['count'] > 0:
                    if col not in separation_issues:
                        separation_issues[col] = []
                    separation_issues[col].append({
                        'value': val,
                        'count': row['count'],
                        'target': int(row['sum'] / row['count'])
                    })
    
    # Print summary
    if len(separation_issues) > 0:
        print(f"\n  WARNING: Found {len(separation_issues)} features with complete separation:")
        for col, issues in separation_issues.items():
            print(f"    {col}: {len(issues)} problematic value groups")
    
    return separation_issues


def handle_missing_data(train_df, val_df, test_df, categorical_features, 
                        boolean_features, numeric_features):
    """
    Handle missing data in categorical and numeric columns.
    
    Categorical: Replace NaN with literal string "missing"
    Numeric: Replace NaN with median computed from train split only
    
    Parameters:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames with missing values handled
    """
    print("\n" + "=" * 80)
    print("HANDLING MISSING DATA")
    print("=" * 80)
    
    all_categorical = categorical_features + boolean_features
    
    # Categorical: Replace NaN with "missing"
    print("\nCategorical features: replacing NaN with 'missing'")
    for col in all_categorical:
        if col in train_df.columns:
            train_missing = train_df[col].isna().sum()
            val_missing = val_df[col].isna().sum()
            test_missing = test_df[col].isna().sum()
            
            if train_missing > 0 or val_missing > 0 or test_missing > 0:
                train_df[col] = train_df[col].fillna('missing')
                val_df[col] = val_df[col].fillna('missing')
                test_df[col] = test_df[col].fillna('missing')
                print(f"  {col}: {train_missing} train, {val_missing} val, {test_missing} test → filled with 'missing'")
    
    # Numeric: Replace NaN with median from train split only
    print("\nNumeric features: replacing NaN with median (computed from train only)")
    for col in numeric_features:
        if col in train_df.columns:
            train_missing = train_df[col].isna().sum()
            val_missing = val_df[col].isna().sum()
            test_missing = test_df[col].isna().sum()
            
            if train_missing > 0 or val_missing > 0 or test_missing > 0:
                train_median = train_df[col].median()
                train_df[col] = train_df[col].fillna(train_median)
                val_df[col] = val_df[col].fillna(train_median)
                test_df[col] = test_df[col].fillna(train_median)
                print(f"  {col}: median={train_median:.2f}, {train_missing} train, {val_missing} val, {test_missing} test → filled")
    
    return train_df, val_df, test_df


def build_preprocessing_pipeline(categorical_features, boolean_features, numeric_features):
    """
    Build sklearn preprocessing pipeline with ColumnTransformer.
    
    Parameters:
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("\n" + "=" * 80)
    print("BUILDING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    all_categorical = categorical_features + boolean_features
    
    # Categorical preprocessing: one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Numeric preprocessing: standardize (missing values already handled)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, all_categorical),
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop'
    )
    
    print(f"\nPreprocessing pipeline created:")
    print(f"  Categorical features ({len(all_categorical)}): {', '.join(all_categorical)}")
    print(f"  Numeric features ({len(numeric_features)}): {', '.join(numeric_features)}")
    
    return preprocessor


def get_feature_names(categorical_features, boolean_features, numeric_features, preprocessor):
    """
    Get feature names after one-hot encoding.
    
    Parameters:
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        preprocessor: Fitted preprocessor pipeline
        
    Returns:
        list: Feature names after encoding
    """
    feature_names = []
    all_categorical = categorical_features + boolean_features
    
    # Get categorical feature names from one-hot encoder
    cat_transformer = preprocessor.transformers_[0][1]
    onehot = cat_transformer.named_steps['onehot']
    
    # Categorical feature names
    for i, feat in enumerate(all_categorical):
        categories = onehot.categories_[i]
        # Drop first category (due to drop='first')
        for cat in categories[1:]:
            feature_names.append(f"{feat}={cat}")
    
    # Numeric feature names
    feature_names.extend(numeric_features)
    
    return feature_names


def remove_constant_post_encoding(X_processed, feature_names):
    """
    Remove constant columns after one-hot encoding.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names after encoding
        
    Returns:
        tuple: (X_cleaned, feature_names_cleaned, removed_features)
    """
    import pandas as pd
    
    # Convert to DataFrame for easier column selection
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Find constant columns
    constant_cols = [col for col in X_df.columns if X_df[col].nunique() <= 1]
    
    if len(constant_cols) > 0:
        print(f"\n  Removing {len(constant_cols)} constant columns after encoding:")
        for col in constant_cols[:10]:  # Show first 10
            print(f"    - {col}")
        if len(constant_cols) > 10:
            print(f"    ... and {len(constant_cols) - 10} more")
        
        # Remove constant columns
        X_df_cleaned = X_df.drop(columns=constant_cols)
        feature_names_cleaned = [f for f in feature_names if f not in constant_cols]
        
        return X_df_cleaned.values, feature_names_cleaned, constant_cols
    else:
        return X_processed, feature_names, []


def remove_highly_correlated_features(X_processed, feature_names, threshold=0.7):
    """
    Remove highly correlated features.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names
        threshold: Correlation threshold (default 0.7)
        
    Returns:
        tuple: (X_cleaned, feature_names_cleaned, removed_features)
    """
    import pandas as pd
    import numpy as np
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Compute correlation matrix
    corr_matrix = X_df.corr().abs()
    
    # Find pairs with correlation > threshold
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'corr': corr_matrix.iloc[i, j]
                })
    
    # Remove one feature from each highly correlated pair
    features_to_remove = set()
    if len(high_corr_pairs) > 0:
        print(f"\n  Found {len(high_corr_pairs)} highly correlated pairs (>{threshold}):")
        for pair in high_corr_pairs[:10]:  # Show first 10
            print(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['corr']:.3f}")
            # Remove the second feature (arbitrary choice)
            features_to_remove.add(pair['feature2'])
        if len(high_corr_pairs) > 10:
            print(f"    ... and {len(high_corr_pairs) - 10} more pairs")
        
        print(f"\n  Removing {len(features_to_remove)} features to eliminate high correlations")
        X_df_cleaned = X_df.drop(columns=list(features_to_remove))
        feature_names_cleaned = [f for f in feature_names if f not in features_to_remove]
        
        return X_df_cleaned.values, feature_names_cleaned, list(features_to_remove)
    else:
        return X_processed, feature_names, []


def train_model(train_df, val_df, categorical_features, boolean_features, 
                numeric_features, preprocessor):
    """
    Train logistic regression model using preprocessing pipeline.
    
    Parameters:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        preprocessor: ColumnTransformer preprocessing pipeline
        
    Returns:
        tuple: (model, full_pipeline, X_train_processed, X_val_processed, feature_names)
    """
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 80)
    
    # Prepare features and target
    all_categorical = categorical_features + boolean_features
    feature_cols = all_categorical + numeric_features
    
    X_train = train_df[feature_cols]
    y_train = train_df['match_label'].astype(int)
    X_val = val_df[feature_cols]
    y_val = val_df['match_label'].astype(int)
    
    print(f"\nTraining set: {len(X_train)} samples, {len(feature_cols)} features")
    print(f"Validation set: {len(X_val)} samples")
    
    # Fit preprocessing on train
    print("\nFitting preprocessing pipeline on train split...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    print(f"  Train features after encoding: {X_train_processed.shape[1]} columns")
    print(f"  Val features after encoding: {X_val_processed.shape[1]} columns")
    
    # Train logistic regression with L2 regularization to handle separation
    print("\nTraining logistic regression model with L2 regularization...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced',  # Handle class imbalance
        C=1.0,  # L2 regularization strength (inverse of lambda)
        penalty='l2'
    )
    
    model.fit(X_train_processed, y_train)
    
    print("✓ Model training complete")
    
    # Note: We return feature_names for potential use, but the pipeline 
    # still uses the original preprocessing
    return model, preprocessor, X_train_processed, X_val_processed


def evaluate_model(model, X_processed, y, split_name="Validation"):
    """
    Evaluate model on a dataset.
    
    Parameters:
        model: Trained logistic regression model
        X_processed: Preprocessed features
        y: Target labels
        split_name: Name of the split for printing
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print(f"\n{split_name.upper()} EVALUATION")
    print("=" * 80)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_processed)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    print(f"\n{split_name} Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    if split_name.lower() == "validation":
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y, y_pred, target_names=['No Crash', 'Crash']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                No Crash  Crash")
        print(f"Actual No Crash   {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"       Crash      {cm[1,0]:5d}   {cm[1,1]:5d}")
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist()
        }
    else:
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }
    
    return metrics


def find_optimal_threshold(y_true, y_proba, criterion='f1'):
    """
    Find optimal probability threshold on validation set.
    criterion: 'f1' or 'youden' (sensitivity + specificity - 1)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if criterion == 'f1':
            score = f1_score(y_true, y_pred)
        else:
            from sklearn.metrics import recall_score
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            score = sens + spec - 1.0
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


def evaluate_with_threshold(model, X_processed, y, threshold, split_name="Validation"):
    y_proba = model.predict_proba(X_processed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba),
        'threshold': threshold
    }


def evaluate_all_splits(full_pipeline, train_df, val_df, test_df,
                        categorical_features, boolean_features, numeric_features):
    """
    Evaluate model on train, validation, and test splits.
    
    Parameters:
        full_pipeline: Complete sklearn pipeline
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        
    Returns:
        dict: Dictionary with metrics for each split
    """
    print("\n" + "=" * 80)
    print("EVALUATING ON ALL SPLITS")
    print("=" * 80)
    
    all_categorical = categorical_features + boolean_features
    feature_cols = all_categorical + numeric_features
    
    all_metrics = {}
    
    # Evaluate on each split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        X = split_df[feature_cols]
        y = split_df['match_label'].astype(int)
        
        # Get preprocessed features (need to extract model from pipeline)
        X_processed = full_pipeline.named_steps['preprocessor'].transform(X)
        model = full_pipeline.named_steps['model']
        
        # Evaluate
        metrics = evaluate_model(model, X_processed, y, split_name=split_name)
        all_metrics[split_name.lower()] = metrics
    
    return all_metrics


def create_comparison_visualizations(all_metrics, output_dir):
    """
    Create comparison visualizations for train/val/test metrics.
    
    Parameters:
        all_metrics: Dictionary with metrics for each split
        output_dir: Output directory for saving figures
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    splits = ['Train', 'Val', 'Test']
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Split': splits,
        'Accuracy': [all_metrics[s.lower()]['accuracy'] for s in splits],
        'Precision': [all_metrics[s.lower()]['precision'] for s in splits],
        'Recall': [all_metrics[s.lower()]['recall'] for s in splits],
        'F1-Score': [all_metrics[s.lower()]['f1_score'] for s in splits],
        'ROC AUC': [all_metrics[s.lower()]['roc_auc'] for s in splits]
    })
    
    # Save metrics to CSV
    metrics_df.to_csv(output_dir / 'baseline_metrics_comparison.csv', index=False)
    print(f"  ✓ Saved metrics comparison to baseline_metrics_comparison.csv")
    
    # 1. Bar plot comparing all metrics across splits
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(splits))
    width = 0.15
    
    # Define colors with test set highlighted
    colors = ['#3498db', '#e74c3c', '#f39c12']  # train (blue), val (red), test (orange)
    
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']):
        values = metrics_df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8, color=colors, edgecolor='white', linewidth=0.5)
        
        # Highlight test set bars
        test_bars = bars[2]  # Test is at index 2
        test_bars.set_edgecolor('#2c3e50')  # Dark blue edge
        test_bars.set_linewidth(2)
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Logistic Regression: Metrics Comparison Across Splits (Test Set Highlighted)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(splits)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']):
        values = metrics_df[metric].values
        for j, v in enumerate(values):
            color = '#2c3e50' if j == 2 else 'black'  # Highlight test values
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9, color=color, fontweight='bold' if j == 2 else 'normal')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_bars.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_bars.png")
    
    # 2. Line plot showing each metric across splits
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']:
        line = ax.plot(splits, metrics_df[metric].values, marker='o', linewidth=2, 
               label=metric, markersize=8)
        # Highlight test point
        ax.plot(splits[2], metrics_df[metric].values[2], marker='o', markersize=12, 
               color=line[0].get_color(), markeredgecolor='#2c3e50', markeredgewidth=2)
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Logistic Regression: Metrics Trends Across Splits (Test Set Highlighted)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_lines.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_lines.png")
    
    # 3. Heatmap of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = metrics_df.set_index('Split')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']].T
    
    if HAS_SEABORN:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1, ax=ax)
    else:
        # Fallback to matplotlib imshow
        im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                text = ax.text(j, i, f'{heatmap_data.values[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
    
    ax.set_title('Baseline Logistic Regression: Metrics Heatmap (Test Set as Primary)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_heatmap.png")
    
    # 4. Individual metric comparison (separate subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_dict = {
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1-Score': 'F1-Score',
        'ROC AUC': 'ROC AUC'
    }
    
    colors = ['#3498db', '#e74c3c', '#f39c12']  # train (blue), val (red), test (orange)
    
    for i, (metric_key, metric_label) in enumerate(metrics_dict.items()):
        ax = axes[i]
        values = metrics_df[metric_key].values
        bars = ax.bar(splits, values, alpha=0.8, color=colors, edgecolor='white', linewidth=0.5)
        
        # Highlight test bar
        bars[2].set_edgecolor('#2c3e50')  # Dark blue edge
        bars[2].set_linewidth(2)
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_label}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            color = '#2c3e50' if j == 2 else 'black'
            weight = 'bold' if j == 2 else 'normal'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, 
                   fontweight=weight, color=color)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    fig.suptitle('Baseline Logistic Regression: Individual Metric Comparisons', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_individual.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_individual.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def generate_visualizations(full_pipeline, train_df, val_df, test_df,
                           categorical_features, boolean_features, numeric_features,
                           output_dir):
    """
    Evaluate on all splits and generate comparison visualizations.
    
    Parameters:
        full_pipeline: Complete sklearn pipeline
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        output_dir: Output directory for saving figures
    """
    # Evaluate on all splits
    all_metrics = evaluate_all_splits(
        full_pipeline, train_df, val_df, test_df,
        categorical_features, boolean_features, numeric_features
    )
    
    # Create visualizations
    create_comparison_visualizations(all_metrics, output_dir)
    
    return all_metrics


def compute_residuals(full_pipeline, train_df, val_df, test_df, 
                     categorical_features, boolean_features, numeric_features):
    """
    Compute residuals for train/val/test splits.
    
    Residual = actual_label - predicted_probability
    
    Parameters:
        full_pipeline: Complete sklearn pipeline (preprocessing + model)
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames with residual column added
    """
    print("\n" + "=" * 80)
    print("COMPUTING RESIDUALS")
    print("=" * 80)
    
    all_categorical = categorical_features + boolean_features
    feature_cols = all_categorical + numeric_features
    
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        X = split_df[feature_cols]
        y = split_df['match_label'].astype(int)
        
        # Get predicted probabilities
        y_pred_proba = full_pipeline.predict_proba(X)[:, 1]
        
        # Compute residuals
        residuals = y - y_pred_proba
        
        # Add to DataFrame
        split_df['residual'] = residuals
        
        # Print statistics
        print(f"\n{split_name} residuals:")
        print(f"  Mean:   {residuals.mean():.4f}")
        print(f"  Std:    {residuals.std():.4f}")
        print(f"  Min:    {residuals.min():.4f}")
        print(f"  Max:    {residuals.max():.4f}")
        print(f"  Positive residuals: {(residuals > 0).sum()} ({(residuals > 0).sum()/len(residuals)*100:.1f}%)")
        print(f"  Negative residuals: {(residuals < 0).sum()} ({(residuals < 0).sum()/len(residuals)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_enriched_datasets(train_df, val_df, test_df, output_dir):
    """
    Save enriched datasets with residuals to CSV files.
    
    Parameters:
        train_df: Training DataFrame with residuals
        val_df: Validation DataFrame with residuals
        test_df: Test DataFrame with residuals
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("SAVING ENRICHED DATASETS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'clusters_train_with_residuals.csv', index=False)
    print(f"  ✓ Saved {len(train_df)} rows to clusters_train_with_residuals.csv")
    
    val_df.to_csv(output_dir / 'clusters_val_with_residuals.csv', index=False)
    print(f"  ✓ Saved {len(val_df)} rows to clusters_val_with_residuals.csv")
    
    test_df.to_csv(output_dir / 'clusters_test_with_residuals.csv', index=False)
    print(f"  ✓ Saved {len(test_df)} rows to clusters_test_with_residuals.csv")


def save_model(full_pipeline, preprocessor, model, categorical_features, 
               boolean_features, numeric_features, metrics, output_dir):
    """
    Save trained model pipeline and metadata.
    
    Parameters:
        full_pipeline: Complete sklearn pipeline
        preprocessor: Preprocessing pipeline
        model: Trained logistic regression model
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        metrics: Dictionary of validation metrics
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL AND METADATA")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full pipeline
    model_path = output_dir / 'baseline_logistic_regression.pkl'
    joblib.dump(full_pipeline, model_path)
    print(f"  ✓ Saved model pipeline to {model_path}")
    
    # Extract feature names after one-hot encoding
    all_categorical = categorical_features + boolean_features
    feature_names = []
    
    # Get categorical feature names from one-hot encoder
    cat_transformer = preprocessor.transformers_[0][1]
    onehot = cat_transformer.named_steps['onehot']
    
    # Categorical feature names
    for i, feat in enumerate(all_categorical):
        categories = onehot.categories_[i]
        # Drop first category (due to drop='first')
        for cat in categories[1:]:
            feature_names.append(f"{feat}={cat}")
    
    # Numeric feature names
    feature_names.extend(numeric_features)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Create metadata
    metadata = {
        'model_type': 'LogisticRegression',
        'features': {
            'categorical': categorical_features,
            'boolean': boolean_features,
            'numeric': numeric_features,
            'total_after_encoding': len(feature_names)
        },
        'hyperparameters': {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'penalty': 'l2',
            'C': 1.0
        },
        'test_metrics': metrics,
        'feature_names': feature_names,
        'coefficients': coefficients.tolist(),
        'intercept': float(model.intercept_[0])
    }
    
    # Save metadata
    metadata_path = output_dir / 'baseline_logistic_regression_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")


def compute_statistical_summary(model, X_train_processed, y_train, feature_names):
    """
    Compute statistical summary including standard errors, z-scores, and p-values.
    
    Note: This is an approximation using the Hessian matrix approach.
    For exact statistics, use statsmodels instead.
    
    Parameters:
        model: Trained logistic regression model
        X_train_processed: Preprocessed training features
        y_train: Training labels
        feature_names: List of feature names
        
    Returns:
        dict: Dictionary with statistical summaries
    """
    from scipy import stats
    from scipy.special import expit
    import numpy as np
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_train_processed)[:, 1]
    
    # Compute Hessian matrix (approximation for variance)
    # For logistic regression, the Hessian is: X^T * diag(p_i * (1-p_i)) * X
    W = y_pred_proba * (1 - y_pred_proba)  # weights
    X_weighted = X_train_processed * np.sqrt(W[:, np.newaxis])
    
    try:
        # Compute covariance matrix: inv(X^T * W * X)
        Hessian = X_weighted.T @ X_weighted
        cov_matrix = np.linalg.inv(Hessian)
        
        # Standard errors are sqrt of diagonal of covariance matrix
        se = np.sqrt(np.diag(cov_matrix))
        
        # Z-scores
        z_scores = model.coef_[0] / se
        
        # P-values (two-tailed test)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
    except np.linalg.LinAlgError:
        print("  WARNING: Could not compute statistical summary (singular matrix)")
        se = np.full(len(feature_names), np.nan)
        z_scores = np.full(len(feature_names), np.nan)
        p_values = np.full(len(feature_names), np.nan)
    
    # Create summary
    summary = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'std_error': se,
        'z_score': z_scores,
        'p_value': p_values,
        'significant': p_values < 0.05  # 5% significance level
    })
    
    return summary, cov_matrix if 'cov_matrix' in locals() else None


def generate_statistical_report(model, X_train_processed, y_train, 
                                categorical_features, boolean_features, 
                                numeric_features, preprocessor, output_dir):
    """
    Generate comprehensive statistical report with p-values and standard errors.
    
    Parameters:
        model: Trained logistic regression model
        X_train_processed: Preprocessed training features
        y_train: Training labels
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        preprocessor: Preprocessing pipeline
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("GENERATING STATISTICAL SUMMARY")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature names
    all_categorical = categorical_features + boolean_features
    feature_names = []
    
    cat_transformer = preprocessor.transformers_[0][1]
    onehot = cat_transformer.named_steps['onehot']
    
    for i, feat in enumerate(all_categorical):
        categories = onehot.categories_[i]
        for cat in categories[1:]:
            feature_names.append(f"{feat}={cat}")
    
    feature_names.extend(numeric_features)
    
    # Compute statistical summary
    summary_df, cov_matrix = compute_statistical_summary(
        model, X_train_processed, y_train, feature_names
    )
    
    # Sort by p-value
    summary_df = summary_df.sort_values('p_value')
    
    # Save statistical report
    report_path = output_dir / 'baseline_logistic_regression_statistics.csv'
    summary_df.to_csv(report_path, index=False)
    print(f"  ✓ Saved statistical report to {report_path}")
    
    # Print significant features
    significant = summary_df[summary_df['significant'] == True]
    print(f"\nSignificant features (p < 0.05): {len(significant)} out of {len(summary_df)}")
    
    if len(significant) > 0:
        print("\nTop 10 most significant features:")
        for idx, row in significant.head(10).iterrows():
            star = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
            print(f"  {row['feature']:40s} β={row['coefficient']:7.4f} "
                  f"SE={row['std_error']:7.4f} p={row['p_value']:.4f} {star}")
    
    # Save covariance matrix if available
    if cov_matrix is not None:
        cov_df = pd.DataFrame(cov_matrix, index=feature_names, columns=feature_names)
        cov_path = output_dir / 'baseline_logistic_regression_covariance_matrix.csv'
        cov_df.to_csv(cov_path)
        print(f"  ✓ Saved covariance matrix to {cov_path}")


def generate_coefficient_report(preprocessor, model, categorical_features, 
                                 boolean_features, numeric_features, output_dir):
    """
    Generate coefficient interpretation report.
    
    Parameters:
        preprocessor: Preprocessing pipeline
        model: Trained logistic regression model
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("GENERATING COEFFICIENT REPORT")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract feature names and coefficients
    all_categorical = categorical_features + boolean_features
    feature_data = []
    
    # Get categorical feature names from one-hot encoder
    cat_transformer = preprocessor.transformers_[0][1]
    onehot = cat_transformer.named_steps['onehot']
    
    coeff_idx = 0
    
    # Categorical features
    for i, feat in enumerate(all_categorical):
        categories = onehot.categories_[i]
        # Drop first category (due to drop='first')
        for cat in categories[1:]:
            feature_name = f"{feat}={cat}"
            coefficient = model.coef_[0][coeff_idx]
            feature_data.append({
                'feature': feature_name,
                'coefficient': coefficient,
                'abs_coefficient': abs(coefficient),
                'feature_type': 'categorical'
            })
            coeff_idx += 1
    
    # Numeric features
    for feat in numeric_features:
        coefficient = model.coef_[0][coeff_idx]
        feature_data.append({
            'feature': feat,
            'coefficient': coefficient,
            'abs_coefficient': abs(coefficient),
            'feature_type': 'numeric'
        })
        coeff_idx += 1
    
    # Create DataFrame and sort by absolute coefficient
    coeff_df = pd.DataFrame(feature_data)
    coeff_df = coeff_df.sort_values('abs_coefficient', ascending=False)
    
    # Save report
    report_path = output_dir / 'baseline_logistic_regression_coefficients.csv'
    coeff_df.to_csv(report_path, index=False)
    print(f"  ✓ Saved coefficient report to {report_path}")
    
    # Print top coefficients
    print("\nTop 10 positive coefficients (increase crash risk):")
    top_positive = coeff_df[coeff_df['coefficient'] > 0].head(10)
    for _, row in top_positive.iterrows():
        print(f"  {row['feature']:40s} {row['coefficient']:8.4f}")
    
    print("\nTop 10 negative coefficients (decrease crash risk):")
    top_negative = coeff_df[coeff_df['coefficient'] < 0].head(10)
    for _, row in top_negative.iterrows():
        print(f"  {row['feature']:40s} {row['coefficient']:8.4f}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("BASELINE LOGISTIC REGRESSION PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and split data
    train_df, val_df, test_df = load_and_split_data(CLUSTERS_DATA_PATH)
    
    # Step 2: Define features
    categorical_features, boolean_features, numeric_features, excluded_features = define_features()
    
    # Step 2.5: Remove constant features
    print("\n" + "=" * 80)
    print("REMOVING CONSTANT FEATURES")
    print("=" * 80)
    all_features = categorical_features + boolean_features + numeric_features
    feature_cols, constant_features = remove_constant_features(train_df, all_features)
    
    # Update feature lists
    categorical_features = [f for f in categorical_features if f in feature_cols]
    boolean_features = [f for f in boolean_features if f in feature_cols]
    numeric_features = [f for f in numeric_features if f in feature_cols]
    
    # Step 2.6: Detect complete separation issues
    print("\n" + "=" * 80)
    print("DETECTING COMPLETE SEPARATION ISSUES")
    print("=" * 80)
    separation_info = detect_separation(train_df, feature_cols, 'match_label')
    
    if len(separation_info) > 0:
        print(f"\n  Note: L2 regularization will be used to handle separation issues")
    
    # Step 3: Handle missing data
    train_df, val_df, test_df = handle_missing_data(
        train_df, val_df, test_df, 
        categorical_features, boolean_features, numeric_features
    )
    
    # Step 4: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        categorical_features, boolean_features, numeric_features
    )
    
    # Step 5: Train model
    model, preprocessor, X_train_processed, X_val_processed = train_model(
        train_df, val_df, categorical_features, boolean_features, 
        numeric_features, preprocessor
    )
    
    # Build full pipeline for later use
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Step 6: Evaluate on validation
    val_y = val_df['match_label'].astype(int)
    val_metrics = evaluate_model(model, X_val_processed, val_y)

    # Step 6.1: Threshold optimization on validation
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION (Validation-based)")
    print("=" * 80)
    val_proba = model.predict_proba(X_val_processed)[:, 1]
    thr_f1, score_f1 = find_optimal_threshold(val_y, val_proba, criterion='f1')
    thr_youden, score_youden = find_optimal_threshold(val_y, val_proba, criterion='youden')
    print(f"  Optimal threshold (F1):     {thr_f1:.2f} (F1={score_f1:.4f})")
    print(f"  Optimal threshold (Youden): {thr_youden:.2f} (J={score_youden:.4f})")
    
    # Step 7: Compute residuals
    train_df, val_df, test_df = compute_residuals(
        full_pipeline, train_df, val_df, test_df,
        categorical_features, boolean_features, numeric_features
    )
    
    # Step 8: Generate visualizations and evaluate all splits
    all_metrics = generate_visualizations(
        full_pipeline, train_df, val_df, test_df,
        categorical_features, boolean_features, numeric_features,
        VIZ_DIR
    )

    # Step 8.1: Evaluate test at optimized thresholds
    print("\n" + "=" * 80)
    print("EVALUATE TEST AT OPTIMIZED THRESHOLDS")
    print("=" * 80)
    # Build processed features for val/test via pipeline
    val_X = val_df[(categorical_features + boolean_features + numeric_features)]
    test_X = test_df[(categorical_features + boolean_features + numeric_features)]
    val_proc = full_pipeline.named_steps['preprocessor'].transform(val_X)
    test_proc = full_pipeline.named_steps['preprocessor'].transform(test_X)
    test_y = test_df['match_label'].astype(int)
    test_at_f1 = evaluate_with_threshold(model, test_proc, test_y, thr_f1, split_name='Test@F1')
    test_at_youden = evaluate_with_threshold(model, test_proc, test_y, thr_youden, split_name='Test@Youden')
    # Save CSV
    opt_df = pd.DataFrame([
        {'Split':'Val','Criterion':'F1','threshold':thr_f1,'F1_or_J':score_f1},
        {'Split':'Val','Criterion':'Youden','threshold':thr_youden,'F1_or_J':score_youden},
        {'Split':'Test','Criterion':'F1','threshold':test_at_f1['threshold'],'accuracy':test_at_f1['accuracy'],'precision':test_at_f1['precision'],'recall':test_at_f1['recall'],'f1_score':test_at_f1['f1_score'],'roc_auc':test_at_f1['roc_auc']},
        {'Split':'Test','Criterion':'Youden','threshold':test_at_youden['threshold'],'accuracy':test_at_youden['accuracy'],'precision':test_at_youden['precision'],'recall':test_at_youden['recall'],'f1_score':test_at_youden['f1_score'],'roc_auc':test_at_youden['roc_auc']}
    ])
    Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)
    (Path(VIZ_DIR) / 'baseline_threshold_optimization.csv').write_text(opt_df.to_csv(index=False))
    print(f"  ✓ Saved threshold optimization to {Path(VIZ_DIR) / 'baseline_threshold_optimization.csv'}")
    
    # Step 9: Save enriched datasets
    save_enriched_datasets(train_df, val_df, test_df, DATA_DIR)
    
    # Step 10: Save model and metadata (use test metrics as primary)
    test_metrics = all_metrics['test']
    save_model(
        full_pipeline, preprocessor, model, 
        categorical_features, boolean_features, numeric_features,
        test_metrics, MODELS_DIR
    )
    
    # Step 11: Generate coefficient report
    generate_coefficient_report(
        preprocessor, model, 
        categorical_features, boolean_features, numeric_features,
        REPORTS_DIR
    )
    
    # Step 12: Generate statistical summary (p-values, standard errors, etc.)
    generate_statistical_report(
        model, X_train_processed, train_df['match_label'].astype(int),
        categorical_features, boolean_features, numeric_features, 
        preprocessor, REPORTS_DIR
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Enriched datasets saved to data/processed/")
    print(f"  - Model pipeline saved to models/baseline_logistic_regression.pkl")
    print(f"  - Metadata saved to models/baseline_logistic_regression_metadata.json")
    print(f"  - Coefficient report saved to reports/baseline_logistic_regression_coefficients.csv")
    print(f"  - Statistical summary saved to reports/baseline_logistic_regression_statistics.csv")
    print(f"  - Covariance matrix saved to reports/baseline_logistic_regression_covariance_matrix.csv")
    print(f"  - Visualizations saved to reports/Regression_beforeCNN/")
    print(f"\nAll splits metrics (Test set highlighted as primary):")
    print(f"  {'Split':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print(f"  {'-'*70}")
    for split_name in ['train', 'val', 'test']:
        m = all_metrics[split_name]
        highlight = "***" if split_name == 'test' else ""
        print(f"  {split_name:<10} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1_score']:<10.4f} {m['roc_auc']:<10.4f} {highlight}")
    
    # Print primary test metrics summary
    test_m = all_metrics['test']
    print(f"\n*** PRIMARY TEST METRICS ***")
    print(f"  Test Accuracy:  {test_m['accuracy']:.4f}")
    print(f"  Test Precision: {test_m['precision']:.4f}")
    print(f"  Test Recall:    {test_m['recall']:.4f}")
    print(f"  Test F1-Score:  {test_m['f1_score']:.4f}")
    print(f"  Test ROC AUC:   {test_m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

