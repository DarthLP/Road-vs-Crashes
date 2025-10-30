#!/usr/bin/env python3
"""
Script: Enhanced Logistic Regression with YOLO Street Quality Features

Purpose:
This script trains an enhanced logistic regression model that includes YOLO-based
street quality features in addition to the baseline infrastructure attributes.
It reuses the same preprocessing and evaluation approach as the baseline model.

Functionality:
- Loads train/test splits with YOLO features from clusters_*_with_yolo_roi.csv
- Uses same feature set as baseline + street_quality_roi_winz
- Applies identical preprocessing (one-hot encoding, missing value handling)
- Trains logistic regression with same hyperparameters as baseline
- Evaluates on train and test splits (skips val per user request)
- Generates comprehensive reports and visualizations

How to run:
    python src/modeling/logistic_regression_with_yolo.py

Prerequisites:
    - Run extract_yolo_features.py first to generate *_with_yolo_roi.csv files
    - Ensure data/processed/clusters_{train,test}_with_yolo_roi.csv exist

Output:
    - models/logistic_regression_with_yolo.pkl
    - models/logistic_regression_with_yolo_metadata.json
    - reports/Regression_withYOLO/ (coefficients, statistics, visualizations)
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
TRAIN_DATA_PATH = DATA_DIR / 'clusters_train_with_yolo_roi.csv'
VAL_DATA_PATH = DATA_DIR / 'clusters_val_with_yolo_roi.csv'
TEST_DATA_PATH = DATA_DIR / 'clusters_test_with_yolo_roi.csv'

# Model paths
MODELS_DIR = project_root / 'models'
MODEL_PATH = MODELS_DIR / 'logistic_regression_with_yolo.pkl'
METADATA_PATH = MODELS_DIR / 'logistic_regression_with_yolo_metadata.json'

# Report paths
REPORTS_DIR = project_root / 'reports' / 'Regression_withYOLO'
COEFFICIENTS_PATH = REPORTS_DIR / 'logistic_regression_with_yolo_coefficients.csv'
STATISTICS_PATH = REPORTS_DIR / 'logistic_regression_with_yolo_statistics.csv'


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_split_data(train_path, val_path, test_path):
    """
    Load the cluster datasets with YOLO features.
    
    Parameters:
        train_path: Path to train CSV
        val_path: Path to val CSV
        test_path: Path to test CSV
        
    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames for each split
    """
    print("=" * 80)
    print("LOADING DATA WITH YOLO FEATURES")
    print("=" * 80)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded {len(train_df)} train rows")
    print(f"Loaded {len(val_df)} val rows")
    print(f"Loaded {len(test_df)} test rows")
    
    # Check target distribution
    print(f"\nTarget distribution (match_label):")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pos_count = split_df['match_label'].sum()
        pos_pct = (pos_count / len(split_df) * 100) if len(split_df) > 0 else 0
        print(f"  {split_name}: {pos_count} positive ({pos_pct:.1f}%), {len(split_df) - pos_count} negative")
    
    # Check YOLO feature availability
    print(f"\nYOLO feature availability:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        with_images = (split_df['num_imgs_used'] > 0).sum()
        print(f"  {split_name}: {with_images} clusters with images ({with_images/len(split_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def define_features():
    """
    Define which features to include in the model.
    Same as baseline + YOLO features.
    
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
    
    # Numeric features (float64 type) + YOLO features
    numeric_features = [
        'road_segment_length_m', 'maxspeed', 'lanes', 'width',
        'street_quality_roi_winz'  # Primary YOLO feature
    ]
    
    # Excluded features (to avoid leakage)
    excluded_features = [
        'lon', 'lat', 'amount_of_matches', 'crash_years', 'captured_at_years',
        'amount_of_images', 'list_of_thumb_1024_url', 'list_of_point_ids',
        'cluster_id', 'osm_way_id', 'split', 'match_label', 'intersection_degree',
        # Exclude other YOLO features (keep only the primary one)
        'yolo_detections_count_roi', 'yolo_damage_area_ratio_roi', 
        'street_quality_roi', 'yolo_damage_density', 'roi_area_px', 
        'num_imgs_used', 'street_quality_roi_iqr'
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


def handle_missing_data(train_df, val_df, test_df, categorical_features, 
                        boolean_features, numeric_features):
    """
    Handle missing data in categorical and numeric columns.
    Same approach as baseline.
    
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
    Same as baseline.
    
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


def train_model(train_df, categorical_features, boolean_features, numeric_features, preprocessor):
    """
    Train logistic regression model using preprocessing pipeline.
    Same hyperparameters as baseline.
    
    Parameters:
        train_df: Training DataFrame
        categorical_features: List of categorical feature names
        boolean_features: List of boolean feature names
        numeric_features: List of numeric feature names
        preprocessor: ColumnTransformer preprocessing pipeline
        
    Returns:
        tuple: (model, full_pipeline, X_train_processed)
    """
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION MODEL WITH YOLO FEATURES")
    print("=" * 80)
    
    # Prepare features and target
    all_categorical = categorical_features + boolean_features
    feature_cols = all_categorical + numeric_features
    
    X_train = train_df[feature_cols]
    y_train = train_df['match_label'].astype(int)
    
    print(f"\nTraining set: {len(X_train)} samples, {len(feature_cols)} features")
    
    # Fit preprocessing on train
    print("\nFitting preprocessing pipeline on train split...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print(f"  Train features after encoding: {X_train_processed.shape[1]} columns")
    
    # Train logistic regression with same hyperparameters as baseline
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
    
    # Build full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return model, full_pipeline, X_train_processed


def evaluate_model(model, X_processed, y, split_name="Test"):
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
    
    if split_name.lower() == "test":
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
    Evaluate model on train, val, and test splits.
    
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
        
        # Get preprocessed features
        X_processed = full_pipeline.named_steps['preprocessor'].transform(X)
        model = full_pipeline.named_steps['model']
        
        # Evaluate
        metrics = evaluate_model(model, X_processed, y, split_name=split_name)
        all_metrics[split_name.lower()] = metrics
    
    return all_metrics


def create_comparison_visualizations(all_metrics, output_dir):
    """
    Create comparison visualizations for train/test metrics.
    Similar to baseline visualizations.
    
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
    metrics_df.to_csv(output_dir / 'logistic_regression_with_yolo_metrics.csv', index=False)
    print(f"  ✓ Saved metrics to logistic_regression_with_yolo_metrics.csv")
    
    # 1. Bar plot comparing all metrics across splits
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(splits))
    width = 0.15
    
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']):
        values = metrics_df[metric].values
        ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced Logistic Regression (with YOLO): Metrics Comparison', 
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
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_bars.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_bars.png")
    
    # 2. Line plot showing each metric across splits
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']:
        ax.plot(splits, metrics_df[metric].values, marker='o', linewidth=2, 
               label=metric, markersize=8)
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Enhanced Logistic Regression (with YOLO): Metrics Trends', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_lines.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_lines.png")
    
    # 3. Heatmap of metrics
    fig, ax = plt.subplots(figsize=(8, 5))
    
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
    
    ax.set_title('Enhanced Logistic Regression (with YOLO): Metrics Heatmap', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_heatmap.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}")


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
    coeff_df.to_csv(COEFFICIENTS_PATH, index=False)
    print(f"  ✓ Saved coefficient report to {COEFFICIENTS_PATH}")
    
    # Print top coefficients
    print("\nTop 10 positive coefficients (increase crash risk):")
    top_positive = coeff_df[coeff_df['coefficient'] > 0].head(10)
    for _, row in top_positive.iterrows():
        print(f"  {row['feature']:40s} {row['coefficient']:8.4f}")
    
    print("\nTop 10 negative coefficients (decrease crash risk):")
    top_negative = coeff_df[coeff_df['coefficient'] < 0].head(10)
    for _, row in top_negative.iterrows():
        print(f"  {row['feature']:40s} {row['coefficient']:8.4f}")
    
    # Highlight YOLO feature
    yolo_coeff = coeff_df[coeff_df['feature'] == 'street_quality_roi_winz']
    if len(yolo_coeff) > 0:
        print(f"\nYOLO Feature Coefficient:")
        print(f"  street_quality_roi_winz: {yolo_coeff.iloc[0]['coefficient']:.4f}")


def compute_statistical_summary(model, X_train_processed, y_train, feature_names):
    """
    Compute statistical summary including standard errors, z-scores, and p-values.
    
    Parameters:
        model: Trained logistic regression model
        X_train_processed: Preprocessed training features
        y_train: Training labels
        feature_names: List of feature names
        
    Returns:
        dict: Dictionary with statistical summaries
    """
    from scipy import stats
    import numpy as np
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_train_processed)[:, 1]
    
    # Compute Hessian matrix (approximation for variance)
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
    summary_df.to_csv(STATISTICS_PATH, index=False)
    print(f"  ✓ Saved statistical report to {STATISTICS_PATH}")
    
    # Print significant features
    significant = summary_df[summary_df['significant'] == True]
    print(f"\nSignificant features (p < 0.05): {len(significant)} out of {len(summary_df)}")
    
    if len(significant) > 0:
        print("\nTop 10 most significant features:")
        for idx, row in significant.head(10).iterrows():
            star = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
            print(f"  {row['feature']:40s} β={row['coefficient']:7.4f} "
                  f"SE={row['std_error']:7.4f} p={row['p_value']:.4f} {star}")
    
    # Highlight YOLO feature significance
    yolo_stats = summary_df[summary_df['feature'] == 'street_quality_roi_winz']
    if len(yolo_stats) > 0:
        print(f"\nYOLO Feature Statistical Summary:")
        row = yolo_stats.iloc[0]
        star = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  street_quality_roi_winz: β={row['coefficient']:7.4f} "
              f"SE={row['std_error']:7.4f} p={row['p_value']:.4f} {star}")


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
        metrics: Dictionary of test metrics
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL AND METADATA")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full pipeline
    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"  ✓ Saved model pipeline to {MODEL_PATH}")
    
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
        'model_type': 'LogisticRegression_with_YOLO',
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
        'intercept': float(model.intercept_[0]),
        'yolo_feature': 'street_quality_roi_winz'
    }
    
    # Save metadata
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {METADATA_PATH}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ENHANCED LOGISTIC REGRESSION WITH YOLO FEATURES")
    print("=" * 80)
    
    # Step 1: Load data
    train_df, val_df, test_df = load_and_split_data(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
    
    # Step 2: Define features
    categorical_features, boolean_features, numeric_features, excluded_features = define_features()
    
    # Step 3: Remove constant features
    print("\n" + "=" * 80)
    print("REMOVING CONSTANT FEATURES")
    print("=" * 80)
    all_features = categorical_features + boolean_features + numeric_features
    feature_cols, constant_features = remove_constant_features(train_df, all_features)
    
    # Update feature lists
    categorical_features = [f for f in categorical_features if f in feature_cols]
    boolean_features = [f for f in boolean_features if f in feature_cols]
    numeric_features = [f for f in numeric_features if f in feature_cols]
    
    # Step 4: Handle missing data
    train_df, val_df, test_df = handle_missing_data(
        train_df, val_df, test_df, 
        categorical_features, boolean_features, numeric_features
    )
    
    # Step 5: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        categorical_features, boolean_features, numeric_features
    )
    
    # Step 6: Train model
    model, full_pipeline, X_train_processed = train_model(
        train_df, categorical_features, boolean_features, 
        numeric_features, preprocessor
    )
    
    # Step 7: Evaluate on all splits
    all_metrics = evaluate_all_splits(
        full_pipeline, train_df, val_df, test_df,
        categorical_features, boolean_features, numeric_features
    )

    # Step 7.1: Threshold optimization on validation and apply to test
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION (Validation-based)")
    print("=" * 80)
    # Build val/test processed via pipeline
    val_X = val_df[(categorical_features + boolean_features + numeric_features)]
    test_X = test_df[(categorical_features + boolean_features + numeric_features)]
    val_proc = full_pipeline.named_steps['preprocessor'].transform(val_X)
    test_proc = full_pipeline.named_steps['preprocessor'].transform(test_X)
    val_y = val_df['match_label'].astype(int)
    test_y = test_df['match_label'].astype(int)
    # Optimize
    val_proba = full_pipeline.named_steps['model'].predict_proba(val_proc)[:, 1]
    thr_f1, score_f1 = find_optimal_threshold(val_y, val_proba, criterion='f1')
    thr_youden, score_youden = find_optimal_threshold(val_y, val_proba, criterion='youden')
    print(f"  Optimal threshold (F1):     {thr_f1:.2f} (F1={score_f1:.4f})")
    print(f"  Optimal threshold (Youden): {thr_youden:.2f} (J={score_youden:.4f})")
    # Evaluate test at optimized thresholds
    test_at_f1 = evaluate_with_threshold(full_pipeline.named_steps['model'], test_proc, test_y, thr_f1, split_name='Test@F1')
    test_at_youden = evaluate_with_threshold(full_pipeline.named_steps['model'], test_proc, test_y, thr_youden, split_name='Test@Youden')
    # Save CSV
    import pandas as pd
    opt_df = pd.DataFrame([
        {'Split':'Val','Criterion':'F1','threshold':thr_f1,'F1_or_J':score_f1},
        {'Split':'Val','Criterion':'Youden','threshold':thr_youden,'F1_or_J':score_youden},
        {'Split':'Test','Criterion':'F1','threshold':test_at_f1['threshold'],'accuracy':test_at_f1['accuracy'],'precision':test_at_f1['precision'],'recall':test_at_f1['recall'],'f1_score':test_at_f1['f1_score'],'roc_auc':test_at_f1['roc_auc']},
        {'Split':'Test','Criterion':'Youden','threshold':test_at_youden['threshold'],'accuracy':test_at_youden['accuracy'],'precision':test_at_youden['precision'],'recall':test_at_youden['recall'],'f1_score':test_at_youden['f1_score'],'roc_auc':test_at_youden['roc_auc']}
    ])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / 'yolo_threshold_optimization.csv').write_text(opt_df.to_csv(index=False))
    print(f"  ✓ Saved threshold optimization to {REPORTS_DIR / 'yolo_threshold_optimization.csv'}")
    
    # Step 8: Create visualizations
    create_comparison_visualizations(all_metrics, REPORTS_DIR)
    
    # Step 9: Generate coefficient report
    generate_coefficient_report(
        preprocessor, model, 
        categorical_features, boolean_features, numeric_features,
        REPORTS_DIR
    )
    
    # Step 10: Generate statistical summary
    generate_statistical_report(
        model, X_train_processed, train_df['match_label'].astype(int),
        categorical_features, boolean_features, numeric_features, 
        preprocessor, REPORTS_DIR
    )
    
    # Step 11: Save model and metadata
    save_model(
        full_pipeline, preprocessor, model, 
        categorical_features, boolean_features, numeric_features,
        all_metrics['test'], MODELS_DIR
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Model saved to {MODEL_PATH}")
    print(f"  - Metadata saved to {METADATA_PATH}")
    print(f"  - Reports saved to {REPORTS_DIR}")
    print(f"\nTest metrics:")
    print(f"  {'Metric':<12} {'Value':<8}")
    print(f"  {'-'*20}")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        print(f"  {metric:<12} {all_metrics['test'][metric]:<8.4f}")


if __name__ == "__main__":
    main()
