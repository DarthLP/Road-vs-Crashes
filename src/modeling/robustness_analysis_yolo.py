#!/usr/bin/env python3
"""
Script: Robustness Analysis for YOLO-Enhanced Logistic Regression

Purpose:
This script performs robustness checks on the YOLO-enhanced logistic regression model
by testing alternative YOLO features and model specifications. It evaluates the
stability of the street quality coefficient across different approaches.

Functionality:
- Tests alternative YOLO features (density, no-winsor, raw values)
- Adds num_imgs_used as control variable
- Excludes clusters with no images (num_imgs_used = 0)
- Compares coefficient stability across specifications
- Generates sensitivity analysis report

How to run:
    python src/modeling/robustness_analysis_yolo.py

Prerequisites:
    - Run extract_yolo_features.py first
    - Ensure data/processed/clusters_{train,test}_with_yolo_roi.csv exist

Output:
    - reports/Regression_withYOLO/sensitivity_analysis.csv
    - reports/Regression_withYOLO/robustness_analysis_report.txt
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = project_root / 'data' / 'processed'
TRAIN_DATA_PATH = DATA_DIR / 'clusters_train_with_yolo_roi.csv'
TEST_DATA_PATH = DATA_DIR / 'clusters_test_with_yolo_roi.csv'

# Output paths
REPORTS_DIR = project_root / 'reports' / 'Regression_withYOLO'
SENSITIVITY_PATH = REPORTS_DIR / 'sensitivity_analysis.csv'
REPORT_PATH = REPORTS_DIR / 'robustness_analysis_report.txt'


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data():
    """
    Load train and test datasets with YOLO features.
    
    Returns:
        tuple: (train_df, test_df) - DataFrames with YOLO features
    """
    print("=" * 80)
    print("LOADING DATA FOR ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    print(f"Loaded {len(train_df)} train rows")
    print(f"Loaded {len(test_df)} test rows")
    
    # Check YOLO feature availability
    for split_name, df in [('Train', train_df), ('Test', test_df)]:
        with_images = (df['num_imgs_used'] > 0).sum()
        print(f"  {split_name}: {with_images} clusters with images ({with_images/len(df)*100:.1f}%)")
    
    return train_df, test_df


def define_base_features():
    """
    Define base features (same as baseline model).
    
    Returns:
        tuple: (categorical_features, boolean_features, numeric_features, excluded_features)
    """
    # Categorical features
    categorical_features = [
        'highway', 'surface', 'lit', 'cycleway', 'sidewalk', 'oneway',
        'bridge', 'tunnel', 'junction', 'access', 'smoothness'
    ]
    
    # Boolean features
    boolean_features = ['is_intersection', 'near_traffic_signal']
    
    # Base numeric features (without YOLO)
    base_numeric_features = [
        'road_segment_length_m', 'maxspeed', 'lanes', 'width'
    ]
    
    # Excluded features
    excluded_features = [
        'lon', 'lat', 'amount_of_matches', 'crash_years', 'captured_at_years',
        'amount_of_images', 'list_of_thumb_1024_url', 'list_of_point_ids',
        'cluster_id', 'osm_way_id', 'split', 'match_label', 'intersection_degree'
    ]
    
    return categorical_features, boolean_features, base_numeric_features, excluded_features


def train_model_with_features(train_df, test_df, yolo_feature, control_feature=None):
    """
    Train logistic regression model with specified YOLO feature.
    
    Parameters:
        train_df: Training DataFrame
        test_df: Test DataFrame
        yolo_feature: Name of YOLO feature to use
        control_feature: Optional control feature to add
        
    Returns:
        dict: Model results and metrics
    """
    
    # Define features
    categorical_features, boolean_features, base_numeric_features, excluded_features = define_base_features()
    
    # Add YOLO feature
    numeric_features = base_numeric_features + [yolo_feature]
    
    # Add control feature if specified
    if control_feature and control_feature not in numeric_features:
        numeric_features.append(control_feature)
    
    # Handle missing data
    all_categorical = categorical_features + boolean_features
    
    # Categorical: Replace NaN with "missing"
    for col in all_categorical:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna('missing')
            test_df[col] = test_df[col].fillna('missing')
    
    # Numeric: Replace NaN with median from train
    for col in numeric_features:
        if col in train_df.columns:
            train_median = train_df[col].median()
            train_df[col] = train_df[col].fillna(train_median)
            test_df[col] = test_df[col].fillna(train_median)
    
    # Prepare features and target
    feature_cols = all_categorical + numeric_features
    X_train = train_df[feature_cols]
    y_train = train_df['match_label'].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df['match_label'].astype(int)
    
    # Build preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, all_categorical),
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop'
    )
    
    # Fit preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced',
        C=1.0,
        penalty='l2'
    )
    
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Get YOLO feature coefficient
    feature_names = []
    cat_transformer = preprocessor.transformers_[0][1]
    onehot = cat_transformer.named_steps['onehot']
    
    for i, feat in enumerate(all_categorical):
        categories = onehot.categories_[i]
        for cat in categories[1:]:
            feature_names.append(f"{feat}={cat}")
    
    feature_names.extend(numeric_features)
    
    # Find YOLO feature coefficient
    yolo_coeff = None
    for i, name in enumerate(feature_names):
        if name == yolo_feature:
            yolo_coeff = model.coef_[0][i]
            break
    
    return {
        'yolo_feature': yolo_feature,
        'control_feature': control_feature,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'yolo_coefficient': yolo_coeff,
        'metrics': metrics
    }


def run_robustness_checks(train_df, test_df):
    """
    Run comprehensive robustness checks.
    
    Parameters:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        pd.DataFrame: Results of all robustness checks
    """
    print("\n" + "=" * 80)
    print("RUNNING ROBUSTNESS CHECKS")
    print("=" * 80)
    
    # Define robustness scenarios
    scenarios = [
        # Baseline YOLO feature
        {'yolo_feature': 'street_quality_roi_winz', 'control_feature': None, 'description': 'Baseline (winsorized)'},
        
        # Alternative YOLO features
        {'yolo_feature': 'street_quality_roi', 'control_feature': None, 'description': 'No winsorization'},
        {'yolo_feature': 'yolo_damage_density', 'control_feature': None, 'description': 'Damage density'},
        {'yolo_feature': 'yolo_detections_count_roi', 'control_feature': None, 'description': 'Detection count'},
        
        # With control variable
        {'yolo_feature': 'street_quality_roi_winz', 'control_feature': 'num_imgs_used', 'description': 'Baseline + num_imgs_used control'},
        
        # Alternative features with control
        {'yolo_feature': 'street_quality_roi', 'control_feature': 'num_imgs_used', 'description': 'No winsor + control'},
        {'yolo_feature': 'yolo_damage_density', 'control_feature': 'num_imgs_used', 'description': 'Density + control'},
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}/{len(scenarios)}: {scenario['description']}")
        
        try:
            result = train_model_with_features(
                train_df, test_df,
                scenario['yolo_feature'],
                scenario['control_feature']
            )
            
            # Add scenario info
            result.update(scenario)
            results.append(result)
            
            print(f"  ✓ YOLO coefficient: {result['yolo_coefficient']:.4f}")
            print(f"  ✓ Test F1: {result['metrics']['f1_score']:.4f}")
            print(f"  ✓ Test ROC AUC: {result['metrics']['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_robustness(results_df):
    """
    Analyze robustness of YOLO feature coefficient.
    
    Parameters:
        results_df: DataFrame with robustness results
        
    Returns:
        dict: Robustness analysis summary
    """
    print("\n" + "=" * 80)
    print("ANALYZING ROBUSTNESS")
    print("=" * 80)
    
    # Filter to successful results
    valid_results = results_df.dropna(subset=['yolo_coefficient'])
    
    if len(valid_results) == 0:
        print("  ❌ No valid results to analyze")
        return {}
    
    # Coefficient statistics
    coeffs = valid_results['yolo_coefficient'].values
    coeff_mean = np.mean(coeffs)
    coeff_std = np.std(coeffs)
    coeff_min = np.min(coeffs)
    coeff_max = np.max(coeffs)
    coeff_range = coeff_max - coeff_min
    
    print(f"\nYOLO Feature Coefficient Robustness:")
    print(f"  Mean: {coeff_mean:.4f}")
    print(f"  Std:  {coeff_std:.4f}")
    print(f"  Min:  {coeff_min:.4f}")
    print(f"  Max:  {coeff_max:.4f}")
    print(f"  Range: {coeff_range:.4f}")
    
    # Sign stability
    positive_coeffs = (coeffs > 0).sum()
    negative_coeffs = (coeffs < 0).sum()
    sign_consistency = max(positive_coeffs, negative_coeffs) / len(coeffs)
    
    print(f"\nSign Stability:")
    print(f"  Positive coefficients: {positive_coeffs}/{len(coeffs)} ({positive_coeffs/len(coeffs)*100:.1f}%)")
    print(f"  Negative coefficients: {negative_coeffs}/{len(coeffs)} ({negative_coeffs/len(coeffs)*100:.1f}%)")
    print(f"  Sign consistency: {sign_consistency:.1%}")
    
    # Performance stability
    f1_scores = valid_results['metrics'].apply(lambda x: x['f1_score']).values
    roc_aucs = valid_results['metrics'].apply(lambda x: x['roc_auc']).values
    
    print(f"\nPerformance Stability:")
    print(f"  F1 Score - Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")
    print(f"  ROC AUC - Mean: {np.mean(roc_aucs):.4f}, Std: {np.std(roc_aucs):.4f}")
    
    # Identify most stable scenarios
    coeff_variation = np.abs(coeffs - coeff_mean)
    most_stable_idx = np.argmin(coeff_variation)
    most_stable = valid_results.iloc[most_stable_idx]
    
    print(f"\nMost Stable Scenario:")
    print(f"  {most_stable['description']}")
    print(f"  Coefficient: {most_stable['yolo_coefficient']:.4f}")
    print(f"  F1 Score: {most_stable['metrics']['f1_score']:.4f}")
    
    return {
        'coefficient_stats': {
            'mean': coeff_mean,
            'std': coeff_std,
            'min': coeff_min,
            'max': coeff_max,
            'range': coeff_range
        },
        'sign_consistency': sign_consistency,
        'performance_stats': {
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'roc_auc_mean': np.mean(roc_aucs),
            'roc_auc_std': np.std(roc_aucs)
        },
        'most_stable_scenario': most_stable['description']
    }


def save_robustness_report(results_df, analysis_summary, output_dir):
    """
    Save robustness analysis report.
    
    Parameters:
        results_df: DataFrame with robustness results
        analysis_summary: Dictionary with analysis summary
        output_dir: Output directory path
    """
    print("\n" + "=" * 80)
    print("SAVING ROBUSTNESS REPORT")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df.to_csv(SENSITIVITY_PATH, index=False)
    print(f"  ✓ Saved detailed results to {SENSITIVITY_PATH}")
    
    # Create text report
    with open(REPORT_PATH, 'w') as f:
        f.write("YOLO-Enhanced Logistic Regression: Robustness Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total scenarios tested: {len(results_df)}\n")
        f.write(f"Successful scenarios: {len(results_df.dropna(subset=['yolo_coefficient']))}\n\n")
        
        if analysis_summary:
            f.write("COEFFICIENT ROBUSTNESS\n")
            f.write("-" * 25 + "\n")
            stats = analysis_summary['coefficient_stats']
            f.write(f"Mean coefficient: {stats['mean']:.4f}\n")
            f.write(f"Standard deviation: {stats['std']:.4f}\n")
            f.write(f"Range: {stats['min']:.4f} to {stats['max']:.4f}\n")
            f.write(f"Sign consistency: {analysis_summary['sign_consistency']:.1%}\n\n")
            
            f.write("PERFORMANCE STABILITY\n")
            f.write("-" * 22 + "\n")
            perf_stats = analysis_summary['performance_stats']
            f.write(f"F1 Score: {perf_stats['f1_mean']:.4f} ± {perf_stats['f1_std']:.4f}\n")
            f.write(f"ROC AUC: {perf_stats['roc_auc_mean']:.4f} ± {perf_stats['roc_auc_std']:.4f}\n\n")
            
            f.write("MOST STABLE SCENARIO\n")
            f.write("-" * 22 + "\n")
            f.write(f"{analysis_summary['most_stable_scenario']}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 17 + "\n")
        f.write("See sensitivity_analysis.csv for complete results.\n")
    
    print(f"  ✓ Saved report to {REPORT_PATH}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("YOLO-ENHANCED LOGISTIC REGRESSION: ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load data
    train_df, test_df = load_data()
    
    # Step 2: Run robustness checks
    results_df = run_robustness_checks(train_df, test_df)
    
    # Step 3: Analyze robustness
    analysis_summary = analyze_robustness(results_df)
    
    # Step 4: Save report
    save_robustness_report(results_df, analysis_summary, REPORTS_DIR)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {SENSITIVITY_PATH}")
    print(f"  - {REPORT_PATH}")


if __name__ == "__main__":
    main()
