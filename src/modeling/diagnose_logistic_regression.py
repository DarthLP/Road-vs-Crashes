#!/usr/bin/env python3
"""
Script: Logistic Regression Diagnostics - Collinearity and Assumptions Check

Purpose:
This script performs comprehensive diagnostics on the baseline logistic regression model
to check for collinearity, separation issues, and other potential problems that could
affect model reliability and coefficient interpretation.

Functionality:
- Loads the trained logistic regression model and preprocessed data
- Computes variance inflation factors (VIF) for all features
- Checks for complete/quasi-complete separation
- Examines feature correlations
- Identifies high-leverage points and influential observations
- Generates diagnostic report with recommendations

How to run:
    python src/modeling/diagnose_logistic_regression.py

Prerequisites:
    - Run baseline_logistic_regression.py first to generate the trained model
    - Ensure models/baseline_logistic_regression.pkl exists
    - Ensure data/processed/clusters_with_crashes.csv exists

Output:
    - reports/Regression_beforeCNN/logistic_regression_diagnostics.txt
    - reports/Regression_beforeCNN/correlation_matrix.csv
    - reports/Regression_beforeCNN/vif_scores.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import VIF calculation
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not available. Install with: conda install statsmodels")

# Path configuration
DATA_DIR = project_root / 'data' / 'processed'
CLUSTERS_DATA_PATH = DATA_DIR / 'clusters_with_crashes.csv'
MODELS_DIR = project_root / 'models'
MODEL_PATH = MODELS_DIR / 'baseline_logistic_regression.pkl'
METADATA_PATH = MODELS_DIR / 'baseline_logistic_regression_metadata.json'
REPORTS_DIR = project_root / 'reports' / 'Regression_beforeCNN'
DIAGNOSTICS_PATH = REPORTS_DIR / 'logistic_regression_diagnostics.txt'


def load_model_and_data():
    """
    Load the trained model pipeline and cluster data.
    
    Returns:
        tuple: (model, preprocessor, metadata, train_df, val_df, test_df, feature_cols)
    """
    print("Loading model and data...")
    
    # Load model pipeline
    full_pipeline = joblib.load(MODEL_PATH)
    preprocessor = full_pipeline.named_steps['preprocessor']
    model = full_pipeline.named_steps['model']
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Load data
    df = pd.read_csv(CLUSTERS_DATA_PATH)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Define features
    categorical_features = metadata['features']['categorical']
    boolean_features = metadata['features']['boolean']
    numeric_features = metadata['features']['numeric']
    feature_cols = categorical_features + boolean_features + numeric_features
    
    return model, preprocessor, metadata, train_df, val_df, test_df, feature_cols


def compute_vif(X_processed, feature_names):
    """
    Compute variance inflation factors for all features.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names after encoding
        
    Returns:
        pd.DataFrame: DataFrame with VIF scores
    """
    print("\nComputing Variance Inflation Factors (VIF)...")
    
    if not HAS_STATSMODELS:
        print("  ERROR: statsmodels not available. Skipping VIF computation.")
        return pd.DataFrame({'feature': feature_names, 'vif': [np.nan] * len(feature_names)})
    
    # VIF computation requires at least as many samples as features
    if X_processed.shape[0] < X_processed.shape[1]:
        print(f"  WARNING: More features ({X_processed.shape[1]}) than samples ({X_processed.shape[0]}).")
        print(f"           VIF cannot be computed reliably.")
        return pd.DataFrame({'feature': feature_names, 'vif': [np.nan] * len(feature_names)})
    
    vif_data = []
    for i in range(X_processed.shape[1]):
        try:
            vif = variance_inflation_factor(X_processed, i)
            vif_data.append({
                'feature': feature_names[i],
                'vif': vif,
                'severity': 'extreme' if vif >= 10 else ('moderate' if vif >= 5 else 'low')
            })
        except Exception as e:
            print(f"  WARNING: Could not compute VIF for feature {i}: {e}")
            vif_data.append({
                'feature': feature_names[i],
                'vif': np.nan,
                'severity': 'error'
            })
    
    vif_df = pd.DataFrame(vif_data)
    
    # Count by severity
    severe_count = (vif_df['vif'] >= 10).sum()
    moderate_count = ((vif_df['vif'] >= 5) & (vif_df['vif'] < 10)).sum()
    low_count = (vif_df['vif'] < 5).sum()
    
    print(f"  VIF Score Distribution:")
    print(f"    Extreme (>= 10): {severe_count} features")
    print(f"    Moderate (5-10): {moderate_count} features")
    print(f"    Low (< 5):       {low_count} features")
    
    if severe_count > 0:
        print(f"\n  WARNING: {severe_count} features with extreme collinearity (VIF >= 10)")
        top_vif = vif_df.nlargest(10, 'vif')
        print("  Top 10 highest VIF scores:")
        for _, row in top_vif.iterrows():
            print(f"    {row['feature']:40s} VIF: {row['vif']:.2f}")
    
    return vif_df


def compute_correlation_matrix(X_processed, feature_names):
    """
    Compute correlation matrix for preprocessed features.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names after encoding
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    print("\nComputing correlation matrix...")
    
    # Compute pairwise correlations
    corr_matrix = np.corrcoef(X_processed.T)
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = abs(corr_matrix[i, j])
            if corr_val > 0.7:
                high_corr_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })
    
    print(f"  Found {len(high_corr_pairs)} feature pairs with |correlation| > 0.7")
    
    if len(high_corr_pairs) > 0:
        print("  Highly correlated pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
            print(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    
    return corr_df, high_corr_pairs


def check_separation(X_processed, y):
    """
    Check for complete or quasi-complete separation in logistic regression.
    
    Complete separation occurs when a feature perfectly predicts the outcome.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        y: Target labels
        
    Returns:
        dict: Separation diagnostic information
    """
    print("\nChecking for complete/quasi-complete separation...")
    
    separation_issues = []
    
    # Check each feature for perfect prediction
    for i in range(X_processed.shape[1]):
        feature_values = X_processed[:, i]
        
        # Check if feature is constant
        if np.std(feature_values) < 1e-10:
            continue
        
        # Check if feature perfectly predicts y
        unique_values = np.unique(feature_values)
        
        for val in unique_values:
            mask = (feature_values == val)
            y_masked = y[mask].values  # Convert to numpy array
            
            if len(y_masked) > 0:
                if np.all(y_masked == 1) or np.all(y_masked == 0):
                    separation_issues.append({
                        'feature_idx': i,
                        'feature_value': val,
                        'perfect_class': int(y_masked[0]),
                        'count': len(y_masked)
                    })
    
    if len(separation_issues) > 0:
        print(f"  WARNING: Found {len(separation_issues)} potential separation issues")
        print("  Complete separation can cause convergence problems and infinite coefficients")
    else:
        print("  No complete separation detected")
    
    return {'separation_issues': separation_issues, 'has_separation': len(separation_issues) > 0}


def check_feature_ranges(X_processed, feature_names):
    """
    Check feature value ranges and detect potential scaling issues.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        feature_names: List of feature names after encoding
        
    Returns:
        pd.DataFrame: Feature statistics
    """
    print("\nChecking feature value ranges...")
    
    stats_data = []
    for i, name in enumerate(feature_names):
        feature = X_processed[:, i]
        stats_data.append({
            'feature': name,
            'min': np.min(feature),
            'max': np.max(feature),
            'mean': np.mean(feature),
            'std': np.std(feature),
            'range': np.max(feature) - np.min(feature),
            'is_constant': np.std(feature) < 1e-10
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Check for extreme ranges (potential scaling issues)
    extreme_ranges = stats_df[stats_df['range'] > 100]
    if len(extreme_ranges) > 0:
        print(f"  WARNING: {len(extreme_ranges)} features with large ranges (> 100):")
        for _, row in extreme_ranges.head(10).iterrows():
            print(f"    {row['feature']:40s} range: {row['range']:.2f}")
    
    # Check for constant features
    constant_features = stats_df[stats_df['is_constant']]
    if len(constant_features) > 0:
        print(f"  WARNING: {len(constant_features)} constant features detected")
        for _, row in constant_features.iterrows():
            print(f"    {row['feature']}")
    
    return stats_df


def compute_leverage_and_influence(X_processed, y, y_pred_proba, feature_names):
    """
    Compute leverage and influence metrics for outlier detection.
    
    Parameters:
        X_processed: Preprocessed feature matrix
        y: Target labels
        y_pred_proba: Predicted probabilities
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: Leverage and influence statistics
    """
    print("\nComputing leverage and influence statistics...")
    
    # Compute hat matrix (leverage)
    X_with_intercept = np.column_stack([np.ones(X_processed.shape[0]), X_processed])
    
    try:
        # Hat matrix: H = X(X'X)^(-1)X'
        XTX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        hat_matrix = X_with_intercept @ XTX_inv @ X_with_intercept.T
        leverage = np.diag(hat_matrix)
        
        # Compute residuals
        residuals = y - y_pred_proba
        
        # Compute Cook's distance
        mse = np.mean(residuals ** 2)
        cook_distance = (residuals ** 2 / (X_processed.shape[1] * mse)) * (leverage / ((1 - leverage) ** 2))
        
        influence_df = pd.DataFrame({
            'observation': range(len(y)),
            'leverage': leverage,
            'cook_distance': cook_distance,
            'residual': residuals
        })
        
        # Flag high leverage points
        high_leverage_threshold = 2 * (X_processed.shape[1] + 1) / X_processed.shape[0]
        high_leverage = influence_df['leverage'] > high_leverage_threshold
        print(f"  High leverage threshold: {high_leverage_threshold:.4f}")
        print(f"  High leverage points: {high_leverage.sum()} ({high_leverage.sum()/len(y)*100:.1f}%)")
        
        # Flag influential points (Cook's distance > 4/n)
        cook_threshold = 4 / X_processed.shape[0]
        influential = influence_df['cook_distance'] > cook_threshold
        print(f"  Influential points threshold: {cook_threshold:.4f}")
        print(f"  Influential points: {influential.sum()} ({influential.sum()/len(y)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ERROR: Could not compute influence statistics: {e}")
        influence_df = pd.DataFrame({
            'observation': range(len(y)),
            'leverage': [np.nan] * len(y),
            'cook_distance': [np.nan] * len(y),
            'residual': residuals
        })
    
    return influence_df


def generate_diagnostic_report(model, preprocessor, metadata, train_df, feature_cols):
    """
    Generate comprehensive diagnostic report.
    
    Parameters:
        model: Trained logistic regression model
        preprocessor: Preprocessing pipeline
        metadata: Model metadata
        train_df: Training DataFrame
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['match_label'].astype(int)
    X_train_processed = preprocessor.transform(X_train)
    
    # Get feature names
    feature_names = metadata['feature_names']
    
    # Run diagnostics
    vif_df = compute_vif(X_train_processed, feature_names)
    corr_df, high_corr_pairs = compute_correlation_matrix(X_train_processed, feature_names)
    separation_info = check_separation(X_train_processed, y_train)
    feature_stats = check_feature_ranges(X_train_processed, feature_names)
    
    # Get predictions for influence analysis
    y_pred_proba = model.predict_proba(X_train_processed)[:, 1]
    influence_df = compute_leverage_and_influence(X_train_processed, y_train, y_pred_proba, feature_names)
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LOGISTIC REGRESSION DIAGNOSTICS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Date: {pd.Timestamp.now()}")
    report_lines.append(f"Dataset: {len(train_df)} training samples")
    report_lines.append(f"Features: {len(feature_names)} features after encoding")
    report_lines.append(f"Positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    report_lines.append("")
    
    # VIF Section
    report_lines.append("-" * 80)
    report_lines.append("1. VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("VIF measures how much the variance of a coefficient increases")
    report_lines.append("due to collinearity with other features.")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("  - VIF < 5:    Low collinearity (acceptable)")
    report_lines.append("  - VIF 5-10:   Moderate collinearity (caution)")
    report_lines.append("  - VIF >= 10:  High collinearity (problematic)")
    report_lines.append("")
    
    severe_vif = vif_df[vif_df['vif'] >= 10]
    moderate_vif = vif_df[(vif_df['vif'] >= 5) & (vif_df['vif'] < 10)]
    
    if len(severe_vif) > 0:
        report_lines.append(f"⚠️  PROBLEM: {len(severe_vif)} features with VIF >= 10")
        report_lines.append("")
        report_lines.append("Top features with extreme collinearity:")
        for _, row in severe_vif.nlargest(10, 'vif').iterrows():
            report_lines.append(f"  {row['feature']:40s} VIF: {row['vif']:.2f}")
        report_lines.append("")
        report_lines.append("Recommendation: Remove or combine highly collinear features")
        report_lines.append("")
    
    if len(moderate_vif) > 0:
        report_lines.append(f"⚠️  CAUTION: {len(moderate_vif)} features with VIF 5-10")
        report_lines.append("")
    
    if len(severe_vif) == 0 and len(moderate_vif) == 0:
        report_lines.append("✓ No collinearity issues detected")
        report_lines.append("")
    
    # Correlation Section
    report_lines.append("-" * 80)
    report_lines.append("2. FEATURE CORRELATION ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if len(high_corr_pairs) > 0:
        report_lines.append(f"⚠️  PROBLEM: {len(high_corr_pairs)} feature pairs with |correlation| > 0.7")
        report_lines.append("")
        report_lines.append("Most highly correlated pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
            report_lines.append(f"  {pair['feature1']}")
            report_lines.append(f"    <-> {pair['feature2']}")
            report_lines.append(f"    correlation: {pair['correlation']:.3f}")
            report_lines.append("")
    else:
        report_lines.append("✓ No highly correlated feature pairs detected")
        report_lines.append("")
    
    # Separation Section
    report_lines.append("-" * 80)
    report_lines.append("3. COMPLETE SEPARATION CHECK")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if separation_info['has_separation']:
        report_lines.append(f"⚠️  PROBLEM: {len(separation_info['separation_issues'])} instances of complete separation")
        report_lines.append("")
        report_lines.append("Complete separation can cause:")
        report_lines.append("  - Convergence problems")
        report_lines.append("  - Infinite or very large coefficients")
        report_lines.append("  - Unreliable model predictions")
        report_lines.append("")
        report_lines.append("Recommendation: Use regularization (L1/L2 penalty) or remove problematic features")
        report_lines.append("")
    else:
        report_lines.append("✓ No complete separation detected")
        report_lines.append("")
    
    # Feature Range Section
    report_lines.append("-" * 80)
    report_lines.append("4. FEATURE SCALING CHECK")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    extreme_ranges = feature_stats[feature_stats['range'] > 100]
    constant_features = feature_stats[feature_stats['is_constant']]
    
    if len(extreme_ranges) > 0:
        report_lines.append(f"⚠️  CAUTION: {len(extreme_ranges)} features with large ranges (> 100)")
        report_lines.append("  These may need scaling for better model performance")
        report_lines.append("")
    
    if len(constant_features) > 0:
        report_lines.append(f"⚠️  PROBLEM: {len(constant_features)} constant features detected")
        report_lines.append("  Constant features provide no information and should be removed")
        report_lines.append("")
    
    if len(extreme_ranges) == 0 and len(constant_features) == 0:
        report_lines.append("✓ No feature scaling issues detected")
        report_lines.append("")
    
    # Influence Section
    report_lines.append("-" * 80)
    report_lines.append("5. OUTLIER AND INFLUENCE ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    high_leverage_threshold = 2 * (X_train_processed.shape[1] + 1) / X_train_processed.shape[0]
    high_leverage = (influence_df['leverage'] > high_leverage_threshold).sum()
    
    cook_threshold = 4 / X_train_processed.shape[0]
    influential = (influence_df['cook_distance'] > cook_threshold).sum()
    
    report_lines.append(f"High leverage points: {high_leverage} ({high_leverage/len(train_df)*100:.1f}%)")
    report_lines.append(f"Influential points: {influential} ({influential/len(train_df)*100:.1f}%)")
    report_lines.append("")
    
    if high_leverage > 0 or influential > 0:
        report_lines.append("Recommendation: Review influential observations and consider removing outliers")
        report_lines.append("")
    
    # Summary
    report_lines.append("-" * 80)
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    issues = []
    if len(severe_vif) > 0:
        issues.append("Collinearity (VIF >= 10)")
    if len(high_corr_pairs) > 0:
        issues.append("High feature correlations")
    if separation_info['has_separation']:
        issues.append("Complete separation")
    if len(constant_features) > 0:
        issues.append("Constant features")
    
    if issues:
        report_lines.append("⚠️  Issues Found:")
        for issue in issues:
            report_lines.append(f"  - {issue}")
        report_lines.append("")
        report_lines.append("Recommendations:")
        report_lines.append("  1. Remove or combine collinear features")
        report_lines.append("  2. Add regularization (L1/L2 penalty) to the model")
        report_lines.append("  3. Consider feature selection or dimensionality reduction")
        report_lines.append("  4. Remove constant features")
    else:
        report_lines.append("✓ No major issues detected")
        report_lines.append("  Model assumptions appear to be satisfied")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIAGNOSTICS_PATH, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved diagnostic report to {DIAGNOSTICS_PATH}")
    
    # Print summary to console
    print("\n" + report_text)
    
    # Save VIF scores
    vif_df.to_csv(REPORTS_DIR / 'vif_scores.csv', index=False)
    print(f"  ✓ Saved VIF scores to vif_scores.csv")
    
    # Save correlation matrix
    corr_df.to_csv(REPORTS_DIR / 'correlation_matrix.csv')
    print(f"  ✓ Saved correlation matrix to correlation_matrix.csv")
    
    # Save top correlations
    if len(high_corr_pairs) > 0:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
        high_corr_df.to_csv(REPORTS_DIR / 'high_correlations.csv', index=False)
        print(f"  ✓ Saved high correlations to high_correlations.csv")
    
    return report_text, vif_df, corr_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("LOGISTIC REGRESSION DIAGNOSTICS")
    print("=" * 80)
    
    # Load model and data
    model, preprocessor, metadata, train_df, val_df, test_df, feature_cols = load_model_and_data()
    
    # Generate diagnostic report
    report_text, vif_df, corr_df = generate_diagnostic_report(
        model, preprocessor, metadata, train_df, feature_cols
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - {DIAGNOSTICS_PATH}")
    print(f"  - {REPORTS_DIR / 'vif_scores.csv'}")
    print(f"  - {REPORTS_DIR / 'correlation_matrix.csv'}")


if __name__ == "__main__":
    main()

