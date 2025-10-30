#!/usr/bin/env python3
"""
Script: Compare Baseline vs Enhanced Logistic Regression with YOLO Features

Purpose:
This script compares the performance of the baseline logistic regression model
(without YOLO features) against the enhanced model (with YOLO features) on
train and test splits. It generates comprehensive comparison visualizations
and calculates performance deltas.

Functionality:
- Loads baseline metrics from existing reports
- Loads enhanced model metrics from YOLO regression output
- Compares train vs train and test vs test performance
- Calculates deltas (ΔAccuracy, ΔPrecision, ΔRecall, ΔF1, ΔROC_AUC)
- Generates comparison visualizations (bars, lines, heatmaps)
- Focuses on test-to-test comparison as primary result

How to run:
    python src/modeling/compare_baseline_vs_yolo.py

Prerequisites:
    - Run baseline_logistic_regression.py first
    - Run logistic_regression_with_yolo.py first
    - Ensure reports/Regression_beforeCNN/baseline_metrics_comparison.csv exists
    - Ensure reports/Regression_withYOLO/logistic_regression_with_yolo_metrics.csv exists

Output:
    - reports/Comparison_Baseline_vs_YOLO/ (comparison visualizations and metrics)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import joblib
from scipy.stats import chi2
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

# Input paths
BASELINE_METRICS_PATH = project_root / 'reports' / 'Regression_beforeCNN' / 'baseline_metrics_comparison.csv'
YOLO_METRICS_PATH = project_root / 'reports' / 'Regression_withYOLO' / 'logistic_regression_with_yolo_metrics.csv'
BASELINE_THRESH_OPT_PATH = project_root / 'reports' / 'Regression_beforeCNN' / 'baseline_threshold_optimization.csv'
YOLO_THRESH_OPT_PATH = project_root / 'reports' / 'Regression_withYOLO' / 'yolo_threshold_optimization.csv'
BASELINE_STATS_PATH = project_root / 'reports' / 'baseline_logistic_regression_statistics.csv'
YOLO_STATS_PATH = project_root / 'reports' / 'Regression_withYOLO' / 'logistic_regression_with_yolo_statistics.csv'

# Output paths
OUTPUT_DIR = project_root / 'reports' / 'Comparison_Baseline_vs_YOLO'
CLUSTERS_ALL_PATH = project_root / 'data' / 'processed' / 'clusters_with_crashes.csv'
YOLO_SPLIT_PATHS = {
    'Train': project_root / 'data' / 'processed' / 'clusters_train_with_yolo_roi.csv',
    'Val': project_root / 'data' / 'processed' / 'clusters_val_with_yolo_roi.csv',
    'Test': project_root / 'data' / 'processed' / 'clusters_test_with_yolo_roi.csv',
}
BASELINE_MODEL_PATH = project_root / 'models' / 'baseline_logistic_regression.pkl'
YOLO_MODEL_PATH = project_root / 'models' / 'logistic_regression_with_yolo.pkl'
BASELINE_META_PATH = project_root / 'models' / 'baseline_logistic_regression_metadata.json'
YOLO_META_PATH = project_root / 'models' / 'logistic_regression_with_yolo_metadata.json'


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_metrics():
    """
    Load metrics from both baseline and YOLO models.
    
    Returns:
        tuple: (baseline_df, yolo_df) - DataFrames with metrics
    """
    print("=" * 80)
    print("LOADING METRICS FROM BOTH MODELS")
    print("=" * 80)
    
    # Load baseline metrics
    if not BASELINE_METRICS_PATH.exists():
        print(f"❌ Error: Baseline metrics not found at {BASELINE_METRICS_PATH}")
        print("Please run baseline_logistic_regression.py first")
        return None, None
    
    baseline_df = pd.read_csv(BASELINE_METRICS_PATH)
    print(f"✓ Loaded baseline metrics: {len(baseline_df)} splits")
    
    # Load YOLO metrics
    if not YOLO_METRICS_PATH.exists():
        print(f"❌ Error: YOLO metrics not found at {YOLO_METRICS_PATH}")
        print("Please run logistic_regression_with_yolo.py first")
        return None, None
    
    yolo_df = pd.read_csv(YOLO_METRICS_PATH)
    print(f"✓ Loaded YOLO metrics: {len(yolo_df)} splits")
    
    # Print summary
    print(f"\nBaseline splits: {list(baseline_df['Split'].values)}")
    print(f"YOLO splits: {list(yolo_df['Split'].values)}")
    
    return baseline_df, yolo_df


def create_comparison_dataframe(baseline_df, yolo_df):
    """
    Create comparison DataFrame with both models' metrics.
    
    Parameters:
        baseline_df: Baseline metrics DataFrame
        yolo_df: YOLO metrics DataFrame
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON DATAFRAME")
    print("=" * 80)
    
    # Merge on Split column
    comparison_df = pd.merge(
        baseline_df, 
        yolo_df, 
        on='Split', 
        suffixes=('_baseline', '_yolo')
    )
    
    # Calculate deltas
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        yolo_col = f"{metric}_yolo"
        delta_col = f"Δ{metric}"
        
        if baseline_col in comparison_df.columns and yolo_col in comparison_df.columns:
            comparison_df[delta_col] = comparison_df[yolo_col] - comparison_df[baseline_col]
    
    print(f"✓ Created comparison DataFrame with {len(comparison_df)} splits")
    print(f"  Metrics compared: {metrics}")
    
    return comparison_df


def create_comparison_visualizations(comparison_df, output_dir):
    """
    Create comprehensive comparison visualizations.
    
    Parameters:
        comparison_df: Comparison DataFrame
        output_dir: Output directory for saving figures
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison data
    comparison_df.to_csv(output_dir / 'metrics_comparison.csv', index=False)
    print(f"  ✓ Saved comparison data to metrics_comparison.csv")
    
    # Extract data for plotting
    splits = comparison_df['Split'].values
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    # 1. Side-by-side bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(splits))
    width = 0.12
    
    for i, metric in enumerate(metrics):
        baseline_values = comparison_df[f"{metric}_baseline"].values
        yolo_values = comparison_df[f"{metric}_yolo"].values
        
        ax.bar(x + i * width, baseline_values, width, 
               label=f'{metric} (Baseline)', alpha=0.7, color=f'C{i}')
        ax.bar(x + i * width + width/2, yolo_values, width, 
               label=f'{metric} (YOLO)', alpha=0.9, color=f'C{i}', 
               edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Enhanced Logistic Regression (with YOLO): Metrics Comparison', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(splits)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_bars.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_bars.png")
    
    # 2. Delta comparison (improvement/degradation)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    delta_metrics = [f"Δ{metric}" for metric in metrics]
    delta_data = comparison_df[delta_metrics].values.T
    
    # Create heatmap
    im = ax.imshow(delta_data, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(splits)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(splits)
    ax.set_yticklabels(metrics)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(splits)):
            text = ax.text(j, i, f'{delta_data[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Delta: YOLO Model - Baseline Model\n(Positive = YOLO Better, Negative = Baseline Better)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Delta Score')
    cbar.ax.axhline(y=0, color='black', linewidth=2)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_delta_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_delta_heatmap.png")
    
    # 3. Line plot showing trends
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, metric in enumerate(metrics):
        baseline_values = comparison_df[f"{metric}_baseline"].values
        yolo_values = comparison_df[f"{metric}_yolo"].values
        
        ax.plot(splits, baseline_values, marker='o', linewidth=2, 
               label=f'{metric} (Baseline)', color=f'C{i}', linestyle='--')
        ax.plot(splits, yolo_values, marker='s', linewidth=2, 
               label=f'{metric} (YOLO)', color=f'C{i}', linestyle='-')
    
    ax.set_xlabel('Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Enhanced Logistic Regression (with YOLO): Performance Trends', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_lines.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_lines.png")
    
    # 4. Individual metric comparisons (subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        baseline_values = comparison_df[f"{metric}_baseline"].values
        yolo_values = comparison_df[f"{metric}_yolo"].values
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_values, width, 
                      label='Baseline', alpha=0.7, color='steelblue')
        bars2 = ax.bar(x + width/2, yolo_values, width, 
                      label='YOLO', alpha=0.9, color='coral')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    fig.suptitle('Baseline vs Enhanced Logistic Regression (with YOLO): Individual Metric Comparisons', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_comparison_individual.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved metrics_comparison_individual.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def create_test_focus_analysis(comparison_df, output_dir):
    """
    Create focused analysis on test set performance (most important comparison).
    
    Parameters:
        comparison_df: Comparison DataFrame
        output_dir: Output directory for saving figures
    """
    print("\n" + "=" * 80)
    print("CREATING TEST-FOCUSED ANALYSIS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    # Filter to test split only
    test_data = comparison_df[comparison_df['Split'] == 'Test'].copy()
    
    if len(test_data) == 0:
        print("  ⚠️ No test data found, skipping test-focused analysis")
        return
    
    # Create test comparison table
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    test_comparison = []
    
    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        yolo_col = f"{metric}_yolo"
        delta_col = f"Δ{metric}"
        
        if all(col in test_data.columns for col in [baseline_col, yolo_col, delta_col]):
            test_comparison.append({
                'Metric': metric,
                'Baseline': test_data[baseline_col].iloc[0],
                'YOLO': test_data[yolo_col].iloc[0],
                'Delta': test_data[delta_col].iloc[0],
                'Improvement': 'Yes' if test_data[delta_col].iloc[0] > 0 else 'No'
            })
    
    test_df = pd.DataFrame(test_comparison)
    test_df.to_csv(output_dir / 'test_metrics_delta.csv', index=False)
    print(f"  ✓ Saved test comparison to test_metrics_delta.csv")
    
    # Create test-focused visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Test metrics comparison
    width = 0.35
    
    baseline_values = [test_df[test_df['Metric'] == m]['Baseline'].iloc[0] for m in metrics]
    yolo_values = [test_df[test_df['Metric'] == m]['YOLO'].iloc[0] for m in metrics]
    # Append absolute Pseudo-R² for Baseline and YOLO
    fit_path = Path(output_dir) / 'fit_stats_test.csv'
    if fit_path.exists():
        fs = pd.read_csv(fit_path)
        if set(['Model','mcfadden_pseudo_r2']).issubset(fs.columns):
            r2_base = fs.loc[fs['Model'] == 'Baseline', 'mcfadden_pseudo_r2']
            r2_yolo = fs.loc[fs['Model'] == 'YOLO', 'mcfadden_pseudo_r2']
            if len(r2_base) and len(r2_yolo):
                baseline_values.append(float(r2_base.iloc[0]))
                yolo_values.append(float(r2_yolo.iloc[0]))
                metrics.append('Pseudo-R²')
    
    # Recompute x after potential extension
    x = np.arange(len(metrics))
    bars1 = ax1.bar(x - width/2, baseline_values, width, 
                   label='Baseline', alpha=0.7, color='steelblue')
    bars2 = ax1.bar(x + width/2, yolo_values, width, 
                   label='YOLO', alpha=0.9, color='coral')
    
    ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Test Set Performance: Baseline vs YOLO Model', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Right plot: Delta values (build once, extended with ΔPseudo-R²)
    base_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    labels = base_labels.copy()
    delta_values = [test_df[test_df['Metric'] == m]['Delta'].iloc[0] for m in base_labels]
    summary_path = Path(output_dir) / 'fit_stats_summary_test.csv'
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        if len(s) > 0:
            dr2 = float(s['delta_pseudo_r2'].iloc[0])
            labels = labels + ['Pseudo-R²']
            delta_values = delta_values + [dr2]
    # Redraw right plot with extended labels
    ax2.clear()
    x = np.arange(len(labels))
    colors = ['green' if d > 0 else 'red' for d in delta_values]
    bars3 = ax2.bar(x, delta_values, color=colors, alpha=0.7)
    ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Delta Score (YOLO - Baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Test Set Improvement: YOLO vs Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    for bar, delta in zip(bars3, delta_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (0.0001 if height >= 0 else -0.0001),
                f'{delta:+.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / 'test_metrics_focus.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved test_metrics_focus.png")
    
    # Print summary
    print(f"\nTest Set Performance Summary:")
    print(f"  {'Metric':<12} {'Baseline':<10} {'YOLO':<10} {'Delta':<8} {'Better':<8}")
    print(f"  {'-'*50}")
    for _, row in test_df.iterrows():
        better = "YOLO" if row['Delta'] > 0 else "Baseline" if row['Delta'] < 0 else "Tie"
        print(f"  {row['Metric']:<12} {row['Baseline']:<10.4f} {row['YOLO']:<10.4f} {row['Delta']:+8.4f} {better:<8}")


def create_test_focus_optimized(output_dir):
    """
    Create a test-focused comparison figure using validation-optimized thresholds
    (F1 and Youden) for both Baseline and YOLO models.
    """
    print("\n" + "=" * 80)
    print("CREATING TEST-FOCUSED ANALYSIS (Optimized Thresholds)")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (BASELINE_THRESH_OPT_PATH.exists() and YOLO_THRESH_OPT_PATH.exists()):
        print("  ⚠️ Threshold optimization files not found; skipping optimized comparison")
        return
    b = pd.read_csv(BASELINE_THRESH_OPT_PATH)
    y = pd.read_csv(YOLO_THRESH_OPT_PATH)

    # Filter Test rows
    b_test = b[(b['Split'] == 'Test')].copy()
    y_test = y[(y['Split'] == 'Test')].copy()

    # For each criterion, build a comparison table
    criteria = ['F1', 'Youden']
    for crit in criteria:
        b_row = b_test[b_test['Criterion'] == crit]
        y_row = y_test[y_test['Criterion'] == crit]
        if len(b_row) == 0 or len(y_row) == 0:
            continue
        metrics = ['accuracy','precision','recall','f1_score','roc_auc']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Build left panel values and append absolute Pseudo-R²
        fit_path = output_dir / 'fit_stats_test.csv'
        metrics_lbl = [m.replace('_',' ').title() if m!='roc_auc' else 'ROC AUC' for m in metrics]
        b_vals = [float(b_row[m].iloc[0]) for m in metrics]
        y_vals = [float(y_row[m].iloc[0]) for m in metrics]
        if fit_path.exists():
            fs = pd.read_csv(fit_path)
            if set(['Model','mcfadden_pseudo_r2']).issubset(fs.columns):
                r2_base = fs.loc[fs['Model'] == 'Baseline', 'mcfadden_pseudo_r2']
                r2_yolo = fs.loc[fs['Model'] == 'YOLO', 'mcfadden_pseudo_r2']
                if len(r2_base) and len(r2_yolo):
                    metrics_lbl = metrics_lbl + ['Pseudo-R²']
                    b_vals = b_vals + [float(r2_base.iloc[0])]
                    y_vals = y_vals + [float(r2_yolo.iloc[0])]
        x = np.arange(len(metrics_lbl))
        width = 0.35
        # b_vals and y_vals already initialized; appended above if R² present
        bars1 = ax1.bar(x - width/2, b_vals, width, label='Baseline', alpha=0.7, color='steelblue')
        bars2 = ax1.bar(x + width/2, y_vals, width, label='YOLO', alpha=0.9, color='coral')
        ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Test Performance at {crit}-optimized threshold (Val-chosen)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_lbl, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.05])
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x()+bar.get_width()/2., h+0.002, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        # Extend improvement bars to include ΔPseudo-R² alongside ΔAUC
        labels = metrics_lbl
        deltas = [y - b for y, b in zip(y_vals, b_vals)]
        ax2.clear()
        x = np.arange(len(labels))
        colors = ['green' if d > 0 else 'red' for d in deltas]
        bars3 = ax2.bar(x, deltas, color=colors, alpha=0.7)
        ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Delta (YOLO - Baseline)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Improvement at {crit}-optimized threshold (Test)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)
        for bar, d in zip(bars3, deltas):
            h = bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2., h+(0.0001 if h>=0 else -0.0001), f'{d:+.3f}', ha='center', va='bottom' if h>=0 else 'top', fontsize=9)
        plt.tight_layout()
        out_path = output_dir / f'test_metrics_focus_optimized_{crit.lower()}.png'
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {out_path.name}")


def print_summary_statistics(comparison_df):
    """
    Print summary statistics of the comparison.
    
    Parameters:
        comparison_df: Comparison DataFrame
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    print(f"\nOverall Performance Comparison:")
    print(f"  {'Split':<8} {'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print(f"  {'-'*80}")
    
    for _, row in comparison_df.iterrows():
        split = row['Split']
        print(f"  {split:<8} {'Baseline':<10} {row['Accuracy_baseline']:<10.4f} {row['Precision_baseline']:<10.4f} {row['Recall_baseline']:<10.4f} {row['F1-Score_baseline']:<10.4f} {row['ROC AUC_baseline']:<10.4f}")
        print(f"  {split:<8} {'YOLO':<10} {row['Accuracy_yolo']:<10.4f} {row['Precision_yolo']:<10.4f} {row['Recall_yolo']:<10.4f} {row['F1-Score_yolo']:<10.4f} {row['ROC AUC_yolo']:<10.4f}")
        print()
    
    # Calculate average improvements
    print(f"Average Improvements (YOLO - Baseline):")
    for metric in metrics:
        delta_col = f"Δ{metric}"
        if delta_col in comparison_df.columns:
            avg_delta = comparison_df[delta_col].mean()
            print(f"  {metric:<12}: {avg_delta:+.4f}")
    
    # Test set focus
    test_data = comparison_df[comparison_df['Split'] == 'Test']
    if len(test_data) > 0:
        print(f"\nTest Set Improvements:")
        for metric in metrics:
            delta_col = f"Δ{metric}"
            if delta_col in test_data.columns:
                test_delta = test_data[delta_col].iloc[0]
                print(f"  {metric:<12}: {test_delta:+.4f}")


def create_coefficients_comparison(output_dir):
    """
    Merge coefficients, std errors, and p-values for Baseline vs YOLO.
    Save as coefficients_comparison.csv. Also collate optimized thresholds.
    """
    print("\n" + "=" * 80)
    print("CREATING COEFFICIENTS COMPARISON TABLE")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (BASELINE_STATS_PATH.exists() and YOLO_STATS_PATH.exists()):
        print("  ⚠️ Stats files not found; skipping coefficients comparison")
        return
    b = pd.read_csv(BASELINE_STATS_PATH)
    y = pd.read_csv(YOLO_STATS_PATH)

    # Select columns and rename
    b2 = b[['feature','coefficient','std_error','p_value']].copy()
    b2.columns = ['feature','beta_baseline','se_baseline','p_baseline']
    y2 = y[['feature','coefficient','std_error','p_value']].copy()
    y2.columns = ['feature','beta_yolo','se_yolo','p_yolo']

    merged = pd.merge(b2, y2, on='feature', how='outer')
    merged.to_csv(output_dir / 'coefficients_comparison.csv', index=False)
    print(f"  ✓ Saved coefficients_comparison.csv")

    # Thresholds summary
    if BASELINE_THRESH_OPT_PATH.exists() and YOLO_THRESH_OPT_PATH.exists():
        tb = pd.read_csv(BASELINE_THRESH_OPT_PATH)
        ty = pd.read_csv(YOLO_THRESH_OPT_PATH)
        tb['model'] = 'Baseline'
        ty['model'] = 'YOLO'
        t = pd.concat([tb, ty], ignore_index=True)
        t.to_csv(output_dir / 'optimized_thresholds_summary.csv', index=False)
        print(f"  ✓ Saved optimized_thresholds_summary.csv")


def compute_log_likelihood(y_true, y_proba):
    eps = 1e-15
    p = np.clip(y_proba, eps, 1 - eps)
    ll = (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).sum()
    return float(ll)


def compute_pseudo_r2(y_true, y_proba):
    ll_model = compute_log_likelihood(y_true, y_proba)
    p_mean = y_true.mean()
    ll_null = compute_log_likelihood(y_true, np.full_like(y_true, p_mean))
    pseudo_r2 = 1.0 - (ll_model / ll_null) if ll_null != 0 else np.nan
    return ll_model, ll_null, pseudo_r2


def prepare_features_for_model(df, meta):
    cats = meta['features']['categorical']
    bools = meta['features']['boolean']
    nums = meta['features']['numeric']
    cols = [c for c in (cats + bools + nums) if c in df.columns]
    return df[cols]


def compute_fit_stats(split, output_dir):
    print("\n" + "=" * 80)
    print(f"FIT STATS AND LR TEST ON {split.upper()}")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models and metadata
    baseline_pipe = joblib.load(BASELINE_MODEL_PATH)
    yolo_pipe = joblib.load(YOLO_MODEL_PATH)
    with open(BASELINE_META_PATH) as f:
        baseline_meta = json.load(f)
    with open(YOLO_META_PATH) as f:
        yolo_meta = json.load(f)

    # Load data and align rows by cluster_id
    # Use YOLO split dataset to ensure identical rows for both models
    yolo_df = pd.read_csv(YOLO_SPLIT_PATHS[split])
    y = yolo_df['match_label'].astype(int).values
    X_base = prepare_features_for_model(yolo_df, baseline_meta)
    X_yolo = prepare_features_for_model(yolo_df, yolo_meta)

    # Basic missing handling to mirror training scripts
    def fill_missing(X_df, meta):
        X_filled = X_df.copy()
        cats = meta['features']['categorical'] + meta['features']['boolean']
        nums = meta['features']['numeric']
        for c in cats:
            if c in X_filled.columns:
                X_filled[c] = X_filled[c].fillna('missing')
        for n in nums:
            if n in X_filled.columns:
                X_filled[n] = X_filled[n].fillna(X_filled[n].median())
        return X_filled

    X_base = fill_missing(X_base, baseline_meta)
    X_yolo = fill_missing(X_yolo, yolo_meta)

    # Predict probabilities
    p_base = baseline_pipe.predict_proba(X_base)[:, 1]
    p_yolo = yolo_pipe.predict_proba(X_yolo)[:, 1]

    # Compute AUC delta on this split
    from sklearn.metrics import roc_auc_score
    auc_base = roc_auc_score(y, p_base)
    auc_yolo = roc_auc_score(y, p_yolo)

    # McFadden's pseudo-R2
    ll_base, ll0_base, r2_base = compute_pseudo_r2(y, p_base)
    ll_yolo, ll0_yolo, r2_yolo = compute_pseudo_r2(y, p_yolo)

    # LR test between baseline (L0) and YOLO (L1) with df=1
    lr_stat = 2.0 * (ll_yolo - ll_base)
    p_val = 1.0 - chi2.cdf(lr_stat, df=1)

    out = pd.DataFrame([
        {'Split': split, 'Model':'Baseline','log_likelihood': ll_base, 'log_likelihood_null': ll0_base, 'mcfadden_pseudo_r2': r2_base, 'auc': auc_base},
        {'Split': split, 'Model':'YOLO','log_likelihood': ll_yolo, 'log_likelihood_null': ll0_yolo, 'mcfadden_pseudo_r2': r2_yolo, 'auc': auc_yolo},
    ])
    fn = output_dir / f'fit_stats_{split.lower()}.csv'
    out.to_csv(fn, index=False)
    print(f"  ✓ Saved {fn.name}")

    summary = pd.DataFrame([
        {'Split': split, 'delta_auc': auc_yolo - auc_base, 'lr_stat': lr_stat, 'lr_df': 1, 'lr_p_value': p_val,
         'r2_base': r2_base, 'r2_yolo': r2_yolo, 'delta_pseudo_r2': r2_yolo - r2_base}
    ])
    fn2 = output_dir / f'fit_stats_summary_{split.lower()}.csv'
    summary.to_csv(fn2, index=False)
    print(f"  ✓ Saved {fn2.name}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("COMPARE BASELINE VS ENHANCED LOGISTIC REGRESSION")
    print("=" * 80)
    
    # Step 1: Load metrics
    baseline_df, yolo_df = load_metrics()
    if baseline_df is None or yolo_df is None:
        return
    
    # Step 2: Create comparison DataFrame
    comparison_df = create_comparison_dataframe(baseline_df, yolo_df)
    
    # Step 3: Create visualizations
    create_comparison_visualizations(comparison_df, OUTPUT_DIR)
    
    # Step 4: Create test-focused analysis
    create_test_focus_analysis(comparison_df, OUTPUT_DIR)
    create_test_focus_optimized(OUTPUT_DIR)

    # Step 4.5: Create coefficients comparison
    create_coefficients_comparison(OUTPUT_DIR)
    
    # Step 5: Print summary statistics
    print_summary_statistics(comparison_df)

    # Step 6: Compute fit stats and LR tests (Train and Test)
    for split in ['Train', 'Test']:
        compute_fit_stats(split, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("Key files:")
    print(f"  - metrics_comparison.csv (full comparison data)")
    print(f"  - test_metrics_delta.csv (test set focus)")
    print(f"  - test_metrics_focus.png (test set visualization)")
    print(f"  - metrics_comparison_bars.png (side-by-side comparison)")
    print(f"  - metrics_delta_heatmap.png (improvement heatmap)")


if __name__ == "__main__":
    main()
