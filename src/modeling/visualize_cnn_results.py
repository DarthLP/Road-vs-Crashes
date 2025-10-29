#!/usr/bin/env python3
"""
Script: Visualize CNN Residual Prediction Results

Purpose:
This script generates visualizations and example images for the CNN residual 
prediction model. It creates loss curves, scatter plots, distributions, and 
high-risk image galleries for the final report.

Functionality:
- Loads training log CSV to plot loss curves
- Creates scatter plots of predicted vs true residuals
- Visualizes residual distributions
- Generates gallery of top high-risk images
- Saves all figures to reports/CNN/figures/

How to run:
    python src/modeling/visualize_cnn_results.py

Prerequisites:
    - Run train_cnn_residual.py first to generate training_log.csv and predictions
    - Ensure reports/CNN/training_log.csv exists
    - Ensure data/processed/cnn_test_predictions.csv exists
    - Ensure data/processed/cnn_test_top_visual_risks.csv exists

Output:
    - reports/CNN/figures/loss_curves.png
    - reports/CNN/figures/residual_scatter.png
    - reports/CNN/figures/residual_distributions.png
    - reports/CNN/figures/top_risky_images.png (if images can be loaded)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Input paths
DATA_DIR = project_root / 'data' / 'processed'
REPORTS_DIR = project_root / 'reports' / 'CNN'
TRAINING_LOG_PATH = REPORTS_DIR / 'training_log.csv'
PREDICTIONS_PATH = DATA_DIR / 'cnn_test_predictions.csv'
TOP_RISKS_PATH = DATA_DIR / 'cnn_test_top_visual_risks.csv'

# Output paths
FIGURES_DIR = REPORTS_DIR / 'figures'


# Set style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.style.use('default')


# =============================================================================
# FUNCTIONS
# =============================================================================

def plot_loss_curves(training_log_path, output_dir):
    """
    Plot training and validation loss curves.
    
    Parameters:
        training_log_path: Path to training_log.csv
        output_dir: Output directory for figures
    """
    print("Creating loss curves...")
    
    df = pd.read_csv(training_log_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE curves
    ax = axes[0]
    ax.plot(df['epoch'], df['train_mse'], label='Train MSE', marker='o', linewidth=2)
    ax.plot(df['epoch'], df['val_mse'], label='Val MSE', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation MSE', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # MAE curves
    ax = axes[1]
    ax.plot(df['epoch'], df['train_mae'], label='Train MAE', marker='o', linewidth=2)
    ax.plot(df['epoch'], df['val_mae'], label='Val MAE', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'loss_curves.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {output_path}")


def plot_residual_scatter(predictions_path, output_dir):
    """
    Plot scatter plot of predicted vs true residuals.
    
    Parameters:
        predictions_path: Path to cnn_test_predictions.csv
        output_dir: Output directory for figures
    """
    print("Creating residual scatter plot...")
    
    df = pd.read_csv(predictions_path)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(df['true_residual'], df['predicted_residual'], alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(df['true_residual'].min(), df['predicted_residual'].min())
    max_val = max(df['true_residual'].max(), df['predicted_residual'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Compute correlation
    corr = np.corrcoef(df['true_residual'], df['predicted_residual'])[0, 1]
    mse = np.mean((df['true_residual'] - df['predicted_residual']) ** 2)
    mae = np.mean(np.abs(df['true_residual'] - df['predicted_residual']))
    
    ax.set_xlabel('True Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Residual', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted vs True Residuals\n(Pearson r={corr:.3f}, MSE={mse:.4f}, MAE={mae:.4f})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'residual_scatter.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {output_path}")


def plot_residual_distributions(predictions_path, output_dir):
    """
    Plot distributions of true and predicted residuals.
    
    Parameters:
        predictions_path: Path to cnn_test_predictions.csv
        output_dir: Output directory for figures
    """
    print("Creating residual distributions...")
    
    df = pd.read_csv(predictions_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution of true residuals
    ax = axes[0]
    ax.hist(df['true_residual'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df['true_residual'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["true_residual"].mean():.4f}')
    ax.set_xlabel('True Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of True Residuals', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Distribution of predicted residuals
    ax = axes[1]
    ax.hist(df['predicted_residual'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(df['predicted_residual'].mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {df["predicted_residual"].mean():.4f}')
    ax.set_xlabel('Predicted Residual', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Predicted Residuals', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'residual_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to: {output_path}")


def plot_top_risky_images(top_risks_path, output_dir, n_images=10):
    """
    Plot gallery of top high-risk images (if URLs can be loaded).
    
    Parameters:
        top_risks_path: Path to cnn_test_top_visual_risks.csv
        output_dir: Output directory for figures
        n_images: Number of top risky images to show
    """
    print(f"Creating top {n_images} risky images gallery...")
    
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        
        df = pd.read_csv(top_risks_path)
        top_n = df.head(n_images)
        
        # Try to load images
        images_loaded = []
        for idx, row in top_n.iterrows():
            try:
                response = requests.get(row['image_url'], timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    images_loaded.append({
                        'image': img,
                        'predicted_residual': row['predicted_residual'],
                        'true_residual': row['true_residual']
                    })
            except:
                continue
        
        if len(images_loaded) == 0:
            print("  ⚠️ Could not load any images from URLs (skipping gallery)")
            return
        
        # Create grid
        n_cols = 5
        n_rows = int(np.ceil(len(images_loaded) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_data in enumerate(images_loaded):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.imshow(img_data['image'])
            ax.set_title(f'Pred: {img_data["predicted_residual"]:.4f}\nTrue: {img_data["true_residual"]:.4f}', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(images_loaded), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(f'Top {len(images_loaded)} High-Risk Images (by Predicted Residual)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = output_dir / 'top_risky_images.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to: {output_path}")
        
    except ImportError:
        print("  ⚠️ Required packages (requests, PIL) not available (skipping gallery)")
    except Exception as e:
        print(f"  ⚠️ Error creating gallery: {e}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("VISUALIZE CNN RESIDUAL PREDICTION RESULTS")
    print("=" * 80)
    
    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if required files exist
    if not TRAINING_LOG_PATH.exists():
        print(f"❌ Error: Training log not found: {TRAINING_LOG_PATH}")
        print("   Please run train_cnn_residual.py first")
        return
    
    if not PREDICTIONS_PATH.exists():
        print(f"❌ Error: Predictions file not found: {PREDICTIONS_PATH}")
        print("   Please run train_cnn_residual.py first")
        return
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_loss_curves(TRAINING_LOG_PATH, FIGURES_DIR)
    plot_residual_scatter(PREDICTIONS_PATH, FIGURES_DIR)
    plot_residual_distributions(PREDICTIONS_PATH, FIGURES_DIR)
    plot_top_risky_images(TOP_RISKS_PATH, FIGURES_DIR, n_images=10)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\n✓ All visualizations saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

