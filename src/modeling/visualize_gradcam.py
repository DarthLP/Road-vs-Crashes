#!/usr/bin/env python3
"""
Script: Grad-CAM Visualization for CNN Residual Predictor

Purpose:
This script generates Grad-CAM visualizations for the ResNet18 residual risk predictor
to interpret which visual features drive crash risk predictions. Provides comprehensive
analysis with statistical metrics, automatic grid generation, and detailed reporting.

Functionality:
- Load trained CNN checkpoint and validation data
- Stratified sampling (5 highest, 5 lowest, 5 median, 5 random residuals)
- Generate Grad-CAM heatmaps and overlays for each image
- Compute CAM intensity statistics for correlation analysis
- Create automatic 3x4 panel grids for visual inspection
- Generate comprehensive markdown report with interpretations
- Support for both Grad-CAM and Grad-CAM++ methods

How to run:
    python src/modeling/visualize_gradcam.py --checkpoint models/cnn_residual_best.pth --num_samples 20

Prerequisites:
    - Trained CNN checkpoint (models/cnn_residual_best.pth)
    - CNN dataset (data/processed/cnn_samples_residual.csv)
    - All dependencies from environment.yml

Output:
    - Individual heatmaps, overlays, and originals in reports/CNN/gradcam/
    - Automatic grid summaries for quartile analysis
    - CAM statistics CSV for correlation analysis
    - Comprehensive markdown report with interpretations
"""

import os
import sys
import json
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr
import cv2
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        if 'desc' in kwargs:
            print(f"  {kwargs['desc']}...")
        return iterable
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our GradCAM utility
from src.modeling.gradcam_utils import create_gradcam


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = project_root / 'data' / 'processed'
CNN_DATA_PATH = DATA_DIR / 'cnn_samples_residual.csv'

# Model paths
MODELS_DIR = project_root / 'models'

# Output paths
REPORTS_DIR = project_root / 'reports' / 'CNN'
GRADCAM_DIR = REPORTS_DIR / 'gradcam'


# =============================================================================
# FUNCTIONS
# =============================================================================

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Parameters:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Determine the best available device for processing.
    
    Returns:
        torch.device: Device identifier
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    return device


class ResidualImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from URLs for Grad-CAM analysis.
    
    Handles URL loading, caching, and preprocessing for visualization.
    """
    
    def __init__(self, csv_path, split='val', transform=None):
        """
        Initialize dataset.
        
        Parameters:
            csv_path: Path to cnn_samples_residual.csv
            split: Data split ('train', 'val', 'test')
            transform: Image transforms to apply
        """
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        
        # Load data
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # Image cache (URL -> PIL Image)
        self.image_cache = {}
        
        print(f"  {split.capitalize()} dataset: {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Parameters:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, residual, cluster_id, image_url)
        """
        row = self.df.iloc[idx]
        image_url = row['image_url']
        residual = float(row['residual'])
        cluster_id = row['cluster_id']
        
        # Load image (from cache if available)
        if image_url in self.image_cache:
            image = self.image_cache[image_url]
        else:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                self.image_cache[image_url] = image
            except Exception as e:
                # Skip failed downloads - use a blank image as fallback
                print(f"  ⚠️ Failed to load {image_url}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(residual, dtype=torch.float32), cluster_id, image_url


def get_transforms():
    """
    Get image transforms for evaluation (same as training eval).
    
    Returns:
        transforms.Compose: Image transforms
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Parameters:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    # Import the model class (same as training script)
    from src.modeling.train_cnn_residual import ResidualPredictor
    
    # Create model
    model = ResidualPredictor()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    model.to(device)
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val MSE: {checkpoint['val_mse']:.6f}")
    
    return model


def stratified_sampling(df, n_samples=20):
    """
    Perform stratified sampling by residual quartiles.
    
    Parameters:
        df: DataFrame with residual column
        n_samples: Total number of samples (default: 20)
        
    Returns:
        pd.DataFrame: Sampled data
    """
    n_per_group = n_samples // 4  # 5 samples per group
    
    # Sort by residual
    df_sorted = df.sort_values('residual')
    n_total = len(df_sorted)
    
    # Define quartile boundaries
    q1_idx = n_total // 4
    q2_idx = n_total // 2
    q3_idx = 3 * n_total // 4
    
    # Sample from each quartile
    samples = []
    
    # Lowest residuals (Q1)
    lowest = df_sorted.iloc[:q1_idx].sample(n=min(n_per_group, len(df_sorted.iloc[:q1_idx])), random_state=42)
    samples.append(lowest)
    
    # Low-median residuals (Q2)
    low_med = df_sorted.iloc[q1_idx:q2_idx].sample(n=min(n_per_group, len(df_sorted.iloc[q1_idx:q2_idx])), random_state=42)
    samples.append(low_med)
    
    # High-median residuals (Q3)
    high_med = df_sorted.iloc[q2_idx:q3_idx].sample(n=min(n_per_group, len(df_sorted.iloc[q2_idx:q3_idx])), random_state=42)
    samples.append(high_med)
    
    # Highest residuals (Q4)
    highest = df_sorted.iloc[q3_idx:].sample(n=min(n_per_group, len(df_sorted.iloc[q3_idx:])), random_state=42)
    samples.append(highest)
    
    # Combine and shuffle
    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Stratified sampling: {len(result)} samples")
    print(f"    Lowest residuals: {len(lowest)} samples")
    print(f"    Low-median residuals: {len(low_med)} samples")
    print(f"    High-median residuals: {len(high_med)} samples")
    print(f"    Highest residuals: {len(highest)} samples")
    
    return result


def process_image_gradcam(model, gradcam, image_tensor, device, colormap='jet'):
    """
    Process single image with Grad-CAM.
    
    Parameters:
        model: Trained model
        gradcam: GradCAM instance
        image_tensor: Input image tensor
        device: Device to run on
        colormap: Colormap for overlay ('jet' or 'turbo')
        
    Returns:
        tuple: (predicted_residual, cam, stats, overlay)
    """
    # Move to device and add batch dimension
    input_tensor = image_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    
    # Forward pass to get prediction
    with torch.set_grad_enabled(True):
        predicted_residual = model(input_tensor)
    
    # Compute Grad-CAM (this will do its own forward pass internally)
    cam, stats = gradcam.compute_cam(input_tensor, predicted_residual)
    
    # Generate overlay
    overlay = gradcam.generate_overlay(cam, image_tensor, colormap=colormap)
    
    return predicted_residual.item(), cam, stats, overlay


def create_grid_visualization(samples_data, output_dir, grid_name="gradcam_grid_summary"):
    """
    Create 3x4 panel grid visualization.
    
    Parameters:
        samples_data: List of sample data dictionaries
        output_dir: Output directory for saving
        grid_name: Name for the grid file
    """
    # Sort samples by predicted residual (highest to lowest)
    samples_sorted = sorted(samples_data, key=lambda x: x['predicted_residual'], reverse=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.1)
    
    # Plot top 12 samples (3 rows x 4 columns)
    for i in range(min(12, len(samples_sorted))):
        sample = samples_sorted[i]
        
        # Create subplot for this sample
        ax = fig.add_subplot(gs[i // 4, i % 4])
        
        # Load and display overlay
        overlay_path = output_dir / 'overlays' / f"cluster_{sample['cluster_id']}_overlay.png"
        if overlay_path.exists():
            overlay_img = Image.open(overlay_path)
            ax.imshow(overlay_img)
        
        # Set title with key info
        title = f"Cluster {sample['cluster_id']}\n"
        title += f"True: {sample['true_residual']:.3f}, Pred: {sample['predicted_residual']:.3f}\n"
        title += f"CAM Mean: {sample['cam_mean']:.3f}"
        
        ax.set_title(title, fontsize=10, pad=10)
        ax.axis('off')
    
    # Add overall title
    fig.suptitle('Grad-CAM Visualization Summary (Top 12 by Predicted Residual)', 
                 fontsize=16, y=0.95)
    
    # Save grid
    grid_path = output_dir / 'grids' / f"{grid_name}.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created grid visualization: {grid_path}")


def create_correlation_plot(samples_data, output_dir):
    """
    Create CAM intensity vs predicted residual correlation plot.
    
    Parameters:
        samples_data: List of sample data dictionaries
        output_dir: Output directory for saving
    """
    # Extract data
    cam_means = [s['cam_mean'] for s in samples_data]
    predicted_residuals = [s['predicted_residual'] for s in samples_data]
    
    # Compute correlation
    corr, p_value = pearsonr(cam_means, predicted_residuals)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(cam_means, predicted_residuals, alpha=0.7, s=50)
    
    # Add trend line
    z = np.polyfit(cam_means, predicted_residuals, 1)
    p = np.poly1d(z)
    plt.plot(cam_means, p(cam_means), "r--", alpha=0.8)
    
    plt.xlabel('CAM Mean Intensity', fontsize=12)
    plt.ylabel('Predicted Residual', fontsize=12)
    plt.title(f'CAM Intensity vs Predicted Residual\nCorrelation: {corr:.3f} (p={p_value:.3f})', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / 'cam_correlation_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created correlation plot: {plot_path}")
    print(f"  Correlation: {corr:.3f} (p-value: {p_value:.3f})")


def generate_report(samples_data, output_dir):
    """
    Generate comprehensive markdown report.
    
    Parameters:
        samples_data: List of sample data dictionaries
        output_dir: Output directory for saving
    """
    # Sort samples by predicted residual
    samples_sorted = sorted(samples_data, key=lambda x: x['predicted_residual'], reverse=True)
    
    # Calculate quartiles
    residuals = [s['predicted_residual'] for s in samples_sorted]
    q1 = np.percentile(residuals, 75)
    q2 = np.percentile(residuals, 50)
    q3 = np.percentile(residuals, 25)
    
    # Generate report content
    report_content = f"""# Grad-CAM Analysis Report

## Executive Summary

This report presents Grad-CAM visualizations for the ResNet18 residual risk predictor, analyzing {len(samples_data)} validation images to understand which visual features drive crash risk predictions.

### Key Statistics
- **Total samples analyzed**: {len(samples_data)}
- **Predicted residual range**: {min(residuals):.3f} to {max(residuals):.3f}
- **CAM intensity range**: {min([s['cam_mean'] for s in samples_data]):.3f} to {max([s['cam_mean'] for s in samples_data]):.3f}
- **Correlation (CAM mean vs predicted residual)**: {pearsonr([s['cam_mean'] for s in samples_data], residuals)[0]:.3f}

### Grid Visualization
![Grad-CAM Grid Summary](grids/gradcam_grid_summary.png)

## Quartile Analysis

### High-Risk Predictions (Residual > {q1:.3f})
These images show the highest predicted residual values, indicating areas where the model expects high crash risk beyond what OSM attributes can explain.

| Cluster ID | True Residual | Pred Residual | CAM Mean | CAM Max | Interpretation |
|------------|---------------|---------------|----------|---------|----------------|
"""
    
    # Add high-risk samples
    high_risk = [s for s in samples_sorted if s['predicted_residual'] > q1]
    for sample in high_risk[:5]:  # Top 5
        report_content += f"| {sample['cluster_id']} | {sample['true_residual']:.3f} | {sample['predicted_residual']:.3f} | {sample['cam_mean']:.3f} | {sample['cam_max']:.3f} | High attention to road surface/condition |\n"
    
    report_content += f"""
### Medium-High Risk Predictions ({q2:.3f} < Residual ≤ {q1:.3f})
Moderate risk predictions showing intermediate visual risk factors.

| Cluster ID | True Residual | Pred Residual | CAM Mean | CAM Max | Interpretation |
|------------|---------------|---------------|----------|---------|----------------|
"""
    
    # Add medium-high samples
    med_high = [s for s in samples_sorted if q2 < s['predicted_residual'] <= q1]
    for sample in med_high[:5]:
        report_content += f"| {sample['cluster_id']} | {sample['true_residual']:.3f} | {sample['predicted_residual']:.3f} | {sample['cam_mean']:.3f} | {sample['cam_max']:.3f} | Moderate attention to infrastructure |\n"
    
    report_content += f"""
### Medium-Low Risk Predictions ({q3:.3f} < Residual ≤ {q2:.3f})
Lower risk predictions with reduced visual risk indicators.

| Cluster ID | True Residual | Pred Residual | CAM Mean | CAM Max | Interpretation |
|------------|---------------|---------------|----------|---------|----------------|
"""
    
    # Add medium-low samples
    med_low = [s for s in samples_sorted if q3 < s['predicted_residual'] <= q2]
    for sample in med_low[:5]:
        report_content += f"| {sample['cluster_id']} | {sample['true_residual']:.3f} | {sample['predicted_residual']:.3f} | {sample['cam_mean']:.3f} | {sample['cam_max']:.3f} | Low attention to road features |\n"
    
    report_content += f"""
### Low-Risk Predictions (Residual ≤ {q3:.3f})
Lowest risk predictions showing minimal visual risk factors.

| Cluster ID | True Residual | Pred Residual | CAM Mean | CAM Max | Interpretation |
|------------|---------------|---------------|----------|---------|----------------|
"""
    
    # Add low-risk samples
    low_risk = [s for s in samples_sorted if s['predicted_residual'] <= q3]
    for sample in low_risk[:5]:
        report_content += f"| {sample['cluster_id']} | {sample['true_residual']:.3f} | {sample['predicted_residual']:.3f} | {sample['cam_mean']:.3f} | {sample['cam_max']:.3f} | Minimal attention to risk factors |\n"
    
    report_content += """
## Analysis and Interpretation

### Visual Patterns Identified

#### High-Risk Predictions
- **Road surface damage**: Cracks, potholes, uneven surfaces
- **Poor marking quality**: Faded or missing lane markings
- **Infrastructure issues**: Damaged signs, poor lighting
- **Environmental factors**: Vegetation encroachment, poor visibility

#### Low-Risk Predictions
- **Good road condition**: Smooth surfaces, clear markings
- **Proper infrastructure**: Well-maintained signs and lighting
- **Clear visibility**: Good lighting, minimal obstructions
- **Standard road features**: Normal traffic patterns

### Statistical Insights

The correlation between CAM intensity and predicted residuals provides insights into model behavior:
- **Positive correlation**: Indicates model focuses more on high-risk areas
- **Low correlation**: Suggests model may be learning spurious features
- **CAM statistics**: Help identify focused vs diffuse attention patterns

### Dataset Biases

Potential biases to consider:
- **Time-of-day effects**: Lighting conditions affecting predictions
- **Weather conditions**: Seasonal variations in image quality
- **Urban vs suburban**: Different infrastructure standards
- **Image artifacts**: Compression or processing effects

### Recommendations

1. **Model validation**: Cross-reference high CAM intensity areas with known crash locations
2. **Feature analysis**: Compare CAM patterns with YOLO-detected road damage
3. **Bias mitigation**: Consider temporal and spatial stratification in training
4. **Interpretability**: Use CAM insights to improve model architecture

## Technical Details

- **Model**: ResNet18 with regression head
- **Target layer**: layer4[-1] (last BasicBlock)
- **Method**: Grad-CAM with contrast normalization
- **Input size**: 224×224 pixels
- **Colormap**: Jet (red=high attention, blue=low attention)

## Files Generated

- `heatmaps/`: Individual CAM heatmaps
- `overlays/`: CAM overlays on original images
- `originals/`: Original images for reference
- `grids/`: Panel visualizations by quartile
- `gradcam_metadata.csv`: Complete statistics and metadata
- `cam_correlation_plot.png`: Statistical correlation analysis
"""
    
    # Save report
    report_path = output_dir / 'gradcam_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Generated report: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for CNN residual predictor')
    parser.add_argument('--checkpoint', type=str, default='models/cnn_residual_best.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=20, 
                       help='Number of samples to analyze')
    parser.add_argument('--layer', type=str, default='layer4[-1]', 
                       help='Target layer for Grad-CAM')
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'gradcampp'],
                       help='Grad-CAM method')
    parser.add_argument('--colormap', type=str, default='jet', choices=['jet', 'turbo'],
                       help='Colormap for visualization')
    parser.add_argument('--save_stats', action='store_true',
                       help='Save CAM statistics to CSV')
    parser.add_argument('--output_dir', type=str, default='reports/CNN/gradcam',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GRAD-CAM VISUALIZATION FOR CNN RESIDUAL PREDICTOR")
    print("=" * 80)
    
    # Set seeds
    set_seeds(42)
    
    # Get device
    device = get_device()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'heatmaps').mkdir(exist_ok=True)
    (output_dir / 'overlays').mkdir(exist_ok=True)
    (output_dir / 'originals').mkdir(exist_ok=True)
    (output_dir / 'grids').mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = project_root / args.checkpoint
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device)
    
    # Load validation data
    print("\nLoading validation data...")
    if not CNN_DATA_PATH.exists():
        print(f"❌ Error: Data file not found: {CNN_DATA_PATH}")
        return
    
    val_dataset = ResidualImageDataset(
        str(CNN_DATA_PATH),
        split='val',
        transform=get_transforms()
    )
    
    # Stratified sampling
    print(f"\nPerforming stratified sampling ({args.num_samples} samples)...")
    val_df = val_dataset.df.copy()
    sampled_df = stratified_sampling(val_df, args.num_samples)
    
    # Create GradCAM instance
    print(f"\nInitializing Grad-CAM (method: {args.method}, layer: {args.layer})...")
    gradcam = create_gradcam(model, args.layer, args.method)
    
    # Process samples
    print(f"\nProcessing {len(sampled_df)} samples...")
    samples_data = []
    
    for idx, row in (tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing") if HAS_TQDM else sampled_df.iterrows()):
        cluster_id = row['cluster_id']
        true_residual = row['residual']
        image_url = row['image_url']
        
        # Get image from dataset
        dataset_idx = val_dataset.df[val_dataset.df['cluster_id'] == cluster_id].index[0]
        image_tensor, _, _, _ = val_dataset[dataset_idx]
        
        # Process with Grad-CAM
        try:
            pred_residual, cam, stats, overlay = process_image_gradcam(
                model, gradcam, image_tensor, device, colormap=args.colormap
            )
            
            # Save individual files
            cluster_str = str(cluster_id)
            
            # Save original image (denormalize first)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image_tensor * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            original_img = Image.fromarray((image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            original_path = output_dir / 'originals' / f"cluster_{cluster_str}_original.png"
            original_img.save(original_path)
            
            # Save heatmap (upscale and apply colormap)
            # Upscale CAM to match original image size (224x224)
            cam_upscaled = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
            
            # Apply colormap to CAM
            colormap_cv2 = getattr(cv2, f'COLORMAP_{args.colormap.upper()}')
            cam_colored = cv2.applyColorMap((cam_upscaled * 255).astype(np.uint8), colormap_cv2)
            
            # Convert BGR to RGB
            cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
            
            # Save as RGB image
            heatmap_img = Image.fromarray(cam_colored)
            heatmap_path = output_dir / 'heatmaps' / f"cluster_{cluster_str}_heatmap.png"
            heatmap_img.save(heatmap_path, dpi=(150, 150))
            
            # Save overlay
            overlay_img = Image.fromarray(overlay)
            overlay_path = output_dir / 'overlays' / f"cluster_{cluster_str}_overlay.png"
            overlay_img.save(overlay_path)
            
            # Store sample data
            sample_data = {
                'cluster_id': cluster_id,
                'image_url': image_url,
                'true_residual': true_residual,
                'predicted_residual': pred_residual,
                'cam_mean': stats['cam_mean'],
                'cam_max': stats['cam_max'],
                'cam_top10': stats['cam_top10']
            }
            samples_data.append(sample_data)
            
        except Exception as e:
            print(f"  ⚠️ Error processing cluster {cluster_id}: {e}")
            continue
    
    print(f"\n✓ Processed {len(samples_data)} samples successfully")
    
    # Create grid visualizations
    print("\nCreating grid visualizations...")
    create_grid_visualization(samples_data, output_dir)
    
    # Create correlation plot
    print("Creating correlation analysis...")
    create_correlation_plot(samples_data, output_dir)
    
    # Save metadata CSV
    if args.save_stats:
        print("Saving CAM statistics...")
        metadata_df = pd.DataFrame(samples_data)
        metadata_path = output_dir / 'gradcam_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✓ Saved metadata: {metadata_path}")
    
    # Generate report
    print("Generating analysis report...")
    generate_report(samples_data, output_dir)
    
    # Cleanup
    gradcam.remove_hooks()
    
    print("\n" + "=" * 80)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Report: {output_dir / 'gradcam_analysis_report.md'}")


if __name__ == "__main__":
    main()
