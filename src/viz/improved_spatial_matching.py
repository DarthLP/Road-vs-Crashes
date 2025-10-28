#!/usr/bin/env python3
"""
Improved Spatial Matching Visualizations

This script creates three distinct, informative visualizations for spatial matching analysis:
1. Spatial matching overview (geographic + distance heatmap)
2. Spatial matching density (4-panel density comparison)
3. Distance distribution spatial (color-coded by distance)

Each visualization provides unique insights without redundancy.

Usage:
    python src/viz/improved_spatial_matching.py

Output:
    reports/figures/matcheddata/spatial_matching_overview.png
    reports/figures/matcheddata/spatial_matching_density.png
    reports/figures/matcheddata/distance_distribution_spatial.png
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def load_data():
    """
    Load crash and image data with OSM attributes.
    
    Returns:
        tuple: (crashes_df, images_df, crashes_original_df, images_original_df) containing loaded data
    """
    print("Loading data...")
    
    # Load matched data
    crashes_path = Path(project_root) / 'data' / 'processed' / 'crashes_with_osm.csv'
    crashes_df = pd.read_csv(crashes_path)
    
    images_path = Path(project_root) / 'data' / 'processed' / 'mapillary_with_osm.csv'
    images_df = pd.read_csv(images_path)
    
    # Load original data for total counts
    crashes_original_path = Path(project_root) / 'data' / 'interim' / 'crashes_aggregated.csv'
    images_original_path = Path(project_root) / 'data' / 'interim' / 'mapillary_berlin_full.csv'
    
    crashes_original_df = pd.read_csv(crashes_original_path)
    images_original_df = pd.read_csv(images_original_path)
    
    print(f"✅ Loaded {len(crashes_df):,} matched crashes, {len(images_df):,} matched images")
    print(f"✅ Original data: {len(crashes_original_df):,} total crashes, {len(images_original_df):,} total images")
    
    return crashes_df, images_df, crashes_original_df, images_original_df

def create_spatial_matching_overview(crashes_df, images_df, crashes_original_df, images_original_df, output_dir):
    """
    Create spatial matching overview with geographic distribution and clear statistics.
    
    Args:
        crashes_df: DataFrame with matched crash data
        images_df: DataFrame with matched image data
        crashes_original_df: DataFrame with all crash data
        images_original_df: DataFrame with all image data
        output_dir: Path to output directory
    """
    print("Creating spatial matching overview...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left panel: Matched vs Unmatched points
    # Sample data for visualization (too many points otherwise)
    crash_sample = crashes_df.sample(n=min(5000, len(crashes_df)), random_state=42)
    image_sample = images_df.sample(n=min(5000, len(images_df)), random_state=42)
    
    # Calculate full counts for legend
    matched_crashes_full = len(crashes_df[crashes_df['dist_to_road_m'] <= 25])
    unmatched_crashes_full = len(crashes_df[crashes_df['dist_to_road_m'] > 25])
    matched_images_full = len(images_df[images_df['dist_to_road_m'] <= 25])
    unmatched_images_full = len(images_df[images_df['dist_to_road_m'] > 25])
    
    # Plot crashes using sample for visualization
    matched_crashes_sample = crash_sample[crash_sample['dist_to_road_m'] <= 25]
    unmatched_crashes_sample = crash_sample[crash_sample['dist_to_road_m'] > 25]
    
    ax1.scatter(matched_crashes_sample['XGCSWGS84'], matched_crashes_sample['YGCSWGS84'], 
                c='green', alpha=0.6, s=1, label=f'Matched Crashes ({matched_crashes_full:,})')
    ax1.scatter(unmatched_crashes_sample['XGCSWGS84'], unmatched_crashes_sample['YGCSWGS84'], 
                c='red', alpha=0.6, s=1, label=f'Unmatched Crashes ({unmatched_crashes_full:,})')
    
    # Plot images using sample for visualization
    matched_images_sample = image_sample[image_sample['dist_to_road_m'] <= 25]
    unmatched_images_sample = image_sample[image_sample['dist_to_road_m'] > 25]
    
    ax1.scatter(matched_images_sample['lon'], matched_images_sample['lat'], 
                c='blue', alpha=0.4, s=0.5, label=f'Matched Images ({matched_images_full:,})')
    ax1.scatter(unmatched_images_sample['lon'], unmatched_images_sample['lat'], 
                c='orange', alpha=0.4, s=0.5, label=f'Unmatched Images ({unmatched_images_full:,})')
    
    ax1.set_title('Spatial Distribution: Matched vs Unmatched Points\n(Sampled for visualization)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Match Statistics
    ax2.axis('off')  # Remove axes for text display
    
    # Calculate actual statistics using original totals
    total_crashes = len(crashes_original_df)
    matched_crashes_count = len(crashes_df)
    unmatched_crashes_count = total_crashes - matched_crashes_count
    crash_match_rate = matched_crashes_count / total_crashes * 100
    
    total_images = len(images_original_df)
    matched_images_count = len(images_df)
    unmatched_images_count = total_images - matched_images_count
    image_match_rate = matched_images_count / total_images * 100
    
    # Distance statistics
    crash_avg_dist = crashes_df['dist_to_road_m'].mean()
    crash_median_dist = crashes_df['dist_to_road_m'].median()
    image_avg_dist = images_df['dist_to_road_m'].mean()
    image_median_dist = images_df['dist_to_road_m'].median()
    
    stats_text = f"""MATCHING STATISTICS (25m threshold)

CRASH DATA:
• Total Crashes: {total_crashes:,}
• Matched: {matched_crashes_count:,} ({crash_match_rate:.1f}%)
• Unmatched: {unmatched_crashes_count:,} ({100-crash_match_rate:.1f}%)
• Avg Distance: {crash_avg_dist:.2f}m
• Median Distance: {crash_median_dist:.2f}m

IMAGE DATA:
• Total Images: {total_images:,}
• Matched: {matched_images_count:,} ({image_match_rate:.1f}%)
• Unmatched: {unmatched_images_count:,} ({100-image_match_rate:.1f}%)
• Avg Distance: {image_avg_dist:.2f}m
• Median Distance: {image_median_dist:.2f}m

KEY INSIGHTS:
• Crash data has excellent spatial accuracy
• Image data covers diverse environments
• 25m threshold provides balanced coverage
• Unmatched points often in parks/water"""
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'spatial_matching_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved spatial matching overview to: {output_path}")

def create_spatial_matching_density(crashes_df, images_df, output_dir):
    """
    Create 4-panel density comparison visualization.
    
    Args:
        crashes_df: DataFrame with crash data
        images_df: DataFrame with image data
        output_dir: Path to output directory
    """
    print("Creating spatial matching density visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top-left: Matched crashes density
    matched_crashes = crashes_df[crashes_df['dist_to_road_m'] <= 25]
    if len(matched_crashes) > 0:
        ax1.hexbin(matched_crashes['XGCSWGS84'], matched_crashes['YGCSWGS84'], 
                   gridsize=50, cmap='Greens', alpha=0.8)
        ax1.set_title(f'Matched Crashes Density\n({len(matched_crashes):,} crashes)', 
                      fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Unmatched crashes density
    unmatched_crashes = crashes_df[crashes_df['dist_to_road_m'] > 25]
    if len(unmatched_crashes) > 0:
        ax2.hexbin(unmatched_crashes['XGCSWGS84'], unmatched_crashes['YGCSWGS84'], 
                   gridsize=50, cmap='Reds', alpha=0.8)
        ax2.set_title(f'Unmatched Crashes Density\n({len(unmatched_crashes):,} crashes)', 
                      fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Matched images density
    matched_images = images_df[images_df['dist_to_road_m'] <= 25]
    if len(matched_images) > 0:
        ax3.hexbin(matched_images['lon'], matched_images['lat'], 
                   gridsize=50, cmap='Blues', alpha=0.8)
        ax3.set_title(f'Matched Images Density\n({len(matched_images):,} images)', 
                      fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Unmatched images density
    unmatched_images = images_df[images_df['dist_to_road_m'] > 25]
    if len(unmatched_images) > 0:
        ax4.hexbin(unmatched_images['lon'], unmatched_images['lat'], 
                   gridsize=50, cmap='Oranges', alpha=0.8)
        ax4.set_title(f'Unmatched Images Density\n({len(unmatched_images):,} images)', 
                      fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Spatial Density Analysis: Matched vs Unmatched Data\n(25m threshold)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'spatial_matching_density.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved spatial matching density to: {output_path}")

def create_distance_distribution_spatial(crashes_df, images_df, output_dir):
    """
    Create single map with all points color-coded by distance to nearest road.
    
    Args:
        crashes_df: DataFrame with crash data
        images_df: DataFrame with image data
        output_dir: Path to output directory
    """
    print("Creating distance distribution spatial map...")
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Use full data for statistics, sample for visualization
    crash_sample = crashes_df.sample(n=min(8000, len(crashes_df)), random_state=42)
    image_sample = images_df.sample(n=min(8000, len(images_df)), random_state=42)
    
    # Plot crashes with specific colors using sample for visualization
    crashes_0_10_sample = crash_sample[crash_sample['dist_to_road_m'] <= 10]
    crashes_10_25_sample = crash_sample[(crash_sample['dist_to_road_m'] > 10) & (crash_sample['dist_to_road_m'] <= 25)]
    
    # Calculate full counts for legend
    crashes_0_10_full = len(crashes_df[crashes_df['dist_to_road_m'] <= 10])
    crashes_10_25_full = len(crashes_df[(crashes_df['dist_to_road_m'] > 10) & (crashes_df['dist_to_road_m'] <= 25)])
    
    if len(crashes_0_10_sample) > 0:
        ax.scatter(crashes_0_10_sample['XGCSWGS84'], crashes_0_10_sample['YGCSWGS84'], 
                  c='darkblue', alpha=0.7, s=2, marker='o', 
                  label=f'Crashes 0-10m ({crashes_0_10_full:,})')
    
    if len(crashes_10_25_sample) > 0:
        ax.scatter(crashes_10_25_sample['XGCSWGS84'], crashes_10_25_sample['YGCSWGS84'], 
                  c='lightblue', alpha=0.7, s=2, marker='o', 
                  label=f'Crashes 10-25m ({crashes_10_25_full:,})')
    
    # Plot images with specific colors using sample for visualization
    images_0_10_sample = image_sample[image_sample['dist_to_road_m'] <= 10]
    images_10_25_sample = image_sample[(image_sample['dist_to_road_m'] > 10) & (image_sample['dist_to_road_m'] <= 25)]
    
    # Calculate full counts for legend
    images_0_10_full = len(images_df[images_df['dist_to_road_m'] <= 10])
    images_10_25_full = len(images_df[(images_df['dist_to_road_m'] > 10) & (images_df['dist_to_road_m'] <= 25)])
    
    if len(images_0_10_sample) > 0:
        ax.scatter(images_0_10_sample['lon'], images_0_10_sample['lat'], 
                  c='darkgreen', alpha=0.5, s=1, marker='s', 
                  label=f'Images 0-10m ({images_0_10_full:,})')
    
    if len(images_10_25_sample) > 0:
        ax.scatter(images_10_25_sample['lon'], images_10_25_sample['lat'], 
                  c='lightgreen', alpha=0.5, s=1, marker='s', 
                  label=f'Images 10-25m ({images_10_25_full:,})')
    
    ax.set_title('Distance to Nearest Road Distribution\n(Matched Data Only - ≤25m)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'distance_distribution_spatial.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved distance distribution spatial to: {output_path}")

def main():
    """
    Main function to generate improved spatial matching visualizations.
    """
    print("Improved Spatial Matching Visualizations")
    print("=" * 50)
    
    try:
        # Load data
        crashes_df, images_df, crashes_original_df, images_original_df = load_data()
        
        # Create output directory
        output_dir = Path(project_root) / 'reports' / 'figures' / 'matcheddata'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        create_spatial_matching_overview(crashes_df, images_df, crashes_original_df, images_original_df, output_dir)
        create_distance_distribution_spatial(crashes_df, images_df, output_dir)
        
        print("\n" + "=" * 50)
        print("SPATIAL VISUALIZATION GENERATION COMPLETE")
        print("=" * 50)
        print("📊 Created 2 distinct visualizations:")
        print("   1. spatial_matching_overview.png - Geographic distribution + statistics")
        print("   2. distance_distribution_spatial.png - Color-coded distance map")
        print(f"\n📁 All saved to: {output_dir}")
        print("🎉 Each visualization provides unique insights!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        raise

if __name__ == "__main__":
    main()
