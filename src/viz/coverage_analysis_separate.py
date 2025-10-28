#!/usr/bin/env python3
"""
Coverage Analysis Detailed - Separate Images Generator

This script creates separate, individual images from the comprehensive coverage analysis,
making it easier to use specific charts in presentations or reports.

Usage:
    python src/viz/coverage_analysis_separate.py

Output:
    reports/figures/matcheddata/coverage_analysis_*.png (multiple separate images)
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
        tuple: (crashes_df, images_df, roads_gdf, crashes_original_df, images_original_df) containing loaded data
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
    
    # Load road network data
    roads_path = Path(project_root) / 'data' / 'interim' / 'osm_berlin_roads.gpkg'
    roads_gdf = gpd.read_file(roads_path)
    
    print(f"✅ Loaded {len(crashes_df):,} matched crashes, {len(images_df):,} matched images, {len(roads_gdf):,} road segments")
    print(f"✅ Original data: {len(crashes_original_df):,} total crashes, {len(images_original_df):,} total images")
    
    return crashes_df, images_df, roads_gdf, crashes_original_df, images_original_df

def create_match_rates_chart(crashes_df, images_df, crashes_original_df, images_original_df, output_dir):
    """
    Create match rates comparison chart.
    """
    print("Creating match rates chart...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate match rates using original totals
    crash_match_rate = len(crashes_df) / len(crashes_original_df) * 100
    image_match_rate = len(images_df) / len(images_original_df) * 100
    
    categories = ['Crashes', 'Images']
    match_rates = [crash_match_rate, image_match_rate]
    colors = ['#2E8B57', '#4169E1']  # Sea green and royal blue
    
    bars = ax.bar(categories, match_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, match_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('OSM Road Matching Success Rates\n(25m threshold)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""Total Data Points:
• Crashes: {len(crashes_original_df):,}
• Images: {len(images_original_df):,}

Matched Points:
• Crashes: {len(crashes_df):,}
• Images: {len(images_df):,}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'coverage_analysis_match_rates.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved match rates chart to: {output_path}")

def create_distance_distribution_chart(crashes_df, images_df, output_dir):
    """
    Create distance distribution histogram.
    """
    print("Creating distance distribution chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Crash distance distribution
    crash_distances = crashes_df['dist_to_road_m']
    ax1.hist(crash_distances, bins=50, alpha=0.7, color='#2E8B57', edgecolor='black')
    ax1.axvline(x=25, color='red', linestyle='--', linewidth=2, label='25m threshold')
    ax1.set_title('Crash Distance Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance to Nearest Road (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Image distance distribution
    image_distances = images_df['dist_to_road_m']
    ax2.hist(image_distances, bins=50, alpha=0.7, color='#4169E1', edgecolor='black')
    ax2.axvline(x=25, color='red', linestyle='--', linewidth=2, label='25m threshold')
    ax2.set_title('Image Distance Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance to Nearest Road (m)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Distance to Nearest Road Distribution\n(All Data Points)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'coverage_analysis_distance_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved distance distribution chart to: {output_path}")


def create_road_type_distribution_chart(roads_gdf, output_dir):
    """
    Create road type distribution chart.
    """
    print("Creating road type distribution chart...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get road type distribution
    road_types = roads_gdf['highway'].value_counts().head(10)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(road_types)))
    bars = ax.bar(range(len(road_types)), road_types.values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('OSM Road Type Distribution\n(Top 10 Types)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Road Type', fontsize=12)
    ax.set_ylabel('Number of Segments', fontsize=12)
    ax.set_xticks(range(len(road_types)))
    ax.set_xticklabels(road_types.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add combined labels on bars (count (percentage))
    total_segments = len(roads_gdf)
    for bar, count in zip(bars, road_types.values):
        height = bar.get_height()
        percentage = count / total_segments * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,} ({percentage:.1f}%)', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'coverage_analysis_road_types.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved road type distribution chart to: {output_path}")

def create_attribute_coverage_chart(roads_gdf, output_dir):
    """
    Create OSM attribute coverage chart.
    """
    print("Creating attribute coverage chart...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate attribute coverage
    attributes = ['surface', 'maxspeed', 'lanes', 'lit', 'cycleway', 'sidewalk', 
                 'crossing', 'smoothness', 'width', 'parking:lane', 'traffic_calming']
    
    coverage_data = {}
    for attr in attributes:
        if attr in roads_gdf.columns:
            coverage = roads_gdf[attr].notna().sum() / len(roads_gdf) * 100
            coverage_data[attr] = coverage
    
    # Sort by coverage
    sorted_attrs = sorted(coverage_data.items(), key=lambda x: x[1], reverse=True)
    attr_names, coverage_values = zip(*sorted_attrs)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(attr_names)))
    bars = ax.bar(range(len(attr_names)), coverage_values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title('OSM Attribute Coverage\n(Percentage of Road Segments with Data)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Attribute', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_xticks(range(len(attr_names)))
    ax.set_xticklabels(attr_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, coverage in zip(bars, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{coverage:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'coverage_analysis_attribute_coverage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved attribute coverage chart to: {output_path}")

def main():
    """
    Main function to generate separate coverage analysis images.
    """
    print("Coverage Analysis - Separate Images Generator")
    print("=" * 50)
    
    try:
        # Load data
        crashes_df, images_df, roads_gdf, crashes_original_df, images_original_df = load_data()
        
        # Create output directory
        output_dir = Path(project_root) / 'reports' / 'figures' / 'matcheddata'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate charts
        create_match_rates_chart(crashes_df, images_df, crashes_original_df, images_original_df, output_dir)
        create_distance_distribution_chart(crashes_df, images_df, output_dir)
        create_road_type_distribution_chart(roads_gdf, output_dir)
        create_attribute_coverage_chart(roads_gdf, output_dir)
        
        print("\n" + "=" * 50)
        print("SEPARATE IMAGES GENERATION COMPLETE")
        print("=" * 50)
        print("📊 Created 4 separate charts:")
        print("   1. coverage_analysis_match_rates.png - Match rates comparison")
        print("   2. coverage_analysis_distance_distribution.png - Distance histograms")
        print("   3. coverage_analysis_road_types.png - Road type distribution")
        print("   4. coverage_analysis_attribute_coverage.png - Attribute coverage")
        print(f"\n📁 All saved to: {output_dir}")
        print("🎉 Ready for individual use in presentations!")
        
    except Exception as e:
        print(f"❌ Error generating separate images: {e}")
        raise

if __name__ == "__main__":
    main()
