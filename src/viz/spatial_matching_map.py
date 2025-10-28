#!/usr/bin/env python3
"""
Spatial Distribution Map: Matched vs Unmatched Images

This script creates a map showing the spatial distribution of Mapillary images,
color-coded by whether they matched to roads (within 10m) or not.

The map will help visualize:
- Geographic patterns in matching success
- Areas with high/low road density
- Berlin vs Brandenburg differences
- Off-road image locations (parks, water, etc.)

Usage:
    python src/viz/spatial_matching_map.py

Output:
    reports/figures/spatial_matching_distribution.png
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import contextily as ctx
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def load_data():
    """
    Load image data and identify matched vs unmatched images.
    
    Returns:
        tuple: (all_images_gdf, matched_images_gdf, unmatched_images_gdf)
    """
    print("Loading image data...")
    
    # Load all images
    all_images_df = pd.read_csv(os.path.join(project_root, 'data', 'interim', 'mapillary_berlin_full.csv'))
    
    # Load matched images
    matched_images_df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'mapillary_with_osm.csv'))
    
    # Create GeoDataFrames
    all_images_gdf = gpd.GeoDataFrame(
        all_images_df,
        geometry=gpd.points_from_xy(all_images_df['lon'], all_images_df['lat']),
        crs='EPSG:4326'
    )
    
    matched_images_gdf = gpd.GeoDataFrame(
        matched_images_df,
        geometry=gpd.points_from_xy(matched_images_df['lon'], matched_images_df['lat']),
        crs='EPSG:4326'
    )
    
    # Find unmatched images
    matched_ids = set(matched_images_df['id'])
    unmatched_images_df = all_images_df[~all_images_df['id'].isin(matched_ids)]
    
    unmatched_images_gdf = gpd.GeoDataFrame(
        unmatched_images_df,
        geometry=gpd.points_from_xy(unmatched_images_df['lon'], unmatched_images_df['lat']),
        crs='EPSG:4326'
    )
    
    print(f"✅ Loaded data:")
    print(f"   - Total images: {len(all_images_gdf):,}")
    print(f"   - Matched images: {len(matched_images_gdf):,}")
    print(f"   - Unmatched images: {len(unmatched_images_gdf):,}")
    
    return all_images_gdf, matched_images_gdf, unmatched_images_gdf

def create_spatial_matching_map(all_images_gdf, matched_images_gdf, unmatched_images_gdf):
    """
    Create a map showing spatial distribution of matched vs unmatched images.
    
    Args:
        all_images_gdf: All images GeoDataFrame
        matched_images_gdf: Matched images GeoDataFrame  
        unmatched_images_gdf: Unmatched images GeoDataFrame
    """
    print("Creating spatial matching distribution map...")
    
    # Convert to UTM for better visualization
    all_images_utm = all_images_gdf.to_crs(epsg=25833)
    matched_images_utm = matched_images_gdf.to_crs(epsg=25833)
    unmatched_images_utm = unmatched_images_gdf.to_crs(epsg=25833)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define colors
    matched_color = '#2E8B57'      # Sea green for matched
    unmatched_color = '#DC143C'    # Crimson for unmatched
    alpha = 0.6
    
    # Plot 1: All images (matched + unmatched)
    ax1.set_title('All Mapillary Images\n(Matched + Unmatched)', fontsize=14, fontweight='bold')
    
    # Plot unmatched images first (so they appear behind matched)
    unmatched_images_utm.plot(ax=ax1, color=unmatched_color, markersize=1, alpha=alpha, label='Unmatched')
    matched_images_utm.plot(ax=ax1, color=matched_color, markersize=1, alpha=alpha, label='Matched')
    
    # Add basemap
    try:
        ctx.add_basemap(ax1, crs=all_images_utm.crs, source=ctx.providers.CartoDB.Positron, alpha=0.7)
    except:
        print("Warning: Could not add basemap, using plain background")
    
    ax1.set_xlabel('Easting (UTM Zone 33N)', fontsize=12)
    ax1.set_ylabel('Northing (UTM Zone 33N)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    matched_patch = mpatches.Patch(color=matched_color, label=f'Matched ({len(matched_images_utm):,})')
    unmatched_patch = mpatches.Patch(color=unmatched_color, label=f'Unmatched ({len(unmatched_images_utm):,})')
    ax1.legend(handles=[matched_patch, unmatched_patch], loc='upper right', fontsize=10)
    
    # Plot 2: Density comparison
    ax2.set_title('Image Density Comparison\n(Matched vs Unmatched)', fontsize=14, fontweight='bold')
    
    # Create density plots
    matched_images_utm.plot(ax=ax2, color=matched_color, markersize=0.8, alpha=0.7, label='Matched')
    unmatched_images_utm.plot(ax=ax2, color=unmatched_color, markersize=0.8, alpha=0.7, label='Unmatched')
    
    # Add basemap
    try:
        ctx.add_basemap(ax2, crs=all_images_utm.crs, source=ctx.providers.CartoDB.Positron, alpha=0.7)
    except:
        print("Warning: Could not add basemap, using plain background")
    
    ax2.set_xlabel('Easting (UTM Zone 33N)', fontsize=12)
    ax2.set_ylabel('Northing (UTM Zone 33N)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend(handles=[matched_patch, unmatched_patch], loc='upper right', fontsize=10)
    
    # Add statistics text
    total_images = len(all_images_utm)
    match_rate = len(matched_images_utm) / total_images * 100
    
    stats_text = f"""Statistics:
Total Images: {total_images:,}
Matched: {len(matched_images_utm):,} ({match_rate:.1f}%)
Unmatched: {len(unmatched_images_utm):,} ({100-match_rate:.1f}%)

Distance Threshold: 10m
Coordinate System: UTM Zone 33N"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'spatial_matching_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Map saved to: {output_path}")
    
    return fig

def create_berlin_brandenburg_comparison(all_images_gdf, matched_images_gdf, unmatched_images_gdf):
    """
    Create a focused comparison map showing Berlin vs Brandenburg differences.
    
    Args:
        all_images_gdf: All images GeoDataFrame
        matched_images_gdf: Matched images GeoDataFrame  
        unmatched_images_gdf: Unmatched images GeoDataFrame
    """
    print("Creating Berlin vs Brandenburg comparison map...")
    
    # Define Berlin proper boundaries
    berlin_lon_min, berlin_lon_max = 13.088, 13.761
    berlin_lat_min, berlin_lat_max = 52.338, 52.675
    
    # Separate Berlin and Brandenburg images
    berlin_mask = (
        (all_images_gdf['lon'] >= berlin_lon_min) & 
        (all_images_gdf['lon'] <= berlin_lon_max) &
        (all_images_gdf['lat'] >= berlin_lat_min) & 
        (all_images_gdf['lat'] <= berlin_lat_max)
    )
    
    berlin_images = all_images_gdf[berlin_mask]
    brandenburg_images = all_images_gdf[~berlin_mask]
    
    # Find matched/unmatched for each region
    matched_ids = set(matched_images_gdf['id'])
    
    berlin_matched = berlin_images[berlin_images['id'].isin(matched_ids)]
    berlin_unmatched = berlin_images[~berlin_images['id'].isin(matched_ids)]
    
    brandenburg_matched = brandenburg_images[brandenburg_images['id'].isin(matched_ids)]
    brandenburg_unmatched = brandenburg_images[~brandenburg_images['id'].isin(matched_ids)]
    
    # Convert to UTM
    berlin_matched_utm = berlin_matched.to_crs(epsg=25833)
    berlin_unmatched_utm = berlin_unmatched.to_crs(epsg=25833)
    brandenburg_matched_utm = brandenburg_matched.to_crs(epsg=25833)
    brandenburg_unmatched_utm = brandenburg_unmatched.to_crs(epsg=25833)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot images
    berlin_unmatched_utm.plot(ax=ax, color='#FF6B6B', markersize=1, alpha=0.6, label='Berlin Unmatched')
    berlin_matched_utm.plot(ax=ax, color='#4ECDC4', markersize=1, alpha=0.6, label='Berlin Matched')
    brandenburg_unmatched_utm.plot(ax=ax, color='#FF8E53', markersize=1, alpha=0.6, label='Brandenburg Unmatched')
    brandenburg_matched_utm.plot(ax=ax, color='#45B7D1', markersize=1, alpha=0.6, label='Brandenburg Matched')
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=berlin_matched_utm.crs, source=ctx.providers.CartoDB.Positron, alpha=0.7)
    except:
        print("Warning: Could not add basemap, using plain background")
    
    ax.set_title('Spatial Matching: Berlin vs Brandenburg\n(Mapillary Images)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Easting (UTM Zone 33N)', fontsize=12)
    ax.set_ylabel('Northing (UTM Zone 33N)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    berlin_matched_patch = mpatches.Patch(color='#4ECDC4', label=f'Berlin Matched ({len(berlin_matched_utm):,})')
    berlin_unmatched_patch = mpatches.Patch(color='#FF6B6B', label=f'Berlin Unmatched ({len(berlin_unmatched_utm):,})')
    brandenburg_matched_patch = mpatches.Patch(color='#45B7D1', label=f'Brandenburg Matched ({len(brandenburg_matched_utm):,})')
    brandenburg_unmatched_patch = mpatches.Patch(color='#FF8E53', label=f'Brandenburg Unmatched ({len(brandenburg_unmatched_utm):,})')
    
    ax.legend(handles=[berlin_matched_patch, berlin_unmatched_patch, 
                      brandenburg_matched_patch, brandenburg_unmatched_patch], 
             loc='upper right', fontsize=10)
    
    # Add statistics
    berlin_match_rate = len(berlin_matched_utm) / len(berlin_images) * 100
    brandenburg_match_rate = len(brandenburg_matched_utm) / len(brandenburg_images) * 100 if len(brandenburg_images) > 0 else 0
    
    stats_text = f"""Regional Match Rates:
Berlin: {berlin_match_rate:.1f}% ({len(berlin_matched_utm):,}/{len(berlin_images):,})
Brandenburg: {brandenburg_match_rate:.1f}% ({len(brandenburg_matched_utm):,}/{len(brandenburg_images):,})

Distance Threshold: 10m"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save the plot
    output_dir = os.path.join(project_root, 'reports', 'figures')
    output_path = os.path.join(output_dir, 'berlin_brandenburg_matching.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Regional comparison map saved to: {output_path}")
    
    return fig

def main():
    """
    Main execution function.
    """
    print("Creating Spatial Matching Distribution Maps")
    print("=" * 50)
    
    try:
        # Load data
        all_images_gdf, matched_images_gdf, unmatched_images_gdf = load_data()
        
        # Create main spatial matching map
        fig1 = create_spatial_matching_map(all_images_gdf, matched_images_gdf, unmatched_images_gdf)
        
        # Create Berlin vs Brandenburg comparison
        fig2 = create_berlin_brandenburg_comparison(all_images_gdf, matched_images_gdf, unmatched_images_gdf)
        
        print(f"\n🎉 Maps created successfully!")
        print(f"📊 Visualized {len(all_images_gdf):,} images with {len(matched_images_gdf)/len(all_images_gdf)*100:.1f}% match rate")
        print(f"🗺️  Maps show geographic patterns in road matching success")
        
    except Exception as e:
        print(f"❌ Error creating maps: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
