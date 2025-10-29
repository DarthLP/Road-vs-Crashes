"""
Script: Match Crashes to Mapillary Images

Purpose:
This script matches crash data to Mapillary images based on spatial proximity and road
characteristics. For each image, it identifies crashes within specified distance thresholds
(5m, 10m, 25m) that occur on the same road type (matching highway and surface attributes).

Functionality:
- Loads processed Mapillary and crash datasets
- Performs spatial matching using multiple distance thresholds
- Filters matches by road attributes (highway, surface)
- Counts crashes before/after image capture dates
- Tracks which crashes are shared between multiple images
- Generates labeled datasets for all thresholds

How to run:
    python src/features/match_crashes_to_images.py
"""

import os
import sys
import csv
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def process_threshold_efficient(images_df, crashes_df, threshold, output_path):
    """
    Efficiently process matching for a specific distance threshold using spatial indexing.
    
    This function uses GeoPandas spatial joins with buffered geometries for efficient
    matching. It filters matches by road attributes (highway and surface) and tracks
    shared crashes between images.
    
    Parameters:
        images_df: DataFrame with image data (must include lon, lat, highway, surface, captured_at, point_id)
        crashes_df: DataFrame with crash data (must include XGCSWGS84, YGCSWGS84, highway, surface, UJAHR, point_id)
        threshold: Distance threshold in meters
        output_path: Path to save output CSV
        
    Returns:
        DataFrame with original image data plus crash matching columns
    """
    print(f"\nProcessing {threshold}m threshold...")
    print(f"  Images: {len(images_df)}")
    print(f"  Crashes: {len(crashes_df)}")
    
    # Create GeoDataFrames
    print("  Creating spatial geometries...")
    images_gdf = gpd.GeoDataFrame(
        images_df,
        geometry=[Point(lon, lat) for lon, lat in zip(images_df['lon'], images_df['lat'])],
        crs='EPSG:4326'
    )
    
    crashes_gdf = gpd.GeoDataFrame(
        crashes_df,
        geometry=[Point(lon, lat) for lon, lat in zip(crashes_df['XGCSWGS84'], crashes_df['YGCSWGS84'])],
        crs='EPSG:4326'
    )
    
    # Convert to UTM (metric CRS) for accurate distance calculations
    print("  Converting to UTM for distance calculations...")
    images_utm = images_gdf.to_crs('EPSG:25833')  # ETRS89 / UTM zone 33N
    crashes_utm = crashes_gdf.to_crs('EPSG:25833')
    
    # Create buffer around each image point
    print("  Creating buffers around images...")
    images_buffered = images_utm.copy()
    images_buffered['geometry'] = images_buffered.geometry.buffer(threshold)
    
    # Create selection GeoDataFrame for spatial join
    images_for_join = gpd.GeoDataFrame(
        images_buffered[['point_id', 'highway', 'surface', 'captured_at']],
        geometry=images_buffered.geometry,
        crs=images_utm.crs
    )
    
    # Perform spatial join to find crashes within threshold
    print("  Performing spatial join...")
    spatial_matches = gpd.sjoin(
        crashes_utm,
        images_for_join,
        how='inner',
        predicate='within'
    )
    
    if len(spatial_matches) == 0:
        print("  No spatial matches found")
        # Create empty results
        results_dict = {
            f'has_crash_{threshold}m': [0] * len(images_df),
            f'crash_count_{threshold}m': [0] * len(images_df),
            f'crash_years_{threshold}m': [''] * len(images_df),
            f'crash_count_before_{threshold}m': [0] * len(images_df),
            f'crash_count_after_{threshold}m': [0] * len(images_df),
            f'all_crash_ids_{threshold}m': [''] * len(images_df),
            f'has_shared_crashes_{threshold}m': [0] * len(images_df),
            f'shared_with_image_ids_{threshold}m': [''] * len(images_df),
        }
        output_df = images_df.copy()
        for col, values in results_dict.items():
            output_df[col] = values
        output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        return output_df
    
    print(f"  Found {len(spatial_matches)} potential matches (before road filtering)")
    
    # Filter by matching road attributes (highway AND surface must match)
    # After spatial join, right dataframe columns have '_right' suffix
    print("  Filtering by road attributes...")
    
    # GeoPandas suffixes both left and right columns when names conflict
    # Check which naming convention is used
    if 'highway_left' in spatial_matches.columns and 'highway_right' in spatial_matches.columns:
        highway_left_col = 'highway_left'
        highway_right_col = 'highway_right'
        surface_left_col = 'surface_left'
        surface_right_col = 'surface_right'
    elif 'highway' in spatial_matches.columns and 'highway_right' in spatial_matches.columns:
        highway_left_col = 'highway'
        highway_right_col = 'highway_right'
        surface_left_col = 'surface'
        surface_right_col = 'surface_right'
    else:
        print(f"  Available columns: {list(spatial_matches.columns)[:30]}...")
        raise ValueError("Could not find highway columns in spatial matches")
    
    # Filter: both highway and surface must match (and neither should be NaN/null)
    spatial_matches['road_match'] = (
        spatial_matches[highway_left_col].notna() &
        spatial_matches[highway_right_col].notna() &
        spatial_matches[surface_left_col].notna() &
        spatial_matches[surface_right_col].notna() &
        (spatial_matches[highway_left_col] == spatial_matches[highway_right_col]) &
        (spatial_matches[surface_left_col] == spatial_matches[surface_right_col])
    )
    matched = spatial_matches[spatial_matches['road_match']].copy()
    
    print(f"  After road filtering: {len(matched)} matches")
    
    # Initialize result columns
    suffix = f"_{threshold}m"
    results = {
        f'has_crash{suffix}': [0] * len(images_df),
        f'crash_count{suffix}': [0] * len(images_df),
        f'crash_years{suffix}': [''] * len(images_df),
        f'crash_count_before{suffix}': [0] * len(images_df),
        f'crash_count_after{suffix}': [0] * len(images_df),
        f'all_crash_ids{suffix}': [''] * len(images_df),
        f'has_shared_crashes{suffix}': [0] * len(images_df),
        f'shared_with_image_ids{suffix}': [''] * len(images_df),
    }
    
    # Group matches by image
    print("  Aggregating matches per image...")
    image_to_crashes = {}
    
    # Determine point_id column names after spatial join
    if 'point_id_left' in matched.columns and 'point_id_right' in matched.columns:
        crash_point_id_col = 'point_id_left'
        image_point_id_col = 'point_id_right'
    elif 'point_id' in matched.columns and 'point_id_right' in matched.columns:
        crash_point_id_col = 'point_id'
        image_point_id_col = 'point_id_right'
    else:
        print(f"  Available point_id columns: {[c for c in matched.columns if 'point_id' in c]}")
        raise ValueError("Could not find point_id columns in matched data")
    
    for img_idx, img_row in images_df.iterrows():
        img_id = img_row['point_id']
        img_matches = matched[matched[image_point_id_col] == img_id].copy()
        
        if len(img_matches) == 0:
            continue
        
        # Get crash info
        crash_ids = sorted(img_matches[crash_point_id_col].astype(int).unique().tolist())
        crash_years = sorted(img_matches['UJAHR'].unique().astype(int).tolist())
        
        img_date = pd.to_datetime(img_row['captured_at'])
        img_year = img_date.year
        
        crash_years_int = img_matches['UJAHR'].values
        
        # Store for results
        results[f'has_crash{suffix}'][img_idx] = 1
        results[f'crash_count{suffix}'][img_idx] = len(crash_ids)
        results[f'crash_years{suffix}'][img_idx] = ','.join(map(str, crash_years))
        results[f'crash_count_before{suffix}'][img_idx] = int(np.sum(crash_years_int < img_year))
        results[f'crash_count_after{suffix}'][img_idx] = int(np.sum(crash_years_int > img_year))
        results[f'all_crash_ids{suffix}'][img_idx] = ','.join(map(str, crash_ids))
        
        # Store for shared crash analysis
        image_to_crashes[img_id] = crash_ids
    
    # Identify shared crashes
    print("  Identifying shared crashes...")
    crash_to_images = {}
    for img_id, crash_ids in image_to_crashes.items():
        for crash_id in crash_ids:
            if crash_id not in crash_to_images:
                crash_to_images[crash_id] = []
            crash_to_images[crash_id].append(img_id)
    
    shared_crashes = {crash_id: images for crash_id, images in crash_to_images.items() if len(images) > 1}
    
    # Update shared crash info
    for img_idx, img_row in images_df.iterrows():
        img_id = img_row['point_id']
        if img_id not in image_to_crashes:
            continue
        
        shared_info = []
        for crash_id in image_to_crashes[img_id]:
            if crash_id in shared_crashes:
                other_images = [i for i in shared_crashes[crash_id] if i != img_id]
                if other_images:
                    shared_info.append(f"c{crash_id}:{','.join([f'i{img_id}' for img_id in other_images])}")
        
        if shared_info:
            results[f'has_shared_crashes{suffix}'][img_idx] = 1
            results[f'shared_with_image_ids{suffix}'][img_idx] = ';'.join(shared_info)
    
    # Combine with original data
    output_df = images_df.copy()
    for col, values in results.items():
        output_df[col] = values
    
    # Save output with proper quoting for all fields (important for URLs and special characters)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"  Saved to {output_path}")
    
    # Print summary statistics
    print(f"  Summary for {threshold}m:")
    has_crash_count = sum(results[f'has_crash{suffix}'])
    print(f"    Images with crashes: {has_crash_count} ({100*has_crash_count/len(output_df):.1f}%)")
    print(f"    Total matched crashes: {sum(results[f'crash_count{suffix}'])}")
    has_shared_count = sum(results[f'has_shared_crashes{suffix}'])
    print(f"    Images with shared crashes: {has_shared_count}")
    print(f"    Crashes before image: {sum(results[f'crash_count_before{suffix}'])}")
    print(f"    Crashes after image: {sum(results[f'crash_count_after{suffix}'])}")
    
    return output_df




def main():
    """Main execution function."""
    print("=" * 80)
    print("MATCH CRASHES TO MAPILLARY IMAGES")
    print("=" * 80)
    
    # File paths
    images_path = project_root / 'data' / 'processed' / 'mapillary_with_osm.csv'
    crashes_path = project_root / 'data' / 'processed' / 'crashes_with_osm.csv'
    
    # Load data
    print("\nLoading data...")
    images_df = pd.read_csv(images_path)
    crashes_df = pd.read_csv(crashes_path)
    print(f"Loaded {len(images_df)} images and {len(crashes_df)} crashes")
    
    # Process each threshold
    thresholds = [5, 10, 25]
    
    for threshold in thresholds:
        output_path = project_root / 'data' / 'processed' / f'mapillary_labeled_{threshold}m.csv'
        process_threshold_efficient(images_df, crashes_df, threshold, output_path)
    
    print("\n" + "=" * 80)
    print("MATCHING COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    for threshold in thresholds:
        print(f"  - data/processed/mapillary_labeled_{threshold}m.csv")


if __name__ == "__main__":
    main()

