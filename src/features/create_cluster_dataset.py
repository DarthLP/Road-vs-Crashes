"""
Script: Create 15m Cluster Dataset with Crash Matching

Purpose:
This script creates 15m × 15m spatial clusters (tiles) from Mapillary images,
aggregates image metadata and OSM attributes per cluster-year, matches crashes
to clusters spatially, generates cluster-level analysis reports, and creates
stratified train/val/test splits.

Functionality:
- Creates 15m UTM grid tiles covering Berlin
- Assigns cluster_id to each image
- Aggregates images per cluster-year with OSM attributes
- Matches crashes to clusters using point-in-polygon (no distance threshold)
- Generates comprehensive cluster-level reports and visualizations
- Splits cluster_ids into train/val/test sets stratified by match_label

How to run:
    python src/features/create_cluster_dataset.py
"""

import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Berlin bounding box in WGS84
BERLIN_BBOX = (13.0884, 52.3383, 13.7612, 52.6755)  # (west, south, east, north)


def create_15m_grid(bbox, tile_size=15):
    """
    Create 15m × 15m grid tiles in UTM coordinates covering the bounding box.
    
    Parameters:
        bbox: Bounding box tuple (west, south, east, north) in WGS84
        tile_size: Tile size in meters (default: 15)
        
    Returns:
        GeoDataFrame with cluster_id and geometry columns
    """
    print(f"Creating {tile_size}m × {tile_size}m grid...")
    
    # Convert bbox to UTM
    west, south, east, north = bbox
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(west, south, east, north)],
        crs='EPSG:4326'
    )
    bbox_utm = bbox_gdf.to_crs('EPSG:25833')
    bounds = bbox_utm.total_bounds  # minx, miny, maxx, maxy
    
    minx, miny, maxx, maxy = bounds
    
    # Calculate number of tiles
    cols = int(np.ceil((maxx - minx) / tile_size))
    rows = int(np.ceil((maxy - miny) / tile_size))
    
    print(f"  Grid size: {cols} columns × {rows} rows = {cols * rows} tiles")
    
    # Create tiles
    tiles = []
    cluster_ids = []
    
    cluster_id = 0
    for row in range(rows):
        for col in range(cols):
            xmin = minx + (col * tile_size)
            xmax = minx + ((col + 1) * tile_size)
            ymin = miny + (row * tile_size)
            ymax = miny + ((row + 1) * tile_size)
            
            # Create tile polygon
            tile_box = box(xmin, ymin, xmax, ymax)
            tiles.append(tile_box)
            cluster_ids.append(cluster_id)
            cluster_id += 1
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(
        {'cluster_id': cluster_ids},
        geometry=tiles,
        crs='EPSG:25833'
    )
    
    print(f"  Created {len(grid_gdf)} tiles")
    return grid_gdf


def assign_cluster_ids(images_gdf, grid_gdf):
    """
    Assign cluster_id to each image using spatial join.
    
    Parameters:
        images_gdf: GeoDataFrame with image points
        grid_gdf: GeoDataFrame with grid tiles
        
    Returns:
        GeoDataFrame with cluster_id assigned
    """
    print("Assigning cluster_ids to images...")
    
    # Convert images to UTM
    images_utm = images_gdf.to_crs('EPSG:25833')
    
    # Spatial join
    joined = gpd.sjoin(images_utm, grid_gdf, how='left', predicate='within')
    
    # Handle images not in any tile (shouldn't happen, but just in case)
    images_with_clusters = joined[joined['cluster_id'].notna()].copy()
    
    print(f"  Assigned cluster_ids to {len(images_with_clusters)} images")
    print(f"  Unique clusters: {images_with_clusters['cluster_id'].nunique()}")
    
    return images_with_clusters


def get_tile_centers(grid_gdf):
    """
    Calculate center coordinates (lon, lat) for each cluster.
    
    Parameters:
        grid_gdf: GeoDataFrame with grid tiles
        
    Returns:
        DataFrame with cluster_id, lon, lat
    """
    # Get centroids in UTM
    centroids_utm = grid_gdf.copy()
    centroids_utm['geometry'] = centroids_utm.geometry.centroid
    
    # Convert to WGS84
    centroids_wgs84 = centroids_utm.to_crs('EPSG:4326')
    
    # Extract coordinates
    centers = pd.DataFrame({
        'cluster_id': centroids_wgs84['cluster_id'],
        'lon': centroids_wgs84.geometry.x,
        'lat': centroids_wgs84.geometry.y
    })
    
    return centers


def aggregate_cluster(images_df, grid_gdf):
    """
    Aggregate images per cluster with OSM attributes and metadata.
    Clusters are year-independent - all images in a cluster are aggregated together.
    
    Parameters:
        images_df: DataFrame with images and cluster_id assigned
        grid_gdf: GeoDataFrame with grid tiles
        
    Returns:
        DataFrame with cluster aggregated data (one row per cluster_id)
    """
    print("Aggregating images per cluster...")
    
    # Get tile centers
    tile_centers = get_tile_centers(grid_gdf)
    
    # Define aggregation functions
    categorical_cols = ['highway', 'surface', 'lit', 'cycleway', 'sidewalk', 'oneway', 
                        'bridge', 'tunnel', 'junction', 'access', 'crossing', 
                        'parking:lane', 'traffic_calming', 'smoothness']
    numeric_cols = ['road_segment_length_m', 'maxspeed', 'lanes', 'width', 'intersection_degree']
    boolean_cols = ['is_intersection', 'near_traffic_signal']
    
    # Helper function for mode
    def mode_func(x):
        if len(x) == 0:
            return np.nan
        # Remove NaN
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return np.nan
        return x_clean.mode().iloc[0] if len(x_clean.mode()) > 0 else np.nan
    
    # Helper function for max (if any True, result is True)
    def max_bool(x):
        if len(x) == 0:
            return False
        return x.max() if x.dtype == bool else (x.astype(bool).max() if x.notna().any() else False)
    
    # Prepare aggregation dict
    agg_dict = {}
    
    # Categorical columns - use mode
    for col in categorical_cols:
        if col in images_df.columns:
            agg_dict[col] = mode_func
    
    # Numeric columns - use mean
    for col in numeric_cols:
        if col in images_df.columns:
            agg_dict[col] = 'mean'
    
    # Boolean columns - use max
    for col in boolean_cols:
        if col in images_df.columns:
            agg_dict[col] = max_bool
    
    # OSM way ID - use mode
    if 'osm_way_id' in images_df.columns:
        agg_dict['osm_way_id'] = mode_func
    
    # Collect image URLs and point IDs
    def collect_urls(x):
        urls = x.dropna().tolist()
        return ','.join([str(url) for url in urls]) if len(urls) > 0 else ''
    
    def collect_point_ids(x):
        point_ids = x.dropna().astype(int).tolist()
        return ','.join([str(pid) for pid in point_ids]) if len(point_ids) > 0 else ''
    
    def collect_years(x):
        years = pd.to_datetime(x.dropna(), format='mixed', errors='coerce').dt.year
        years_clean = years.dropna().unique().astype(int).tolist()
        return ','.join([str(y) for y in sorted(years_clean)]) if len(years_clean) > 0 else ''
    
    agg_dict['thumb_1024_url'] = collect_urls
    agg_dict['point_id'] = collect_point_ids
    agg_dict['captured_at'] = collect_years
    
    # Group by cluster_id only (no year grouping)
    grouped = images_df.groupby('cluster_id')
    
    # Aggregate
    aggregated = grouped.agg(agg_dict).reset_index()
    
    # Add amount_of_images using size()
    image_counts = grouped.size().reset_index(name='amount_of_images')
    aggregated = aggregated.merge(image_counts, on='cluster_id', how='left')
    
    # Rename columns
    aggregated = aggregated.rename(columns={
        'thumb_1024_url': 'list_of_thumb_1024_url',
        'point_id': 'list_of_point_ids',
        'captured_at': 'captured_at_years'
    })
    
    # Merge with tile centers
    aggregated = aggregated.merge(tile_centers, on='cluster_id', how='left')
    
    # Reorder columns (no 'year' column)
    col_order = ['cluster_id'] + categorical_cols + numeric_cols + boolean_cols
    if 'osm_way_id' in aggregated.columns:
        col_order.insert(1, 'osm_way_id')
    col_order.extend(['lon', 'lat', 'captured_at_years', 'list_of_thumb_1024_url', 
                      'list_of_point_ids', 'amount_of_images'])
    
    # Only include columns that exist
    col_order = [col for col in col_order if col in aggregated.columns]
    aggregated = aggregated[col_order]
    
    print(f"  Created {len(aggregated)} clusters")
    print(f"  Unique clusters: {aggregated['cluster_id'].nunique()}")
    
    return aggregated


def match_crashes_to_clusters(clusters_df, crashes_path, grid_gdf):
    """
    Match crashes to clusters using point-in-polygon (no distance threshold, no year filtering).
    
    Parameters:
        clusters_df: DataFrame with cluster data (one row per cluster_id)
        crashes_path: Path to crashes CSV file
        grid_gdf: GeoDataFrame with grid tiles
        
    Returns:
        DataFrame with match_label, amount_of_matches, and crash_years added.
        All crashes within a cluster are matched regardless of crash year.
        crash_years contains comma-separated list of years when crashes occurred.
    """
    print("Matching crashes to clusters...")
    
    # Load crashes
    crashes_df = pd.read_csv(crashes_path)
    print(f"  Loaded {len(crashes_df)} crashes")
    
    # Create GeoDataFrame for crashes
    crashes_gdf = gpd.GeoDataFrame(
        crashes_df,
        geometry=[Point(lon, lat) for lon, lat in zip(crashes_df['XGCSWGS84'], crashes_df['YGCSWGS84'])],
        crs='EPSG:4326'
    )
    
    # Convert to UTM
    crashes_utm = crashes_gdf.to_crs('EPSG:25833')
    
    # Spatial join: crashes within cluster polygons (NO YEAR FILTERING)
    crashes_in_clusters = gpd.sjoin(crashes_utm, grid_gdf, how='inner', predicate='within')
    
    print(f"  Found {len(crashes_in_clusters)} crashes within cluster bounds")
    
    # Count crashes per cluster (all crashes, regardless of year)
    if len(crashes_in_clusters) > 0:
        # Count total crashes per cluster
        crash_counts_per_cluster = crashes_in_clusters.groupby('cluster_id').agg({
            'UJAHR': lambda x: ','.join(sorted([str(int(y)) for y in x.dropna().unique()])),  # List of crash years
            'point_id': 'count'  # Total crash count
        }).reset_index()
        crash_counts_per_cluster = crash_counts_per_cluster.rename(columns={
            'UJAHR': 'crash_years',
            'point_id': 'amount_of_matches'
        })
        
        # Merge with clusters_df (match on cluster_id only, not year)
        clusters_df = clusters_df.merge(crash_counts_per_cluster, on='cluster_id', how='left')
        clusters_df['amount_of_matches'] = clusters_df['amount_of_matches'].fillna(0).astype(int)
        clusters_df['crash_years'] = clusters_df['crash_years'].fillna('')
        clusters_df['match_label'] = clusters_df['amount_of_matches'] > 0
    else:
        # No crashes found
        clusters_df['match_label'] = False
        clusters_df['amount_of_matches'] = 0
        clusters_df['crash_years'] = ''
    
    # Reorder columns: move match_label, amount_of_matches, and crash_years after cluster_id
    cols = list(clusters_df.columns)
    cols.remove('match_label')
    cols.remove('amount_of_matches')
    if 'crash_years' in cols:
        cols.remove('crash_years')
    if 'cluster_id' in cols:
        insert_idx = cols.index('cluster_id') + 1
        cols.insert(insert_idx, 'match_label')
        cols.insert(insert_idx + 1, 'amount_of_matches')
        cols.insert(insert_idx + 2, 'crash_years')
    else:
        cols.insert(1, 'match_label')
        cols.insert(2, 'amount_of_matches')
        cols.insert(3, 'crash_years')
    clusters_df = clusters_df[cols]
    
    print(f"  Clusters with crashes: {clusters_df['match_label'].sum()}")
    print(f"  Total matched crashes: {clusters_df['amount_of_matches'].sum()}")
    print(f"  Note: Crashes matched to clusters regardless of crash year")
    
    return clusters_df


def generate_cluster_reports(clusters_df, output_dir):
    """
    Generate cluster-level analysis reports and visualizations.
    
    Parameters:
        clusters_df: DataFrame with cluster-year data and crash matches
        output_dir: Path to output directory
    """
    print("\n" + "=" * 80)
    print("GENERATING CLUSTER-LEVEL REPORTS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary Statistics
    print("\nGenerating summary statistics...")
    total_clusters = len(clusters_df)
    clusters_with_crashes = clusters_df[clusters_df['match_label']].shape[0]
    match_rate = (clusters_with_crashes / total_clusters * 100) if total_clusters > 0 else 0
    
    matched_clusters = clusters_df[clusters_df['match_label']]
    avg_crashes = matched_clusters['amount_of_matches'].mean() if len(matched_clusters) > 0 else 0
    median_crashes = matched_clusters['amount_of_matches'].median() if len(matched_clusters) > 0 else 0
    max_crashes = matched_clusters['amount_of_matches'].max() if len(matched_clusters) > 0 else 0
    
    avg_images = clusters_df['amount_of_images'].mean()
    median_images = clusters_df['amount_of_images'].median()
    max_images = clusters_df['amount_of_images'].max()
    
    summary_stats = pd.DataFrame([{
        'total_clusters': total_clusters,
        'clusters_with_crashes': clusters_with_crashes,
        'match_rate_pct': round(match_rate, 2),
        'total_matched_crashes': clusters_df['amount_of_matches'].sum(),
        'avg_crashes_per_matched_cluster': round(avg_crashes, 2),
        'median_crashes_per_matched_cluster': round(median_crashes, 2),
        'max_crashes_per_cluster': int(max_crashes),
        'avg_images_per_cluster': round(avg_images, 2),
        'median_images_per_cluster': round(median_images, 2),
        'max_images_per_cluster': int(max_images)
    }])
    
    summary_stats.to_csv(output_dir / 'cluster_summary_statistics.csv', index=False)
    print(f"  ✓ Saved cluster_summary_statistics.csv")
    
    # 2. Crash Distributions
    print("\nGenerating crash count distributions...")
    crash_counts = clusters_df['amount_of_matches'].value_counts().sort_index()
    
    distribution_rows = []
    for crash_count, frequency in crash_counts.items():
        pct = (frequency / len(clusters_df) * 100) if len(clusters_df) > 0 else 0
        distribution_rows.append({
            'crash_count': int(crash_count),
            'frequency': int(frequency),
            'percentage': round(pct, 2)
        })
    
    crash_dist_df = pd.DataFrame(distribution_rows)
    crash_dist_df.to_csv(output_dir / 'cluster_crash_distributions.csv', index=False)
    print(f"  ✓ Saved cluster_crash_distributions.csv")
    
    # 3. Temporal Distribution (based on crash_years info)
    print("\nGenerating temporal distribution...")
    # Parse crash_years to extract year information
    all_crash_years = []
    for crash_years_str in clusters_df[clusters_df['crash_years'] != '']['crash_years']:
        years = [int(y) for y in str(crash_years_str).split(',') if y.strip()]
        all_crash_years.extend(years)
    
    if len(all_crash_years) > 0:
        year_counts = pd.Series(all_crash_years).value_counts().sort_index()
        temporal_rows = []
        for year, count in year_counts.items():
            temporal_rows.append({
                'crash_year': year,
                'crash_count': int(count)
            })
        temporal_df = pd.DataFrame(temporal_rows)
        temporal_df.to_csv(output_dir / 'cluster_temporal_distribution.csv', index=False)
        print(f"  ✓ Saved cluster_temporal_distribution.csv")
    else:
        temporal_df = pd.DataFrame(columns=['crash_year', 'crash_count'])
        temporal_df.to_csv(output_dir / 'cluster_temporal_distribution.csv', index=False)
        print(f"  ✓ Saved cluster_temporal_distribution.csv (empty)")
    
    # 4. Image Statistics
    print("\nGenerating image statistics...")
    images_per_cluster_buckets = {
        '1': len(clusters_df[clusters_df['amount_of_images'] == 1]),
        '2-5': len(clusters_df[(clusters_df['amount_of_images'] >= 2) & (clusters_df['amount_of_images'] <= 5)]),
        '6-10': len(clusters_df[(clusters_df['amount_of_images'] >= 6) & (clusters_df['amount_of_images'] <= 10)]),
        '11-20': len(clusters_df[(clusters_df['amount_of_images'] >= 11) & (clusters_df['amount_of_images'] <= 20)]),
        '21+': len(clusters_df[clusters_df['amount_of_images'] >= 21])
    }
    
    image_stats_rows = []
    for bucket, count in images_per_cluster_buckets.items():
        pct = (count / total_clusters * 100) if total_clusters > 0 else 0
        image_stats_rows.append({
            'images_per_cluster': bucket,
            'frequency': count,
            'percentage': round(pct, 2)
        })
    
    # Also add average
    image_stats_rows.append({
        'images_per_cluster': 'avg_per_cluster',
        'frequency': round(avg_images, 2),
        'percentage': np.nan
    })
    
    image_stats_df = pd.DataFrame(image_stats_rows)
    image_stats_df.to_csv(output_dir / 'cluster_image_statistics.csv', index=False)
    print(f"  ✓ Saved cluster_image_statistics.csv")
    
    # Generate Visualizations
    print("\nGenerating visualizations...")
    
    # 1. Crash count distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    top_counts = crash_dist_df.head(20)  # Top 20 crash counts
    ax.bar(top_counts['crash_count'], top_counts['frequency'], alpha=0.8, color='steelblue')
    ax.set_xlabel('Number of Crashes per Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Crash Counts per Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / 'cluster_crash_count_distribution.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved cluster_crash_count_distribution.png")
    
    # 2. Images per cluster distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    buckets = ['1', '2-5', '6-10', '11-20', '21+']
    counts = [images_per_cluster_buckets[b] for b in buckets]
    ax.bar(buckets, counts, alpha=0.8, color='coral')
    ax.set_xlabel('Images per Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Images per Cluster', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / 'images_per_cluster_distribution.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved images_per_cluster_distribution.png")
    
    # 3. Match rates
    fig, ax = plt.subplots(figsize=(8, 6))
    match_counts = clusters_df['match_label'].value_counts()
    ax.bar(['No Crashes', 'With Crashes'], [match_counts.get(False, 0), match_counts.get(True, 0)], 
           alpha=0.8, color=['lightcoral', 'steelblue'])
    ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster Match Distribution\n(Match Rate: {match_rate:.1f}%)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / 'cluster_match_rates.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved cluster_match_rates.png")
    
    # 4. Temporal trends
    if len(temporal_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(temporal_df['crash_year'], temporal_df['crash_count'], 
                marker='o', linewidth=2, label='Crashes', color='steelblue')
        ax.set_xlabel('Crash Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
        ax.set_title('Temporal Distribution of Crashes (from crash_years)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(figures_dir / 'cluster_crashes_per_year_timeseries.png', bbox_inches='tight')
        plt.close(fig)
        print("  ✓ Saved cluster_crashes_per_year_timeseries.png")
    else:
        print("  ⚠ Skipped temporal visualization (no crash year data)")
    
    # 5. Summary dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Match rates
    ax1 = fig.add_subplot(gs[0, 0])
    match_counts = clusters_df['match_label'].value_counts()
    ax1.bar(['No Crashes', 'With Crashes'], [match_counts.get(False, 0), match_counts.get(True, 0)], 
           alpha=0.7, color=['lightcoral', 'steelblue'])
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Match Distribution', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Crash statistics
    ax2 = fig.add_subplot(gs[0, 1])
    if len(matched_clusters) > 0:
        ax2.bar(['Avg', 'Median', 'Max'], 
               [avg_crashes, median_crashes, max_crashes], 
               alpha=0.7, color='steelblue')
    ax2.set_ylabel('Crashes', fontweight='bold')
    ax2.set_title('Crash Count Statistics', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Image statistics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(['Avg', 'Median', 'Max'], 
           [avg_images, median_images, max_images], 
           alpha=0.7, color='coral')
    ax3.set_ylabel('Images', fontweight='bold')
    ax3.set_title('Images per Cluster', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Crash distribution
    ax4 = fig.add_subplot(gs[1, :2])
    top_counts = crash_dist_df.head(15)
    ax4.bar(top_counts['crash_count'], top_counts['frequency'], alpha=0.7, color='steelblue')
    ax4.set_xlabel('Crashes per Cluster', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Crash Count Distribution', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Temporal
    ax5 = fig.add_subplot(gs[1, 2])
    if len(temporal_df) > 0:
        ax5.plot(temporal_df['crash_year'], temporal_df['crash_count'], 
                marker='o', linewidth=2, color='steelblue')
        ax5.set_xlabel('Crash Year', fontweight='bold')
        ax5.set_ylabel('Crashes', fontweight='bold')
        ax5.set_title('Temporal Trends', fontweight='bold')
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No temporal data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Temporal Trends', fontweight='bold')
    
    # Images per cluster distribution
    ax6 = fig.add_subplot(gs[2, :2])
    buckets = ['1', '2-5', '6-10', '11-20', '21+']
    counts = [images_per_cluster_buckets[b] for b in buckets]
    ax6.bar(buckets, counts, alpha=0.7, color='coral')
    ax6.set_xlabel('Images per Cluster', fontweight='bold')
    ax6.set_ylabel('Number of Clusters', fontweight='bold')
    ax6.set_title('Images per Cluster Distribution', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    summary_text = f"""SUMMARY

Total Clusters: {total_clusters}
Clusters with Crashes: {clusters_with_crashes}
Match Rate: {match_rate:.1f}%

Total Matched Crashes: {clusters_df['amount_of_matches'].sum()}
Avg Crashes/Match: {avg_crashes:.2f}
Max Crashes/Match: {max_crashes}

Avg Images/Cluster: {avg_images:.2f}
Max Images/Cluster: {max_images}
"""
    ax7.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax7.transAxes)
    
    fig.suptitle('Cluster-Level Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(figures_dir / 'cluster_summary_dashboard.png', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved cluster_summary_dashboard.png")
    
    print(f"\n✓ All reports saved to: {output_dir}")


def create_stratified_split(clusters_df, random_seed=42):
    """
    Create stratified train/val/test split by match_label.
    
    Parameters:
        clusters_df: DataFrame with cluster-year data
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with split column added
    """
    print("\n" + "=" * 80)
    print("CREATING STRATIFIED TRAIN/VAL/TEST SPLIT")
    print("=" * 80)
    
    # Get unique cluster_ids
    unique_clusters = clusters_df['cluster_id'].unique()
    
    # Determine match_label for each cluster (if any cluster-year has match, cluster is positive)
    cluster_labels = {}
    for cluster_id in unique_clusters:
        cluster_data = clusters_df[clusters_df['cluster_id'] == cluster_id]
        # Cluster is positive if any cluster-year has a match
        cluster_labels[cluster_id] = cluster_data['match_label'].any()
    
    # Separate positive and negative clusters
    positive_clusters = [c for c in unique_clusters if cluster_labels[c]]
    negative_clusters = [c for c in unique_clusters if not cluster_labels[c]]
    
    print(f"  Positive clusters (with crashes): {len(positive_clusters)}")
    print(f"  Negative clusters (no crashes): {len(negative_clusters)}")
    
    # Split positive clusters: 70% train, 15% val, 15% test
    np.random.seed(random_seed)
    positive_shuffled = np.random.permutation(positive_clusters)
    n_pos = len(positive_shuffled)
    n_train_pos = int(n_pos * 0.7)
    n_val_pos = int(n_pos * 0.15)
    
    train_pos = positive_shuffled[:n_train_pos]
    val_pos = positive_shuffled[n_train_pos:n_train_pos + n_val_pos]
    test_pos = positive_shuffled[n_train_pos + n_val_pos:]
    
    # Split negative clusters: 70% train, 15% val, 15% test
    negative_shuffled = np.random.permutation(negative_clusters)
    n_neg = len(negative_shuffled)
    n_train_neg = int(n_neg * 0.7)
    n_val_neg = int(n_neg * 0.15)
    
    train_neg = negative_shuffled[:n_train_neg]
    val_neg = negative_shuffled[n_train_neg:n_train_neg + n_val_neg]
    test_neg = negative_shuffled[n_train_neg + n_val_neg:]
    
    # Combine
    train_clusters = set(np.concatenate([train_pos, train_neg]))
    val_clusters = set(np.concatenate([val_pos, val_neg]))
    test_clusters = set(np.concatenate([test_pos, test_neg]))
    
    print(f"\n  Train clusters: {len(train_clusters)} ({len(train_pos)} pos, {len(train_neg)} neg)")
    print(f"  Val clusters: {len(val_clusters)} ({len(val_pos)} pos, {len(val_neg)} neg)")
    print(f"  Test clusters: {len(test_clusters)} ({len(test_pos)} pos, {len(test_neg)} neg)")
    
    # Assign split to dataframe
    clusters_df['split'] = 'train'
    clusters_df.loc[clusters_df['cluster_id'].isin(val_clusters), 'split'] = 'val'
    clusters_df.loc[clusters_df['cluster_id'].isin(test_clusters), 'split'] = 'test'
    
    # Verify split distribution
    print("\n  Split distribution:")
    print(clusters_df.groupby('split').agg({
        'cluster_id': 'nunique',
        'match_label': lambda x: (x == True).sum()
    }))
    
    return clusters_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("CREATE 15M CLUSTER DATASET WITH CRASH MATCHING")
    print("=" * 80)
    
    # File paths
    images_path = project_root / 'data' / 'processed' / 'mapillary_with_osm.csv'
    crashes_path = project_root / 'data' / 'processed' / 'crashes_with_osm.csv'
    output_path = project_root / 'data' / 'processed' / 'clusters_with_crashes.csv'
    reports_dir = project_root / 'reports' / 'Clusters'
    
    # Load images
    print("\nLoading images...")
    images_df = pd.read_csv(images_path)
    print(f"Loaded {len(images_df)} images")
    
    # Create GeoDataFrame
    images_gdf = gpd.GeoDataFrame(
        images_df,
        geometry=[Point(lon, lat) for lon, lat in zip(images_df['lon'], images_df['lat'])],
        crs='EPSG:4326'
    )
    
    # Step 1: Create grid and assign cluster_ids
    grid_gdf = create_15m_grid(BERLIN_BBOX, tile_size=15)
    images_with_clusters = assign_cluster_ids(images_gdf, grid_gdf)
    
    # Step 2: Aggregate per cluster (no year grouping)
    clusters_df = aggregate_cluster(images_with_clusters, grid_gdf)
    
    # Step 3: Match crashes to clusters
    clusters_df = match_crashes_to_clusters(clusters_df, crashes_path, grid_gdf)
    
    # Step 4: Generate reports
    generate_cluster_reports(clusters_df, reports_dir)
    
    # Step 5: Create stratified split
    clusters_df = create_stratified_split(clusters_df, random_seed=42)
    
    # Save final output
    print("\n" + "=" * 80)
    print("SAVING FINAL OUTPUT")
    print("=" * 80)
    clusters_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved final dataset to: {output_path}")
    print(f"  Total rows: {len(clusters_df)}")
    print(f"  Columns: {len(clusters_df.columns)}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

