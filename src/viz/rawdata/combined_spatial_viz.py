#!/usr/bin/env python3
"""
Combined Spatial Visualization Script

This script creates a combined spatial visualization showing both Mapillary images
and traffic accident data on the same map with different colored dots.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from shapely.geometry import Point
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent.parent.parent
data_dir = project_root / 'data' / 'processed'
output_dir = project_root / 'reports' / 'figures' / 'rawdata'

def load_mapillary_data():
    """Load Mapillary data from CSV file."""
    print("Loading Mapillary data...")
    mapillary_path = project_root / 'data' / 'raw' / 'mapillary_berlin_full.csv'
    df = pd.read_csv(mapillary_path)
    
    # Create geometry column for spatial operations
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    print(f"Loaded {len(gdf)} Mapillary images")
    return gdf

def load_crash_data():
    """Load crash data from CSV file."""
    print("Loading crash data...")
    crash_path = data_dir / 'crashes_aggregated.csv'
    df = pd.read_csv(crash_path)
    
    # Create geometry column for spatial operations (WGS84 coordinates)
    geometry = [Point(xy) for xy in zip(df['XGCSWGS84'], df['YGCSWGS84'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    print(f"Loaded {len(gdf)} crashes")
    return gdf

def spatial_aggregation(gdf, grid_size=0.001):
    """
    Aggregate nearby points into clusters for better visualization.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing spatial data
    grid_size : float
        Grid cell size in degrees for spatial binning
        
    Returns:
    --------
    gpd.GeoDataFrame
        Aggregated GeoDataFrame with cluster counts
    """
    print("Performing spatial aggregation...")
    
    # Create grid cells
    gdf['grid_lon'] = (gdf.geometry.x / grid_size).astype(int) * grid_size
    gdf['grid_lat'] = (gdf.geometry.y / grid_size).astype(int) * grid_size
    
    # Aggregate by grid cells
    if 'id' in gdf.columns:
        # Mapillary data
        aggregated = gdf.groupby(['grid_lon', 'grid_lat']).agg({
            'id': 'count',
            'lon': 'mean',
            'lat': 'mean'
        }).reset_index()
        aggregated.rename(columns={'id': 'image_count'}, inplace=True)
    else:
        # Crash data
        aggregated = gdf.groupby(['grid_lon', 'grid_lat']).agg({
            'UJAHR': 'count',
            'XGCSWGS84': 'mean',
            'YGCSWGS84': 'mean'
        }).reset_index()
        aggregated.rename(columns={'UJAHR': 'crash_count'}, inplace=True)
    
    # Create geometry for aggregated points
    if 'image_count' in aggregated.columns:
        geometry = [Point(xy) for xy in zip(aggregated['lon'], aggregated['lat'])]
    else:
        geometry = [Point(xy) for xy in zip(aggregated['XGCSWGS84'], aggregated['YGCSWGS84'])]
    
    aggregated_gdf = gpd.GeoDataFrame(aggregated, geometry=geometry, crs='EPSG:4326')
    
    if 'image_count' in aggregated_gdf.columns:
        print(f"Aggregated {len(gdf)} images into {len(aggregated_gdf)} clusters")
        print(f"Max images per cluster: {aggregated_gdf['image_count'].max()}")
    else:
        print(f"Aggregated {len(gdf)} crashes into {len(aggregated_gdf)} clusters")
        print(f"Max crashes per cluster: {aggregated_gdf['crash_count'].max()}")
    
    return aggregated_gdf

def create_combined_spatial_visualization(mapillary_gdf, crash_gdf, output_path):
    """
    Create a combined spatial scatter plot showing both Mapillary images and crash data.
    
    Parameters:
    -----------
    mapillary_gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data with geometry column
    crash_gdf : gpd.GeoDataFrame
        GeoDataFrame containing crash data with geometry column
    output_path : str
        Path where to save the visualization
    """
    print("Creating combined spatial visualization...")
    
    # Perform spatial aggregation for both datasets
    aggregated_mapillary = spatial_aggregation(mapillary_gdf)
    aggregated_crashes = spatial_aggregation(crash_gdf)
    
    # Convert to Web Mercator for contextily
    mapillary_web = aggregated_mapillary.to_crs('EPSG:3857')
    crashes_web = aggregated_crashes.to_crs('EPSG:3857')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot Mapillary points (blue dots)
    min_size_mapillary = 10
    max_size_mapillary = 150
    sizes_mapillary = np.interp(aggregated_mapillary['image_count'], 
                               (aggregated_mapillary['image_count'].min(), aggregated_mapillary['image_count'].max()),
                               (min_size_mapillary, max_size_mapillary))
    
    scatter_mapillary = ax.scatter(mapillary_web.geometry.x, mapillary_web.geometry.y, 
                                  s=sizes_mapillary, alpha=0.5, c='blue', edgecolors='darkblue', linewidth=0.3)
    
    # Plot crash points (red dots)
    min_size_crashes = 10
    max_size_crashes = 150
    sizes_crashes = np.interp(aggregated_crashes['crash_count'], 
                             (aggregated_crashes['crash_count'].min(), aggregated_crashes['crash_count'].max()),
                             (min_size_crashes, max_size_crashes))
    
    scatter_crashes = ax.scatter(crashes_web.geometry.x, crashes_web.geometry.y, 
                                s=sizes_crashes, alpha=0.6, c='red', edgecolors='darkred', linewidth=0.3)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=crashes_web.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        print("Plotting without basemap...")
    
    # Customize the plot
    ax.set_title('Mapillary Images and Traffic Accidents in Berlin\n(Dot size proportional to density)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create combined legend
    legend_elements = [
        plt.scatter([], [], s=50, c='blue', alpha=0.5, edgecolors='darkblue', label='Mapillary Images'),
        plt.scatter([], [], s=50, c='red', alpha=0.6, edgecolors='darkred', label='Traffic Accidents')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title='Data Type', 
             title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined spatial visualization saved to {output_path}")
    plt.close()

def main():
    """Main function to create combined spatial visualization."""
    print("=== Creating Combined Spatial Visualization ===")
    
    # Load data
    mapillary_gdf = load_mapillary_data()
    crash_gdf = load_crash_data()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create combined visualization
    output_path = output_dir / 'combined_spatial_distribution.png'
    create_combined_spatial_visualization(mapillary_gdf, crash_gdf, output_path)
    
    print("=== Combined Visualization Complete! ===")

if __name__ == "__main__":
    main()
