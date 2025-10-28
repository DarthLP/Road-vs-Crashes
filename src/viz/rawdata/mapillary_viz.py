#!/usr/bin/env python3
"""
Mapillary Data Visualization Script

This script creates two key visualizations for the Mapillary street image dataset:
1. Time series plot showing the distribution of images by year
2. Spatial scatter plot on Berlin street map with dot sizes proportional to image density

The script loads data from the Mapillary CSV file, processes temporal and spatial information,
and generates publication-ready visualizations saved to the reports/figures/rawdata directory.

Usage:
    python src/viz/rawdata/mapillary_viz.py

Dependencies:
    - pandas, geopandas, matplotlib, seaborn, contextily
    - Data file: data/raw/mapillary_berlin_full.csv
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from shapely.geometry import Point
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_mapillary_data(csv_path):
    """
    Load and preprocess Mapillary CSV data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the Mapillary CSV file
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with geometry column and parsed dates
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Parse captured_at to datetime (handle microseconds)
    df['captured_at'] = pd.to_datetime(df['captured_at'], format='ISO8601')
    df['year'] = df['captured_at'].dt.year
    df['quarter'] = df['captured_at'].dt.to_period('Q')
    
    # Filter out data before 2013
    df = df[df['year'] >= 2013]
    
    # Create geometry column for spatial operations
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    print(f"Loaded {len(gdf)} images")
    print(f"Date range: {df['captured_at'].min()} to {df['captured_at'].max()}")
    print(f"Year range: {df['year'].min()} to {df['year'].max()}")
    
    return gdf

def create_year_visualization(gdf, output_path):
    """
    Create a time series line plot showing image counts by year.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data with year column
    output_path : str
        Path where to save the visualization
    """
    print("Creating year visualization...")
    
    # Count images per year
    yearly_counts = gdf['year'].value_counts().sort_index()
    total_images = len(gdf)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line with markers
    ax.plot(yearly_counts.index, yearly_counts.values, 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    # Customize the plot
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Mapillary Images in Berlin by Year (Total: {total_images:,} images)', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.set_xticks(yearly_counts.index)
    ax.tick_params(axis='x', rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for x, y in zip(yearly_counts.index, yearly_counts.values):
        ax.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Year visualization saved to {output_path}")
    plt.close()

def create_quarterly_visualization(gdf, output_path):
    """
    Create a time series line plot showing image counts by quarter (2013 onwards).
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data with quarter column
    output_path : str
        Path where to save the visualization
    """
    print("Creating quarterly visualization...")
    
    # Count images per quarter
    quarterly_counts = gdf['quarter'].value_counts().sort_index()
    total_images = len(gdf)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot line with markers
    ax.plot(range(len(quarterly_counts)), quarterly_counts.values, 
            marker='o', linewidth=2, markersize=6, color='#E63946')
    
    # Customize the plot
    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Mapillary Images in Berlin by Quarter (2013-2025, Total: {total_images:,} images)', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis with quarter labels
    quarter_labels = [str(q) for q in quarterly_counts.index]
    ax.set_xticks(range(len(quarterly_counts)))
    ax.set_xticklabels(quarter_labels, rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add value labels on every 4th point to avoid crowding
    for i, (x, y) in enumerate(zip(range(len(quarterly_counts)), quarterly_counts.values)):
        if i % 4 == 0:  # Show every 4th label
            ax.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Quarterly visualization saved to {output_path}")
    plt.close()

def create_monthly_visualization(gdf, output_path):
    """
    Create a time series line plot showing image counts by month for years 2018-2024.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data with captured_at column
    output_path : str
        Path where to save the visualization
    """
    print("Creating monthly visualization for 2018-2024...")
    
    # Filter for years 2018-2024
    monthly_data = gdf[gdf['year'].isin([2018, 2019, 2020, 2021, 2022, 2023, 2024])].copy()
    
    if len(monthly_data) == 0:
        print("No data found for years 2018-2024")
        return
    
    # Create year-month column
    monthly_data['year_month'] = monthly_data['captured_at'].dt.to_period('M')
    
    # Count images per month
    monthly_counts = monthly_data['year_month'].value_counts().sort_index()
    total_images = len(monthly_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(22, 8))
    
    # Plot line with markers
    ax.plot(range(len(monthly_counts)), monthly_counts.values, 
            marker='o', linewidth=2, markersize=4, color='#F77F00')
    
    # Customize the plot
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Mapillary Images in Berlin by Month (2018-2024, Total: {total_images:,} images)', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis with month labels
    month_labels = [str(m) for m in monthly_counts.index]
    ax.set_xticks(range(len(monthly_counts)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add value labels on every 8th point to avoid crowding (more data points now)
    for i, (x, y) in enumerate(zip(range(len(monthly_counts)), monthly_counts.values)):
        if i % 8 == 0:  # Show every 8th label
            ax.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Monthly visualization saved to {output_path}")
    plt.close()

def spatial_aggregation(gdf, grid_size=0.001):
    """
    Aggregate nearby points into clusters for better visualization.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data
    grid_size : float
        Grid cell size in degrees for spatial binning
        
    Returns:
    --------
    gpd.GeoDataFrame
        Aggregated GeoDataFrame with cluster counts
    """
    print("Performing spatial aggregation...")
    
    # Create grid cells
    gdf['grid_lon'] = (gdf['lon'] / grid_size).astype(int) * grid_size
    gdf['grid_lat'] = (gdf['lat'] / grid_size).astype(int) * grid_size
    
    # Aggregate by grid cells
    aggregated = gdf.groupby(['grid_lon', 'grid_lat']).agg({
        'id': 'count',
        'lon': 'mean',
        'lat': 'mean'
    }).reset_index()
    
    # Rename count column
    aggregated.rename(columns={'id': 'image_count'}, inplace=True)
    
    # Create geometry for aggregated points
    geometry = [Point(xy) for xy in zip(aggregated['lon'], aggregated['lat'])]
    aggregated_gdf = gpd.GeoDataFrame(aggregated, geometry=geometry, crs='EPSG:4326')
    
    print(f"Aggregated {len(gdf)} points into {len(aggregated_gdf)} clusters")
    print(f"Max images per cluster: {aggregated_gdf['image_count'].max()}")
    
    return aggregated_gdf

def create_spatial_visualization_by_year(gdf, output_path):
    """
    Create a spatial scatter plot on Berlin street map with year-based color coding.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data with year column
    output_path : str
        Path where to save the visualization
    """
    print("Creating spatial visualization by year...")
    
    # Filter to 2018-2024 to match crash data
    gdf_filtered = gdf[gdf['year'].between(2018, 2024)].copy()
    
    # Convert to Web Mercator for contextily
    gdf_web = gdf_filtered.to_crs('EPSG:3857')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors for each year
    years = sorted(gdf_filtered['year'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
    year_colors = dict(zip(years, colors))
    
    # Plot points for each year with different colors
    for year in years:
        year_data = gdf_web[gdf_web['year'] == year]
        ax.scatter(year_data.geometry.x, year_data.geometry.y, 
                  s=15, alpha=0.7, c=[year_colors[year]], 
                  label=f'{year}', edgecolors='white', linewidth=0.3)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf_web.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        print("Plotting without basemap...")
    
    # Customize the plot
    ax.set_title('Mapillary Images Distribution by Year (2018-2024)\n(Color-coded by year)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
    # Add statistics text
    total_images = len(gdf_filtered)
    year_counts = gdf_filtered['year'].value_counts().sort_index()
    stats_text = f'Total Images: {total_images:,}\n'
    for year, count in year_counts.items():
        stats_text += f'{year}: {count:,}\n'
    
    ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Spatial visualization by year saved to {output_path}")


def create_spatial_visualization(gdf, output_path):
    """
    Create a spatial scatter plot on Berlin street map with size-coded dots.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing Mapillary data
    output_path : str
        Path where to save the visualization
    """
    print("Creating spatial visualization...")
    
    # Perform spatial aggregation
    aggregated_gdf = spatial_aggregation(gdf)
    
    # Convert to Web Mercator for contextily
    gdf_web = aggregated_gdf.to_crs('EPSG:3857')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot points with size based on image count
    min_size = 20
    max_size = 300
    sizes = np.interp(aggregated_gdf['image_count'], 
                     (aggregated_gdf['image_count'].min(), aggregated_gdf['image_count'].max()),
                     (min_size, max_size))
    
    scatter = ax.scatter(gdf_web.geometry.x, gdf_web.geometry.y, 
                        s=sizes, alpha=0.6, c='red', edgecolors='darkred', linewidth=0.5)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=gdf_web.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        print("Plotting without basemap...")
    
    # Customize the plot
    ax.set_title('Mapillary Images Distribution in Berlin\n(Dot size proportional to image density)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend for dot sizes
    legend_sizes = [1, 5, 10, 25, 50]
    legend_labels = [f'{size} images' for size in legend_sizes]
    legend_sizes_scaled = np.interp(legend_sizes, 
                                  (aggregated_gdf['image_count'].min(), aggregated_gdf['image_count'].max()),
                                  (min_size, max_size))
    
    legend_elements = [plt.scatter([], [], s=size, c='red', alpha=0.6, edgecolors='darkred', 
                                 label=label) for size, label in zip(legend_sizes_scaled, legend_labels)]
    
    ax.legend(handles=legend_elements, loc='upper right', title='Image Density', 
             title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Spatial visualization saved to {output_path}")
    plt.close()

def main():
    """
    Main function to execute the visualization pipeline.
    """
    # Set up paths
    project_root = Path(__file__).parent.parent.parent.parent
    csv_path = project_root / 'data' / 'raw' / 'mapillary_berlin_full.csv'
    output_dir = project_root / 'reports' / 'figures' / 'rawdata'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data file exists
    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        return
    
    # Load data
    gdf = load_mapillary_data(csv_path)
    
    # Create visualizations
    year_output = output_dir / 'images_by_year.png'
    quarterly_output = output_dir / 'images_by_quarter.png'
    monthly_output = output_dir / 'images_by_month_2018_2024.png'
    spatial_output = output_dir / 'images_spatial_distribution.png'
    spatial_by_year_output = output_dir / 'images_spatial_distribution_by_year.png'
    
    create_year_visualization(gdf, year_output)
    create_quarterly_visualization(gdf, quarterly_output)
    create_monthly_visualization(gdf, monthly_output)
    create_spatial_visualization(gdf, spatial_output)
    create_spatial_visualization_by_year(gdf, spatial_by_year_output)
    
    print("\nVisualization complete!")
    print(f"Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
