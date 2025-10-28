#!/usr/bin/env python3
"""
Traffic Accident Data Visualization Script

This script creates comprehensive visualizations for the Berlin traffic accident dataset (2018-2024).
It mirrors the structure and style of the Mapillary visualization script, creating temporal
and spatial plots plus additional variable-specific visualizations.

The script loads aggregated crash data, processes temporal and spatial information,
and generates publication-ready visualizations saved to the reports/figures/rawdata directory.

Visualizations include:
1. Temporal analysis: yearly, quarterly, and monthly trends
2. Spatial distribution: crash density map on Berlin basemap
3. Severity analysis: distribution by accident severity
4. Accident types: collision patterns and vehicle involvement
5. Temporal patterns: hour-of-day and day-of-week analysis
6. Environmental factors: lighting and road conditions
7. Advanced analysis: heatmaps and district comparisons

Usage:
    python src/viz/rawdata/crashes_viz.py

Dependencies:
    - pandas, geopandas, matplotlib, seaborn, contextily
    - Data file: data/processed/crashes_aggregated.csv
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

# Set style for better-looking plots (matching Mapillary style)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define consistent color scheme
COLORS = {
    'primary': '#2E86AB',      # Blue for main lines
    'secondary': '#E63946',    # Red for secondary lines  
    'tertiary': '#F77F00',     # Orange for tertiary lines
    'accent': '#2E86AB',       # Blue accent
    'severity': ['#d62728', '#ff7f0e', '#2ca02c'],  # Red, Orange, Green for severity
    'pie': ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0'],  # Soft colors for pie charts
    'lighting': ['#ffd700', '#ff8c00', '#2f4f4f'],  # Gold, Orange, Dark gray
    'road': ['#90EE90', '#4169E1', '#B0C4DE']  # Light green, Blue, Light blue
}


def load_crash_data(csv_path):
    """
    Load and preprocess aggregated crash CSV data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the aggregated crash CSV file
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with parsed dates and geometry
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create year-month column for temporal analysis
    df['year_month'] = pd.to_datetime(df['UJAHR'].astype(str) + '-' + df['UMONAT'].astype(str).str.zfill(2) + '-01')
    df['quarter'] = df['year_month'].dt.to_period('Q')
    
    # Create geometry column for spatial operations (WGS84 coordinates)
    # Note: XGCSWGS84/YGCSWGS84 are WGS84 lat/lon coordinates
    geometry = [Point(xy) for xy in zip(df['XGCSWGS84'], df['YGCSWGS84'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')  # WGS84
    
    print(f"Loaded {len(gdf)} crashes")
    print(f"Date range: {df['UJAHR'].min()} to {df['UJAHR'].max()}")
    print(f"Year range: {df['UJAHR'].min()} to {df['UJAHR'].max()}")
    
    return gdf


def create_year_visualization(df, output_path):
    """
    Create a time series line plot showing crash counts by year.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with UJAHR column
    output_path : str
        Path where to save the visualization
    """
    print("Creating year visualization...")
    
    # Count crashes per year
    yearly_counts = df['UJAHR'].value_counts().sort_index()
    total_crashes = len(df)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line with markers
    ax.plot(yearly_counts.index, yearly_counts.values, 
            marker='o', linewidth=2, markersize=8, color=COLORS['secondary'])
    
    # Customize the plot
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title(f'Traffic Accidents in Berlin by Year (Total: {total_crashes:,} crashes)', 
                fontsize=14, fontweight='bold', pad=20)
    
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


def create_quarterly_visualization(df, output_path):
    """
    Create a time series line plot showing crash counts by quarter.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with quarter column
    output_path : str
        Path where to save the visualization
    """
    print("Creating quarterly visualization...")
    
    # Count crashes per quarter
    quarterly_counts = df['quarter'].value_counts().sort_index()
    total_crashes = len(df)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot line with markers
    ax.plot(range(len(quarterly_counts)), quarterly_counts.values, 
            marker='o', linewidth=2, markersize=6, color=COLORS['tertiary'])
    
    # Customize the plot
    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title(f'Traffic Accidents in Berlin by Quarter (2018-2024, Total: {total_crashes:,} crashes)', 
                fontsize=14, fontweight='bold', pad=20)
    
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


def create_monthly_visualization(df, output_path):
    """
    Create a time series line plot showing crash counts by month for years 2018-2024.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with year_month column
    output_path : str
        Path where to save the visualization
    """
    print("Creating monthly visualization for 2018-2024...")
    
    # Count crashes per month
    monthly_counts = df['year_month'].value_counts().sort_index()
    total_crashes = len(df)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Plot line with markers
    ax.plot(range(len(monthly_counts)), monthly_counts.values, 
            marker='o', linewidth=2, markersize=4, color=COLORS['primary'])
    
    # Customize the plot
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title(f'Traffic Accidents in Berlin by Month (2018-2024, Total: {total_crashes:,} crashes)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis with month labels (remove time component)
    month_labels = [str(m).split(' ')[0] for m in monthly_counts.index]  # Remove time component
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
    Aggregate nearby crash points into clusters for better visualization.
    Uses the same approach as Mapillary spatial aggregation.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing crash data
    grid_size : float
        Grid cell size in degrees for spatial binning
        
    Returns:
    --------
    gpd.GeoDataFrame
        Aggregated GeoDataFrame with cluster counts
    """
    print("Performing spatial aggregation...")
    
    # Create grid cells (same as Mapillary approach)
    gdf['grid_lon'] = (gdf.geometry.x / grid_size).astype(int) * grid_size
    gdf['grid_lat'] = (gdf.geometry.y / grid_size).astype(int) * grid_size
    
    # Aggregate by grid cells
    aggregated = gdf.groupby(['grid_lon', 'grid_lat']).agg({
        'UJAHR': 'count',
        'geometry': lambda x: Point(x.x.mean(), x.y.mean())
    }).reset_index()
    
    # Rename count column
    aggregated.rename(columns={'UJAHR': 'crash_count'}, inplace=True)
    
    # Create geometry for aggregated points
    aggregated_gdf = gpd.GeoDataFrame(aggregated, geometry='geometry', crs='EPSG:4326')
    
    print(f"Aggregated {len(gdf)} crashes into {len(aggregated_gdf)} clusters")
    print(f"Max crashes per cluster: {aggregated_gdf['crash_count'].max()}")
    
    return aggregated_gdf


def create_combined_spatial_visualization(gdf, mapillary_gdf, output_path):
    """
    Create a spatial scatter plot on Berlin street map with size-coded dots.
    Uses the exact same approach as Mapillary spatial visualization.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing crash data
    output_path : str
        Path where to save the visualization
    """
    print("Creating spatial visualization...")
    
    # Perform spatial aggregation (same as Mapillary)
    aggregated_gdf = spatial_aggregation(gdf)
    
    # Convert to Web Mercator for contextily (same as Mapillary)
    gdf_web = aggregated_gdf.to_crs('EPSG:3857')
    
    # Create the plot (same size as Mapillary)
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot points with size based on crash count (smaller dots)
    min_size = 15
    max_size = 200
    sizes = np.interp(aggregated_gdf['crash_count'], 
                     (aggregated_gdf['crash_count'].min(), aggregated_gdf['crash_count'].max()),
                     (min_size, max_size))
    
    scatter = ax.scatter(gdf_web.geometry.x, gdf_web.geometry.y, 
                        s=sizes, alpha=0.6, c='red', edgecolors='darkred', linewidth=0.5)
    
    # Add basemap (same as Mapillary)
    try:
        ctx.add_basemap(ax, crs=gdf_web.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not add basemap: {e}")
        print("Plotting without basemap...")
    
    # Customize the plot (same style as Mapillary)
    ax.set_title('Traffic Accidents Distribution in Berlin\n(Dot size proportional to crash density)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner look (same as Mapillary)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend for dot sizes (updated for smaller dots)
    legend_sizes = [1, 5, 10, 25, 50]
    legend_labels = [f'{size} crashes' for size in legend_sizes]
    legend_sizes_scaled = np.interp(legend_sizes, 
                                  (aggregated_gdf['crash_count'].min(), aggregated_gdf['crash_count'].max()),
                                  (min_size, max_size))
    
    legend_elements = [plt.scatter([], [], s=size, c='red', alpha=0.6, edgecolors='darkred', 
                                 label=label) for size, label in zip(legend_sizes_scaled, legend_labels)]
    
    ax.legend(handles=legend_elements, loc='upper right', title='Crash Density', 
             title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Spatial visualization saved to {output_path}")
    plt.close()


def create_severity_visualization(df, output_path):
    """
    Create a bar chart showing crash distribution by severity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with UKATEGORIE column
    output_path : str
        Path where to save the visualization
    """
    print("Creating severity visualization...")
    
    # Count crashes by severity
    severity_counts = df['UKATEGORIE'].value_counts().sort_index()
    severity_labels = ['Fatal', 'Severe Injury', 'Light Injury']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(severity_counts)), severity_counts.values, 
                  color=COLORS['severity'], alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Accident Severity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Accidents by Severity', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(severity_counts)))
    ax.set_xticklabels(severity_labels)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, severity_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Severity visualization saved to {output_path}")
    plt.close()


def create_accident_type_visualization(df, output_path):
    """
    Create a bar chart showing crash distribution by accident type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with UART column
    output_path : str
        Path where to save the visualization
    """
    print("Creating accident type visualization...")
    
    # Count crashes by accident type
    accident_counts = df['UART'].value_counts().sort_index()
    accident_labels = [
        'Collision with stationary vehicle',
        'Collision with vehicle ahead',
        'Side collision same direction',
        'Head-on collision',
        'Collision with turning vehicle',
        'Collision with pedestrian',
        'Collision with obstacle',
        'Run off road right',
        'Run off road left',
        'Other accident type'
    ]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar chart for better readability
    bars = ax.barh(range(len(accident_counts)), accident_counts.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(accident_counts))))
    
    # Customize the plot
    ax.set_xlabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accident Type', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Accidents by Type', fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels
    ax.set_yticks(range(len(accident_counts)))
    ax.set_yticklabels([f'{i}: {label}' for i, label in zip(accident_counts.index, accident_labels)])
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, accident_counts.values)):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accident type visualization saved to {output_path}")
    plt.close()


def create_vehicle_type_visualization(df, output_path):
    """
    Create a stacked bar chart showing vehicle involvement in crashes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with vehicle type columns
    output_path : str
        Path where to save the visualization
    """
    print("Creating vehicle type visualization...")
    
    # Count vehicle involvement
    vehicle_counts = {
        'Bicycle': df['IstRad'].sum(),
        'Car': df['IstPKW'].sum(),
        'Pedestrian': df['IstFuss'].sum(),
        'Motorcycle': df['IstKrad'].sum(),
        'Truck': df['IstGkfz'].sum(),
        'Other': df['IstSonstig'].sum()
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create pie chart with combined labels
    colors = COLORS['pie']
    # Create labels that include both name and count
    combined_labels = [f'{name}\n{count:,}' for name, count in vehicle_counts.items()]
    wedges, texts, autotexts = ax.pie(vehicle_counts.values(), labels=combined_labels,
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Customize the plot
    ax.set_title('Vehicle Involvement in Traffic Accidents', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Vehicle type visualization saved to {output_path}")
    plt.close()


def create_hour_visualization(df, output_path):
    """
    Create a line plot showing crash distribution by hour of day.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with USTUNDE column
    output_path : str
        Path where to save the visualization
    """
    print("Creating hour visualization...")
    
    # Count crashes by hour
    hour_counts = df['USTUNDE'].value_counts().sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot line with markers
    ax.plot(hour_counts.index, hour_counts.values, 
            marker='o', linewidth=2, markersize=6, color=COLORS['accent'])
    
    # Customize the plot
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Accidents by Hour of Day', fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Remove peak hour highlighting as requested
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Hour visualization saved to {output_path}")
    plt.close()


def create_weekday_visualization(df, output_path):
    """
    Create a bar chart showing crash distribution by day of week.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with UWOCHENTAG column
    output_path : str
        Path where to save the visualization
    """
    print("Creating weekday visualization...")
    
    # Count crashes by weekday
    weekday_counts = df['UWOCHENTAG'].value_counts().sort_index()
    weekday_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(weekday_counts)), weekday_counts.values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(weekday_counts))))
    
    # Customize the plot
    ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Accidents by Day of Week', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(range(len(weekday_counts)))
    ax.set_xticklabels(weekday_labels)
    
    # Add value labels on bars
    for bar, count in zip(bars, weekday_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Weekday visualization saved to {output_path}")
    plt.close()


def create_lighting_visualization(df, output_path):
    """
    Create a pie chart showing crash distribution by lighting conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with ULICHTVERH column
    output_path : str
        Path where to save the visualization
    """
    print("Creating lighting visualization...")
    
    # Count crashes by lighting conditions
    lighting_counts = df['ULICHTVERH'].value_counts().sort_index()
    lighting_labels = ['Daylight', 'Twilight', 'Darkness']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with combined labels
    colors = COLORS['lighting']
    # Create labels that include both name and count
    combined_labels = [f'{label}\n{count:,}' for label, count in zip(lighting_labels, lighting_counts.values)]
    wedges, texts, autotexts = ax.pie(lighting_counts.values, labels=combined_labels,
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Customize the plot
    ax.set_title('Traffic Accidents by Lighting Conditions', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Lighting visualization saved to {output_path}")
    plt.close()


def create_road_condition_visualization(df, output_path):
    """
    Create a pie chart showing crash distribution by road conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with STRZUSTAND column
    output_path : str
        Path where to save the visualization
    """
    print("Creating road condition visualization...")
    
    # Count crashes by road conditions
    road_counts = df['STRZUSTAND'].value_counts().sort_index()
    road_labels = ['Dry', 'Wet/Slippery', 'Icy']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    colors = COLORS['road']
    wedges, texts, autotexts = ax.pie(road_counts.values, labels=road_labels,
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Customize the plot
    ax.set_title('Traffic Accidents by Road Conditions', fontsize=14, fontweight='bold', pad=20)
    
    # Add count labels
    for i, (wedge, count) in enumerate(zip(wedges, road_counts.values)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 1.2 * np.cos(np.radians(angle))
        y = 1.2 * np.sin(np.radians(angle))
        ax.text(x, y, f'{count:,}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Road condition visualization saved to {output_path}")
    plt.close()


def create_heatmap_visualization(df, output_path):
    """
    Create a heatmap showing crash distribution by hour and day of week.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with USTUNDE and UWOCHENTAG columns
    output_path : str
        Path where to save the visualization
    """
    print("Creating heatmap visualization...")
    
    # Create pivot table for heatmap
    heatmap_data = df.groupby(['UWOCHENTAG', 'USTUNDE']).size().unstack(fill_value=0)
    
    # Create weekday labels
    weekday_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    heatmap_data.index = weekday_labels
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Number of Crashes'}, ax=ax)
    
    # Customize the plot
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Day of Week', fontsize=12, fontweight='bold')
    ax.set_title('Traffic Accidents Heatmap: Hour vs Day of Week', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap visualization saved to {output_path}")
    plt.close()


def create_district_visualization(df, output_path):
    """
    Create a bar chart showing crash distribution by district/county (UKREIS).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing crash data with UKREIS column
    output_path : str
        Path where to save the visualization
    """
    print("Creating district visualization...")
    
    # Count crashes by district
    district_counts = df['UKREIS'].value_counts().sort_index()
    
    # Berlin district names (codes 1-12)
    berlin_districts = {
        1: 'Mitte', 2: 'Friedrichshain-Kreuzberg', 3: 'Pankow', 4: 'Charlottenburg-Wilmersdorf',
        5: 'Spandau', 6: 'Steglitz-Zehlendorf', 7: 'Tempelhof-Schöneberg', 8: 'Neukölln',
        9: 'Treptow-Köpenick', 10: 'Marzahn-Hellersdorf', 11: 'Lichtenberg', 12: 'Reinickendorf'
    }
    
    # Create labels for all districts/counties
    district_labels = []
    for ukreis in district_counts.index:
        if ukreis in berlin_districts:
            district_labels.append(f'{berlin_districts[ukreis]} (Berlin)')
        else:
            district_labels.append(f'County {ukreis} (Brandenburg)')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create horizontal bar chart for better readability
    bars = ax.barh(range(len(district_counts)), district_counts.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(district_counts))))
    
    # Customize the plot
    ax.set_xlabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_ylabel('District/County', fontsize=12, fontweight='bold')
    ax.set_title(f'Traffic Accidents by District/County (2018-2024, Total: {district_counts.sum():,} crashes)', fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis labels
    ax.set_yticks(range(len(district_counts)))
    ax.set_yticklabels(district_labels)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, district_counts.values)):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"District visualization saved to {output_path}")
    plt.close()


def main():
    """
    Main function to execute the visualization pipeline.
    """
    # Set up paths
    project_root = Path(__file__).parent.parent.parent.parent
    csv_path = project_root / 'data' / 'processed' / 'crashes_aggregated.csv'
    output_dir = project_root / 'reports' / 'figures' / 'rawdata'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data file exists
    if not csv_path.exists():
        print(f"Error: Data file not found at {csv_path}")
        print("Please run the aggregation script first: python src/features/aggregate_crashes.py")
        return
    
    # Load data
    gdf = load_crash_data(csv_path)
    
    # Create visualizations
    print("\n=== Creating Visualizations ===\n")
    
    # Temporal visualizations
    create_year_visualization(gdf, output_dir / 'crashes_by_year.png')
    create_quarterly_visualization(gdf, output_dir / 'crashes_by_quarter.png')
    create_monthly_visualization(gdf, output_dir / 'crashes_by_month_2018_2024.png')
    
    # Spatial visualization
    create_combined_spatial_visualization(gdf, None, output_dir / 'crashes_spatial_distribution.png')
    
    # Variable-specific visualizations
    create_severity_visualization(gdf, output_dir / 'crashes_by_severity.png')
    create_accident_type_visualization(gdf, output_dir / 'crashes_by_accident_type.png')
    create_vehicle_type_visualization(gdf, output_dir / 'crashes_by_vehicle_type.png')
    create_hour_visualization(gdf, output_dir / 'crashes_by_hour.png')
    create_weekday_visualization(gdf, output_dir / 'crashes_by_weekday.png')
    create_lighting_visualization(gdf, output_dir / 'crashes_by_lighting.png')
    create_road_condition_visualization(gdf, output_dir / 'crashes_by_road_condition.png')
    
    # Advanced visualizations
    create_heatmap_visualization(gdf, output_dir / 'crashes_heatmap_hour_weekday.png')
    create_district_visualization(gdf, output_dir / 'crashes_by_district.png')
    
    print("\n=== Visualization Complete! ===")
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
