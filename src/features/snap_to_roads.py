#!/usr/bin/env python3
"""
Spatial Point-to-Road Matching Script

This script matches crash and Mapillary image coordinates to the nearest road segments
from the OSM network, extracting infrastructure attributes for each point.

The script:
1. Loads crash and Mapillary point data
2. Loads OSM road network
3. Uses spatial indexing for efficient nearest neighbor search
4. Matches points to roads within 10m threshold
5. Extracts OSM attributes for matched points
6. Saves enriched datasets

Key features:
- Efficient spatial indexing using rtree
- 10m distance threshold for matching
- Handles both crash and image data
- Generates validation statistics
- Saves enriched data with all OSM attributes

Usage:
    python src/features/snap_to_roads.py

Input files:
    data/interim/crashes_aggregated.csv
    data/interim/mapillary_berlin_full.csv
    data/interim/osm_berlin_roads.gpkg

Output files:
    data/processed/crashes_with_osm.csv
    data/processed/mapillary_with_osm.csv
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def load_data():
    """
    Load crash, image, and OSM road data.
    
    Returns:
        tuple: (crashes_gdf, images_gdf, roads_gdf)
    """
    print("Loading data files...")
    
    # Define file paths
    crashes_path = os.path.join(project_root, 'data', 'interim', 'crashes_aggregated.csv')
    images_path = os.path.join(project_root, 'data', 'interim', 'mapillary_berlin_full.csv')
    roads_path = os.path.join(project_root, 'data', 'interim', 'osm_berlin_roads.gpkg')
    
    # Check if files exist
    for path, name in [(crashes_path, 'crashes'), (images_path, 'images'), (roads_path, 'roads')]:
        if not os.path.exists(path):
            print(f"❌ Error: {name} file not found at {path}")
            sys.exit(1)
    
    # Load crash data
    print("Loading crash data...")
    crashes_df = pd.read_csv(crashes_path)
    
    # Create GeoDataFrame for crashes
    crashes_gdf = gpd.GeoDataFrame(
        crashes_df,
        geometry=gpd.points_from_xy(crashes_df['XGCSWGS84'], crashes_df['YGCSWGS84']),
        crs='EPSG:4326'
    )
    
    # Load image data
    print("Loading Mapillary image data...")
    images_df = pd.read_csv(images_path)
    
    # Create GeoDataFrame for images
    images_gdf = gpd.GeoDataFrame(
        images_df,
        geometry=gpd.points_from_xy(images_df['lon'], images_df['lat']),
        crs='EPSG:4326'
    )
    
    # Load road data
    print("Loading OSM road network...")
    roads_gdf = gpd.read_file(roads_path)
    
    print(f"✅ Loaded data:")
    print(f"   - Crashes: {len(crashes_gdf):,} points")
    print(f"   - Images: {len(images_gdf):,} points")
    print(f"   - Roads: {len(roads_gdf):,} segments")
    
    return crashes_gdf, images_gdf, roads_gdf

def convert_to_project_crs(crashes_gdf, images_gdf, roads_gdf):
    """
    Convert all data to project CRS (EPSG:25833) for accurate distance calculations.
    
    Args:
        crashes_gdf: Crash points in WGS84
        images_gdf: Image points in WGS84
        roads_gdf: Road segments in UTM
        
    Returns:
        tuple: (crashes_utm, images_utm, roads_utm)
    """
    print("Converting to project CRS (EPSG:25833)...")
    
    # Convert points to UTM
    crashes_utm = crashes_gdf.to_crs(epsg=25833)
    images_utm = images_gdf.to_crs(epsg=25833)
    
    # Roads are already in UTM, but ensure consistency
    roads_utm = roads_gdf.to_crs(epsg=25833)
    
    print("✅ Converted all data to UTM coordinates")
    return crashes_utm, images_utm, roads_utm

def create_spatial_index(roads_gdf):
    """
    Create spatial index for efficient nearest neighbor search.
    
    Args:
        roads_gdf: GeoDataFrame with road segments
        
    Returns:
        shapely.strtree.STRtree: Spatial index
    """
    print("Creating spatial index for road segments...")
    
    # Create spatial index using road segment geometries
    spatial_index = STRtree(roads_gdf.geometry)
    
    print("✅ Created spatial index")
    return spatial_index

def match_points_to_roads(points_gdf, roads_gdf, spatial_index, max_distance=10.0):
    """
    Match points to nearest road segments within distance threshold using GeoPandas.
    
    Args:
        points_gdf: GeoDataFrame with points to match
        roads_gdf: GeoDataFrame with road segments
        spatial_index: Spatial index for roads (not used in this implementation)
        max_distance: Maximum distance for matching (meters)
        
    Returns:
        pandas.DataFrame: Matched points with road attributes
    """
    print(f"Matching {len(points_gdf):,} points to roads (max distance: {max_distance}m)...")
    
    # Use GeoPandas spatial join with distance-based matching
    # First, create a buffer around each point
    points_buffered = points_gdf.copy()
    points_buffered.geometry = points_gdf.geometry.buffer(max_distance)
    
    # Perform spatial join to find roads within buffer
    joined = gpd.sjoin(points_buffered, roads_gdf, how='inner', predicate='intersects')
    
    if len(joined) == 0:
        print("❌ No roads found within distance threshold")
        return pd.DataFrame()
    
    # Calculate actual distances for all matches
    matched_data = []
    
    # Group by point to find the nearest road for each point
    for point_id, group in joined.groupby(joined.index):
        point_geom = points_gdf.loc[point_id].geometry
        
        # Calculate distances to all matched roads
        distances = []
        road_indices = []
        
        for road_idx in group.index_right:
            road_geom = roads_gdf.loc[road_idx].geometry
            distance = point_geom.distance(road_geom)
            distances.append(distance)
            road_indices.append(road_idx)
        
        # Find the nearest road
        min_distance_idx = np.argmin(distances)
        nearest_road_idx = road_indices[min_distance_idx]
        min_distance = distances[min_distance_idx]
        
        # Get road attributes
        road_row = roads_gdf.loc[nearest_road_idx]
        point_row = points_gdf.loc[point_id]
        
        # Create combined record
        matched_record = {
            'point_id': point_id,
            'dist_to_road_m': min_distance,
            'road_segment_length_m': road_row['road_segment_length_m'],
            'osm_way_id': road_row['osm_way_id']
        }
        
        # Add all OSM attributes
        osm_attrs = [
            'highway', 'surface', 'maxspeed', 'lanes', 'lit',
            'cycleway', 'sidewalk', 'oneway', 'bridge', 'tunnel',
            'junction', 'access', 'crossing', 'width', 'parking:lane',
            'traffic_calming', 'smoothness', 'is_intersection',
            'intersection_degree', 'near_traffic_signal'
        ]
        
        for attr in osm_attrs:
            if attr in road_row:
                matched_record[attr] = road_row[attr]
            else:
                matched_record[attr] = None
        
        # Add original point data
        for col in points_gdf.columns:
            if col != 'geometry':
                matched_record[col] = point_row[col]
        
        matched_data.append(matched_record)
    
    # Convert to DataFrame
    matched_df = pd.DataFrame(matched_data)
    
    # Calculate unmatched count
    matched_point_ids = set(matched_df['point_id'])
    total_points = len(points_gdf)
    unmatched_count = total_points - len(matched_point_ids)
    
    print(f"✅ Matching completed:")
    print(f"   - Matched: {len(matched_df):,} points")
    print(f"   - Unmatched: {unmatched_count:,} points")
    print(f"   - Match rate: {len(matched_df)/total_points*100:.1f}%")
    
    return matched_df

def save_enriched_data(crashes_matched, images_matched):
    """
    Save enriched datasets to processed folder.
    
    Args:
        crashes_matched: DataFrame with crash data + OSM attributes
        images_matched: DataFrame with image data + OSM attributes
    """
    print("Saving enriched datasets...")
    
    # Create output directory
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save crash data
    crashes_path = os.path.join(output_dir, 'crashes_with_osm.csv')
    crashes_matched.to_csv(crashes_path, index=False)
    print(f"✅ Saved enriched crash data: {crashes_path}")
    
    # Save image data
    images_path = os.path.join(output_dir, 'mapillary_with_osm.csv')
    images_matched.to_csv(images_path, index=False)
    print(f"✅ Saved enriched image data: {images_path}")

def generate_validation_report(crashes_matched, images_matched):
    """
    Generate validation report with statistics and coverage analysis.
    
    Args:
        crashes_matched: DataFrame with matched crash data
        images_matched: DataFrame with matched image data
    """
    print("\n" + "="*60)
    print("SPATIAL MATCHING VALIDATION REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"Crash Data:")
    print(f"  - Total crashes: {len(crashes_matched):,}")
    print(f"  - Average distance to road: {crashes_matched['dist_to_road_m'].mean():.2f} m")
    print(f"  - Median distance to road: {crashes_matched['dist_to_road_m'].median():.2f} m")
    
    print(f"\nImage Data:")
    print(f"  - Total images: {len(images_matched):,}")
    print(f"  - Average distance to road: {images_matched['dist_to_road_m'].mean():.2f} m")
    print(f"  - Median distance to road: {images_matched['dist_to_road_m'].median():.2f} m")
    
    # Distance distribution
    print(f"\nDistance Distribution (both datasets):")
    all_distances = pd.concat([crashes_matched['dist_to_road_m'], images_matched['dist_to_road_m']])
    print(f"  - 95th percentile: {all_distances.quantile(0.95):.2f} m")
    print(f"  - 99th percentile: {all_distances.quantile(0.99):.2f} m")
    print(f"  - Max distance: {all_distances.max():.2f} m")
    
    # Attribute coverage for crashes
    print(f"\nOSM Attribute Coverage (Crash Data):")
    print("-" * 40)
    osm_attrs = ['highway', 'surface', 'maxspeed', 'lanes', 'lit', 'cycleway', 'sidewalk']
    for attr in osm_attrs:
        if attr in crashes_matched.columns:
            coverage = (crashes_matched[attr].notna().sum() / len(crashes_matched)) * 100
            print(f"{attr:15s}: {coverage:6.1f}%")
    
    # Road type distribution
    if 'highway' in crashes_matched.columns:
        print(f"\nRoad Type Distribution (Crash Data):")
        print("-" * 40)
        highway_counts = crashes_matched['highway'].value_counts().head(8)
        for road_type, count in highway_counts.items():
            pct = (count / len(crashes_matched)) * 100
            print(f"{road_type:15s}: {count:6,} ({pct:5.1f}%)")

def main():
    """
    Main execution function.
    """
    print("Spatial Point-to-Road Matching")
    print("=" * 50)
    
    try:
        # Step 1: Load data
        crashes_gdf, images_gdf, roads_gdf = load_data()
        
        # Step 2: Convert to project CRS
        crashes_utm, images_utm, roads_utm = convert_to_project_crs(crashes_gdf, images_gdf, roads_gdf)
        
        # Step 3: Create spatial index
        spatial_index = create_spatial_index(roads_utm)
        
        # Step 4: Match crashes to roads
        print("\n" + "-"*50)
        print("MATCHING CRASH DATA")
        print("-"*50)
        crashes_matched = match_points_to_roads(crashes_utm, roads_utm, spatial_index, max_distance=25.0)
        
        # Step 5: Match images to roads
        print("\n" + "-"*50)
        print("MATCHING IMAGE DATA")
        print("-"*50)
        images_matched = match_points_to_roads(images_utm, roads_utm, spatial_index, max_distance=25.0)
        
        # Step 6: Save enriched data
        save_enriched_data(crashes_matched, images_matched)
        
        # Step 7: Generate validation report
        generate_validation_report(crashes_matched, images_matched)
        
        print(f"\n🎉 Spatial matching completed successfully!")
        print(f"📊 Ready for infrastructure-aware crash analysis")
        
    except Exception as e:
        print(f"❌ Error during spatial matching: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
