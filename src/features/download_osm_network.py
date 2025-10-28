#!/usr/bin/env python3
"""
Efficient OSM Network Download Script

This script downloads Berlin's street network using a more efficient approach:
1. Uses OSMnx's graph_from_place for better performance
2. Downloads in smaller chunks if needed
3. Focuses on essential attributes only
4. Uses cached data when available

Usage:
    python src/features/download_osm_network.py

Output:
    data/interim/osm_berlin_roads.gpkg - Berlin road network with attributes
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def download_berlin_network_efficient():
    """
    Download Berlin street network using efficient OSMnx approach.
    
    Returns:
        networkx.MultiDiGraph: Street network graph
    """
    print("Downloading Berlin street network (efficient approach)...")
    
    try:
        # Method 1: Try using graph_from_place (more efficient for cities)
        print("Attempting to download using city name...")
        G = ox.graph_from_place(
            "Berlin, Germany",
            network_type='drive',
            simplify=True
        )
        print(f"✅ Successfully downloaded network with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
        
    except Exception as e:
        print(f"City-based download failed: {e}")
        print("Falling back to bounding box approach...")
        
        # Method 2: Use smaller bounding box
        # Central Berlin area for testing
        north, south, east, west = 52.52, 52.48, 13.45, 13.35
        
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type='drive',
            simplify=True
        )
        print(f"✅ Downloaded central Berlin network with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G

def extract_essential_attributes(G):
    """
    Extract essential road attributes from network graph.
    
    Args:
        G: NetworkX graph with street network
        
    Returns:
        geopandas.GeoDataFrame: Road segments with essential attributes
    """
    print("Extracting essential road attributes...")
    
    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, fill_edge_geoms=True)
    
    # Focus on edges (road segments)
    roads_gdf = edges.copy()
    
    # Essential attributes only (most commonly available)
    essential_attrs = [
        'highway', 'surface', 'maxspeed', 'lanes', 'lit',
        'cycleway', 'sidewalk', 'oneway', 'bridge', 'tunnel'
    ]
    
    # Ensure all expected columns exist
    for attr in essential_attrs:
        if attr not in roads_gdf.columns:
            roads_gdf[attr] = None
    
    # Select only essential columns
    keep_cols = ['geometry'] + essential_attrs
    roads_gdf = roads_gdf[keep_cols].copy()
    
    # Add unique identifier
    roads_gdf['osm_way_id'] = roads_gdf.index.get_level_values('osmid')
    
    # Reset index
    roads_gdf = roads_gdf.reset_index()
    
    # Add derived attributes
    roads_gdf = add_intersection_attributes(roads_gdf, G)
    
    print(f"Extracted {len(roads_gdf)} road segments")
    return roads_gdf

def add_intersection_attributes(roads_gdf, G):
    """
    Add intersection detection attributes.
    
    Args:
        roads_gdf: GeoDataFrame with road segments
        G: NetworkX graph
        
    Returns:
        geopandas.GeoDataFrame: Roads with intersection attributes
    """
    print("Computing intersection attributes...")
    
    # Initialize columns
    roads_gdf['is_intersection'] = False
    roads_gdf['intersection_degree'] = 0
    
    # Calculate node degrees
    node_degrees = dict(G.degree())
    
    # Check intersection status for each road segment
    for idx, row in roads_gdf.iterrows():
        u, v = row.name  # Get node IDs
        
        # Check if either endpoint is an intersection (3+ connections)
        u_degree = node_degrees.get(u, 0)
        v_degree = node_degrees.get(v, 0)
        max_degree = max(u_degree, v_degree)
        
        roads_gdf.loc[idx, 'intersection_degree'] = max_degree
        roads_gdf.loc[idx, 'is_intersection'] = max_degree >= 3
    
    intersection_count = roads_gdf['is_intersection'].sum()
    print(f"Found {intersection_count} road segments at intersections")
    
    return roads_gdf

def convert_and_save(roads_gdf):
    """
    Convert to project CRS and save to file.
    
    Args:
        roads_gdf: GeoDataFrame with road segments
    """
    print("Converting to project CRS and saving...")
    
    # Convert to UTM zone 33N (EPSG:25833)
    roads_utm = roads_gdf.to_crs(epsg=25833)
    
    # Calculate road segment lengths
    roads_utm['road_segment_length_m'] = roads_utm.geometry.length
    
    # Save to GeoPackage
    output_dir = os.path.join(project_root, 'data', 'interim')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'osm_berlin_roads.gpkg')
    roads_utm.to_file(output_path, driver='GPKG')
    
    print(f"✅ Saved road network to: {output_path}")
    
    # Generate summary
    total_length_km = roads_utm['road_segment_length_m'].sum() / 1000
    print(f"📊 Summary:")
    print(f"   - Road segments: {len(roads_utm):,}")
    print(f"   - Total length: {total_length_km:,.1f} km")
    print(f"   - Average segment length: {roads_utm['road_segment_length_m'].mean():.1f} m")
    
    # Attribute coverage
    print(f"\n📈 Attribute Coverage:")
    for attr in ['highway', 'surface', 'maxspeed', 'lanes', 'lit', 'cycleway', 'sidewalk']:
        if attr in roads_utm.columns:
            coverage = (roads_utm[attr].notna().sum() / len(roads_utm)) * 100
            print(f"   - {attr}: {coverage:.1f}%")

def main():
    """
    Main execution function.
    """
    print("OSM Berlin Road Network Download (Efficient)")
    print("=" * 50)
    
    try:
        # Download network
        G = download_berlin_network_efficient()
        
        # Extract attributes
        roads_gdf = extract_essential_attributes(G)
        
        # Convert and save
        convert_and_save(roads_gdf)
        
        print(f"\n🎉 Successfully processed Berlin road network!")
        print(f"📁 Ready for spatial matching with crash and image data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()