#!/usr/bin/env python3
"""
Mapillary Image Fetcher for Berlin

This script fetches street-level image metadata from Mapillary Graph API v4
for Berlin's bounding box, optimized for ML-based pothole detection.

Features:
- 1024x768 pixel images (thumb_1024_url) for optimal ML processing
- Filters out panoramic images (is_pano=True) for better object detection
- Extracts essential metadata: GPS coordinates, timestamp, image dimensions, compass angle, sequence ID
- Tiled approach to cover all of Berlin efficiently

Usage:
    python src/fetch/mapillary_fetch.py

Environment Variables:
    MAPILLARY_ACCESS_TOKEN: Your Mapillary API access token
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import time
import json
from dotenv import load_dotenv

# Constants
# Note: Mapillary API has a limit of 0.010 square degrees per request
BERLIN_BBOX = (13.0884, 52.3383, 13.7612, 52.6755)  # (west, south, east, north) - Full Berlin
BASE_URL = "https://graph.mapillary.com/images"
DEFAULT_FIELDS = "id,captured_at,geometry,thumb_1024_url,width,height,compass_angle,sequence,is_pano"
DEFAULT_LIMIT = 2000  # Max API limit per request

# Tiling parameters
TILE_SIZE = 0.005  # sq degrees, smaller tiles to avoid hitting 2000 image limit
MAX_IMAGES_PER_TILE = 2000
REQUEST_DELAY = 1.5  # seconds between tiles


def validate_token(access_token: str) -> bool:
    """
    Test API connectivity with a small query to validate the token.
    
    Args:
        access_token: Mapillary API access token
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    print("🔑 Validating Mapillary API token...")
    
    # Test with a very small query using a tiny test area
    test_bbox = (13.405, 52.520, 13.410, 52.525)  # Tiny test area
    params = {
        'access_token': access_token,
        'fields': 'id',
        'bbox': f"{test_bbox[0]},{test_bbox[1]},{test_bbox[2]},{test_bbox[3]}",
        'limit': 1
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                print("✅ Token validation successful!")
                return True
            else:
                print("❌ Token validation failed: Invalid response structure")
                return False
        elif response.status_code == 401:
            print("❌ Token validation failed: Invalid or expired token")
            return False
        elif response.status_code == 429:
            print("❌ Token validation failed: Rate limit exceeded")
            return False
        else:
            print(f"❌ Token validation failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Token validation failed: Network error - {e}")
        return False


def fetch_images_graph(bbox: Tuple[float, float, float, float], 
                      access_token: str, 
                      fields: str = DEFAULT_FIELDS,
                      limit: int = DEFAULT_LIMIT,
                      max_retries: int = 3) -> List[Dict]:
    """
    Fetch image metadata from Mapillary Graph API for a given bounding box.
    
    Args:
        bbox: Bounding box as (west, south, east, north) in WGS84
        access_token: Mapillary API access token
        fields: Comma-separated list of fields to retrieve
        limit: Maximum number of images to fetch
        max_retries: Maximum number of retry attempts for failed requests
        
    Returns:
        List of image metadata dictionaries
    """
    print(f"📡 Fetching images from Mapillary API (limit: {limit})...")
    
    all_images = []
    after_cursor = None
    total_fetched = 0
    no_more_pages = False
    
    while total_fetched < limit and not no_more_pages:
        # Calculate how many more we need
        remaining = limit - total_fetched
        current_limit = min(remaining, 2000)  # API max is 2000 per request
        
        params = {
            'access_token': access_token,
            'fields': fields,
            'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            'limit': current_limit
        }
        
        if after_cursor:
            params['after'] = after_cursor
        
        # Retry logic for failed requests
        retry_count = 0
        request_success = False
        
        while retry_count <= max_retries and not request_success:
            try:
                response = requests.get(BASE_URL, params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data and data['data']:
                        images = data['data']
                        all_images.extend(images)
                        total_fetched += len(images)
                        
                        print(f"  📸 Fetched {len(images)} images (total: {total_fetched})")
                        
                        # Check for pagination
                        if 'paging' in data and 'cursors' in data['paging'] and 'after' in data['paging']['cursors']:
                            after_cursor = data['paging']['cursors']['after']
                        else:
                            print("  📄 No more pages available")
                            no_more_pages = True  # Exit main pagination loop
                            request_success = True  # Mark as successful to exit retry loop
                            break
                            
                    else:
                        print("  📄 No more images found")
                        no_more_pages = True  # Exit main pagination loop
                        request_success = True  # Mark as successful to exit retry loop
                        break
                    
                    request_success = True
                        
                elif response.status_code == 429:
                    print("  ⏳ Rate limit hit, waiting 5 seconds...")
                    time.sleep(5)
                    continue
                    
                elif response.status_code == 500:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                        print(f"  ⚠️ Server error (HTTP 500), retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ API request failed after {max_retries} retries: HTTP {response.status_code}")
                        break
                        
                else:
                    print(f"  ❌ API request failed: HTTP {response.status_code}")
                    break
                    
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 2 ** retry_count
                    print(f"  ⚠️ Network error, retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Network error after {max_retries} retries: {e}")
                    break
    
    print(f"✅ Successfully fetched {len(all_images)} images")
    return all_images


def hydrate_images(images_data: List[Dict]) -> pd.DataFrame:
    """
    Parse API response into structured DataFrame.
    
    Args:
        images_data: List of image metadata dictionaries from API
        
    Returns:
        DataFrame with columns: id, lon, lat, captured_at, thumb_1024_url, width, height, compass_angle, sequence, is_pano
    """
    print("🔄 Processing image data...")
    
    processed_data = []
    
    for img in images_data:
        try:
            # Skip panoramic images (they have distortion and are harder to process)
            if img.get('is_pano', False):
                continue
            
            # Extract coordinates from geometry
            if 'geometry' in img and 'coordinates' in img['geometry']:
                lon, lat = img['geometry']['coordinates']
            else:
                continue  # Skip images without valid geometry
            
            # Convert timestamp from milliseconds to ISO format
            captured_at = img.get('captured_at', '')
            if captured_at and isinstance(captured_at, (int, float)):
                # Convert from milliseconds to seconds and then to ISO format
                timestamp_seconds = captured_at / 1000
                captured_at = pd.to_datetime(timestamp_seconds, unit='s').isoformat()
            
            processed_data.append({
                'id': img.get('id', ''),
                'lon': float(lon),
                'lat': float(lat),
                'captured_at': captured_at,
                'thumb_1024_url': img.get('thumb_1024_url', ''),
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'compass_angle': img.get('compass_angle', 0),
                'sequence': img.get('sequence', ''),
                'is_pano': img.get('is_pano', False)
            })
            
        except (ValueError, KeyError) as e:
            print(f"  ⚠️ Skipping invalid image data: {e}")
            continue
    
    df = pd.DataFrame(processed_data)
    
    # Remove duplicates based on image ID (safety measure)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"✅ Processed {initial_count} images, removed {initial_count - final_count} duplicates")
    else:
        print(f"✅ Processed {final_count} valid images (no duplicates)")
    
    return df


def print_preview(df: pd.DataFrame, n: int = 10) -> None:
    """
    Print a formatted preview of the DataFrame.
    
    Args:
        df: DataFrame to preview
        n: Number of rows to show
    """
    print(f"\n📋 Preview of first {min(n, len(df))} images:")
    print("=" * 80)
    
    if df.empty:
        print("No images to display")
        return
    
    for i, row in df.head(n).iterrows():
        print(f"ID: {row['id']}")
        print(f"  📍 Location: {row['lat']:.6f}, {row['lon']:.6f}")
        print(f"  📅 Captured: {row['captured_at']}")
        print(f"  📐 Size: {row['width']}x{row['height']} pixels")
        print(f"  🧭 Compass: {row['compass_angle']:.1f}°")
        print(f"  🔗 Sequence: {row['sequence']}")
        print(f"  🖼️  1024px: {row['thumb_1024_url'][:60]}...")
        print()


def create_berlin_tiles(full_bbox: Tuple[float, float, float, float], 
                       tile_size: float = TILE_SIZE) -> List[Tuple[float, float, float, float]]:
    """
    Divide Berlin into non-overlapping tiles that respect API limit (0.010 sq degrees).
    
    Args:
        full_bbox: Full Berlin bounding box (west, south, east, north)
        tile_size: Size of each tile in square degrees (default: 0.008)
        
    Returns:
        List of tile bounding boxes as (west, south, east, north) tuples
    """
    west, south, east, north = full_bbox
    
    # Calculate tile dimensions
    tile_side = tile_size ** 0.5  # Square tiles: side = sqrt(area)
    
    # Calculate number of tiles needed (no overlap)
    width = east - west
    height = north - south
    
    cols = int(width / tile_side) + 1
    rows = int(height / tile_side) + 1
    
    tiles = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate tile bounds (no overlap)
            tile_west = west + (col * tile_side)
            tile_east = min(tile_west + tile_side, east)
            tile_south = south + (row * tile_side)
            tile_north = min(tile_south + tile_side, north)
            
            # Skip if tile is too small
            if (tile_east - tile_west) < 0.001 or (tile_north - tile_south) < 0.001:
                continue
                
            tiles.append((tile_west, tile_south, tile_east, tile_north))
    
    return tiles


def fetch_all_berlin_images(access_token: str, 
                           fields: str = DEFAULT_FIELDS,
                           max_per_tile: int = MAX_IMAGES_PER_TILE) -> List[Dict]:
    """
    Fetch images from all Berlin tiles with progress tracking.
    
    Args:
        access_token: Mapillary API access token
        fields: Comma-separated list of fields to retrieve
        max_per_tile: Maximum images to fetch per tile
        
    Returns:
        List of all image metadata dictionaries
    """
    print("📐 Creating tiles for Berlin...")
    tiles = create_berlin_tiles(BERLIN_BBOX)
    print(f"   Full bbox: {BERLIN_BBOX}")
    print(f"   Tile size: {TILE_SIZE} sq degrees")
    print(f"   Total tiles: {len(tiles)}")
    print()
    
    all_images = []
    successful_tiles = 0
    failed_tiles = 0
    
    for i, tile in enumerate(tiles, 1):
        print(f"📡 Fetching from tile {i}/{len(tiles)}...")
        print(f"   Bbox: {tile}")
        
        max_retries = 3
        retry_count = 0
        tile_success = False
        
        while retry_count < max_retries and not tile_success:
            try:
                # Fetch images from this tile
                tile_images = fetch_images_graph(
                    bbox=tile,
                    access_token=access_token,
                    fields=fields,
                    limit=max_per_tile
                )
                
                all_images.extend(tile_images)
                successful_tiles += 1
                tile_success = True
                
                if retry_count > 0:
                    print(f"   📸 Fetched {len(tile_images)} images (total: {len(all_images)}) - SUCCESS on retry {retry_count}")
                else:
                    print(f"   📸 Fetched {len(tile_images)} images (total: {len(all_images)})")
                
                # Add delay between requests to respect rate limits
                if i < len(tiles):  # Don't delay after last tile
                    time.sleep(REQUEST_DELAY)
                    
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"   ⚠️ Failed to fetch from tile {i} (attempt {retry_count}): {e}")
                    print(f"   🔄 Retrying in {REQUEST_DELAY * 2} seconds...")
                    time.sleep(REQUEST_DELAY * 2)  # Longer delay on retry
                else:
                    print(f"   ❌ Failed to fetch from tile {i} after {max_retries} attempts: {e}")
                    failed_tiles += 1
                    break
    
    print(f"\n✅ Tile processing complete!")
    print(f"   Successful tiles: {successful_tiles}")
    print(f"   Failed tiles: {failed_tiles}")
    print(f"   Total images: {len(all_images)}")
    
    return all_images


def print_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the fetched data.
    
    Args:
        df: DataFrame with image data
    """
    if df.empty:
        print("📊 No data to summarize")
        return
    
    print("\n📊 Summary Statistics:")
    print("=" * 40)
    print(f"Total images: {len(df)}")
    
    if 'captured_at' in df.columns and not df['captured_at'].isna().all():
        # Convert to datetime with proper ISO format handling
        df_temp = df.copy()
        df_temp['captured_at_dt'] = pd.to_datetime(df_temp['captured_at'], format='ISO8601', errors='coerce')
        df_temp['year'] = df_temp['captured_at_dt'].dt.year
        year_counts = df_temp['year'].value_counts().sort_index()
        print(f"Date range: {df_temp['captured_at_dt'].min()} to {df_temp['captured_at_dt'].max()}")
        print("Images per year:")
        for year, count in year_counts.items():
            if pd.notna(year):  # Skip NaN years
                print(f"  {int(year)}: {count} images")
    
    # Spatial bounds
    if 'lat' in df.columns and 'lon' in df.columns:
        print(f"Latitude range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
        print(f"Longitude range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")


def main():
    """
    Main execution function.
    """
    print("🗺️  Mapillary Full Berlin Coverage Fetch")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Load access token from environment
    access_token = os.getenv('MAPILLARY_ACCESS_TOKEN')
    if not access_token:
        print("❌ Error: MAPILLARY_ACCESS_TOKEN environment variable not set")
        print("Please set your token in .env file or export MAPILLARY_ACCESS_TOKEN='your_token_here'")
        return
    
    # Validate token
    if not validate_token(access_token):
        print("❌ Cannot proceed without valid token")
        return
    
    # Fetch images from all Berlin tiles
    images_data = fetch_all_berlin_images(
        access_token=access_token,
        fields=DEFAULT_FIELDS,
        max_per_tile=MAX_IMAGES_PER_TILE
    )
    
    if not images_data:
        print("❌ No images fetched")
        return
    
    # Process into DataFrame
    df = hydrate_images(images_data)
    
    if df.empty:
        print("❌ No valid images processed")
        return
    
    # Use all images (no year filtering)
    df_filtered = df.copy()
    print(f"✅ Using all {len(df_filtered)} images (no year filtering)")
    
    # Print summary and preview
    print_summary(df_filtered)
    print_preview(df_filtered, n=10)
    
    # Save to CSV
    output_path = "data/raw/mapillary_berlin_full.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_filtered.to_csv(output_path, index=False)
    print(f"💾 Saved {len(df_filtered)} images to {output_path}")
    
    print("\n✅ Mapillary full Berlin fetch completed successfully!")


if __name__ == "__main__":
    main()
