"""
Script: Filter Mapillary Labeled Files by Distance to Road

Purpose:
This script filters the mapillary_labeled_Xm.csv files to only include images
where dist_to_road < 10m, replacing the original files with filtered versions.

Functionality:
- Loads each mapillary_labeled_Xm.csv file (5m, 10m, 25m)
- Filters to only include rows where dist_to_road < 10m
- Saves filtered files back to same location
- Preserves all existing columns and data structure

How to run:
    python src/features/filter_labeled_files.py
"""

import os
import sys
import pandas as pd
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def filter_labeled_file(file_path, threshold):
    """
    Filter a labeled file to only include images with dist_to_road < 10m.
    
    Parameters:
        file_path: Path to the CSV file to filter
        threshold: Distance threshold (5, 10, or 25) for logging
        
    Returns:
        Tuple of (original_count, filtered_count)
    """
    print(f"\nProcessing mapillary_labeled_{threshold}m.csv...")
    
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return None, None
    
    # Load the file
    print(f"  Loading {file_path}...")
    df = pd.read_csv(file_path)
    original_count = len(df)
    print(f"  Original row count: {original_count:,}")
    
    # Check if dist_to_road_m column exists
    if 'dist_to_road_m' not in df.columns:
        print(f"  Warning: 'dist_to_road_m' column not found in file")
        print(f"  Available columns: {list(df.columns)[:10]}...")
        return original_count, original_count
    
    # Filter to dist_to_road < 10m
    print(f"  Filtering to dist_to_road < 10m...")
    filtered_df = df[df['dist_to_road_m'] < 10].copy()
    filtered_count = len(filtered_df)
    print(f"  Filtered row count: {filtered_count:,}")
    print(f"  Removed: {original_count - filtered_count:,} rows ({100 * (original_count - filtered_count) / original_count:.1f}%)")
    
    # Save filtered file
    print(f"  Saving filtered file...")
    filtered_df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"  Saved to {file_path}")
    
    return original_count, filtered_count


def main():
    """Main execution function."""
    print("=" * 80)
    print("FILTER MAPILLARY LABELED FILES BY DISTANCE TO ROAD")
    print("=" * 80)
    print("\nFiltering all mapillary_labeled_Xm.csv files to dist_to_road < 10m")
    
    # File paths
    thresholds = [5, 10, 25]
    results = {}
    
    for threshold in thresholds:
        file_path = project_root / 'data' / 'processed' / f'mapillary_labeled_{threshold}m.csv'
        original, filtered = filter_labeled_file(file_path, threshold)
        if original is not None:
            results[threshold] = {'original': original, 'filtered': filtered}
    
    # Print summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    for threshold in thresholds:
        if threshold in results:
            r = results[threshold]
            print(f"\n{threshold}m threshold:")
            print(f"  Original: {r['original']:,} images")
            print(f"  Filtered: {r['filtered']:,} images")
            print(f"  Removed: {r['original'] - r['filtered']:,} images ({100 * (r['original'] - r['filtered']) / r['original']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print("\nNext step: Run analyze_match_results.py to regenerate Matches reports")


if __name__ == "__main__":
    main()

