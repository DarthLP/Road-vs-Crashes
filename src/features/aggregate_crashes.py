#!/usr/bin/env python3
"""
Traffic Accident Data Aggregation Script

This script aggregates and cleans Germany-wide traffic accident data from multiple years (2018-2024),
filters it to match the geographic bounds of Mapillary image data (Berlin + Brandenburg region),
and combines all years into a single processed dataset.

Purpose:
    - Combine multiple yearly accident datasets into one unified file
    - Filter crashes to match image data coverage (lon: 13.088-13.761, lat: 52.338-52.676)
    - Remove redundant columns (LAND, LOR_ab_2021, STRASSE, LOR, OBJECTID)
    - Standardize coordinate format (keep WGS84 coordinates)
    - Handle German CSV format (semicolon delimiter, comma decimal separator)
    - Prepare clean data for visualization and analysis

Output:
    - data/processed/crashes_aggregated.csv: Combined and cleaned dataset filtered to image bounds

Usage:
    python src/features/aggregate_crashes.py
    
Dependencies:
    - pandas
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_crash_data(csv_path):
    """
    Load a single crash dataset CSV file with proper German formatting.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file to load
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame with parsed numeric columns
    """
    print(f"Loading {csv_path.name}...")
    
    # Load with semicolon delimiter and handle German decimal format
    df = pd.read_csv(csv_path, sep=';', encoding='latin-1')
    
    # Convert comma decimals to dots for WGS84 coordinate columns only
    wgs84_cols = ['XGCSWGS84', 'YGCSWGS84']
    for col in wgs84_cols:
        if col in df.columns:
            print(f"  Converting {col} from comma to dot format")
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            print(f"  {col} range after conversion: {df[col].min():.3f} to {df[col].max():.3f}")
    
    # Clean coordinate data - remove rows with swapped coordinates
    if 'XGCSWGS84' in df.columns and 'YGCSWGS84' in df.columns:
        # Remove rows where XGCSWGS84 (longitude) is > 20 (should be ~13 for Berlin)
        # or YGCSWGS84 (latitude) is < 50 (should be ~52 for Berlin)
        before_count = len(df)
        df = df[(df['XGCSWGS84'] < 20) & (df['YGCSWGS84'] > 50)]
        after_count = len(df)
        if before_count != after_count:
            print(f"  Removed {before_count - after_count} rows with invalid coordinates")
    
    # Clean binary columns (should only contain 0 or 1)
    binary_cols = ['IstRad', 'IstPKW', 'IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstig', 'IstSonstige']
    for col in binary_cols:
        if col in df.columns:
            # Convert to string, keep only 0 and 1, convert back to int
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(['0', '1']), '0').astype(int)
    
    # Clean categorical columns (should only contain 0, 1, 2)
    categorical_cols = ['STRZUSTAND', 'USTRZUSTAND', 'IstStrassenzustand']
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string, keep only 0, 1, 2, convert back to int
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(['0', '1', '2']), '0').astype(int)
    
    print(f"  Loaded {len(df)} records")
    return df


def filter_to_image_bounds(df):
    """
    Filter crash data to match the geographic bounds of Mapillary image data.
    
    Image data bounds:
    - Longitude: 13.088403 to 13.761199
    - Latitude: 52.338301 to 52.675498
    - Coverage: Berlin + small part of Brandenburg
    
    Parameters:
    -----------
    df : pd.DataFrame
        Crash data DataFrame with XGCSWGS84 and YGCSWGS84 columns
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only crashes within image bounds
    """
    if 'XGCSWGS84' not in df.columns or 'YGCSWGS84' not in df.columns:
        print("  Warning: Coordinate columns not found, skipping geographic filter")
        return df
    
    before_count = len(df)
    
    # Filter to image data bounds
    df_filtered = df[
        (df['XGCSWGS84'] >= 13.088403) & (df['XGCSWGS84'] <= 13.761199) &
        (df['YGCSWGS84'] >= 52.338301) & (df['YGCSWGS84'] <= 52.675498)
    ]
    
    after_count = len(df_filtered)
    removed_count = before_count - after_count
    
    print(f"  Geographic filtering:")
    print(f"    Before: {before_count:,} crashes")
    print(f"    After: {after_count:,} crashes")
    print(f"    Removed: {removed_count:,} crashes ({removed_count/before_count*100:.1f}%)")
    
    if after_count > 0:
        print(f"    Filtered bounds: X {df_filtered['XGCSWGS84'].min():.3f}-{df_filtered['XGCSWGS84'].max():.3f}, Y {df_filtered['YGCSWGS84'].min():.3f}-{df_filtered['YGCSWGS84'].max():.3f}")
    
    return df_filtered


def clean_crash_data(df):
    """
    Remove unnecessary columns from crash dataset and standardize column names.
    
    Removes:
    - LAND (always 11 for Berlin)
    - LOR_ab_2021 (new LOR system, keeping BEZ)
    - STRASSE (street name, inconsistent)
    - LOR (old LOR system)
    - OBJECTID (just row identifier)
    - XGCSWGS84, YGCSWGS84 (GK3 coordinates, keeping UTM)
    
    Standardizes:
    - IstSonstige -> IstSonstig
    - USTRZUSTAND -> STRZUSTAND
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw crash data DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with only essential columns
    """
    columns_to_remove = ['OBJECTID', 'LAND', 'LOR', 'STRASSE', 'LOR_ab_2021', 
                         'LINREFX', 'LINREFY', 'UREGBEZ', 'BEZ', 'ULAND', 'PLST', 
                         'OBJECTID_1', 'UIDENTSTLAE', 'ï»¿OID_']  # Remove unnecessary columns
    
    # Remove columns that exist in the dataframe
    cols_to_drop = [col for col in columns_to_remove if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # Standardize column names
    column_mapping = {
        'IstSonstige': 'IstSonstig',
        'USTRZUSTAND': 'STRZUSTAND',
        'IstStrassenzustand': 'STRZUSTAND',
        'UREGBEZ': 'BEZ'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    print(f"  Removed {len(cols_to_drop)} unnecessary columns")
    print(f"  Standardized column names")
    print(f"  Retained columns: {list(df_clean.columns)}")
    
    return df_clean


def aggregate_all_years(data_dir):
    """
    Load and aggregate crash data from all available years (2018-2024).
    
    Parameters:
    -----------
    data_dir : Path
        Directory containing raw crash data CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all years of crash data
    """
    print("\n=== Aggregating Traffic Accident Data ===\n")
    
    # Find all crash CSV files - both old format and new format
    old_format_files = sorted(data_dir.glob('AfSBBB_BE_LOR_Strasse_Strassenverkehrsunfaelle_*_Datensatz.csv'))
    new_format_files = sorted(data_dir.glob('Unfallorte*_LinRef.csv'))
    
    crash_files = old_format_files + new_format_files
    
    if not crash_files:
        raise FileNotFoundError(f"No crash data files found in {data_dir}")
    
    print(f"Found {len(crash_files)} data files:")
    for f in crash_files:
        print(f"  - {f.name}")
    print()
    
    # Load and clean each file
    all_data = []
    for csv_path in crash_files:
        df = load_crash_data(csv_path)
        df_clean = clean_crash_data(df)
        df_filtered = filter_to_image_bounds(df_clean)
        all_data.append(df_filtered)
    
    # Combine all years
    print("\nCombining all years...")
    print("Checking coordinate ranges before concatenation:")
    for i, df in enumerate(all_data):
        print(f"  Year {i+1}: XGCSWGS84 {df['XGCSWGS84'].min():.3f}-{df['XGCSWGS84'].max():.3f}, YGCSWGS84 {df['YGCSWGS84'].min():.3f}-{df['YGCSWGS84'].max():.3f}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nAfter concatenation:")
    print(f"  XGCSWGS84 range: {combined_df['XGCSWGS84'].min():.3f} to {combined_df['XGCSWGS84'].max():.3f}")
    print(f"  YGCSWGS84 range: {combined_df['YGCSWGS84'].min():.3f} to {combined_df['YGCSWGS84'].max():.3f}")
    
    # Sort by year and month
    combined_df = combined_df.sort_values(['UJAHR', 'UMONAT']).reset_index(drop=True)
    
    print(f"\nTotal records: {len(combined_df)}")
    print(f"Year range: {combined_df['UJAHR'].min()} - {combined_df['UJAHR'].max()}")
    print(f"Total crashes by year:")
    print(combined_df['UJAHR'].value_counts().sort_index())
    
    return combined_df


def save_aggregated_data(df, output_path):
    """
    Save aggregated crash data to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined crash data
    output_path : Path
        Path where to save the processed CSV file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with comma delimiter (standard CSV)
    df.to_csv(output_path, index=False)
    print(f"\nAggregated data saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def print_data_summary(df):
    """
    Print summary statistics of the aggregated dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Aggregated crash data
    """
    print("\n=== Data Summary ===\n")
    
    print(f"Total accidents: {len(df):,}")
    print(f"\nAccident severity (UKATEGORIE):")
    print(f"  1 (Fatal): {(df['UKATEGORIE'] == 1).sum():,}")
    print(f"  2 (Severe): {(df['UKATEGORIE'] == 2).sum():,}")
    print(f"  3 (Light): {(df['UKATEGORIE'] == 3).sum():,}")
    
    print(f"\nVehicle involvement:")
    print(f"  Bicycle (IstRad): {df['IstRad'].sum():,}")
    print(f"  Car (IstPKW): {df['IstPKW'].sum():,}")
    print(f"  Pedestrian (IstFuss): {df['IstFuss'].sum():,}")
    print(f"  Motorcycle (IstKrad): {df['IstKrad'].sum():,}")
    print(f"  Truck (IstGkfz): {df['IstGkfz'].sum():,}")
    
    print(f"\nSpatial coverage:")
    print(f"  Districts/Counties (UKREIS): {df['UKREIS'].nunique()}")
    print(f"  Municipalities (UGEMEINDE): {df['UGEMEINDE'].nunique()}")
    print(f"  Coordinate range X: {df['XGCSWGS84'].min():.1f} - {df['XGCSWGS84'].max():.1f}")
    print(f"  Coordinate range Y: {df['YGCSWGS84'].min():.1f} - {df['YGCSWGS84'].max():.1f}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  None")


def main():
    """
    Main function to execute the aggregation pipeline.
    """
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    output_path = project_root / 'data' / 'interim' / 'crashes_aggregated.csv'
    
    # Check if raw data directory exists
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory not found at {raw_data_dir}")
        return
    
    # Aggregate data
    df_aggregated = aggregate_all_years(raw_data_dir)
    
    # Print summary
    print_data_summary(df_aggregated)
    
    # Save processed data
    save_aggregated_data(df_aggregated, output_path)
    
    print("\n=== Aggregation Complete! ===\n")


if __name__ == "__main__":
    main()

