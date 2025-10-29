#!/usr/bin/env python3
"""
Script: Prepare CNN Dataset from Residual CSV Files

Purpose:
This script transforms residual CSV files (from baseline logistic regression) into 
per-cluster samples for CNN training. It selects one representative image per cluster 
(the first URL from list_of_thumb_1024_url) to ensure clean 1:1 mapping between 
clusters and residuals.

Functionality:
- Loads clusters_{train,val,test}_with_residuals.csv files
- Extracts first image URL per cluster from list_of_thumb_1024_url
- Creates one row per cluster with image_url, residual, split, cluster_id
- Optionally filters dead URLs (checks HTTP status codes)
- Saves consolidated dataset for CNN training

How to run:
    python src/modeling/prepare_cnn_dataset.py

Prerequisites:
    - Run baseline_logistic_regression.py first to generate *_with_residuals.csv files
    - Ensure data/processed/clusters_{train,val,test}_with_residuals.csv exist

Output:
    - data/processed/cnn_samples_residual.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import requests
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Input paths
DATA_DIR = project_root / 'data' / 'processed'
CLUSTERS_TRAIN_PATH = DATA_DIR / 'clusters_train_with_residuals.csv'
CLUSTERS_VAL_PATH = DATA_DIR / 'clusters_val_with_residuals.csv'
CLUSTERS_TEST_PATH = DATA_DIR / 'clusters_test_with_residuals.csv'

# Output paths
CNN_SAMPLES_PATH = DATA_DIR / 'cnn_samples_residual.csv'


# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_first_image_url(url_string):
    """
    Extract the first URL from a comma-separated string of URLs.
    
    Parameters:
        url_string: Comma-separated string of image URLs
        
    Returns:
        str: First URL, or empty string if none found
    """
    if pd.isna(url_string) or url_string == '':
        return ''
    
    # Split by comma and take first
    urls = str(url_string).split(',')
    first_url = urls[0].strip()
    
    return first_url if first_url else ''


def check_url_alive(url, timeout=5):
    """
    Check if a URL is accessible (optional filtering step).
    
    Parameters:
        url: URL to check
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if URL is accessible, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except:
        return False


def prepare_cnn_dataset(check_urls=False):
    """
    Prepare CNN dataset by selecting one image per cluster.
    
    Parameters:
        check_urls: If True, filter out dead URLs (slower but safer)
        
    Returns:
        DataFrame: Prepared dataset with one image per cluster
    """
    print("=" * 80)
    print("PREPARING CNN DATASET")
    print("=" * 80)
    
    # Load all three splits
    print("\nLoading residual CSV files...")
    train_df = pd.read_csv(CLUSTERS_TRAIN_PATH)
    val_df = pd.read_csv(CLUSTERS_VAL_PATH)
    test_df = pd.read_csv(CLUSTERS_TEST_PATH)
    
    print(f"  Train: {len(train_df)} clusters")
    print(f"  Val:   {len(val_df)} clusters")
    print(f"  Test:  {len(test_df)} clusters")
    
    # Combine all splits
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"\n  Total: {len(all_df)} clusters")
    
    # Extract first image URL per cluster
    print("\nExtracting first image URL per cluster...")
    all_df['image_url'] = all_df['list_of_thumb_1024_url'].apply(extract_first_image_url)
    
    # Count clusters with valid URLs
    valid_urls = (all_df['image_url'] != '') & (all_df['image_url'].notna())
    print(f"  Clusters with URLs: {valid_urls.sum()} ({valid_urls.sum()/len(all_df)*100:.1f}%)")
    print(f"  Clusters without URLs: {(~valid_urls).sum()}")
    
    # Filter out rows without URLs
    df_with_urls = all_df[valid_urls].copy()
    
    # Optional: Check URL accessibility
    if check_urls:
        print("\nChecking URL accessibility (this may take a while)...")
        alive_urls = []
        url_iter = df_with_urls['image_url'].items()
        if HAS_TQDM:
            url_iter = tqdm(url_iter, total=len(df_with_urls))
        for idx, url in url_iter:
            is_alive = check_url_alive(url)
            alive_urls.append(is_alive)
        
        df_with_urls['url_alive'] = alive_urls
        alive_count = df_with_urls['url_alive'].sum()
        print(f"\n  Alive URLs: {alive_count} ({alive_count/len(df_with_urls)*100:.1f}%)")
        print(f"  Dead URLs: {(~df_with_urls['url_alive']).sum()}")
        
        # Filter to only alive URLs
        df_with_urls = df_with_urls[df_with_urls['url_alive']].copy()
    
    # Select only required columns
    cnn_dataset = df_with_urls[['cluster_id', 'image_url', 'residual', 'split']].copy()
    
    # Verify split distribution
    print("\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        split_count = len(cnn_dataset[cnn_dataset['split'] == split])
        print(f"  {split.capitalize()}: {split_count} samples")
    
    # Check residual statistics
    print("\nResidual statistics:")
    print(f"  Mean:   {cnn_dataset['residual'].mean():.4f}")
    print(f"  Std:    {cnn_dataset['residual'].std():.4f}")
    print(f"  Min:    {cnn_dataset['residual'].min():.4f}")
    print(f"  Max:    {cnn_dataset['residual'].max():.4f}")
    
    return cnn_dataset


def main():
    """Main execution function."""
    print("=" * 80)
    print("PREPARE CNN DATASET FROM RESIDUAL CSV FILES")
    print("=" * 80)
    
    # Prepare dataset (set check_urls=True to filter dead URLs)
    cnn_dataset = prepare_cnn_dataset(check_urls=False)
    
    # Save output
    CNN_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    cnn_dataset.to_csv(CNN_SAMPLES_PATH, index=False)
    print(f"\n✓ Saved {len(cnn_dataset)} samples to: {CNN_SAMPLES_PATH}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

