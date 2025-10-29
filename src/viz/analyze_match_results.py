"""
Script: Analyze and Summarize Crash-to-Image Matching Results

Purpose:
This script analyzes the matched crash-to-image datasets and generates comprehensive
summary statistics in CSV format and visualizations for each distance threshold (5m, 10m, 25m).

Functionality:
- Loads all three threshold datasets
- Calculates comprehensive matching statistics
- Computes time differences between images and crashes
- Analyzes shared crash patterns
- Generates year-wise breakdowns
- Creates visualizations comparing thresholds
- Saves all summaries to CSV files and figures to reports/Matches/

How to run:
    python src/viz/analyze_match_results.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def calculate_time_differences(df, threshold):
    """
    Calculate time difference statistics between image capture and crash years.
    
    Parameters:
        df: DataFrame with image and crash data
        threshold: Distance threshold (5, 10, or 25)
        
    Returns:
        Dictionary with time difference statistics
    """
    suffix = f"_{threshold}m"
    
    # Filter to images with crashes
    with_crashes = df[df[f'has_crash{suffix}'] == 1].copy()
    
    if len(with_crashes) == 0:
        return {
            'avg_time_diff': np.nan,
            'median_time_diff': np.nan,
            'crashes_before': 0,
            'crashes_after': 0,
            'crashes_same_year': 0,
            'time_diff_list': []
        }
    
    # Parse image capture dates (handle mixed formats)
    with_crashes['img_date'] = pd.to_datetime(with_crashes['captured_at'], format='mixed', errors='coerce')
    with_crashes['img_year'] = with_crashes['img_date'].dt.year
    
    # Parse crash years and calculate differences
    time_diffs = []
    crashes_before = 0
    crashes_after = 0
    crashes_same_year = 0
    
    for idx, row in with_crashes.iterrows():
        crash_years_str = row[f'crash_years{suffix}']
        if pd.isna(crash_years_str) or crash_years_str == '':
            continue
        
        crash_years = [int(y) for y in str(crash_years_str).split(',')]
        img_year = row['img_year']
        
        # Calculate time difference for each crash
        for crash_year in crash_years:
            time_diff = crash_year - img_year
            time_diffs.append(abs(time_diff))
            
            if time_diff < 0:
                crashes_before += 1
            elif time_diff > 0:
                crashes_after += 1
            else:
                crashes_same_year += 1
    
    return {
        'avg_time_diff': np.mean(time_diffs) if len(time_diffs) > 0 else np.nan,
        'median_time_diff': np.median(time_diffs) if len(time_diffs) > 0 else np.nan,
        'crashes_before': crashes_before,
        'crashes_after': crashes_after,
        'crashes_same_year': crashes_same_year,
        'time_diff_list': time_diffs
    }


def analyze_shared_crashes(df, threshold):
    """
    Analyze shared crash patterns.
    
    Parameters:
        df: DataFrame with image and crash data
        threshold: Distance threshold (5, 10, or 25)
        
    Returns:
        Dictionary with shared crash statistics
    """
    suffix = f"_{threshold}m"
    
    # Build crash-to-images mapping from shared_with_image_ids
    crash_to_images = {}
    
    for idx, row in df.iterrows():
        shared_info = row[f'shared_with_image_ids{suffix}']
        current_image_id = row['point_id']
        
        if pd.isna(shared_info) or shared_info == '':
            continue
        
        # Parse format: "c123:i456,i789;c456:i123"
        for crash_entry in str(shared_info).split(';'):
            if ':' not in crash_entry or crash_entry.strip() == '':
                continue
            
            crash_part, images_part = crash_entry.split(':', 1)
            crash_id = int(crash_part.replace('c', ''))
            image_ids = [int(img.replace('i', '')) for img in images_part.split(',') if img.replace('i', '').isdigit()]
            
            # Initialize if not exists
            if crash_id not in crash_to_images:
                crash_to_images[crash_id] = []
            
            # Add current image
            crash_to_images[crash_id].append(current_image_id)
            # Add other images
            crash_to_images[crash_id].extend(image_ids)
    
    # Remove duplicates and count
    for crash_id in crash_to_images:
        crash_to_images[crash_id] = list(set(crash_to_images[crash_id]))
    
    if len(crash_to_images) == 0:
        return {
            'total_shared_crashes': 0,
            'avg_images_per_shared_crash': 0,
            'max_images_per_crash': 0,
            'shared_by_2_images': 0,
            'shared_by_3_images': 0,
            'shared_by_4_plus_images': 0,
            'images_involved_in_sharing': 0
        }
    
    # Calculate statistics
    image_counts = [len(images) for images in crash_to_images.values()]
    unique_images = set()
    for images in crash_to_images.values():
        unique_images.update(images)
    
    return {
        'total_shared_crashes': len(crash_to_images),
        'avg_images_per_shared_crash': np.mean(image_counts),
        'max_images_per_crash': max(image_counts),
        'shared_by_2_images': sum(1 for c in image_counts if c == 2),
        'shared_by_3_images': sum(1 for c in image_counts if c == 3),
        'shared_by_4_plus_images': sum(1 for c in image_counts if c >= 4),
        'images_involved_in_sharing': len(unique_images)
    }


def get_matched_crashes_per_year(df, threshold):
    """
    Get year-wise breakdown of matched crashes.
    
    Parameters:
        df: DataFrame with image and crash data
        threshold: Distance threshold (5, 10, or 25)
        
    Returns:
        DataFrame with year-wise statistics
    """
    suffix = f"_{threshold}m"
    
    # Parse image capture dates (handle mixed formats)
    df_copy = df.copy()
    df_copy['img_date'] = pd.to_datetime(df_copy['captured_at'], format='mixed', errors='coerce')
    df_copy['img_year'] = df_copy['img_date'].dt.year
    
    # Initialize year tracking
    year_data = {year: {'matched_crashes': 0, 'images_count': 0, 'same_year_count': 0} 
                 for year in range(2018, 2025)}
    
    # Process images with crashes
    with_crashes = df_copy[df_copy[f'has_crash{suffix}'] == 1]
    
    for idx, row in with_crashes.iterrows():
        crash_years_str = row[f'crash_years{suffix}']
        if pd.isna(crash_years_str) or crash_years_str == '':
            continue
        
        crash_years = [int(y) for y in str(crash_years_str).split(',')]
        img_year = row['img_year']
        
        # Track images with crashes from this year
        images_with_this_year = False
        
        for crash_year in crash_years:
            if crash_year in year_data:
                year_data[crash_year]['matched_crashes'] += 1
                
                if crash_year == img_year:
                    year_data[crash_year]['same_year_count'] += 1
                
                if not images_with_this_year:
                    year_data[crash_year]['images_count'] += 1
                    images_with_this_year = True
    
    # Convert to DataFrame
    year_rows = []
    for year, data in year_data.items():
        year_rows.append({
            'threshold': f'{threshold}m',
            'crash_year': year,
            'matched_crashes_count': data['matched_crashes'],
            'images_with_crashes_this_year': data['images_count'],
            'crashes_same_year_as_image': data['same_year_count']
        })
    
    return pd.DataFrame(year_rows)


def generate_summary_statistics(df_5m, df_10m, df_25m):
    """Generate summary statistics CSV comparing all thresholds."""
    print("=" * 80)
    print("GENERATING MATCHING SUMMARY STATISTICS")
    print("=" * 80)
    
    datasets = {
        '5m': (5, df_5m),
        '10m': (10, df_10m),
        '25m': (25, df_25m)
    }
    
    summary_rows = []
    
    for threshold_name, (threshold, df) in datasets.items():
        suffix = f"_{threshold}m"
        
        print(f"\nProcessing {threshold_name} threshold...")
        
        # Basic statistics
        total_images = len(df)
        images_with_crashes = df[f'has_crash{suffix}'].sum()
        match_rate = (images_with_crashes / total_images * 100) if total_images > 0 else 0
        
        # Crash statistics
        total_crash_matches = df[f'crash_count{suffix}'].sum()
        matched_images = df[df[f'has_crash{suffix}'] == 1]
        
        crash_counts = matched_images[f'crash_count{suffix}']
        avg_crashes = crash_counts.mean() if len(matched_images) > 0 else 0
        median_crashes = crash_counts.median() if len(matched_images) > 0 else 0
        max_crashes = crash_counts.max() if len(matched_images) > 0 else 0
        std_crashes = crash_counts.std() if len(matched_images) > 0 else 0
        
        # Time differences
        time_stats = calculate_time_differences(df, threshold)
        
        # Shared crashes
        shared_stats = analyze_shared_crashes(df, threshold)
        
        # Unique crash IDs
        all_crash_ids = []
        for crash_ids_str in df[df[f'all_crash_ids{suffix}'] != ''][f'all_crash_ids{suffix}']:
            if pd.notna(crash_ids_str) and crash_ids_str != '':
                all_crash_ids.extend([int(x) for x in str(crash_ids_str).split(',')])
        unique_crash_ids = len(set(all_crash_ids)) if all_crash_ids else 0
        
        summary_rows.append({
            'threshold': threshold_name,
            'total_images': total_images,
            'images_with_crashes': images_with_crashes,
            'match_rate_pct': round(match_rate, 2),
            'total_crash_matches': total_crash_matches,
            'unique_crash_ids_matched': unique_crash_ids,
            'unique_images_involved': images_with_crashes,
            'avg_crashes_per_matched_image': round(avg_crashes, 2),
            'median_crashes_per_matched_image': round(median_crashes, 2),
            'max_crashes_per_image': int(max_crashes) if pd.notna(max_crashes) else 0,
            'std_crashes_per_image': round(std_crashes, 2),
            'avg_time_diff_years': round(time_stats['avg_time_diff'], 2) if pd.notna(time_stats['avg_time_diff']) else np.nan,
            'median_time_diff_years': round(time_stats['median_time_diff'], 2) if pd.notna(time_stats['median_time_diff']) else np.nan,
            'crashes_before_image': time_stats['crashes_before'],
            'crashes_after_image': time_stats['crashes_after'],
            'crashes_same_year_as_image': time_stats['crashes_same_year'],
            'images_with_shared_crashes': shared_stats['images_involved_in_sharing'],
            'total_shared_crash_relationships': shared_stats['total_shared_crashes'],
            'avg_images_per_shared_crash': round(shared_stats['avg_images_per_shared_crash'], 2),
            'max_images_per_crash': shared_stats['max_images_per_crash'],
            'crashes_shared_by_2': shared_stats['shared_by_2_images'],
            'crashes_shared_by_3': shared_stats['shared_by_3_images'],
            'crashes_shared_by_4_plus': shared_stats['shared_by_4_plus_images']
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    output_dir = project_root / 'reports' / 'Matches'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'matching_summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved summary statistics to: {output_path}")
    
    return summary_df


def generate_shared_crashes_statistics(df_5m, df_10m, df_25m):
    """Generate detailed shared crashes statistics CSV."""
    print("\n" + "=" * 80)
    print("GENERATING SHARED CRASHES STATISTICS")
    print("=" * 80)
    
    datasets = {
        '5m': (5, df_5m),
        '10m': (10, df_10m),
        '25m': (25, df_25m)
    }
    
    shared_rows = []
    
    for threshold_name, (threshold, df) in datasets.items():
        print(f"\nProcessing {threshold_name} threshold...")
        
        shared_stats = analyze_shared_crashes(df, threshold)
        images_with_shared = df[f'has_shared_crashes_{threshold}m'].sum()
        
        total_shared = shared_stats['total_shared_crashes']
        
        shared_rows.append({
            'threshold': threshold_name,
            'total_shared_crashes': total_shared,
            'avg_images_per_shared_crash': round(shared_stats['avg_images_per_shared_crash'], 2),
            'max_images_per_crash': shared_stats['max_images_per_crash'],
            'crashes_shared_by_2_images': shared_stats['shared_by_2_images'],
            'crashes_shared_by_3_images': shared_stats['shared_by_3_images'],
            'crashes_shared_by_4_plus_images': shared_stats['shared_by_4_plus_images'],
            'images_involved_in_sharing': shared_stats['images_involved_in_sharing'],
            'pct_crashes_shared_by_2': round(shared_stats['shared_by_2_images'] / total_shared * 100, 2) if total_shared > 0 else 0,
            'pct_crashes_shared_by_3': round(shared_stats['shared_by_3_images'] / total_shared * 100, 2) if total_shared > 0 else 0,
            'pct_crashes_shared_by_4_plus': round(shared_stats['shared_by_4_plus_images'] / total_shared * 100, 2) if total_shared > 0 else 0
        })
    
    # Create DataFrame
    shared_df = pd.DataFrame(shared_rows)
    
    # Save to CSV
    output_dir = project_root / 'reports' / 'Matches'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'shared_crashes_statistics.csv'
    shared_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved shared crashes statistics to: {output_path}")
    
    return shared_df


def generate_crash_count_distributions(df_5m, df_10m, df_25m):
    """Generate crash count distribution CSV."""
    print("\n" + "=" * 80)
    print("GENERATING CRASH COUNT DISTRIBUTIONS")
    print("=" * 80)
    
    datasets = {
        '5m': (5, df_5m),
        '10m': (10, df_10m),
        '25m': (25, df_25m)
    }
    
    distribution_rows = []
    
    for threshold_name, (threshold, df) in datasets.items():
        print(f"\nProcessing {threshold_name} threshold...")
        
        suffix = f"_{threshold}m"
        matched = df[df[f'has_crash{suffix}'] == 1]
        
        if len(matched) > 0:
            crash_counts = matched[f'crash_count{suffix}'].value_counts().sort_index()
            
            for crash_count, frequency in crash_counts.items():
                pct = (frequency / len(matched) * 100) if len(matched) > 0 else 0
                distribution_rows.append({
                    'threshold': threshold_name,
                    'crash_count': int(crash_count),
                    'frequency': int(frequency),
                    'percentage': round(pct, 2)
                })
    
    # Create DataFrame
    dist_df = pd.DataFrame(distribution_rows)
    
    # Save to CSV
    output_dir = project_root / 'reports' / 'Matches'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'crash_count_distributions.csv'
    dist_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved crash count distributions to: {output_path}")
    
    return dist_df


def generate_matched_crashes_per_year(df_5m, df_10m, df_25m):
    """Generate year-wise breakdown CSV."""
    print("\n" + "=" * 80)
    print("GENERATING MATCHED CRASHES PER YEAR")
    print("=" * 80)
    
    year_dfs = []
    
    for threshold_name, threshold_num in [('5m', 5), ('10m', 10), ('25m', 25)]:
        print(f"\nProcessing {threshold_name} threshold...")
        
        if threshold_name == '5m':
            df = df_5m
        elif threshold_name == '10m':
            df = df_10m
        else:
            df = df_25m
        
        year_df = get_matched_crashes_per_year(df, threshold_num)
        year_dfs.append(year_df)
    
    # Combine all
    combined_df = pd.concat(year_dfs, ignore_index=True)
    
    # Save to CSV
    output_dir = project_root / 'reports' / 'Matches'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'matched_crashes_per_year.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved matched crashes per year to: {output_path}")
    
    return combined_df


def generate_temporal_distribution(df_5m, df_10m, df_25m):
    """Generate time difference distribution CSV."""
    print("\n" + "=" * 80)
    print("GENERATING TEMPORAL DISTRIBUTION")
    print("=" * 80)
    
    datasets = {
        '5m': (5, df_5m),
        '10m': (10, df_10m),
        '25m': (25, df_25m)
    }
    
    temporal_rows = []
    
    for threshold_name, (threshold, df) in datasets.items():
        print(f"\nProcessing {threshold_name} threshold...")
        
        time_stats = calculate_time_differences(df, threshold)
        time_diffs = time_stats['time_diff_list']
        
        if len(time_diffs) == 0:
            continue
        
        # Create buckets: 0, 1, 2, 3+ years
        buckets = {
            0: sum(1 for d in time_diffs if d == 0),
            1: sum(1 for d in time_diffs if d == 1),
            2: sum(1 for d in time_diffs if d == 2),
            3: sum(1 for d in time_diffs if d >= 3)
        }
        
        total = len(time_diffs)
        
        for years, frequency in buckets.items():
            label = f"{years}+" if years == 3 else str(years)
            pct = (frequency / total * 100) if total > 0 else 0
            temporal_rows.append({
                'threshold': threshold_name,
                'time_diff_years': label,
                'frequency': frequency,
                'percentage': round(pct, 2)
            })
    
    # Create DataFrame
    temporal_df = pd.DataFrame(temporal_rows)
    
    # Save to CSV
    output_dir = project_root / 'reports' / 'Matches'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'temporal_distribution.csv'
    temporal_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved temporal distribution to: {output_path}")
    
    return temporal_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_crash_count_distribution(dist_df):
    """Plot crash count distributions comparing all thresholds."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    thresholds = ['5m', '10m', '25m']
    crash_counts = sorted(dist_df['crash_count'].unique())
    
    x = np.arange(len(crash_counts))
    width = 0.25
    
    for i, threshold in enumerate(thresholds):
        threshold_data = dist_df[dist_df['threshold'] == threshold]
        percentages = []
        for count in crash_counts:
            match = threshold_data[threshold_data['crash_count'] == count]
            if len(match) > 0:
                percentages.append(match['percentage'].iloc[0])
            else:
                percentages.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, percentages, width, label=f'{threshold} threshold', alpha=0.8)
    
    ax.set_xlabel('Number of Crashes per Image', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Matched Images (%)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Crash Counts per Image by Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}+' if c == max(crash_counts) and c >= 5 else str(c) for c in crash_counts])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_matched_crashes_per_year(year_df):
    """Plot temporal distribution of matched crashes by year."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    thresholds = ['5m', '10m', '25m']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    years = sorted(year_df['crash_year'].unique())
    
    x = np.arange(len(years))
    width = 0.25
    
    for i, threshold in enumerate(thresholds):
        threshold_data = year_df[year_df['threshold'] == threshold]
        counts = []
        for year in years:
            match = threshold_data[threshold_data['crash_year'] == year]
            if len(match) > 0:
                counts.append(match['matched_crashes_count'].iloc[0])
            else:
                counts.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, counts, width, label=f'{threshold} threshold', alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Crash Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Matched Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Matched Crashes per Year by Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_time_difference_histogram(temporal_df):
    """Plot histogram of time differences between images and crashes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    thresholds = ['5m', '10m', '25m']
    time_diff_labels = ['0', '1', '2', '3+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        threshold_data = temporal_df[temporal_df['threshold'] == threshold]
        
        percentages = []
        for label in time_diff_labels:
            match = threshold_data[threshold_data['time_diff_years'] == label]
            if len(match) > 0:
                percentages.append(match['percentage'].iloc[0])
            else:
                percentages.append(0)
        
        ax.bar(time_diff_labels, percentages, alpha=0.8, color=colors[i])
        ax.set_xlabel('Time Difference (Years)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{threshold} Threshold', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Distribution of Time Differences Between Images and Crashes', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_shared_crashes_comparison(shared_df):
    """Compare shared crash patterns across thresholds."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    thresholds = ['5m', '10m', '25m']
    categories = ['2 images', '3 images', '4+ images']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, threshold in enumerate(thresholds):
        threshold_data = shared_df[shared_df['threshold'] == threshold].iloc[0]
        counts = [
            threshold_data['crashes_shared_by_2_images'],
            threshold_data['crashes_shared_by_3_images'],
            threshold_data['crashes_shared_by_4_plus_images']
        ]
        
        offset = (i - 1) * width
        ax.bar(x + offset, counts, width, label=f'{threshold} threshold', alpha=0.8)
    
    ax.set_xlabel('Sharing Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Shared Crash Patterns by Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_matching_rates(summary_df):
    """Plot match rates and related percentages across thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = summary_df['threshold'].values
    match_rates = summary_df['match_rate_pct'].values
    shared_rates = (summary_df['images_with_shared_crashes'] / summary_df['total_images'] * 100).values
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, match_rates, width, label='Images with crashes', alpha=0.8)
    bars2 = ax.bar(x + width/2, shared_rates, width, label='Images with shared crashes', alpha=0.8)
    
    ax.set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Matching Rates Comparison Across Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_crash_count_statistics(summary_df):
    """Plot average, median, max crashes per image across thresholds."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    thresholds = summary_df['threshold'].values
    x = np.arange(len(thresholds))
    width = 0.25
    
    avg_crashes = summary_df['avg_crashes_per_matched_image'].values
    median_crashes = summary_df['median_crashes_per_matched_image'].values
    max_crashes = summary_df['max_crashes_per_image'].values
    
    ax.bar(x - width, avg_crashes, width, label='Average', alpha=0.8)
    ax.bar(x, median_crashes, width, label='Median', alpha=0.8)
    ax.bar(x + width, max_crashes, width, label='Maximum', alpha=0.8)
    
    ax.set_xlabel('Distance Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Crash Count Statistics per Matched Image by Threshold', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_same_year_matches(year_df):
    """Plot crashes that occurred in same year as image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    thresholds = ['5m', '10m', '25m']
    years = sorted(year_df['crash_year'].unique())
    
    x = np.arange(len(years))
    width = 0.25
    
    for i, threshold in enumerate(thresholds):
        threshold_data = year_df[year_df['threshold'] == threshold]
        same_year_counts = []
        for year in years:
            match = threshold_data[threshold_data['crash_year'] == year]
            if len(match) > 0:
                same_year_counts.append(match['crashes_same_year_as_image'].iloc[0])
            else:
                same_year_counts.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, same_year_counts, width, label=f'{threshold} threshold', alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Crashes', fontsize=12, fontweight='bold')
    ax.set_title('Crashes Occurring in Same Year as Image Capture', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_dashboard(summary_df, shared_df, dist_df, year_df, temporal_df):
    """Create multi-panel dashboard with key metrics."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Matching rates
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = summary_df['threshold'].values
    match_rates = summary_df['match_rate_pct'].values
    ax1.bar(thresholds, match_rates, alpha=0.7, color='steelblue')
    ax1.set_ylabel('Match Rate (%)', fontweight='bold')
    ax1.set_title('Match Rates', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Crash count statistics
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(thresholds))
    width = 0.25
    ax2.bar(x - width, summary_df['avg_crashes_per_matched_image'], width, label='Avg', alpha=0.7)
    ax2.bar(x, summary_df['median_crashes_per_matched_image'], width, label='Median', alpha=0.7)
    ax2.bar(x + width, summary_df['max_crashes_per_image'], width, label='Max', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(thresholds)
    ax2.set_ylabel('Crashes per Image', fontweight='bold')
    ax2.set_title('Crash Count Statistics', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Shared crashes
    ax3 = fig.add_subplot(gs[0, 2])
    shared_totals = shared_df['total_shared_crashes'].values
    ax3.bar(thresholds, shared_totals, alpha=0.7, color='coral')
    ax3.set_ylabel('Shared Crashes', fontweight='bold')
    ax3.set_title('Total Shared Crash Relationships', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Temporal - same year
    ax4 = fig.add_subplot(gs[1, 0])
    same_year = summary_df['crashes_same_year_as_image'].values
    ax4.bar(thresholds, same_year, alpha=0.7, color='green')
    ax4.set_ylabel('Number of Crashes', fontweight='bold')
    ax4.set_title('Crashes Same Year as Image', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Time differences
    ax5 = fig.add_subplot(gs[1, 1])
    avg_time = summary_df['avg_time_diff_years'].fillna(0).values
    ax5.bar(thresholds, avg_time, alpha=0.7, color='purple')
    ax5.set_ylabel('Average Years', fontweight='bold')
    ax5.set_title('Average Time Difference', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Matched crashes per year (line plot)
    ax6 = fig.add_subplot(gs[1, 2])
    years = sorted(year_df['crash_year'].unique())
    for threshold in thresholds:
        threshold_data = year_df[year_df['threshold'] == threshold]
        counts = [threshold_data[threshold_data['crash_year'] == y]['matched_crashes_count'].iloc[0] 
                 if len(threshold_data[threshold_data['crash_year'] == y]) > 0 else 0 for y in years]
        ax6.plot(years, counts, marker='o', label=threshold, linewidth=2)
    ax6.set_xlabel('Year', fontweight='bold')
    ax6.set_ylabel('Matched Crashes', fontweight='bold')
    ax6.set_title('Matched Crashes Over Time', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # 7. Crash count distribution (top 5 counts)
    ax7 = fig.add_subplot(gs[2, :2])
    threshold_5m = dist_df[dist_df['threshold'] == '5m'].head(5)
    threshold_10m = dist_df[dist_df['threshold'] == '10m'].head(5)
    threshold_25m = dist_df[dist_df['threshold'] == '25m'].head(5)
    
    x = np.arange(5)
    width = 0.25
    ax7.bar(x - width, threshold_5m['percentage'].values, width, label='5m', alpha=0.7)
    ax7.bar(x, threshold_10m['percentage'].values, width, label='10m', alpha=0.7)
    ax7.bar(x + width, threshold_25m['percentage'].values, width, label='25m', alpha=0.7)
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"{int(c)}" for c in threshold_5m['crash_count'].values])
    ax7.set_xlabel('Crash Count per Image', fontweight='bold')
    ax7.set_ylabel('Percentage (%)', fontweight='bold')
    ax7.set_title('Top 5 Crash Count Distributions', fontweight='bold')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Summary text
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""SUMMARY (All Thresholds)
    
Total Images: {summary_df['total_images'].iloc[0]}
Total Matches: {summary_df['total_crash_matches'].sum()}
Unique Crashes: {summary_df['unique_crash_ids_matched'].sum()}

Match Rates:
  5m: {summary_df[summary_df['threshold']=='5m']['match_rate_pct'].iloc[0]:.1f}%
  10m: {summary_df[summary_df['threshold']=='10m']['match_rate_pct'].iloc[0]:.1f}%
  25m: {summary_df[summary_df['threshold']=='25m']['match_rate_pct'].iloc[0]:.1f}%

Shared Crashes:
  5m: {shared_df[shared_df['threshold']=='5m']['total_shared_crashes'].iloc[0]}
  10m: {shared_df[shared_df['threshold']=='10m']['total_shared_crashes'].iloc[0]}
  25m: {shared_df[shared_df['threshold']=='25m']['total_shared_crashes'].iloc[0]}
"""
    ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax8.transAxes)
    
    fig.suptitle('Crash-to-Image Matching Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def generate_all_visualizations(summary_df, shared_df, dist_df, year_df, temporal_df):
    """Generate all visualization files."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = project_root / 'reports' / 'Matches' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating crash count distribution comparison...")
    fig1 = plot_crash_count_distribution(dist_df)
    fig1.savefig(output_dir / 'crash_count_distribution_comparison.png', bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Saved crash_count_distribution_comparison.png")
    
    print("\nCreating matched crashes per year timeseries...")
    fig2 = plot_matched_crashes_per_year(year_df)
    fig2.savefig(output_dir / 'matched_crashes_per_year_timeseries.png', bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ Saved matched_crashes_per_year_timeseries.png")
    
    print("\nCreating time difference histogram...")
    fig3 = plot_time_difference_histogram(temporal_df)
    fig3.savefig(output_dir / 'time_difference_histogram.png', bbox_inches='tight')
    plt.close(fig3)
    print("  ✓ Saved time_difference_histogram.png")
    
    print("\nCreating shared crashes comparison...")
    fig4 = plot_shared_crashes_comparison(shared_df)
    fig4.savefig(output_dir / 'shared_crashes_comparison.png', bbox_inches='tight')
    plt.close(fig4)
    print("  ✓ Saved shared_crashes_comparison.png")
    
    print("\nCreating matching rates comparison...")
    fig5 = plot_matching_rates(summary_df)
    fig5.savefig(output_dir / 'matching_rates_comparison.png', bbox_inches='tight')
    plt.close(fig5)
    print("  ✓ Saved matching_rates_comparison.png")
    
    print("\nCreating crash count statistics comparison...")
    fig6 = plot_crash_count_statistics(summary_df)
    fig6.savefig(output_dir / 'crash_count_statistics_comparison.png', bbox_inches='tight')
    plt.close(fig6)
    print("  ✓ Saved crash_count_statistics_comparison.png")
    
    print("\nCreating same year matches by year...")
    fig7 = plot_same_year_matches(year_df)
    fig7.savefig(output_dir / 'same_year_matches_by_year.png', bbox_inches='tight')
    plt.close(fig7)
    print("  ✓ Saved same_year_matches_by_year.png")
    
    print("\nCreating summary dashboard...")
    fig8 = create_summary_dashboard(summary_df, shared_df, dist_df, year_df, temporal_df)
    fig8.savefig(output_dir / 'summary_dashboard.png', bbox_inches='tight')
    plt.close(fig8)
    print("  ✓ Saved summary_dashboard.png")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ANALYZE MATCH RESULTS")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    df_5m = pd.read_csv(project_root / 'data' / 'processed' / 'mapillary_labeled_5m.csv', low_memory=False)
    df_10m = pd.read_csv(project_root / 'data' / 'processed' / 'mapillary_labeled_10m.csv', low_memory=False)
    df_25m = pd.read_csv(project_root / 'data' / 'processed' / 'mapillary_labeled_25m.csv', low_memory=False)
    print(f"Loaded {len(df_5m)} images (5m), {len(df_10m)} images (10m), {len(df_25m)} images (25m)")
    
    # Generate all CSV summaries
    summary_df = generate_summary_statistics(df_5m, df_10m, df_25m)
    shared_df = generate_shared_crashes_statistics(df_5m, df_10m, df_25m)
    dist_df = generate_crash_count_distributions(df_5m, df_10m, df_25m)
    year_df = generate_matched_crashes_per_year(df_5m, df_10m, df_25m)
    temporal_df = generate_temporal_distribution(df_5m, df_10m, df_25m)
    
    # Generate all visualizations
    generate_all_visualizations(summary_df, shared_df, dist_df, year_df, temporal_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to reports/Matches/:")
    print("  CSV Files:")
    print("    - matching_summary_statistics.csv")
    print("    - shared_crashes_statistics.csv")
    print("    - crash_count_distributions.csv")
    print("    - matched_crashes_per_year.csv")
    print("    - temporal_distribution.csv")
    print("  Figures:")
    print("    - figures/crash_count_distribution_comparison.png")
    print("    - figures/matched_crashes_per_year_timeseries.png")
    print("    - figures/time_difference_histogram.png")
    print("    - figures/shared_crashes_comparison.png")
    print("    - figures/matching_rates_comparison.png")
    print("    - figures/crash_count_statistics_comparison.png")
    print("    - figures/same_year_matches_by_year.png")
    print("    - figures/summary_dashboard.png")
    
    # Print preview
    print("\n" + "=" * 80)
    print("PREVIEW: Summary Statistics")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

