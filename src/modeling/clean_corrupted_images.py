#!/usr/bin/env python3
"""
Clean Corrupted Images Script

This script removes label files that don't have corresponding images due to
corrupted image files that failed during compression.

Key Features:
- Identifies orphaned label files (labels without corresponding images)
- Removes orphaned label files to maintain dataset consistency
- Provides statistics on cleaned dataset
- Works across all splits (train/val/test)

Usage:
    python src/modeling/clean_corrupted_images.py

Prerequisites:
    - Compressed dataset must exist at data/rdd_yolo_cz_no_compressed/
    - Images and labels directories must exist

Output:
    - Removes orphaned label files
    - Prints cleaning statistics
"""

import os
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

DATA_DIR = "data/rdd_yolo_cz_no_compressed"
SPLITS = ["train", "val", "test"]


# =============================================================================
# FUNCTIONS
# =============================================================================

def clean_orphaned_labels(split):
    """
    Remove label files that don't have corresponding images.
    
    Args:
        split (str): Dataset split name ('train', 'val', 'test')
        
    Returns:
        tuple: (removed_count, remaining_count)
    """
    images_dir = Path(DATA_DIR) / 'images' / split
    labels_dir = Path(DATA_DIR) / 'labels' / split
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️  {split} directories not found")
        return 0, 0
    
    # Get all image files (without extension for matching)
    image_files = set()
    for img_file in images_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_files.add(img_file.stem)
    
    # Get all label files
    label_files = list(labels_dir.glob('*.txt'))
    
    removed_count = 0
    remaining_count = 0
    
    print(f"\\n📁 Cleaning {split} split...")
    print(f"   Images: {len(image_files)}")
    print(f"   Labels before: {len(label_files)}")
    
    # Remove orphaned label files
    for label_file in tqdm(label_files, desc=f"  {split}"):
        label_stem = label_file.stem
        
        if label_stem in image_files:
            remaining_count += 1
        else:
            # Remove orphaned label file
            label_file.unlink()
            removed_count += 1
    
    print(f"   Labels removed: {removed_count}")
    print(f"   Labels remaining: {remaining_count}")
    
    return removed_count, remaining_count


def clean_all_splits():
    """
    Clean orphaned labels from all dataset splits.
    
    Returns:
        dict: Cleaning statistics
    """
    print("🧹 Cleaning corrupted images and orphaned labels...")
    
    stats = {
        'total_removed': 0,
        'total_remaining': 0,
        'split_stats': {}
    }
    
    for split in SPLITS:
        removed, remaining = clean_orphaned_labels(split)
        
        stats['split_stats'][split] = {
            'removed': removed,
            'remaining': remaining
        }
        stats['total_removed'] += removed
        stats['total_remaining'] += remaining
    
    return stats


def print_cleaning_summary(stats):
    """
    Print a summary of the cleaning process.
    
    Args:
        stats (dict): Cleaning statistics
    """
    print("\\n" + "="*60)
    print("🧹 CLEANING SUMMARY")
    print("="*60)
    
    print(f"\\n📁 Total labels removed: {stats['total_removed']}")
    print(f"📁 Total labels remaining: {stats['total_remaining']}")
    
    print("\\n📂 Per-split statistics:")
    for split, split_stats in stats['split_stats'].items():
        print(f"\\n  {split.upper()}:")
        print(f"    Removed: {split_stats['removed']}")
        print(f"    Remaining: {split_stats['remaining']}")
    
    print("\\n✅ Dataset cleaned and consistent!")
    print("\\n📊 Final dataset stats:")
    print(f"   Total images: {stats['total_remaining']}")
    print(f"   Train: {stats['split_stats']['train']['remaining']}")
    print(f"   Val: {stats['split_stats']['val']['remaining']}")
    print(f"   Test: {stats['split_stats']['test']['remaining']}")


def main():
    """
    Main function to clean corrupted images and orphaned labels.
    """
    print("🚀 Starting dataset cleaning...")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        print("   Please run the compression script first")
        return
    
    # Clean all splits
    stats = clean_all_splits()
    
    # Print summary
    print_cleaning_summary(stats)
    
    print("\\n" + "="*60)
    print("✅ Cleaning complete!")
    print("="*60)
    print("\\n🚀 Dataset is now ready for training!")
    print("   Run: python src/modeling/train_yolo.py")


if __name__ == "__main__":
    main()
