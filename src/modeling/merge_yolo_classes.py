#!/usr/bin/env python3
"""
YOLO Class Merging Script

This script merges all YOLO label classes (crack, other corruption, Pothole) into
a single "road_damage" class by converting all class IDs (0, 1, 2) to 0 in all
label files across train/val/test splits.

Key Features:
- Processes all label files in train/val/test directories
- Merges classes 0, 1, 2 → 0 (road_damage)
- Preserves bounding box coordinates
- Creates backup of original labels before modification
- Provides progress feedback during processing

Usage:
    python src/modeling/merge_yolo_classes.py

Prerequisites:
    - YOLO dataset must exist at data/rdd_yolo_cz_no_compressed/
    - Label files must be in YOLO format (class_id x y w h)

Output:
    - All label files updated with merged class IDs
    - Backup of original labels saved to labels_backup/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

DATA_DIR = "data/rdd_yolo_cz_no_compressed"
LABELS_DIR = os.path.join(DATA_DIR, "labels")
BACKUP_DIR = os.path.join(DATA_DIR, "labels_backup")
SPLITS = ["train", "val", "test"]


# =============================================================================
# FUNCTIONS
# =============================================================================

def backup_labels():
    """
    Create a backup of the original label files before modification.
    
    Returns:
        bool: True if backup successful, False otherwise
    """
    print("📦 Creating backup of original labels...")
    
    if not os.path.exists(LABELS_DIR):
        print(f"❌ Labels directory not found: {LABELS_DIR}")
        return False
    
    # Remove existing backup if it exists
    if os.path.exists(BACKUP_DIR):
        print(f"⚠️  Removing existing backup: {BACKUP_DIR}")
        shutil.rmtree(BACKUP_DIR)
    
    # Create backup
    try:
        shutil.copytree(LABELS_DIR, BACKUP_DIR)
        print(f"✅ Backup created at: {BACKUP_DIR}")
        return True
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False


def merge_classes_in_file(label_file_path):
    """
    Merge all class IDs (0, 1, 2) to 0 in a single label file.
    
    Args:
        label_file_path (str): Path to the label file to process
        
    Returns:
        tuple: (num_annotations, num_merged) - Number of annotations processed and merged
    """
    try:
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        
        merged_lines = []
        num_annotations = 0
        num_merged = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            bbox_coords = parts[1:5]
            
            num_annotations += 1
            
            # Merge all classes (0, 1, 2) to 0 (road_damage)
            if class_id in [0, 1, 2]:
                new_class_id = 0
                if class_id != 0:
                    num_merged += 1
            else:
                # Keep unknown class IDs as-is (shouldn't happen)
                new_class_id = class_id
            
            merged_lines.append(f"{new_class_id} {' '.join(bbox_coords)}")
        
        # Write merged annotations back to file
        with open(label_file_path, 'w') as f:
            f.write('\n'.join(merged_lines) + '\n')
        
        return num_annotations, num_merged
    
    except Exception as e:
        print(f"⚠️  Error processing {label_file_path}: {e}")
        return 0, 0


def merge_all_labels():
    """
    Process all label files across train/val/test splits to merge classes.
    
    Returns:
        dict: Statistics about the merging process
    """
    print("🔄 Merging classes in all label files...")
    
    stats = {
        'total_files': 0,
        'total_annotations': 0,
        'total_merged': 0,
        'split_stats': {}
    }
    
    for split in SPLITS:
        split_labels_dir = os.path.join(LABELS_DIR, split)
        
        if not os.path.exists(split_labels_dir):
            print(f"⚠️  Split directory not found: {split_labels_dir}")
            continue
        
        split_stats = {
            'files': 0,
            'annotations': 0,
            'merged': 0
        }
        
        # Get all label files in split
        label_files = list(Path(split_labels_dir).glob("*.txt"))
        
        if len(label_files) == 0:
            print(f"⚠️  No label files found in {split_labels_dir}")
            continue
        
        print(f"\n📂 Processing {split} split: {len(label_files)} label files")
        
        # Process each label file
        for label_file in tqdm(label_files, desc=f"  {split}"):
            num_annotations, num_merged = merge_classes_in_file(str(label_file))
            
            split_stats['files'] += 1
            split_stats['annotations'] += num_annotations
            split_stats['merged'] += num_merged
        
        stats['split_stats'][split] = split_stats
        stats['total_files'] += split_stats['files']
        stats['total_annotations'] += split_stats['annotations']
        stats['total_merged'] += split_stats['merged']
    
    return stats


def print_summary(stats):
    """
    Print a summary of the class merging process.
    
    Args:
        stats (dict): Statistics from merge_all_labels()
    """
    print("\n" + "="*60)
    print("📊 CLASS MERGING SUMMARY")
    print("="*60)
    
    print(f"\n📁 Total label files processed: {stats['total_files']}")
    print(f"🏷️  Total annotations: {stats['total_annotations']}")
    print(f"🔄 Total class IDs merged: {stats['total_merged']}")
    
    print("\n📂 Per-split statistics:")
    for split, split_stats in stats['split_stats'].items():
        print(f"\n  {split.upper()}:")
        print(f"    Files: {split_stats['files']}")
        print(f"    Annotations: {split_stats['annotations']}")
        print(f"    Merged: {split_stats['merged']}")
    
    print("\n✅ All classes successfully merged to: road_damage (class 0)")
    print(f"📦 Backup saved at: {BACKUP_DIR}")
    print("\n⚠️  IMPORTANT: Update data.yaml to reflect single class configuration")


def main():
    """
    Main function to merge YOLO classes across all label files.
    """
    print("🚀 Starting YOLO class merging...")
    print("="*60)
    
    # Verify labels directory exists
    if not os.path.exists(LABELS_DIR):
        print(f"❌ Labels directory not found: {LABELS_DIR}")
        print("   Please ensure the YOLO dataset exists at data/rdd_yolo_cz_no_compressed/")
        return
    
    # Create backup
    if not backup_labels():
        print("❌ Backup failed. Aborting to prevent data loss.")
        return
    
    # Merge classes in all label files
    stats = merge_all_labels()
    
    # Print summary
    print_summary(stats)
    
    print("\n" + "="*60)
    print("✅ Class merging complete!")
    print("="*60)
    print("\n📋 Next steps:")
    print("   1. Update data/rdd_yolo_cz_no_compressed/data.yaml")
    print("   2. Change 'names:' section to: 0: road_damage")
    print("   3. Run train_yolo.py to train with merged classes")


if __name__ == "__main__":
    main()

