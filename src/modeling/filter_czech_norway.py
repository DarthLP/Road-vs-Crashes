#!/usr/bin/env python3
"""
Czech + Norway Dataset Filter

This script filters the RDD2022 dataset to keep only Czech Republic and Norway data,
removing all other countries to create a smaller, more manageable dataset.

Key Features:
- Keeps only Czech and Norway images and labels
- Removes all other country data to save space
- Creates a new filtered dataset directory
- Updates data.yaml configuration
- Provides size comparison before/after

Usage:
    python filter_czech_norway.py

Output:
    Creates data/rdd_yolo_cz_no/ directory with filtered dataset
    Updates data.yaml for the new dataset
"""

import os
import shutil
from pathlib import Path


def filter_dataset_by_countries(source_dir, target_dir, countries=['Czech', 'Norway']):
    """
    Filter dataset to keep only specified countries.
    
    Args:
        source_dir (str): Source dataset directory
        target_dir (str): Target directory for filtered dataset
        countries (list): List of country prefixes to keep
    """
    print(f"🔄 Filtering dataset to keep only: {', '.join(countries)}")
    
    # Create target directory structure
    os.makedirs(os.path.join(target_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'test'), exist_ok=True)
    
    total_kept = 0
    total_removed = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Processing {split} split...")
        
        source_images_dir = os.path.join(source_dir, 'images', split)
        source_labels_dir = os.path.join(source_dir, 'labels', split)
        target_images_dir = os.path.join(target_dir, 'images', split)
        target_labels_dir = os.path.join(target_dir, 'labels', split)
        
        if not os.path.exists(source_images_dir):
            print(f"⚠️  {source_images_dir} not found, skipping...")
            continue
        
        kept_count = 0
        removed_count = 0
        
        # Process all images in the split
        for img_file in os.listdir(source_images_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Check if image is from target countries
            is_target_country = any(img_file.startswith(country) for country in countries)
            
            if is_target_country:
                # Copy image
                source_img_path = os.path.join(source_images_dir, img_file)
                target_img_path = os.path.join(target_images_dir, img_file)
                shutil.copy2(source_img_path, target_img_path)
                
                # Copy corresponding label file
                label_file = os.path.splitext(img_file)[0] + '.txt'
                source_label_path = os.path.join(source_labels_dir, label_file)
                target_label_path = os.path.join(target_labels_dir, label_file)
                
                if os.path.exists(source_label_path):
                    shutil.copy2(source_label_path, target_label_path)
                
                kept_count += 1
            else:
                removed_count += 1
        
        print(f"  ✅ {split}: {kept_count} images kept, {removed_count} images removed")
        total_kept += kept_count
        total_removed += removed_count
    
    print(f"\n📊 Total: {total_kept} images kept, {total_removed} images removed")
    return total_kept, total_removed


def create_filtered_data_yaml(target_dir):
    """
    Create data.yaml configuration for the filtered dataset.
    
    Args:
        target_dir (str): Target dataset directory
    """
    data_yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write("""path: data/rdd_yolo_cz_no
train: images/train
val: images/val
test: images/test
names:
  0: crack
  1: other corruption
  2: Pothole
""")
    
    print(f"📝 Created configuration file: {data_yaml_path}")


def get_directory_size(path):
    """
    Get the size of a directory in GB.
    
    Args:
        path (str): Directory path
        
    Returns:
        float: Size in GB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert to GB


def main():
    """
    Main function to filter dataset to Czech Republic and Norway only.
    """
    print("🚀 Starting Czech + Norway dataset filtering...")
    
    source_dir = "data/rdd_yolo"
    target_dir = "data/rdd_yolo_cz_no"
    
    # Check if source dataset exists
    if not os.path.exists(source_dir):
        print(f"❌ Source dataset not found: {source_dir}")
        print("   Please run convert_rdd_to_yolo.py first")
        return
    
    # Get original size
    original_size = get_directory_size(source_dir)
    print(f"📏 Original dataset size: {original_size:.2f} GB")
    
    # Filter dataset
    kept_count, removed_count = filter_dataset_by_countries(source_dir, target_dir)
    
    # Create configuration file
    create_filtered_data_yaml(target_dir)
    
    # Get filtered size
    filtered_size = get_directory_size(target_dir)
    size_reduction = ((original_size - filtered_size) / original_size) * 100
    
    print(f"\n🎉 Filtering complete!")
    print(f"📊 Results:")
    print(f"   Images kept: {kept_count}")
    print(f"   Images removed: {removed_count}")
    print(f"   Size reduction: {size_reduction:.1f}%")
    print(f"   New dataset size: {filtered_size:.2f} GB")
    print(f"📁 Filtered dataset: {target_dir}")
    print(f"⚙️  Configuration: {target_dir}/data.yaml")
    
    print(f"\n🚀 Ready for training!")
    print(f"   Training command: python src/modeling/train_yolo_cz_no.py")
    print(f"   Expected training time: ~20-30 minutes on Colab GPU")


if __name__ == "__main__":
    main()
