#!/usr/bin/env python3
"""
RDD2022 Dataset to YOLO Format Converter

This script downloads the RDD2022 road damage dataset from Kaggle and converts it
to YOLO format for training YOLOv8 models. It consolidates crack types into a
single 'crack' class and filters out drone images to focus on street-level data.

Key Features:
- Downloads dataset using kagglehub
- Parses XML annotations and converts to YOLO format
- Consolidates crack types: longitudinal, transverse, alligator → 'crack'
- Filters out drone images (filename contains 'drone' or 'Drone')
- Uses existing train/val split, merges test set into training
- Creates proper directory structure for YOLO training

Usage:
    python convert_rdd_to_yolo.py

Output:
    Creates data/rdd_yolo/ directory with images/ and labels/ subdirectories
    containing train/ and val/ splits in YOLO format.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
import kagglehub
from PIL import Image


def download_rdd_dataset():
    """
    Download the RDD2022 dataset from Kaggle using kagglehub.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    print("📥 Downloading RDD2022 dataset from Kaggle...")
    path = kagglehub.dataset_download("aliabdelmenam/rdd-2022")
    print(f"✅ Dataset downloaded to: {path}")
    return path


def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file and extract bounding box information.
    
    Args:
        xml_path (str): Path to the XML annotation file
        
    Returns:
        list: List of tuples (class_name, bbox) where bbox is (xmin, ymin, xmax, ymax)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            objects.append((class_name, (xmin, ymin, xmax, ymax)))
        
        return objects
    except Exception as e:
        print(f"⚠️  Error parsing {xml_path}: {e}")
        return []


def convert_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box from (xmin, ymin, xmax, ymax) to YOLO format (cx, cy, w, h).
    
    Args:
        bbox (tuple): Bounding box in format (xmin, ymin, xmax, ymax)
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        tuple: Normalized YOLO format (cx, cy, w, h)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center coordinates and dimensions
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Normalize by image dimensions
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height
    
    return (cx, cy, w, h)


def consolidate_classes(class_name):
    """
    Consolidate crack types into a single 'crack' class.
    
    Args:
        class_name (str): Original class name
        
    Returns:
        str: Consolidated class name or None if class should be ignored
    """
    # Class mapping: consolidate all crack types
    class_map = {
        'longitudinal crack': 'crack',
        'transverse crack': 'crack', 
        'alligator crack': 'crack',
        'other corruption': 'other corruption',
        'pothole': 'Pothole'
    }
    
    return class_map.get(class_name)


def is_drone_image(filename):
    """
    Check if image filename indicates it's from a drone.
    
    Args:
        filename (str): Image filename
        
    Returns:
        bool: True if image appears to be from drone
    """
    return 'drone' in filename.lower()


def process_dataset_split(dataset_path, split_name, output_dir):
    """
    Process a dataset split (train/val/test) and convert to YOLO format.
    
    Args:
        dataset_path (str): Path to the dataset split directory
        split_name (str): Name of the split ('train', 'val', 'test')
        output_dir (str): Output directory for YOLO format files
        
    Returns:
        tuple: (num_images_processed, num_images_skipped)
    """
    print(f"🔄 Processing {split_name} split...")
    
    images_dir = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_dir):
        print(f"⚠️  {images_dir} not found, skipping...")
        return 0, 0
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, 'images', split_name)
    output_labels_dir = os.path.join(output_dir, 'labels', split_name)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    processed = 0
    skipped = 0
    
    # Process all images in the split
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Skip drone images
        if is_drone_image(img_file):
            skipped += 1
            continue
            
        img_path = os.path.join(images_dir, img_file)
        
        # Get corresponding label file (RDD2022 already has YOLO format .txt files)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(dataset_path, 'labels', label_file)
        
        # Copy image to output directory
        output_img_path = os.path.join(output_images_dir, img_file)
        shutil.copy2(img_path, output_img_path)
        
        # Process annotations if label file exists
        if os.path.exists(label_path):
            # Read existing YOLO annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Process and consolidate annotations
            yolo_annotations = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_idx = int(parts[0])
                    bbox_coords = parts[1:5]
                    
                    # Map old class indices to new consolidated classes
                    # RDD2022 original classes: 0=longitudinal crack, 1=transverse crack, 2=alligator crack, 3=other corruption, 4=Pothole
                    class_mapping = {
                        0: 0,  # longitudinal crack -> crack
                        1: 0,  # transverse crack -> crack  
                        2: 0,  # alligator crack -> crack
                        3: 1,  # other corruption -> other corruption
                        4: 2   # Pothole -> Pothole
                    }
                    
                    if old_class_idx in class_mapping:
                        new_class_idx = class_mapping[old_class_idx]
                        yolo_annotations.append(f"{new_class_idx} {' '.join(bbox_coords)}")
            
            # Write consolidated YOLO annotation file
            output_label_path = os.path.join(output_labels_dir, label_file)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        processed += 1
        
        if processed % 100 == 0:
            print(f"  Processed {processed} images...")
    
    print(f"✅ {split_name}: {processed} images processed, {skipped} drone images skipped")
    return processed, skipped


def main():
    """
    Main function to download RDD2022 dataset and convert to YOLO format.
    """
    print("🚀 Starting RDD2022 to YOLO conversion...")
    
    # Download dataset
    dataset_path = download_rdd_dataset()
    
    # Create output directory
    output_dir = "data/rdd_yolo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    total_processed = 0
    total_skipped = 0
    
    # Process train and val splits as-is (dataset is in RDD_SPLIT subdirectory)
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_path, 'RDD_SPLIT', split)
        processed, skipped = process_dataset_split(split_path, split, output_dir)
        total_processed += processed
        total_skipped += skipped
    
    # Process test split as separate test set
    test_path = os.path.join(dataset_path, 'RDD_SPLIT', 'test')
    if os.path.exists(test_path):
        print("🔄 Processing test split...")
        processed, skipped = process_dataset_split(test_path, 'test', output_dir)
        total_processed += processed
        total_skipped += skipped
    
    # Create data.yaml configuration file
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write("""path: data/rdd_yolo
train: images/train
val: images/val
test: images/test
names:
  0: crack
  1: other corruption
  2: Pothole
""")
    
    print(f"\n🎉 Conversion complete!")
    print(f"📊 Total images processed: {total_processed}")
    print(f"🚁 Drone images skipped: {total_skipped}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Configuration file: {data_yaml_path}")
    print(f"🏷️  Classes: crack, other corruption, Pothole")


if __name__ == "__main__":
    main()
