#!/usr/bin/env python3
"""
Image Compression Script for Czech + Norway Dataset

This script compresses images larger than 1024x768 pixels to match Mapillary quality,
keeping smaller images at their original resolution to avoid unnecessary upscaling.

Key Features:
- Only resizes images larger than 1024x768 pixels
- Keeps smaller images at original resolution (no upscaling)
- Compresses with quality=85 (good balance of size/quality)
- Maintains aspect ratio with padding if needed
- Provides before/after size comparison

Usage:
    python compress_cz_no_dataset.py

Output:
    Creates compressed dataset in data/rdd_yolo_cz_no_compressed/
    Updates data.yaml configuration
"""

import os
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import argparse


def resize_and_compress_image(input_path, output_path, target_size=(1024, 768), quality=85):
    """
    Resize and compress an image only if it's larger than target dimensions.
    Keeps smaller images at their original resolution.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save compressed image
        target_size (tuple): Target (width, height) dimensions
        quality (int): JPEG compression quality (1-100)
        
    Returns:
        tuple: (original_size_bytes, compressed_size_bytes, was_resized)
    """
    try:
        # Open and process image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check if image is larger than target size
            img_width, img_height = img.size
            target_width, target_height = target_size
            
            # Only resize if image is larger than target dimensions
            if img_width > target_width or img_height > target_height:
                # Resize with aspect ratio preservation
                img_processed = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                was_resized = True
            else:
                # Keep original size
                img_processed = img
                was_resized = False
            
            # Save with compression
            img_processed.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            # Get file sizes
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            
            return original_size, compressed_size, was_resized
            
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return None, None, False


def compress_dataset(source_dir, target_dir, target_size=(1024, 768), quality=85):
    """
    Compress the Czech + Norway dataset by resizing Norway images.
    
    Args:
        source_dir (str): Source dataset directory
        target_dir (str): Target directory for compressed dataset
        target_size (tuple): Target image dimensions
        quality (int): JPEG compression quality
        
    Returns:
        dict: Compression statistics
    """
    print(f"🔄 Compressing dataset...")
    print(f"   Target size: {target_size[0]}x{target_size[1]} pixels")
    print(f"   Quality: {quality}")
    
    # Create target directory structure
    os.makedirs(os.path.join(target_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'val'), exist_ok=True)
    
    stats = {
        'czech_images': 0,
        'norway_images': 0,
        'czech_resized': 0,
        'norway_resized': 0,
        'total_original_size': 0,
        'total_compressed_size': 0,
        'czech_original_size': 0,
        'czech_compressed_size': 0,
        'norway_original_size': 0,
        'norway_compressed_size': 0
    }
    
    # Process each split
    for split in ['train', 'val']:
        print(f"\n📁 Processing {split} split...")
        
        source_images_dir = os.path.join(source_dir, 'images', split)
        source_labels_dir = os.path.join(source_dir, 'labels', split)
        target_images_dir = os.path.join(target_dir, 'images', split)
        target_labels_dir = os.path.join(target_dir, 'labels', split)
        
        if not os.path.exists(source_images_dir):
            print(f"⚠️  {source_images_dir} not found, skipping...")
            continue
        
        czech_count = 0
        norway_count = 0
        
        # Process all images in the split
        for img_file in os.listdir(source_images_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            source_img_path = os.path.join(source_images_dir, img_file)
            target_img_path = os.path.join(target_images_dir, img_file)
            
            # Copy corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            source_label_path = os.path.join(source_labels_dir, label_file)
            target_label_path = os.path.join(target_labels_dir, label_file)
            
            if os.path.exists(source_label_path):
                shutil.copy2(source_label_path, target_label_path)
            
            # Process ALL images (both Czech and Norway) with compression
            if img_file.startswith('Czech'):
                # Czech images: compress only if larger than target
                original_size, compressed_size, was_resized = resize_and_compress_image(
                    source_img_path, target_img_path, target_size, quality
                )
                if original_size and compressed_size:
                    stats['czech_images'] += 1
                    if was_resized:
                        stats['czech_resized'] += 1
                    stats['czech_original_size'] += original_size
                    stats['czech_compressed_size'] += compressed_size
                    czech_count += 1
                    
                    if czech_count % 100 == 0:
                        print(f"  Processed {czech_count} Czech images...")
                
            elif img_file.startswith('Norway'):
                # Norway images: compress only if larger than target
                original_size, compressed_size, was_resized = resize_and_compress_image(
                    source_img_path, target_img_path, target_size, quality
                )
                if original_size and compressed_size:
                    stats['norway_images'] += 1
                    if was_resized:
                        stats['norway_resized'] += 1
                    stats['norway_original_size'] += original_size
                    stats['norway_compressed_size'] += compressed_size
                    norway_count += 1
                    
                    if norway_count % 100 == 0:
                        print(f"  Processed {norway_count} Norway images...")
        
        print(f"  ✅ {split}: {czech_count} Czech processed, {norway_count} Norway processed")
    
    # Calculate totals
    stats['total_original_size'] = stats['czech_original_size'] + stats['norway_original_size']
    stats['total_compressed_size'] = stats['czech_compressed_size'] + stats['norway_compressed_size']
    
    return stats


def create_compressed_data_yaml(target_dir):
    """
    Create data.yaml configuration for the compressed dataset.
    
    Args:
        target_dir (str): Target dataset directory
    """
    data_yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write("""path: data/rdd_yolo_cz_no_compressed
train: images/train
val: images/val
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


def print_compression_stats(stats):
    """
    Print detailed compression statistics.
    
    Args:
        stats (dict): Compression statistics dictionary
    """
    print("\n" + "="*60)
    print("📊 COMPRESSION STATISTICS")
    print("="*60)
    
    # Czech images (copied)
    czech_reduction = ((stats['czech_original_size'] - stats['czech_compressed_size']) / 
                       stats['czech_original_size'] * 100) if stats['czech_original_size'] > 0 else 0
    
    print(f"🇨🇿 Czech Republic:")
    print(f"   Images: {stats['czech_images']} ({stats['czech_resized']} resized)")
    print(f"   Original size: {stats['czech_original_size'] / (1024**3):.2f} GB")
    print(f"   Compressed size: {stats['czech_compressed_size'] / (1024**3):.2f} GB")
    print(f"   Size reduction: {czech_reduction:.1f}%")
    
    # Norway images (compressed)
    norway_reduction = ((stats['norway_original_size'] - stats['norway_compressed_size']) / 
                       stats['norway_original_size'] * 100) if stats['norway_original_size'] > 0 else 0
    
    print(f"\n🇳🇴 Norway:")
    print(f"   Images: {stats['norway_images']} ({stats['norway_resized']} resized)")
    print(f"   Original size: {stats['norway_original_size'] / (1024**3):.2f} GB")
    print(f"   Compressed size: {stats['norway_compressed_size'] / (1024**3):.2f} GB")
    print(f"   Size reduction: {norway_reduction:.1f}%")
    
    # Total statistics
    total_reduction = ((stats['total_original_size'] - stats['total_compressed_size']) / 
                      stats['total_original_size'] * 100) if stats['total_original_size'] > 0 else 0
    
    print(f"\n📈 Total Dataset:")
    print(f"   Images: {stats['czech_images'] + stats['norway_images']}")
    print(f"   Original size: {stats['total_original_size'] / (1024**3):.2f} GB")
    print(f"   Compressed size: {stats['total_compressed_size'] / (1024**3):.2f} GB")
    print(f"   Total reduction: {total_reduction:.1f}%")


def main():
    """
    Main function to compress the Czech + Norway dataset.
    """
    print("🚀 Starting Czech + Norway dataset compression...")
    
    source_dir = "data/rdd_yolo_cz_no"
    target_dir = "data/rdd_yolo_cz_no_compressed"
    
    # Check if source dataset exists
    if not os.path.exists(source_dir):
        print(f"❌ Source dataset not found: {source_dir}")
        print("   Please run filter_czech_norway.py first")
        return
    
    # Get original size
    original_size = get_directory_size(source_dir)
    print(f"📏 Original dataset size: {original_size:.2f} GB")
    
    # Compress dataset
    stats = compress_dataset(source_dir, target_dir)
    
    # Create configuration file
    create_compressed_data_yaml(target_dir)
    
    # Get compressed size
    compressed_size = get_directory_size(target_dir)
    
    # Print statistics
    print_compression_stats(stats)
    
    print(f"\n🎉 Compression complete!")
    print(f"📁 Compressed dataset: {target_dir}")
    print(f"⚙️  Configuration: {target_dir}/data.yaml")
    
    print(f"\n🚀 Ready for training!")
    print(f"   Training command: python src/modeling/train_yolo_cz_no_compressed.py")
    print(f"   Expected training time: ~15-20 minutes on Colab GPU")


if __name__ == "__main__":
    main()
