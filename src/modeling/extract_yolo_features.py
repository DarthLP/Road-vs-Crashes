#!/usr/bin/env python3
"""
Script: Extract YOLO Street Quality Features with ROI Filtering

Purpose:
This script extracts robust road-focused damage metrics from YOLO detections for each cluster.
It uses a trapezoid ROI (bottom-centered) to reduce framing bias and computes per-cluster
aggregated metrics for street quality assessment.

Functionality:
- Loads train/val/test splits from clusters_*_with_residuals.csv
- Extracts first URL from list_of_thumb_1024_url per cluster
- Runs YOLO inference with ROI filtering (trapezoid mask)
- Computes per-image ROI metrics, aggregates per cluster
- Winsorizes metrics per split independently
- Saves enriched datasets with YOLO features

How to run:
    python src/modeling/extract_yolo_features.py

Prerequisites:
    - YOLO model weights at runs/detect/train/weights/best.pt
    - Run baseline_logistic_regression.py first to generate *_with_residuals.csv files
    - Ensure data/processed/clusters_{train,val,test}_with_residuals.csv exist

Output:
    - data/processed/clusters_{train,val,test}_with_yolo_roi.csv
    - Additional metrics: damage_conf_sum, damage_max_conf, damage_area_frac
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
import cv2
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# =============================================================================
# CONFIGURATION
# =============================================================================

# YOLO model path (configurable)
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"

# YOLO inference settings (recall-oriented for Mapillary)
YOLO_IMGSZ = 416
YOLO_CONF = 0.10  # Lower confidence for more detections
YOLO_IOU = 0.45

# ROI parameters (trapezoid for road focus)
ROI_BOTTOM_FRAC = 1.0   # Bottom edge of ROI (100% down from top)
ROI_TOP_FRAC = 0.4      # Top edge of ROI (40% down from top) - bottom 60%
ROI_HMARGIN_FRAC = 0.15 # Horizontal margin (15% from each side) - 70% width

# Dataset paths
DATA_DIR = project_root / 'data' / 'processed'
TRAIN_PATH = DATA_DIR / 'clusters_train_with_residuals.csv'
VAL_PATH = DATA_DIR / 'clusters_val_with_residuals.csv'
TEST_PATH = DATA_DIR / 'clusters_test_with_residuals.csv'

# Output paths
TRAIN_OUTPUT = DATA_DIR / 'clusters_train_with_yolo_roi.csv'
VAL_OUTPUT = DATA_DIR / 'clusters_val_with_yolo_roi.csv'
TEST_OUTPUT = DATA_DIR / 'clusters_test_with_yolo_roi.csv'


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


def load_image_from_url(url, timeout=3):
    """
    Load image from URL and convert to OpenCV format.
    
    Parameters:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        np.ndarray: Image in BGR format, or None if failed
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Convert to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    except Exception as e:
        print(f"  ⚠️ Failed to load {url[:50]}...: {e}")
        return None


def create_roi_mask(image_shape, bottom_frac, top_frac, hmargin_frac):
    """
    Create trapezoid ROI mask for road focus.
    
    Parameters:
        image_shape: (height, width) of image
        bottom_frac: Bottom edge position (0-1)
        top_frac: Top edge position (0-1)
        hmargin_frac: Horizontal margin (0-1)
        
    Returns:
        np.ndarray: Binary mask (1 = inside ROI, 0 = outside)
    """
    height, width = image_shape[:2]
    
    # Calculate pixel coordinates
    top_y = int(height * top_frac)
    bottom_y = int(height * bottom_frac)
    left_x = int(width * hmargin_frac)
    right_x = int(width * (1 - hmargin_frac))
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define trapezoid vertices (bottom wider than top)
    pts = np.array([
        [left_x, bottom_y],      # Bottom left
        [right_x, bottom_y],     # Bottom right
        [right_x, top_y],        # Top right
        [left_x, top_y]          # Top left
    ], np.int32)
    
    # Fill trapezoid
    cv2.fillPoly(mask, [pts], 1)
    
    return mask


def compute_roi_metrics(detections, roi_mask, confidences=None):
    """
    Compute damage metrics within ROI only.
    
    Parameters:
        detections: YOLO detection results (xyxy format)
        roi_mask: Binary ROI mask
        confidences: Detection confidences (optional)
        
    Returns:
        dict: ROI metrics
    """
    if detections is None or len(detections) == 0:
        return {
            'count_roi': 0,
            'ratio_roi': 0.0,
            'percent_roi': 0.0,
            'density': 0.0,
            'roi_area_px': int(np.sum(roi_mask)),
            'damage_conf_sum': 0.0,
            'damage_max_conf': 0.0,
            'damage_area_frac': 0.0
        }
    
    # Get ROI area in pixels
    roi_area_px = int(np.sum(roi_mask))
    
    # Initialize metrics
    count_roi = 0
    total_intersected_area = 0
    conf_sum = 0.0
    max_conf = 0.0
    total_normalized_area = 0.0
    
    # Process each detection
    for i, detection in enumerate(detections):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = detection[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        height, width = roi_mask.shape
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Create detection box mask
        det_mask = np.zeros_like(roi_mask)
        det_mask[y1:y2, x1:x2] = 1
        
        # Compute intersection with ROI
        intersection = cv2.bitwise_and(roi_mask, det_mask)
        intersected_area = np.sum(intersection)
        
        if intersected_area > 0:
            count_roi += 1
            total_intersected_area += intersected_area
            
            # Confidence metrics
            if confidences is not None and i < len(confidences):
                conf = confidences[i]
                conf_sum += conf
                max_conf = max(max_conf, conf)
            
            # Normalized area (w*h in normalized coordinates)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            total_normalized_area += box_width * box_height
    
    # Compute ratios
    ratio_roi = total_intersected_area / roi_area_px if roi_area_px > 0 else 0.0
    percent_roi = 100.0 * ratio_roi
    
    # Compute density (detections per ROI megapixel)
    roi_megapixels = roi_area_px / 1e6
    density = count_roi / roi_megapixels if roi_megapixels > 0 else 0.0
    
    return {
        'count_roi': count_roi,
        'ratio_roi': ratio_roi,
        'percent_roi': percent_roi,
        'density': density,
        'roi_area_px': roi_area_px,
        'damage_conf_sum': conf_sum,
        'damage_max_conf': max_conf,
        'damage_area_frac': total_normalized_area
    }


def process_cluster_images(cluster_df, yolo_model):
    """
    Process the first image for a cluster and compute metrics.
    
    Parameters:
        cluster_df: DataFrame with image URLs for one cluster
        yolo_model: Loaded YOLO model
        
    Returns:
        dict: Cluster metrics from first image
    """
    # Extract first image URL only
    urls = cluster_df['list_of_thumb_1024_url'].iloc[0]
    if pd.isna(urls) or urls == '':
        return {
            'yolo_detections_count_roi': 0,
            'yolo_damage_area_ratio_roi': 0.0,
            'street_quality_roi': 0.0,
            'street_quality_roi_winz': 0.0,
            'yolo_damage_density': 0.0,
            'roi_area_px': 0,
            'num_imgs_used': 1,
            'street_quality_roi_iqr': 0.0,
            'damage_conf_sum': 0.0,
            'damage_max_conf': 0.0,
            'damage_area_frac': 0.0
        }
    
    # Get first URL only
    url_list = str(urls).split(',')
    first_url = url_list[0].strip() if url_list else ''
    
    if not first_url:
        return {
            'yolo_detections_count_roi': 0,
            'yolo_damage_area_ratio_roi': 0.0,
            'street_quality_roi': 0.0,
            'street_quality_roi_winz': 0.0,
            'yolo_damage_density': 0.0,
            'roi_area_px': 0,
            'num_imgs_used': 1,
            'street_quality_roi_iqr': 0.0,
            'damage_conf_sum': 0.0,
            'damage_max_conf': 0.0,
            'damage_area_frac': 0.0
        }
    
    # Load image
    print(f"    📥 Loading image from URL...")
    image = load_image_from_url(first_url)
    if image is None:
        print(f"    ❌ Image loading failed, returning zeros")
        return {
            'yolo_detections_count_roi': 0,
            'yolo_damage_area_ratio_roi': 0.0,
            'street_quality_roi': 0.0,
            'street_quality_roi_winz': 0.0,
            'yolo_damage_density': 0.0,
            'roi_area_px': 0,
            'num_imgs_used': 1,
            'street_quality_roi_iqr': 0.0,
            'damage_conf_sum': 0.0,
            'damage_max_conf': 0.0,
            'damage_area_frac': 0.0
        }
    print(f"    ✅ Image loaded successfully: {image.shape}")
    
    # Create ROI mask
    roi_mask = create_roi_mask(
        image.shape, 
        ROI_BOTTOM_FRAC, 
        ROI_TOP_FRAC, 
        ROI_HMARGIN_FRAC
    )
    
    # Run YOLO inference (balanced speed/quality)
    try:
        print(f"    🔍 Running YOLO inference (conf={YOLO_CONF}, imgsz={YOLO_IMGSZ})...")
        results = yolo_model(image, 
                           imgsz=YOLO_IMGSZ, 
                           conf=YOLO_CONF, 
                           iou=YOLO_IOU, 
                           agnostic_nms=False,
                           augment=False,
                           max_det=100,
                           verbose=False)
        
        # Extract detections and confidences
        detections = []
        confidences = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            print(f"    🎯 Found {len(boxes)} detections")
            for box, conf in zip(boxes, confs):
                detections.append(box)
                confidences.append(conf)
        else:
            print(f"    ❌ No detections found")
        
        # Compute ROI metrics
        metrics = compute_roi_metrics(detections, roi_mask, confidences)
        
        return {
            'yolo_detections_count_roi': int(metrics['count_roi']),
            'yolo_damage_area_ratio_roi': float(metrics['ratio_roi']),
            'street_quality_roi': float(metrics['percent_roi']),
            'street_quality_roi_winz': float(metrics['percent_roi']),  # Will be winsorized later
            'yolo_damage_density': float(metrics['density']),
            'roi_area_px': int(metrics['roi_area_px']),
            'num_imgs_used': 1,
            'street_quality_roi_iqr': 0.0,  # No IQR for single image
            'damage_conf_sum': float(metrics['damage_conf_sum']),
            'damage_max_conf': float(metrics['damage_max_conf']),
            'damage_area_frac': float(metrics['damage_area_frac'])
        }
        
    except Exception as e:
        print(f"  ⚠️ YOLO inference failed for {first_url}: {e}")
        return {
            'yolo_detections_count_roi': 0,
            'yolo_damage_area_ratio_roi': 0.0,
            'street_quality_roi': 0.0,
            'street_quality_roi_winz': 0.0,
            'yolo_damage_density': 0.0,
            'roi_area_px': 0,
            'num_imgs_used': 1,
            'street_quality_roi_iqr': 0.0,
            'damage_conf_sum': 0.0,
            'damage_max_conf': 0.0,
            'damage_area_frac': 0.0
        }


def winsorize_per_split(df, column='street_quality_roi_winz'):
    """
    Winsorize a column at 1%/99% tails per split.
    
    Parameters:
        df: DataFrame with 'split' column
        column: Column name to winsorize
        
    Returns:
        DataFrame: With winsorized column
    """
    df = df.copy()
    
    for split in df['split'].unique():
        mask = df['split'] == split
        values = df.loc[mask, column]
        
        if len(values) > 0:
            # Compute 1% and 99% percentiles
            p1 = np.percentile(values, 1)
            p99 = np.percentile(values, 99)
            
            # Winsorize
            winsorized = mstats.winsorize(values, limits=(0.01, 0.01))
            df.loc[mask, column] = winsorized
    
    return df


def process_split(split_name, input_path, output_path, yolo_model):
    """
    Process a single split (train/val/test).
    
    Parameters:
        split_name: Name of split for logging
        input_path: Path to input CSV
        output_path: Path to output CSV
        yolo_model: Loaded YOLO model
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} clusters")
    
    # Process each cluster
    yolo_features = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:  # Progress updates every 100 clusters
            print(f"  Processing cluster {idx+1}/{len(df)}...")
        
        # Process cluster images
        cluster_df = pd.DataFrame([row])
        metrics = process_cluster_images(cluster_df, yolo_model)
        yolo_features.append(metrics)
    
    # Add YOLO features to DataFrame
    yolo_df = pd.DataFrame(yolo_features)
    result_df = pd.concat([df, yolo_df], axis=1)
    
    # Winsorize per split
    result_df = winsorize_per_split(result_df, 'street_quality_roi_winz')
    
    # Print summary statistics
    print(f"\n{split_name.capitalize()} summary:")
    print(f"  Clusters processed: {len(result_df)}")
    print(f"  Clusters with images: {(result_df['num_imgs_used'] > 0).sum()}")
    print(f"  Street quality (median): {result_df['street_quality_roi'].median():.2f}")
    print(f"  Street quality (winsorized median): {result_df['street_quality_roi_winz'].median():.2f}")
    print(f"  Street quality range: {result_df['street_quality_roi'].min():.2f} - {result_df['street_quality_roi'].max():.2f}")
    
    # Save result
    result_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    
    return result_df


def main():
    """Main execution function."""
    print("=" * 80)
    print("EXTRACT YOLO STREET QUALITY FEATURES")
    print("=" * 80)
    
    # Check if YOLO model exists
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ Error: YOLO model not found at {YOLO_MODEL_PATH}")
        print("Please ensure the model weights exist or update YOLO_MODEL_PATH")
        return
    
    # Load YOLO model
    print(f"\nLoading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✓ YOLO model loaded successfully")
    
    # Check input files
    input_files = [TRAIN_PATH, VAL_PATH, TEST_PATH]
    for path in input_files:
        if not path.exists():
            print(f"❌ Error: Input file not found: {path}")
            return
    
    # Process each split
    splits = [
        ('train', TRAIN_PATH, TRAIN_OUTPUT),
        ('val', VAL_PATH, VAL_OUTPUT),
        ('test', TEST_PATH, TEST_OUTPUT)
    ]
    
    all_results = {}
    
    for split_name, input_path, output_path in splits:
        result_df = process_split(split_name, input_path, output_path, yolo_model)
        all_results[split_name] = result_df
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for split_name, df in all_results.items():
        print(f"\n{split_name.capitalize()}:")
        print(f"  Total clusters: {len(df)}")
        print(f"  With images: {(df['num_imgs_used'] > 0).sum()}")
        print(f"  Street quality (winsorized): {df['street_quality_roi_winz'].describe()}")
    
    print(f"\n✓ All splits processed successfully!")
    print(f"Output files:")
    print(f"  - {TRAIN_OUTPUT}")
    print(f"  - {VAL_OUTPUT}")
    print(f"  - {TEST_OUTPUT}")


if __name__ == "__main__":
    main()
