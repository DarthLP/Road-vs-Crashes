#!/usr/bin/env python3
"""
YOLOv8 Training Script for RDD2022 Road Damage Detection

This script trains a YOLOv8s model on the RDD2022 dataset to detect road damage
as a single unified class. It uses pretrained weights and optimized hyperparameters
for road damage detection with early stopping.

Key Features:
- Loads pretrained YOLOv8s weights
- Trains with single "road_damage" class (merged from crack, other corruption, Pothole)
- Trains for 30 epochs with early stopping (patience=5)
- Uses image size 416x416 for better detection quality
- Trains with Mac-optimized settings (MPS device)
- Runs inference on validation images after training
- Saves prediction visualizations for verification
- Provides comprehensive training progress feedback

Usage:
    python src/modeling/train_yolo.py

Prerequisites:
    - Run merge_yolo_classes.py to merge all classes to "road_damage"
    - Run filter_czech_norway.py first to create filtered dataset
    - Run compress_cz_no_dataset.py to compress images
    - Install ultralytics: pip install ultralytics
    - Ensure data/rdd_yolo_cz_no_compressed/ directory exists with proper structure

Output:
    - Trained model weights saved to runs/detect/train/weights/best.pt
    - Training metrics and plots in runs/detect/train/
    - Inference results on validation images
"""

import os
import random
from pathlib import Path
from ultralytics import YOLO
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = "data/rdd_yolo_cz_no_compressed"
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
TRAIN_DIR = os.path.join(DATA_DIR, "images", "train")
VAL_DIR = os.path.join(DATA_DIR, "images", "val")
LABEL_DIR = os.path.join(DATA_DIR, "labels")

# Model paths
MODEL_WEIGHTS = "yolov8n.pt"  # Using nano model for speed (faster than 's' variant)
BEST_WEIGHTS_PATH = "runs/detect/train/weights/best.pt"
LAST_WEIGHTS_PATH = "runs/detect/train/weights/last.pt"

# Output paths
OUTPUT_DIR = "runs/detect"
PROJECT_NAME = "train"
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, PROJECT_NAME, "predictions")


# =============================================================================
# FUNCTIONS
# =============================================================================

def check_dataset_exists():
    """
    Verify that the YOLO dataset directory and files exist.
    
    Returns:
        bool: True if dataset is properly set up, False otherwise
    """
    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset configuration file not found: {DATA_YAML}")
        print("   Please run filter_czech_norway.py and compress_cz_no_dataset.py first")
        return False
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training images directory not found: {TRAIN_DIR}")
        print("   Please run filter_czech_norway.py and compress_cz_no_dataset.py first")
        return False
    
    if not os.path.exists(VAL_DIR):
        print(f"❌ Validation images directory not found: {VAL_DIR}")
        print("   Please run filter_czech_norway.py and compress_cz_no_dataset.py first")
        return False
    
    # Count images in each split
    train_count = len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(VAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"📊 Dataset verification:")
    print(f"   Training images: {train_count}")
    print(f"   Validation images: {val_count}")
    
    if train_count == 0 or val_count == 0:
        print("❌ No images found in dataset directories")
        return False
    
    return True


def setup_device():
    """
    Determine the best available device for training.
    
    Returns:
        str: Device identifier ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("🚀 Using CUDA GPU acceleration")
    else:
        device = 'cpu'
        print("💻 Using CPU (training will be slower)")
    
    return device


def train_model(data_yaml, device, epochs=30, imgsz=416, batch=32, lr0=0.01, patience=5):
    """
    Train the YOLOv8s model on the RDD2022 dataset.
    
    Args:
        data_yaml (str): Path to the dataset configuration file
        device (str): Device to use for training ('mps', 'cuda', or 'cpu')
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size for training
        lr0 (float): Initial learning rate
        patience (int): Early stopping patience (epochs without improvement)
        
    Returns:
        YOLO: Trained model object
    """
    print("🤖 Loading YOLOv8 pretrained weights...")
    model = YOLO(MODEL_WEIGHTS)
    
    print("🏋️ Starting training...")
    print(f"   Dataset: {data_yaml}")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Learning rate: {lr0}")
    print(f"   Early stopping patience: {patience}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        project=OUTPUT_DIR,
        name=PROJECT_NAME,
        save=True,
        plots=True,
        val=True,
        patience=patience
    )
    
    print("✅ Training completed!")
    return model


def run_inference_test(model, val_dir, num_images=3):
    """
    Run inference on a few validation images to test the trained model.
    
    Args:
        model (YOLO): Trained YOLO model
        val_dir (str): Path to validation images directory
        num_images (int): Number of images to test
        
    Returns:
        list: Paths to saved prediction images
    """
    print(f"🔍 Running inference test on {num_images} validation images...")
    
    # Get list of validation images
    val_images = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(val_images) == 0:
        print("⚠️  No validation images found for inference test")
        return []
    
    # Select random images for testing
    test_images = random.sample(val_images, min(num_images, len(val_images)))
    prediction_paths = []
    
    # Create output directory for predictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    for img_name in test_images:
        img_path = os.path.join(val_dir, img_name)
        
        print(f"   Testing: {img_name}")
        
        # Run inference
        results = model.predict(
            source=img_path,
            save=True,
            save_txt=True,
            conf=0.25,  # Confidence threshold
            project=PREDICTIONS_DIR,
            name=f"test_{os.path.splitext(img_name)[0]}",
            exist_ok=True
        )
        
        # Get the saved prediction path
        pred_path = os.path.join(PREDICTIONS_DIR, f"test_{os.path.splitext(img_name)[0]}", img_name)
        if os.path.exists(pred_path):
            prediction_paths.append(pred_path)
    
    print(f"✅ Inference test completed! Predictions saved to {PREDICTIONS_DIR}")
    return prediction_paths


def print_training_summary():
    """
    Print a summary of the training results and file locations.
    """
    print("\n" + "="*60)
    print("🎉 YOLOv8 TRAINING COMPLETE!")
    print("="*60)
    
    if os.path.exists(BEST_WEIGHTS_PATH):
        print(f"🏆 Best weights: {BEST_WEIGHTS_PATH}")
    else:
        print(f"⚠️  Best weights not found at expected location")
    
    if os.path.exists(LAST_WEIGHTS_PATH):
        print(f"📁 Last weights: {LAST_WEIGHTS_PATH}")
    
    print(f"📊 Training results: runs/detect/train/")
    print(f"📈 Metrics plots: runs/detect/train/")
    print(f"🔍 Test predictions: {PREDICTIONS_DIR}/")
    
    print("\n📋 Model Classes:")
    print("   0: road_damage (merged from crack, other corruption, Pothole)")
    
    print("\n🚀 Next Steps:")
    print("   - Use best.pt for inference on new images")
    print("   - Check prediction visualizations in predictions/ folder")
    print("   - Review training metrics in runs/detect/train/")


def main():
    """
    Main function to train YOLOv8 model on RDD2022 dataset.
    """
    print("🚀 Starting YOLOv8 training on RDD2022 dataset...")
    
    # Verify dataset exists
    if not check_dataset_exists():
        return
    
    # Setup device
    device = setup_device()
    
    # Training configuration
    epochs = 30  # Increased for better convergence
    imgsz = 416  # Increased image size for better detection quality
    batch = 16   # Keep small batch for memory efficiency
    lr0 = 0.01
    patience = 5  # Early stopping patience (stops if no improvement for 5 epochs)
    
    # Train the model
    model = train_model(DATA_YAML, device, epochs, imgsz, batch, lr0, patience)
    
    # Run inference test
    prediction_paths = run_inference_test(model, VAL_DIR, num_images=3)
    
    # Print summary
    print_training_summary()
    
    print("\n✅ Finetuning complete — best weights saved to runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
