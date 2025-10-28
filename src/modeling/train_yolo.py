#!/usr/bin/env python3
"""
YOLOv8 Training Script for RDD2022 Road Damage Detection

This script trains a YOLOv8s model on the RDD2022 dataset to detect road damage
including cracks, potholes, and other corruption. It uses pretrained weights
and optimized hyperparameters for road damage detection.

Key Features:
- Loads pretrained YOLOv8s weights
- Trains with Mac-optimized settings (MPS device)
- Runs inference on validation images after training
- Saves prediction visualizations for verification
- Provides comprehensive training progress feedback

Usage:
    python src/modeling/train_yolo.py

Prerequisites:
    - Run convert_rdd_to_yolo.py first to prepare the dataset
    - Install ultralytics: pip install ultralytics
    - Ensure data/rdd_yolo/ directory exists with proper structure

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


def check_dataset_exists():
    """
    Verify that the YOLO dataset directory and files exist.
    
    Returns:
        bool: True if dataset is properly set up, False otherwise
    """
    data_yaml = "data/rdd_yolo/data.yaml"
    train_dir = "data/rdd_yolo/images/train"
    val_dir = "data/rdd_yolo/images/val"
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset configuration file not found: {data_yaml}")
        print("   Please run convert_rdd_to_yolo.py first")
        return False
    
    if not os.path.exists(train_dir):
        print(f"❌ Training images directory not found: {train_dir}")
        print("   Please run convert_rdd_to_yolo.py first")
        return False
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation images directory not found: {val_dir}")
        print("   Please run convert_rdd_to_yolo.py first")
        return False
    
    # Count images in each split
    train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
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


def train_model(data_yaml, device, epochs=30, imgsz=416, batch=32, lr0=0.01):
    """
    Train the YOLOv8s model on the RDD2022 dataset.
    
    Args:
        data_yaml (str): Path to the dataset configuration file
        device (str): Device to use for training ('mps', 'cuda', or 'cpu')
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size for training
        lr0 (float): Initial learning rate
        
    Returns:
        YOLO: Trained model object
    """
    print("🤖 Loading YOLOv8s pretrained weights...")
    model = YOLO('yolov8s.pt')
    
    print("🏋️ Starting training...")
    print(f"   Dataset: {data_yaml}")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Learning rate: {lr0}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        project='runs/detect',
        name='train',
        save=True,
        plots=True,
        val=True
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
    pred_dir = "runs/detect/train/predictions"
    os.makedirs(pred_dir, exist_ok=True)
    
    for img_name in test_images:
        img_path = os.path.join(val_dir, img_name)
        
        print(f"   Testing: {img_name}")
        
        # Run inference
        results = model.predict(
            source=img_path,
            save=True,
            save_txt=True,
            conf=0.25,  # Confidence threshold
            project=pred_dir,
            name=f"test_{os.path.splitext(img_name)[0]}",
            exist_ok=True
        )
        
        # Get the saved prediction path
        pred_path = os.path.join(pred_dir, f"test_{os.path.splitext(img_name)[0]}", img_name)
        if os.path.exists(pred_path):
            prediction_paths.append(pred_path)
    
    print(f"✅ Inference test completed! Predictions saved to {pred_dir}")
    return prediction_paths


def print_training_summary():
    """
    Print a summary of the training results and file locations.
    """
    best_weights_path = "runs/detect/train/weights/best.pt"
    last_weights_path = "runs/detect/train/weights/last.pt"
    
    print("\n" + "="*60)
    print("🎉 YOLOv8 TRAINING COMPLETE!")
    print("="*60)
    
    if os.path.exists(best_weights_path):
        print(f"🏆 Best weights: {best_weights_path}")
    else:
        print(f"⚠️  Best weights not found at expected location")
    
    if os.path.exists(last_weights_path):
        print(f"📁 Last weights: {last_weights_path}")
    
    print(f"📊 Training results: runs/detect/train/")
    print(f"📈 Metrics plots: runs/detect/train/")
    print(f"🔍 Test predictions: runs/detect/train/predictions/")
    
    print("\n📋 Model Classes:")
    print("   0: crack (consolidated from longitudinal, transverse, alligator)")
    print("   1: other corruption")
    print("   2: Pothole")
    
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
    
    # Training configuration (optimized for speed)
    data_yaml = "data/rdd_yolo/data.yaml"
    epochs = 30
    imgsz = 416
    batch = 32
    lr0 = 0.01
    
    # Train the model
    model = train_model(data_yaml, device, epochs, imgsz, batch, lr0)
    
    # Run inference test
    val_dir = "data/rdd_yolo/images/val"
    prediction_paths = run_inference_test(model, val_dir, num_images=3)
    
    # Print summary
    print_training_summary()
    
    print("\n✅ Finetuning complete — best weights saved to runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    main()
