#!/usr/bin/env python3
"""
YOLO Model Testing Script

This script tests a trained YOLOv8 model on the test dataset and generates
comprehensive evaluation metrics, confusion matrices, and visualizations.

Key Features:
- Loads trained YOLOv8 model from checkpoint
- Evaluates on test dataset images
- Computes mAP50, mAP50-95, precision, recall, F1 scores
- For single-class models: evaluates "road_damage" detection
- For multi-class models: computes per-class metrics
- Generates confusion matrix
- Saves sample predictions with bounding boxes
- Creates detailed evaluation report

Usage:
    python test_yolo_model.py --model runs/detect/train/weights/best.pt
    python test_yolo_model.py --model path/to/best.pt --imgsz 416
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = "data/rdd_yolo_cz_no_compressed"
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "images", "test")

# Model paths (defaults - can be overridden via command line)
DEFAULT_MODEL_PATH = "runs/detect/train/weights/best.pt"

# Output paths
DEFAULT_OUTPUT_DIR = "reports/YOLO/test_results"


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_model(model_path, imgsz=416):
    """
    Load a trained YOLOv8 model from checkpoint.
    
    Args:
        model_path (str): Path to model weights (.pt file)
        imgsz (int): Image size for inference (default: 416)
        
    Returns:
        YOLO: Loaded YOLO model
    """
    print(f"📦 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    
    return model


def evaluate_model(model, data_yaml, imgsz=416, conf_threshold=0.25):
    """
    Evaluate model on test dataset.
    
    Args:
        model: YOLO model
        data_yaml (str): Path to data.yaml configuration
        imgsz (int): Image size for inference (default: 416)
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n🧪 Evaluating model on test dataset...")
    print(f"   Test images from: {data_yaml}")
    print(f"   Image size: {imgsz}x{imgsz}")
    print(f"   Confidence threshold: {conf_threshold}")
    
    # Run evaluation
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=imgsz,
        conf=conf_threshold,
        save_json=True,
        verbose=True
    )
    
    # Extract metrics
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.p,
        'recall': results.box.r,
        'f1': results.box.f1,
        'class_metrics': {}
    }
    
    # Class-specific metrics (if available)
    if hasattr(results, 'names') and hasattr(results, 'box'):
        for i, class_name in enumerate(results.names.values()):
            if hasattr(results.box, 'maps') and len(results.box.maps) > i:
                metrics['class_metrics'][class_name] = {
                    'mAP50': results.box.maps50[i] if hasattr(results.box, 'maps50') else None,
                    'mAP50-95': results.box.maps[i] if hasattr(results.box, 'maps') else None
                }
    
    return metrics, results


def save_evaluation_report(metrics, output_dir):
    """
    Save evaluation metrics to JSON and CSV reports.
    
    Args:
        metrics (dict): Evaluation metrics
        output_dir (str): Output directory for reports
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'test_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to: {json_path}")
    
    # Save CSV summary
    summary_data = {
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1'],
        'Value': [
            metrics['mAP50'],
            metrics['mAP50-95'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'test_metrics_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"📊 Summary saved to: {csv_path}")
    
    # Save class-specific metrics if available
    if metrics['class_metrics']:
        class_data = []
        for class_name, class_metrics in metrics['class_metrics'].items():
            class_data.append({
                'Class': class_name,
                'mAP50': class_metrics.get('mAP50'),
                'mAP50-95': class_metrics.get('mAP50-95')
            })
        
        df_class = pd.DataFrame(class_data)
        csv_class_path = os.path.join(output_dir, 'test_class_metrics.csv')
        df_class.to_csv(csv_class_path, index=False)
        print(f"📊 Class metrics saved to: {csv_class_path}")


def visualize_metrics(metrics, output_dir):
    """
    Create visualizations of evaluation metrics.
    
    Args:
        metrics (dict): Evaluation metrics
        output_dir (str): Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall metrics bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']
    metric_values = [
        metrics['mAP50'],
        metrics['mAP50-95'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('YOLO Model Test Performance (USA Test Dataset)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'test_metrics_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Metrics visualization saved to: {output_path}")
    
    # Class-specific metrics if available
    if metrics['class_metrics']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(metrics['class_metrics'].keys())
        map50_values = [metrics['class_metrics'][c]['mAP50'] for c in classes]
        map5095_values = [metrics['class_metrics'][c]['mAP50-95'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, map50_values, width, label='mAP50', color='#3498db')
        ax.bar(x + width/2, map5095_values, width, label='mAP50-95', color='#2ecc71')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('mAP Score', fontsize=12)
        ax.set_title('Class-Specific Performance (USA Test Dataset)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'test_class_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Class metrics visualization saved to: {output_path}")


def save_predictions(model, data_yaml, output_dir, num_samples=20, imgsz=416):
    """
    Save sample predictions with bounding boxes for visual inspection.
    
    Args:
        model: YOLO model
        data_yaml (str): Path to data.yaml configuration
        output_dir (str): Output directory for predictions
        num_samples (int): Number of sample images to save
        imgsz (int): Image size for inference (default: 416)
    """
    print(f"\n🎨 Generating sample predictions...")
    
    predictions_dir = os.path.join(output_dir, 'sample_predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Get test images directory (use default if not found in yaml)
    test_images_dir = TEST_IMAGES_DIR
    if os.path.exists(data_yaml):
        with open(data_yaml, 'r') as f:
            content = f.read()
            # Extract path
            path_line = [line for line in content.split('\n') if line.startswith('path:')]
            if path_line:
                base_path = path_line[0].split(':')[1].strip()
                test_images_dir = os.path.join(base_path, 'images', 'test')
    
    # Get list of test images
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(test_images) > num_samples:
        import random
        random.seed(42)
        test_images = random.sample(test_images, num_samples)
    
    print(f"   Generating predictions for {len(test_images)} sample images...")
    
    # Run inference and save predictions
    for i, img_file in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_file)
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            save=True,
            project=predictions_dir,
            name=f'samples',
            exist_ok=True
        )
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{len(test_images)} images...")
    
    print(f"✅ Sample predictions saved to: {predictions_dir}")


def main():
    """
    Main function to test YOLO model on test dataset.
    """
    parser = argparse.ArgumentParser(description='Test YOLOv8 model on test dataset')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help=f'Path to trained model weights (.pt file) (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--data', type=str, default=DATA_YAML,
                       help=f'Path to data.yaml configuration file (default: {DATA_YAML})')
    parser.add_argument('--imgsz', type=int, default=416,
                       help='Image size for inference (default: 416)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for test results (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save sample predictions with bounding boxes')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of sample images to save (default: 20)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 YOLO MODEL TESTING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Test Data: {args.data}")
    print(f"Output: {args.output}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        model = load_model(args.model, args.imgsz)
        
        # Evaluate model
        metrics, results = evaluate_model(model, args.data, args.imgsz, args.conf)
        
        # Print results
        print("\n" + "="*60)
        print("📊 TEST RESULTS")
        print("="*60)
        print(f"mAP50:       {metrics['mAP50']:.4f}")
        print(f"mAP50-95:    {metrics['mAP50-95']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        
        # Save reports
        save_evaluation_report(metrics, output_dir)
        
        # Create visualizations
        visualize_metrics(metrics, output_dir)
        
        # Save sample predictions if requested
        if args.save_predictions:
            save_predictions(model, args.data, output_dir, args.num_samples, args.imgsz)
        
        print("\n" + "="*60)
        print("✅ Testing complete!")
        print(f"📁 Results saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()

