#!/usr/bin/env python3
"""
Script: Train CNN for Residual Risk Prediction

Purpose:
This script trains a ResNet18 CNN to predict residual crash risk (unexplained by 
OSM attributes) from street-level Mapillary images. The model uses transfer learning 
with a two-phase training approach (frozen backbone then fine-tuning) and evaluates 
on test set to identify visually high-risk locations.

Functionality:
- Loads CNN dataset (one image per cluster)
- Creates PyTorch dataset with URL loading and caching
- Trains ResNet18 with regression head (two-phase: frozen then fine-tuning)
- Tracks training metrics (MSE, MAE) per epoch
- Implements early stopping based on validation MSE
- Evaluates on test set and ranks images by predicted residual
- Saves model in eval mode for Grad-CAM compatibility
- Generates all required outputs for analysis and visualization

How to run:
    python src/modeling/train_cnn_residual.py [--max_epochs 15]

Prerequisites:
    - Run prepare_cnn_dataset.py first to generate cnn_samples_residual.csv
    - Ensure data/processed/cnn_samples_residual.csv exists

Output:
    - models/cnn_residual_best.pth (best model weights)
    - models/cnn_residual_metadata.json (model metadata)
    - reports/CNN/training_log.csv (epoch-by-epoch metrics)
    - data/processed/cnn_test_predictions.csv (test predictions)
    - data/processed/cnn_test_top_visual_risks.csv (ranked high-risk images)
"""

import os
import sys
import json
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        if 'desc' in kwargs:
            print(f"  {kwargs['desc']}...")
        return iterable
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dataset paths
DATA_DIR = project_root / 'data' / 'processed'
CNN_DATA_PATH = DATA_DIR / 'cnn_samples_residual.csv'

# Model paths
MODELS_DIR = project_root / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'cnn_residual_best.pth'
METADATA_PATH = MODELS_DIR / 'cnn_residual_metadata.json'

# Output paths
REPORTS_DIR = project_root / 'reports' / 'CNN'
TRAINING_LOG_PATH = REPORTS_DIR / 'training_log.csv'
PREDICTIONS_PATH = DATA_DIR / 'cnn_test_predictions.csv'
TOP_RISKS_PATH = DATA_DIR / 'cnn_test_top_visual_risks.csv'


# =============================================================================
# FUNCTIONS
# =============================================================================

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Parameters:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Determine the best available device for training.
    
    Returns:
        torch.device: Device identifier
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    return device


class ResidualImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from URLs and predicting residuals.
    
    Handles URL loading, caching, and preprocessing for training/validation/test splits.
    """
    
    def __init__(self, csv_path, split='train', transform=None):
        """
        Initialize dataset.
        
        Parameters:
            csv_path: Path to cnn_samples_residual.csv
            split: Data split ('train', 'val', 'test')
            transform: Image transforms to apply
        """
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        
        # Load data
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # Image cache (URL -> PIL Image)
        self.image_cache = {}
        
        print(f"  {split.capitalize()} dataset: {len(self.df)} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Parameters:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, residual)
        """
        row = self.df.iloc[idx]
        image_url = row['image_url']
        residual = float(row['residual'])
        
        # Load image (from cache if available)
        if image_url in self.image_cache:
            image = self.image_cache[image_url]
        else:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                self.image_cache[image_url] = image
            except Exception as e:
                # Skip failed downloads - use a blank image as fallback
                print(f"  ⚠️ Failed to load {image_url}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(residual, dtype=torch.float32)


class ResidualPredictor(nn.Module):
    """
    ResNet18-based model for predicting residual risk from images.
    
    Uses pretrained ResNet18 backbone with a custom regression head.
    Preserves ResNet layer structure for Grad-CAM compatibility.
    """
    
    def __init__(self):
        """
        Initialize model architecture.
        """
        super(ResidualPredictor, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Preserve ResNet structure for Grad-CAM access
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # This is the target layer for Grad-CAM
        
        # Remove the original avgpool and fc, replace with our own
        self.avgpool = resnet.avgpool
        
        # Add regression head (no activation - linear output)
        self.fc = nn.Linear(512, 1)
        
        # Initialize new head weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
            x: Input image tensor
            
        Returns:
            torch.Tensor: Predicted residual
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Grad-CAM target: self.layer4[-1] accesses last BasicBlock
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


def get_transforms(split='train'):
    """
    Get image transforms for a given split.
    
    Parameters:
        split: Data split ('train', 'val', 'test')
        
    Returns:
        transforms.Compose: Image transforms
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Parameters:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        tuple: (average_loss, average_mae)
    """
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0
    
    for images, residuals in (tqdm(dataloader, desc="  Training", leave=False) if HAS_TQDM else dataloader):
        images = images.to(device)
        residuals = residuals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, residuals)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        mae = torch.mean(torch.abs(predictions - residuals))
        running_mae += mae.item() * images.size(0)
        n_samples += images.size(0)
    
    avg_loss = running_loss / n_samples
    avg_mae = running_mae / n_samples
    
    return avg_loss, avg_mae


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Parameters:
        model: PyTorch model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, average_mae, all_predictions, all_residuals)
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0
    
    all_predictions = []
    all_residuals = []
    
    with torch.no_grad():
        for images, residuals in (tqdm(dataloader, desc="  Evaluating", leave=False) if HAS_TQDM else dataloader):
            images = images.to(device)
            residuals = residuals.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, residuals)
            
            running_loss += loss.item() * images.size(0)
            mae = torch.mean(torch.abs(predictions - residuals))
            running_mae += mae.item() * images.size(0)
            n_samples += images.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_residuals.extend(residuals.cpu().numpy())
    
    avg_loss = running_loss / n_samples
    avg_mae = running_mae / n_samples
    
    return avg_loss, avg_mae, np.array(all_predictions), np.array(all_residuals)


def train_model(config):
    """
    Main training function.
    
    Parameters:
        config: Configuration dictionary with hyperparameters
    """
    print("=" * 80)
    print("TRAINING CNN RESIDUAL PREDICTOR")
    print("=" * 80)
    
    # Set seeds
    set_seeds(config['seed'])
    
    # Get device
    device = get_device()
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='train',
        transform=get_transforms('train')
    )
    val_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='val',
        transform=get_transforms('val')
    )
    test_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='test',
        transform=get_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with URL loading
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    model = ResidualPredictor().to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training tracking
    training_log = []
    best_val_mse = float('inf')
    epochs_without_improvement = 0
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("TRAINING LOOP")
    print("=" * 80)
    
    # Phase 1: Freeze backbone (epochs 1-2)
    print("\nPhase 1: Frozen backbone (epochs 1-2)")
    # Freeze all layers except the regression head (fc)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    # Phase 2: Unfreeze backbone (epochs 3+)
    unfreeze_epoch = 3
    
    # Training loop
    for epoch in range(1, config['max_epochs'] + 1):
        # Unfreeze at epoch 3 (if we have enough epochs)
        if epoch == unfreeze_epoch and config['max_epochs'] >= unfreeze_epoch:
            print("\nPhase 2: Fine-tuning (unfreezing backbone)")
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
        
        print(f"\nEpoch {epoch}/{config['max_epochs']}")
        
        # Train
        train_mse, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_mse, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        training_log.append({
            'epoch': epoch,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'lr': current_lr
        })
        
        # Print progress
        print(f"  Train MSE: {train_mse:.6f}, Train MAE: {train_mae:.6f}")
        print(f"  Val MSE:   {val_mse:.6f}, Val MAE:   {val_mae:.6f}")
        print(f"  Best Val MSE so far: {best_val_mse:.6f}")
        
        # Save best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_without_improvement = 0
            
            # Save model in eval mode
            model.eval()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_mse,
                'val_mae': val_mae,
            }, BEST_MODEL_PATH)
            print(f"  ✓ Saved new best model (Val MSE: {val_mse:.6f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= 3:
            print(f"\nEarly stopping triggered (no improvement for 3 epochs)")
            break
    
    # Save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(TRAINING_LOG_PATH, index=False)
    print(f"\n✓ Saved training log to: {TRAINING_LOG_PATH}")
    
    # Load best model for evaluation
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on test set
    test_mse, test_mae, test_predictions, test_residuals = evaluate(model, test_loader, criterion, device)
    
    # Compute Pearson correlation
    pearson_corr, pearson_p = pearsonr(test_residuals, test_predictions)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    
    # Save predictions
    test_df = pd.read_csv(config['data_path'])
    test_df = test_df[test_df['split'] == 'test'].reset_index(drop=True)
    
    predictions_df = pd.DataFrame({
        'image_url': test_df['image_url'],
        'cluster_id': test_df['cluster_id'],
        'true_residual': test_residuals,
        'predicted_residual': test_predictions
    })
    
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n✓ Saved test predictions to: {PREDICTIONS_PATH}")
    
    # Create ranked high-risk images
    predictions_df['predicted_residual_abs'] = predictions_df['predicted_residual'].abs()
    top_risks_df = predictions_df.sort_values('predicted_residual', ascending=False).reset_index(drop=True)
    
    top_risks_df.to_csv(TOP_RISKS_PATH, index=False)
    print(f"✓ Saved top visual risks to: {TOP_RISKS_PATH}")
    
    # Save metadata
    metadata = {
        'model_type': 'ResNet18',
        'hyperparameters': {
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'batch_size': config['batch_size'],
            'max_epochs': config['max_epochs'],
            'seed': config['seed']
        },
        'training_metrics': {
            'best_val_mse': float(best_val_mse),
            'best_val_mae': float(checkpoint['val_mae']),
            'best_epoch': int(checkpoint['epoch'])
        },
        'test_metrics': {
            'test_mse': float(test_mse),
            'test_mae': float(test_mae),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p)
        },
        'architecture': {
            'backbone': 'ResNet18 (pretrained ImageNet)',
            'head': 'Linear(512, 1) - no activation',
            'output': 'Single residual value',
            'gradcam_target_layer': 'layer4[-1]',
            'note': 'Model preserves ResNet layer structure for Grad-CAM compatibility. Access layer4[-1] for Grad-CAM visualization.'
        }
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {METADATA_PATH}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train CNN for residual risk prediction')
    parser.add_argument('--max_epochs', type=int, default=4, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': CNN_DATA_PATH,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed
    }
    
    # Check if data file exists
    if not CNN_DATA_PATH.exists():
        print(f"❌ Error: Data file not found: {CNN_DATA_PATH}")
        print("   Please run prepare_cnn_dataset.py first")
        return
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()

