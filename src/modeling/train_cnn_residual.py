#!/usr/bin/env python3
"""
Script: Train CNN for Residual Risk Prediction

Purpose:
This script trains a ResNet18 CNN to predict residual crash risk (unexplained by 
OSM attributes) from street-level Mapillary images. The model uses transfer learning 
with a three-phase training approach: frozen backbone, fine-tuning, and progressive 
resizing from 224x224 → 424x424 for better street scene detail capture.

Functionality:
- Loads CNN dataset (one image per cluster)
- **Parallel downloads**: 16 concurrent downloads for fast initial caching (much faster)
- Creates PyTorch dataset with persistent disk caching for fast image loading
- Pre-downloads all images to disk cache on first run (optimized, resized, lower quality)
- **Optimized caching**: Images resized to 224x224 and saved at 85% quality for speed
- Uses multiprocessing (4 workers) with prefetching (2 batches ahead) for parallel loading
- Trains ResNet18 with regression head using three-phase approach:
  * Phase 1 (epochs 1-6): Frozen backbone at 224x224
  * Phase 2 (epochs 7-14): Fine-tuning at 224x224
  * Phase 3 (epochs 15-26): Progressive resizing to 424x424 for better detail
- Supports checkpoint resuming with --resume flag
- Tracks training metrics (MSE, MAE) per epoch
- Implements early stopping based on validation MSE
- Evaluates on test set at 424x424 and ranks images by predicted residual
- Saves model in eval mode for Grad-CAM compatibility
- Generates all required outputs for analysis and visualization

How to run:
    python src/modeling/train_cnn_residual.py [--max_epochs 15]
    python src/modeling/train_cnn_residual.py --resume  # Resume from last checkpoint

Prerequisites:
    - Run prepare_cnn_dataset.py first to generate cnn_samples_residual.csv
    - Ensure data/processed/cnn_samples_residual.csv exists

Output:
    - models/cnn_residual_best.pth (best model weights)
    - models/cnn_residual_checkpoint.pth (last checkpoint for resuming)
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
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CACHE_DIR = project_root / 'data' / 'cache' / 'images'

# Model paths
MODELS_DIR = project_root / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'cnn_residual_best.pth'
CHECKPOINT_PATH = MODELS_DIR / 'cnn_residual_checkpoint.pth'
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
    
    Handles URL loading, persistent disk caching, and preprocessing for training/validation/test splits.
    Supports dynamic resolution changes for progressive resizing.
    """
    
    def __init__(self, csv_path, split='train', transform=None, cache_dir=CACHE_DIR):
        """
        Initialize dataset.
        
        Parameters:
            csv_path: Path to cnn_samples_residual.csv
            split: Data split ('train', 'val', 'test')
            transform: Image transforms to apply
            cache_dir: Directory for caching images
        """
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.cache_dir = cache_dir
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = pd.read_csv(csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # In-memory cache for already loaded images in this session
        self.image_cache = {}
        
        print(f"  {split.capitalize()} dataset: {len(self.df)} samples")
        
        # Pre-download and cache all images if not already cached
        self._cache_all_images()
    
    def _get_cache_path(self, url):
        """
        Generate cache file path from URL.
        
        Parameters:
            url: Image URL
            
        Returns:
            Path: Cache file path
        """
        # Create hash of URL for filename
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"
    
    def _download_and_cache_image(self, url):
        """
        Download a single image and save to cache.
        
        Parameters:
            url: Image URL
            
        Returns:
            tuple: (success, url, cache_path)
        """
        cache_path = self._get_cache_path(url)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            # Optimize: Resize to target resolution and use lower quality for faster loading
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            image.save(cache_path, 'JPEG', quality=85, optimize=True)
            return (True, url, cache_path)
        except Exception as e:
            # Create a placeholder image
            placeholder = Image.new('RGB', (224, 224), color='black')
            placeholder.save(cache_path, 'JPEG', quality=85, optimize=True)
            return (False, url, cache_path)
    
    def _cache_all_images(self):
        """
        Pre-download and cache all images to disk using parallel downloads.
        
        Downloads images that aren't already cached to speed up training.
        """
        print(f"  Checking cache for {len(self.df)} images...")
        
        images_to_download = []
        for idx, row in self.df.iterrows():
            url = row['image_url']
            cache_path = self._get_cache_path(url)
            if not cache_path.exists():
                images_to_download.append(url)
        
        if images_to_download:
            print(f"  ⏬ Downloading {len(images_to_download)} images in parallel (faster)...")
            # Use ThreadPoolExecutor for parallel downloads (up to 16 concurrent downloads)
            max_workers = min(16, len(images_to_download))
            success_count = 0
            fail_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_url = {executor.submit(self._download_and_cache_image, url): url 
                               for url in images_to_download}
                
                # Process completed downloads with progress bar
                for future in (tqdm(as_completed(future_to_url), total=len(images_to_download), 
                                    desc="  Downloading", leave=False) if HAS_TQDM else as_completed(future_to_url)):
                    success, url, cache_path = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        print(f"  ⚠️ Failed to download {url}")
            
            print(f"  ✓ Cached {success_count} images ({fail_count} failed)")
        else:
            print(f"  ✓ All {len(self.df)} images already cached")
    
    def update_transform(self, new_transform):
        """
        Update the transform for progressive resizing.
        
        Parameters:
            new_transform: New image transforms to apply
        """
        self.transform = new_transform
        
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
        
        # Load image from disk cache (much faster than downloading)
        cache_path = self._get_cache_path(image_url)
        try:
            image = Image.open(cache_path).convert('RGB')
        except Exception as e:
            # If cache fails, use a blank image as fallback
            print(f"  ⚠️ Failed to load cached image {cache_path}: {e}")
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


def get_transforms(split='train', resolution=224):
    """
    Get image transforms for a given split and resolution.
    
    Parameters:
        split: Data split ('train', 'val', 'test')
        resolution: Image resolution (224, 424, or 632)
        
    Returns:
        transforms.Compose: Image transforms
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
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
    
    # Check if resuming from checkpoint
    start_epoch = 1
    training_log = []
    best_val_mse = float('inf')
    epochs_without_improvement = 0
    
    if config.get('resume', False) and CHECKPOINT_PATH.exists():
        print(f"\n📂 Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint['epoch'] + 1
        best_val_mse = checkpoint['best_val_mse']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        training_log = checkpoint['training_log']
        print(f"  Starting from epoch {start_epoch}")
        print(f"  Best Val MSE so far: {best_val_mse:.6f}")
        print(f"  Epochs without improvement: {epochs_without_improvement}")
        print(f"  Previously trained {len(training_log)} epochs")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='train',
        transform=get_transforms('train', resolution=224)  # Start with 224x224
    )
    val_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='val',
        transform=get_transforms('val', resolution=224)
    )
    test_dataset = ResidualImageDataset(
        str(config['data_path']),
        split='test',
        transform=get_transforms('test', resolution=424)  # Test at highest resolution
    )
    
    # Create dataloaders
    # Use multiprocessing for faster image loading (now that images are cached)
    num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
    print(f"  Using {num_workers} workers for data loading")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches for faster training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Create model
    print("\nCreating model...")
    model = ResidualPredictor().to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Load model and optimizer state if resuming
    if config.get('resume', False) and CHECKPOINT_PATH.exists():
        print(f"\n📂 Loading model and optimizer state from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ Model and optimizer states loaded successfully")
    
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
    
    # Phase 2: Unfreeze backbone (epochs 7+)
    unfreeze_epoch = 7
    
    # Progressive resizing: switch to 424x424 at epoch 15
    resolution_switch_epoch = 15  # 224x224 → 424x424
    
    # Training loop
    for epoch in range(start_epoch, config['max_epochs'] + 1):
        # Unfreeze at epoch 7 (if we have enough epochs)
        if epoch == unfreeze_epoch and config['max_epochs'] >= unfreeze_epoch:
            print("\nPhase 2: Fine-tuning (unfreezing backbone)")
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
        
        # Progressive resizing: switch to 424x424
        if epoch == resolution_switch_epoch and config['max_epochs'] >= resolution_switch_epoch:
            print(f"\nPhase 3: Progressive resizing (switching to 424x424)")
            # Update transforms to higher resolution
            train_dataset.update_transform(get_transforms('train', resolution=424))
            val_dataset.update_transform(get_transforms('val', resolution=424))
            # Recreate dataloaders with new transforms
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
        
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
        
        # Save checkpoint (for resuming training)
        model.eval()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_mse': best_val_mse,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'epochs_without_improvement': epochs_without_improvement,
            'training_log': training_log
        }, CHECKPOINT_PATH)
        model.train()
        
        # Early stopping (increased patience for better learning)
        if epochs_without_improvement >= 8:
            print(f"\nEarly stopping triggered (no improvement for 8 epochs)")
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
            'progressive_resizing': {
                'phase_1': 'Frozen backbone at 224x224 (epochs 1-6)',
                'phase_2': 'Fine-tuning at 224x224 (epochs 7-14)',
                'phase_3': 'Progressive resizing to 424x424 (epochs 15-26)',
                'test_resolution': '424x424'
            },
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
    parser.add_argument('--max_epochs', type=int, default=26, help='Maximum number of training epochs (progressive resizing at epoch 15)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (increased from 8 for faster training)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': CNN_DATA_PATH,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'resume': args.resume
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

