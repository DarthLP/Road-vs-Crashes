#!/usr/bin/env python3
"""
Script: Grad-CAM Utility Module

Purpose:
This module provides production-ready Grad-CAM (Gradient-weighted Class Activation Mapping) 
functionality for CNN interpretability. Supports both standard Grad-CAM and Grad-CAM++ 
methods with statistical analysis, batch-safe hook management, and flexible visualization.

Functionality:
- GradCAM class with batch-safe hook registration and cleanup
- Support for both Grad-CAM and Grad-CAM++ methods
- CAM intensity statistics (mean, max, top10 quantile)
- Contrast normalization for consistent visualization
- Flexible colormap support (jet, turbo)
- Memory-safe hook management to prevent leaks

How to use:
    from gradcam_utils import GradCAM
    
    gradcam = GradCAM(model, target_layer=model.layer4[-1], method='gradcam')
    cam, stats = gradcam.compute_cam(input_tensor, target_score)
    overlay = gradcam.generate_overlay(cam, image_tensor, colormap='jet')

Prerequisites:
    - PyTorch model with accessible target layer
    - Input tensor with requires_grad=True
    - Target score for backward pass

Output:
    - CAM heatmap (normalized to [0,1])
    - CAM statistics dictionary
    - Overlay visualization (RGB numpy array)
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    
    Provides interpretability for CNN predictions by visualizing which parts of the input
    the model focuses on when making predictions. Supports both standard Grad-CAM and
    Grad-CAM++ methods with statistical analysis.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, method: str = 'gradcam'):
        """
        Initialize GradCAM.
        
        Parameters:
            model: PyTorch model (should be in eval mode)
            target_layer: Target layer for activation extraction (e.g., model.layer4[-1])
            method: 'gradcam' or 'gradcampp' for different weighting schemes
        """
        self.model = model
        self.target_layer = target_layer
        self.method = method.lower()
        
        if self.method not in ['gradcam', 'gradcampp']:
            raise ValueError(f"Method must be 'gradcam' or 'gradcampp', got {method}")
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Hook handles for cleanup
        self.handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def save_activations(module, input, output):
            """Save activations from forward pass."""
            self.activations = output.detach()
        
        def save_gradients(module, grad_input, grad_output):
            """Save gradients from backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Register hooks and store handles
        handle_f = self.target_layer.register_forward_hook(save_activations)
        handle_b = self.target_layer.register_full_backward_hook(save_gradients)
        
        self.handles = [handle_f, handle_b]
    
    def __del__(self):
        """Clean up hooks to prevent memory leaks."""
        self.remove_hooks()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def compute_cam(self, input_tensor: torch.Tensor, target_score: torch.Tensor) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute Grad-CAM heatmap and statistics.
        
        Parameters:
            input_tensor: Input image tensor (1, C, H, W) with requires_grad=True
            target_score: Target score for backward pass (e.g., predicted residual)
            
        Returns:
            tuple: (cam_heatmap, statistics_dict)
                - cam_heatmap: Normalized CAM heatmap (H, W) in range [0, 1]
                - statistics_dict: CAM intensity statistics
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass (activations will be captured by hook)
        with torch.set_grad_enabled(True):
            _ = self.model(input_tensor)
        
        # Check if we have activations
        if self.activations is None:
            raise RuntimeError("No activations captured. Ensure forward pass completed.")
        
        # Backward pass on target score
        target_score.backward(retain_graph=True)
        
        # Check if we have gradients
        if self.gradients is None:
            raise RuntimeError("No gradients captured. Ensure backward pass completed.")
        
        # Compute weights based on method
        if self.method == 'gradcam':
            # Standard Grad-CAM: Global Average Pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        elif self.method == 'gradcampp':
            # Grad-CAM++: Normalized gradients squared
            gradients_squared = self.gradients ** 2
            gradients_cubed = self.gradients ** 3
            
            # Global average pooling
            sum_gradients_squared = torch.sum(gradients_squared, dim=(2, 3), keepdim=True)
            sum_gradients_cubed = torch.sum(gradients_cubed, dim=(2, 3), keepdim=True)
            
            # Normalize weights
            weights = gradients_squared / (2 * sum_gradients_squared + 
                                         sum_gradients_cubed * self.activations + 1e-8)
            weights = torch.sum(weights, dim=(2, 3), keepdim=True)
        
        # Compute CAM: weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU to get positive activations only
        
        # Convert to numpy and squeeze batch dimension
        cam = cam.squeeze().cpu().numpy()  # (H, W)
        
        # Contrast normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)
        
        # Compute statistics
        stats = {
            'cam_mean': float(cam.mean()),
            'cam_max': float(cam.max()),
            'cam_top10': float(np.quantile(cam.flatten(), 0.9))
        }
        
        return cam, stats
    
    def generate_overlay(self, cam: np.ndarray, image_tensor: torch.Tensor, 
                        colormap: str = 'jet') -> np.ndarray:
        """
        Generate overlay visualization of CAM on original image.
        
        Parameters:
            cam: CAM heatmap (H, W) in range [0, 1]
            image_tensor: Original image tensor (C, H, W) normalized with ImageNet stats
            colormap: Colormap name ('jet' or 'turbo')
            
        Returns:
            np.ndarray: Overlay image (H, W, 3) in RGB format
        """
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Denormalize: pixel = pixel * std + mean
        image_denorm = image_tensor * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # Convert to numpy and transpose to (H, W, C)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8
        image_uint8 = (image_np * 255).astype(np.uint8)
        
        # Apply colormap to CAM
        colormap_cv2 = getattr(cv2, f'COLORMAP_{colormap.upper()}')
        cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap_cv2)
        
        # Convert BGR to RGB
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Resize CAM to match image size if needed
        if cam_colored.shape[:2] != image_uint8.shape[:2]:
            cam_colored = cv2.resize(cam_colored, (image_uint8.shape[1], image_uint8.shape[0]))
        
        # Blend images (50/50 alpha)
        overlay = cv2.addWeighted(image_uint8, 0.5, cam_colored, 0.5, 0)
        
        return overlay


def get_target_layer(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """
    Get target layer from model using string path.
    
    Parameters:
        model: PyTorch model
        layer_path: String path to layer (e.g., 'layer4[-1]' or 'layer4.2')
        
    Returns:
        torch.nn.Module: Target layer
    """
    # Handle special case for ResNet layer4[-1]
    if layer_path == 'layer4[-1]':
        return model.layer4[-1]
    
    # Parse layer path
    parts = layer_path.split('.')
    layer = model
    
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            # Handle indexing like [0] or [-1]
            index = int(part[1:-1])
            layer = layer[index]
        else:
            # Handle attribute access
            layer = getattr(layer, part)
    
    return layer


def create_gradcam(model: torch.nn.Module, layer_path: str = 'layer4[-1]', 
                  method: str = 'gradcam') -> GradCAM:
    """
    Convenience function to create GradCAM instance.
    
    Parameters:
        model: PyTorch model
        layer_path: String path to target layer
        method: 'gradcam' or 'gradcampp'
        
    Returns:
        GradCAM: Configured GradCAM instance
    """
    target_layer = get_target_layer(model, layer_path)
    return GradCAM(model, target_layer, method)
