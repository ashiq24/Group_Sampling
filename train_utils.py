"""
Training Utilities for Group Equivariant CNN Training Pipeline

This module provides comprehensive training utilities for both classification and
segmentation tasks using Group Equivariant CNNs. It includes:

CORE FEATURES:
- Data augmentation strategies for 3D medical images
- Loss functions for classification and segmentation tasks
- Evaluation metrics and logging functions
- Data loading utilities and batch processing
- Model initialization and optimization strategies
- Training configuration helpers

LOSS FUNCTIONS:
- Classification: Cross-entropy loss
- Segmentation: Focal Dice, Soft Dice, Tversky loss
- Custom loss functions with mathematical formulations

EVALUATION METRICS:
- Classification: Accuracy, Precision, Recall, F1-Score
- Segmentation: Dice coefficient, IoU, Hausdorff distance
- Comprehensive metric computation and logging

DATA AUGMENTATION:
- 3D spatial transformations (rotation, scaling, flipping)
- Intensity transformations (noise, contrast, brightness)
- Group-equivariant augmentations

MATHEMATICAL FOUNDATIONS:
- Dice Coefficient: 2|A∩B| / (|A| + |B|)
- Focal Loss: -α(1-p_t)^γ log(p_t)
- Tversky Loss: |A∩B| / (|A∩B| + α|A\B| + β|B\A|)
- IoU: |A∩B| / |A∪B|

USAGE:
    # Loss functions
    loss_fn = LossFunctions.get_loss_function("focal_dice")
    
    # Evaluation metrics
    metrics = EvaluationMetrics(task_type="segmentation")
    
    # Data augmentation
    aug = DataAugmentation3D(rotation_range=15, scale_range=0.1)
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
)
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from data.medmnist_loader import (
        MedMNIST3DDataset, 
        MedMNIST3DDataModule, 
        MEDMNIST_3D_DATASETS
    )
    MEDMNIST_AVAILABLE = True
except ImportError as e:
    MEDMNIST_AVAILABLE = False
    warnings.warn(f"Could not import MedMNIST modules: {e}")
    # Define placeholder classes for testing
    class MedMNIST3DDataModule:
        pass


class DataAugmentation3D:
    """
    3D data augmentation strategies for medical imaging.
    
    Implements various augmentation techniques suitable for 3D medical data
    including rotations, scaling, flipping, and intensity transformations.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_probability: float = 0.5,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        noise_std: float = 0.01
    ):
        """
        Initialize 3D data augmentation.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            scale_range: Range for scaling (min, max)
            flip_probability: Probability of flipping along each axis
            intensity_shift: Maximum intensity shift
            intensity_scale: Maximum intensity scaling
            noise_std: Standard deviation of Gaussian noise
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_probability = flip_probability
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.noise_std = noise_std
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D data augmentation to input tensor.
        
        Args:
            img: Input tensor of shape (C, H, W, D)
            
        Returns:
            Augmented tensor of same shape
        """
        if img.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {img.dim()}D")
        
        # Apply augmentations
        img = self._random_flip(img)
        img = self._random_rotation(img)
        img = self._random_scaling(img)
        img = self._intensity_transform(img)
        img = self._add_noise(img)
        
        return img
    
    def _random_flip(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random flipping along each axis."""
        if torch.rand(1) < self.flip_probability:
            img = torch.flip(img, dims=[1])  # Flip H
        if torch.rand(1) < self.flip_probability:
            img = torch.flip(img, dims=[2])  # Flip W
        if torch.rand(1) < self.flip_probability:
            img = torch.flip(img, dims=[3])  # Flip D
        return img
    
    def _random_rotation(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random 3D rotation."""
        if self.rotation_range > 0:
            # Simple rotation by permuting axes (approximation)
            if torch.rand(1) < 0.5:
                img = img.permute(0, 2, 3, 1)  # (C, W, D, H)
            if torch.rand(1) < 0.5:
                img = img.permute(0, 3, 1, 2)  # (C, D, H, W)
        return img
    
    def _random_scaling(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            # Simple scaling by interpolation (approximation)
            if scale != 1.0:
                img = F.interpolate(
                    img.unsqueeze(0), 
                    scale_factor=scale.item(), 
                    mode='trilinear', 
                    align_corners=False
                ).squeeze(0)
        return img
    
    def _intensity_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply intensity transformations."""
        # Intensity shift
        if self.intensity_shift > 0:
            shift = (torch.rand(1) * 2 - 1) * self.intensity_shift
            img = img + shift
        
        # Intensity scaling
        if self.intensity_scale > 0:
            scale = 1.0 + (torch.rand(1) * 2 - 1) * self.intensity_scale
            img = img * scale
        
        # Clamp to valid range
        img = torch.clamp(img, 0, 1)
        return img
    
    def _add_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise
            img = torch.clamp(img, 0, 1)
        return img


class LossFunctions:
    """
    Collection of loss functions for medical image classification and segmentation.
    
    This class provides various loss functions optimized for 3D medical image analysis:
    - Cross-entropy loss for classification tasks
    - Focal loss for handling class imbalance in classification
    - Dice loss for segmentation with smooth gradients
    - Focal Dice loss for handling class imbalance in segmentation
    - Soft Dice loss for segmentation with smooth gradients
    - Tversky loss for controlling false positive/negative trade-offs
    - Weighted cross-entropy for class imbalance in classification
    
    Mathematical Foundations:
    - Cross-Entropy: CE = -Σ y_i * log(ŷ_i)
    - Dice Coefficient: DSC = 2|A∩B| / (|A| + |B|)
    - Focal Loss: FL = -α(1-p)^γ * log(p)
    - Tversky Index: TI = |A∩B| / (|A∩B| + α|A\B| + β|B\A|)
    
    All loss functions are implemented as PyTorch modules for seamless integration
    with the training pipeline and automatic gradient computation.
    """
    
    @staticmethod
    def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
        """
        Get loss function by name with automatic parameter passing.
        
        This factory method provides a unified interface for creating loss functions
        with their specific parameters. It automatically handles parameter validation
        and passes the correct arguments to each loss function constructor.
        
        Args:
            loss_name: Name of the loss function to create
                      Available options:
                      - 'cross_entropy': Standard cross-entropy loss
                      - 'focal': Focal loss for class imbalance
                      - 'dice': Dice loss for segmentation
                      - 'weighted_cross_entropy': Weighted cross-entropy loss
                      - 'focal_dice': Combined focal and dice loss
                      - 'soft_dice': Soft dice loss for segmentation
                      - 'tversky': Tversky loss for FP/FN control
            **kwargs: Additional arguments specific to each loss function
                     Common parameters:
                     - alpha: Focal loss alpha parameter (default: 1.0)
                     - gamma: Focal loss gamma parameter (default: 2.0)
                     - smooth: Smoothing factor for dice/tversky (default: 1e-6)
                     - class_weights: Tensor of class weights for weighted losses
                     
        Returns:
            PyTorch module implementing the specified loss function
            
        Raises:
            ValueError: If loss_name is not recognized
            
        Example:
            # Create focal dice loss with custom parameters
            loss_fn = LossFunctions.get_loss_function(
                'focal_dice',
                alpha=0.5,
                gamma=2.0,
                smooth=1e-6
            )
            
            # Create weighted cross-entropy loss
            class_weights = torch.tensor([1.0, 2.0, 3.0])  # Higher weight for rare classes
            loss_fn = LossFunctions.get_loss_function(
                'weighted_cross_entropy',
                class_weights=class_weights
            )
        """
        loss_functions = {
            'cross_entropy': LossFunctions.cross_entropy_loss,
            'focal': LossFunctions.focal_loss,
            'dice': LossFunctions.dice_loss,
            'weighted_cross_entropy': LossFunctions.weighted_cross_entropy_loss,
            'focal_dice': LossFunctions.focal_dice_loss,
            'soft_dice': LossFunctions.soft_dice_loss,
            'tversky': LossFunctions.tversky_loss
        }
        
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
        
        return loss_functions[loss_name](**kwargs)
    
    @staticmethod
    def cross_entropy_loss(**kwargs) -> nn.Module:
        """
        Create standard cross-entropy loss for classification tasks.
        
        Cross-entropy loss measures the difference between predicted probability
        distribution and true distribution. For multi-class classification:
        CE = -Σ y_i * log(ŷ_i)
        where y_i is true label and ŷ_i is predicted probability.
        
        Args:
            **kwargs: Additional arguments passed to nn.CrossEntropyLoss
                     Common parameters:
                     - weight: Optional class weights tensor
                     - reduction: Reduction method ('mean', 'sum', 'none')
                     - ignore_index: Index to ignore in loss computation
                     
        Returns:
            PyTorch CrossEntropyLoss module
            
        Mathematical Details:
            For input logits x and target class indices y:
            CE = -log(softmax(x)[y]) = -log(exp(x[y]) / Σ exp(x[i]))
            
        Usage:
            loss_fn = LossFunctions.cross_entropy_loss()
            loss = loss_fn(predictions, targets)
        """
        return nn.CrossEntropyLoss(**kwargs)
    
    @staticmethod
    def focal_loss(alpha: float = 1.0, gamma: float = 2.0, **kwargs) -> nn.Module:
        """
        Create Focal loss for handling class imbalance in classification tasks.
        
        Focal loss addresses class imbalance by down-weighting easy examples and
        focusing on hard examples. It modifies cross-entropy loss with a modulating
        factor: FL = -α(1-p)^γ * log(p)
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
                   - α=1: no class weighting
                   - α>1: up-weight rare class
            gamma: Focusing parameter (default: 2.0)
                   - γ=0: equivalent to cross-entropy
                   - γ=2: moderate focus on hard examples
                   - γ>2: strong focus on hard examples
            **kwargs: Additional arguments passed to underlying CrossEntropyLoss
                     
        Returns:
            PyTorch module implementing Focal loss
            
        Mathematical Details:
            For predicted probability p_t = exp(-CE_loss):
            FL = -α(1-p_t)^γ * log(p_t)
            
            Where:
            - p_t is the predicted probability for the true class
            - α controls class weighting
            - γ controls focus on hard examples
            
        Benefits:
            - Automatically handles class imbalance
            - Focuses learning on hard examples
            - Reduces contribution of easy examples
            - Improves performance on imbalanced datasets
            
        Usage:
            loss_fn = LossFunctions.focal_loss(alpha=1.0, gamma=2.0)
            loss = loss_fn(predictions, targets)
        """
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha  # Class weighting factor
                self.gamma = gamma  # Focusing parameter
                self.ce = nn.CrossEntropyLoss(reduction='none', **kwargs)
            
            def forward(self, inputs, targets):
                # Compute cross-entropy loss for each sample
                ce_loss = self.ce(inputs, targets)
                # Convert to probability: p_t = exp(-CE_loss)
                pt = torch.exp(-ce_loss)
                # Apply focal weighting: α(1-p_t)^γ * CE_loss
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    @staticmethod
    def dice_loss(smooth: float = 1e-6, **kwargs) -> nn.Module:
        """
        Create Dice loss for segmentation tasks.
        
        Dice loss directly optimizes the Dice coefficient, which is a common metric
        for segmentation tasks. It measures the overlap between predicted and ground
        truth masks: DSC = 2|A∩B| / (|A| + |B|)
        
        Args:
            smooth: Smoothing factor to avoid division by zero (default: 1e-6)
                   - Prevents numerical instability
                   - Higher values provide more smoothing
            **kwargs: Additional arguments (unused, for compatibility)
                     
        Returns:
            PyTorch module implementing Dice loss
            
        Mathematical Details:
            For predicted probabilities p and target masks t:
            DSC = 2 * Σ(p * t) / (Σ(p) + Σ(t))
            Dice Loss = 1 - DSC
            
            Where:
            - p: predicted probabilities (after softmax)
            - t: target masks (one-hot encoded)
            - Σ: sum over all spatial locations
            
        Benefits:
            - Directly optimizes Dice coefficient
            - Handles class imbalance naturally
            - Smooth gradients for stable training
            - Well-suited for medical image segmentation
            
        Usage:
            loss_fn = LossFunctions.dice_loss(smooth=1e-6)
            loss = loss_fn(predictions, targets)
        """
        class DiceLoss(nn.Module):
            def __init__(self, smooth=smooth):
                super().__init__()
                self.smooth = smooth  # Smoothing factor for numerical stability
            
            def forward(self, inputs, targets):
                # Convert targets to one-hot if needed
                if targets.dim() == 1:
                    targets = F.one_hot(targets, num_classes=inputs.size(1))
                
                # Apply softmax to inputs to get probabilities
                inputs = F.softmax(inputs, dim=1)
                
                # Flatten spatial dimensions for computation
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                
                # Compute Dice coefficient
                intersection = (inputs * targets).sum()
                dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
                
                # Return Dice loss (1 - Dice coefficient)
                return 1 - dice
        
        return DiceLoss()
    
    @staticmethod
    def weighted_cross_entropy_loss(class_weights: Optional[torch.Tensor] = None, **kwargs) -> nn.Module:
        """
        Create weighted cross-entropy loss for handling class imbalance.
        
        Weighted cross-entropy loss applies different weights to different classes
        during loss computation. This is particularly useful for imbalanced datasets
        where some classes are underrepresented.
        
        Args:
            class_weights: Optional tensor of class weights of shape (num_classes,)
                          - Higher weights for underrepresented classes
                          - Weights are applied as: loss = weight[class] * CE_loss
                          - If None, returns standard cross-entropy loss
            **kwargs: Additional arguments passed to nn.CrossEntropyLoss
                     Common parameters:
                     - reduction: Reduction method ('mean', 'sum', 'none')
                     - ignore_index: Index to ignore in loss computation
                     
        Returns:
            PyTorch CrossEntropyLoss module with optional class weights
            
        Mathematical Details:
            For class weights w and cross-entropy loss CE:
            Weighted CE = w[y] * CE(x, y)
            
            Where:
            - w[y] is the weight for class y
            - CE(x, y) is the standard cross-entropy loss
            - Higher weights increase the loss contribution for that class
            
        Benefits:
            - Directly addresses class imbalance
            - Simple and effective approach
            - Compatible with standard optimization techniques
            - Easy to tune and interpret
            
        Usage:
            # Create weighted loss for imbalanced dataset
            class_weights = torch.tensor([1.0, 2.0, 3.0])  # Higher weight for rare classes
            loss_fn = LossFunctions.weighted_cross_entropy_loss(class_weights=class_weights)
            loss = loss_fn(predictions, targets)
            
            # Create standard cross-entropy loss
            loss_fn = LossFunctions.weighted_cross_entropy_loss()
            loss = loss_fn(predictions, targets)
        """
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
        return nn.CrossEntropyLoss(**kwargs)
    
    @staticmethod
    def focal_dice_loss(alpha: float = 0.5, gamma: float = 2.0, smooth: float = 1e-6, **kwargs) -> nn.Module:
        """
        Create combined Focal and Dice loss for segmentation with class imbalance handling.
        
        This loss combines the benefits of both Focal loss (handles class imbalance)
        and Dice loss (optimizes segmentation metric). It's particularly effective
        for medical image segmentation where class imbalance is common.
        
        Args:
            alpha: Weighting factor between focal and dice components (default: 0.5)
                   - α=0: pure dice loss
                   - α=1: pure focal loss
                   - α=0.5: balanced combination
            gamma: Focal loss focusing parameter (default: 2.0)
                   - γ=0: no focusing (equivalent to cross-entropy)
                   - γ=2: moderate focus on hard examples
                   - γ>2: strong focus on hard examples
            smooth: Smoothing factor for dice computation (default: 1e-6)
                    - Prevents division by zero
                    - Higher values provide more smoothing
            **kwargs: Additional arguments passed to underlying CrossEntropyLoss
                     
        Returns:
            PyTorch module implementing combined Focal-Dice loss
            
        Mathematical Details:
            Total Loss = α * Focal_Loss + (1-α) * Dice_Loss
            
            Where:
            - Focal_Loss = -α_focal * (1-p_t)^γ * log(p_t)
            - Dice_Loss = 1 - (2|A∩B| + smooth) / (|A| + |B| + smooth)
            - α controls the balance between components
            
        Benefits:
            - Handles class imbalance through focal component
            - Optimizes Dice coefficient through dice component
            - Automatically handles size mismatches
            - Well-suited for medical image segmentation
            - Provides smooth gradients for stable training
            
        Usage:
            loss_fn = LossFunctions.focal_dice_loss(alpha=0.5, gamma=2.0)
            loss = loss_fn(predictions, targets)
        """
        class FocalDiceLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma, smooth=smooth):
                super().__init__()
                self.alpha = alpha  # Weighting between focal and dice
                self.gamma = gamma  # Focal loss focusing parameter
                self.smooth = smooth  # Dice loss smoothing factor
                self.ce = nn.CrossEntropyLoss(reduction='none', **kwargs)
            
            def forward(self, inputs, targets):
                # Handle size mismatch between inputs and targets
                if inputs.shape[2:] != targets.shape[1:]:
                    print(f"Size mismatch: inputs {inputs.shape[2:]} vs targets {targets.shape[1:]}, interpolating...")
                    inputs = F.interpolate(inputs, size=targets.shape[1:], mode='trilinear', align_corners=False)
                
                # Focal loss component
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                # Dice loss component
                if targets.dim() == 1:
                    targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
                else:
                    targets_one_hot = targets
                
                inputs_softmax = F.softmax(inputs, dim=1)
                intersection = (inputs_softmax * targets_one_hot).sum()
                dice = (2. * intersection + self.smooth) / (inputs_softmax.sum() + targets_one_hot.sum() + self.smooth)
                dice_loss = 1 - dice
                
                # Combine losses with weighting
                total_loss = focal_loss.mean() + (1 - self.alpha) * dice_loss
                return total_loss
        
        return FocalDiceLoss()
    
    @staticmethod
    def soft_dice_loss(smooth: float = 1e-6, **kwargs) -> nn.Module:
        """
        Create Soft Dice loss for segmentation tasks with per-class computation.
        
        Soft Dice loss computes the Dice coefficient for each class separately and
        averages them. This approach is particularly effective for multi-class
        segmentation tasks where different classes may have different importance.
        
        Args:
            smooth: Smoothing factor to avoid division by zero (default: 1e-6)
                   - Prevents numerical instability
                   - Higher values provide more smoothing
            **kwargs: Additional arguments (unused, for compatibility)
                     
        Returns:
            PyTorch module implementing Soft Dice loss
            
        Mathematical Details:
            For each class c:
            DSC_c = 2 * Σ(p_c * t_c) / (Σ(p_c) + Σ(t_c))
            Soft Dice Loss = 1 - mean(DSC_c)
            
            Where:
            - p_c: predicted probabilities for class c (after softmax)
            - t_c: target mask for class c (one-hot encoded)
            - Σ: sum over all spatial locations
            - mean: average across all classes
            
        Benefits:
            - Per-class Dice computation handles class imbalance
            - Smooth gradients enable stable training
            - Directly optimizes Dice coefficient
            - Well-suited for multi-class segmentation
            - Handles both 4D and 5D target formats automatically
            
        Usage:
            loss_fn = LossFunctions.soft_dice_loss(smooth=1e-6)
            loss = loss_fn(predictions, targets)
        """
        class SoftDiceLoss(nn.Module):
            def __init__(self, smooth=smooth):
                super().__init__()
                self.smooth = smooth  # Smoothing factor for numerical stability
            
            def forward(self, inputs, targets):
                # Convert targets to one-hot if needed
                if targets.dim() == 4:  # (B, D, H, W) - segmentation labels
                    targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
                    # Move class dimension to position 1: (B, D, H, W, C) -> (B, C, D, H, W)
                    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)
                elif targets.dim() == 5:  # Already one-hot (B, C, D, H, W)
                    targets_one_hot = targets
                else:
                    targets_one_hot = targets
                
                # Apply softmax to inputs to get probabilities
                inputs_softmax = F.softmax(inputs, dim=1)
                
                # Flatten spatial dimensions for per-class computation
                inputs_flat = inputs_softmax.view(inputs_softmax.size(0), inputs_softmax.size(1), -1)  # (B, C, D*H*W)
                targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # (B, C, D*H*W)
                
                # Compute dice per class
                intersection = (inputs_flat * targets_flat).sum(dim=2)  # (B, C)
                union = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                
                # Return mean dice loss (1 - dice)
                return 1 - dice.mean()
        
        return SoftDiceLoss()
    
    @staticmethod
    def tversky_loss(alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6, **kwargs) -> nn.Module:
        """
        Create Tversky loss for segmentation with controllable false positive/negative trade-offs.
        
        Tversky loss generalizes Dice loss by allowing different weights for false positives
        and false negatives. This is particularly useful when false positives and false
        negatives have different clinical implications in medical image segmentation.
        
        Args:
            alpha: False positive weight (default: 0.3)
                   - Higher alpha penalizes false positives more
                   - α=0.5, β=0.5: equivalent to Dice loss
                   - α < β: more sensitive to false negatives (better recall)
            beta: False negative weight (default: 0.7)
                  - Higher beta penalizes false negatives more
                  - β > α: more sensitive to false negatives (better recall)
                  - α > β: more sensitive to false positives (better precision)
            smooth: Smoothing factor to avoid division by zero (default: 1e-6)
                    - Prevents numerical instability
                    - Higher values provide more smoothing
            **kwargs: Additional arguments (unused, for compatibility)
                     
        Returns:
            PyTorch module implementing Tversky loss
            
        Mathematical Details:
            For each class c:
            TI_c = |A_c ∩ B_c| / (|A_c ∩ B_c| + α|A_c \ B_c| + β|B_c \ A_c|)
            Tversky Loss = 1 - mean(TI_c)
            
            Where:
            - A_c: predicted mask for class c
            - B_c: target mask for class c
            - |A_c ∩ B_c|: true positives
            - |A_c \ B_c|: false positives
            - |B_c \ A_c|: false negatives
            - α, β: weights for false positives and false negatives
            
        Clinical Applications:
            - α < β: Prioritize recall over precision (e.g., tumor detection)
            - α > β: Prioritize precision over recall (e.g., healthy tissue preservation)
            - α = β = 0.5: Balanced approach (equivalent to Dice loss)
            
        Benefits:
            - Controllable trade-off between precision and recall
            - Handles class imbalance through per-class computation
            - Smooth gradients for stable training
            - Well-suited for medical image segmentation
            - Handles both 4D and 5D target formats automatically
            
        Usage:
            # Prioritize recall (detect more tumors)
            loss_fn = LossFunctions.tversky_loss(alpha=0.3, beta=0.7)
            loss = loss_fn(predictions, targets)
            
            # Prioritize precision (avoid false alarms)
            loss_fn = LossFunctions.tversky_loss(alpha=0.7, beta=0.3)
            loss = loss_fn(predictions, targets)
        """
        class TverskyLoss(nn.Module):
            def __init__(self, alpha=alpha, beta=beta, smooth=smooth):
                super().__init__()
                self.alpha = alpha  # False positive weight
                self.beta = beta    # False negative weight
                self.smooth = smooth  # Smoothing factor for numerical stability
            
            def forward(self, inputs, targets):
                # Convert targets to one-hot if needed
                if targets.dim() == 4:  # (B, D, H, W) - segmentation labels
                    targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
                    # Move class dimension to position 1: (B, D, H, W, C) -> (B, C, D, H, W)
                    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)
                elif targets.dim() == 5:  # Already one-hot (B, C, D, H, W)
                    targets_one_hot = targets
                else:
                    targets_one_hot = targets
                
                # Apply softmax to inputs to get probabilities
                inputs_softmax = F.softmax(inputs, dim=1)
                
                # Flatten spatial dimensions for per-class computation
                inputs_flat = inputs_softmax.view(inputs_softmax.size(0), inputs_softmax.size(1), -1)  # (B, C, D*H*W)
                targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)  # (B, C, D*H*W)
                
                # Compute tversky per class
                intersection = (inputs_flat * targets_flat).sum(dim=2)  # (B, C) - true positives
                false_positive = (inputs_flat * (1 - targets_flat)).sum(dim=2)  # (B, C) - false positives
                false_negative = ((1 - inputs_flat) * targets_flat).sum(dim=2)  # (B, C) - false negatives
                
                # Compute Tversky index with smoothing
                tversky = (intersection + self.smooth) / (intersection + self.alpha * false_positive + self.beta * false_negative + self.smooth)
                
                # Return Tversky loss (1 - mean Tversky index)
                return 1 - tversky.mean()
        
        return TverskyLoss()


class EvaluationMetrics:
    """
    Collection of evaluation metrics for medical image classification and segmentation.
    
    Implements standard classification metrics and medical-specific metrics
    with proper handling of multi-class scenarios.
    """
    
    def __init__(self, num_classes: int, device: torch.device = torch.device('cpu'), task_type: str = 'classification'):
        """
        Initialize evaluation metrics.
        
        Args:
            num_classes: Number of classes
            device: Device for computations
            task_type: Type of task ('classification' or 'segmentation')
        """
        self.num_classes = num_classes
        self.device = device
        self.task_type = task_type
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: Model predictions (logits)
            targets: Ground truth targets
        """
        # Convert to CPU if needed
        if predictions.device != self.device:
            predictions = predictions.cpu()
        if targets.device != self.device:
            targets = targets.cpu()
        
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Store for later analysis
        self.predictions.append(pred_classes)
        self.targets.append(targets)
        
        # Update confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += torch.sum((pred_classes == i) & (targets == j)).item()
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        if not self.predictions:
            return {}
        
        # Find the maximum spatial dimensions across all predictions
        max_dims = [0, 0, 0]  # D, H, W
        for pred in self.predictions:
            if pred.dim() == 5:  # (B, C, D, H, W)
                max_dims[0] = max(max_dims[0], pred.shape[2])
                max_dims[1] = max(max_dims[1], pred.shape[3])
                max_dims[2] = max(max_dims[2], pred.shape[4])
            elif pred.dim() == 4:  # (B, D, H, W)
                max_dims[0] = max(max_dims[0], pred.shape[1])
                max_dims[1] = max(max_dims[1], pred.shape[2])
                max_dims[2] = max(max_dims[2], pred.shape[3])

        # Interpolate all predictions and targets to the same size
        interpolated_predictions = []
        interpolated_targets = []
        
        for pred, target in zip(self.predictions, self.targets):
            if pred.dim() == 5:  # (B, C, D, H, W)
                if pred.shape[2:] != tuple(max_dims):
                    pred = F.interpolate(pred, size=max_dims, mode='trilinear', align_corners=False)
                if target.shape[1:] != tuple(max_dims):
                    target = F.interpolate(target.unsqueeze(1).float(), size=max_dims, mode='nearest').squeeze(1).long()
            elif pred.dim() == 4:  # (B, D, H, W)
                if pred.shape[1:] != tuple(max_dims):
                    pred = F.interpolate(pred.unsqueeze(1).float(), size=max_dims, mode='trilinear', align_corners=False).squeeze(1)
                if target.shape[1:] != tuple(max_dims):
                    target = F.interpolate(target.unsqueeze(1).float(), size=max_dims, mode='nearest').squeeze(1).long()
            
            interpolated_predictions.append(pred)
            interpolated_targets.append(target)

        # Concatenate all predictions and targets
        all_predictions = torch.cat(interpolated_predictions)
        all_targets = torch.cat(interpolated_targets)
        
        # Compute metrics
        metrics = {}
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics.update(self._compute_classification_metrics(all_predictions, all_targets))
        elif self.task_type == 'segmentation':
            # Segmentation metrics
            metrics.update(self._compute_segmentation_metrics(all_predictions, all_targets))
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        return metrics
    
    def _compute_classification_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute classification-specific metrics."""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (predictions == targets).float().mean().item()
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_mask = targets == i
            if class_mask.sum() > 0:
                # Precision
                tp = torch.sum((predictions == i) & class_mask).item()
                fp = torch.sum((predictions == i) & ~class_mask).item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics[f'precision_class_{i}'] = precision
                
                # Recall
                fn = torch.sum((predictions != i) & class_mask).item()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics[f'recall_class_{i}'] = recall
                
                # F1-score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                metrics[f'f1_class_{i}'] = f1
        
        # Macro averages
        metrics['macro_precision'] = np.mean([metrics.get(f'precision_class_{i}', 0) for i in range(self.num_classes)])
        metrics['macro_recall'] = np.mean([metrics.get(f'recall_class_{i}', 0) for i in range(self.num_classes)])
        metrics['macro_f1'] = np.mean([metrics.get(f'f1_class_{i}', 0) for i in range(self.num_classes)])
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.confusion_matrix.cpu().numpy()
        
        return metrics
    
    def _compute_segmentation_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation-specific metrics."""
        metrics = {}
        
        # Convert to class indices for one-hot encoding
        if predictions.dim() == 5:  # (B, C, D, H, W) - logits
            predictions = torch.argmax(predictions, dim=1)  # (B, D, H, W)
        elif predictions.dim() == 4:  # (B, D, H, W) - already indices
            predictions = predictions.long()
        else:
            raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
        
        # Ensure targets are long integers
        targets = targets.long()
        
        # Convert to one-hot for segmentation metrics
        predictions_one_hot = F.one_hot(predictions, num_classes=self.num_classes)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        
        # Per-class Dice coefficient
        dice_scores = []
        for i in range(self.num_classes):
            if i == 0:  # Skip background
                continue
                
            pred_class = predictions_one_hot[..., i]
            target_class = targets_one_hot[..., i]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            dice = (2.0 * intersection) / (union + 1e-6)
            dice_scores.append(dice.item())
            metrics[f'dice_class_{i}'] = dice.item()
        
        # Mean Dice (excluding background)
        metrics['dice_mean'] = np.mean(dice_scores) if dice_scores else 0.0
        
        # Overall accuracy
        metrics['accuracy'] = (predictions == targets).float().mean().item()
        
        # Per-class IoU
        for i in range(self.num_classes):
            if i == 0:  # Skip background
                continue
                
            pred_class = predictions_one_hot[..., i]
            target_class = targets_one_hot[..., i]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum() - intersection
            iou = intersection / (union + 1e-6)
            metrics[f'iou_class_{i}'] = iou.item()
        
        # Mean IoU (excluding background)
        iou_scores = [metrics.get(f'iou_class_{i}', 0) for i in range(1, self.num_classes)]
        metrics['iou_mean'] = np.mean(iou_scores) if iou_scores else 0.0
        
        return metrics
    
    def get_confusion_matrix(self) -> torch.Tensor:
        """Get the confusion matrix."""
        return self.confusion_matrix


class OptimizerFactory:
    """
    Factory for creating optimizers with different configurations.
    
    Supports common optimizers with configurable parameters and
    learning rate scheduling strategies.
    """
    
    @staticmethod
    def get_optimizer(
        model: nn.Module,
        optimizer_name: str = 'adam',
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Get optimizer by name.
        
        Args:
            model: PyTorch model
            optimizer_name: Name of the optimizer
            **kwargs: Optimizer parameters
            
        Returns:
            Optimizer instance
        """
        # Get model parameters
        if 'weight_decay' in kwargs:
            # Convert weight_decay to float if it's a string
            weight_decay = float(kwargs['weight_decay']) if isinstance(kwargs['weight_decay'], str) else kwargs['weight_decay']
            
            if weight_decay > 0:
                # Separate weight decay for different parameter groups
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'weight_decay': weight_decay
                    },
                    {
                        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0
                    }
                ]
                kwargs.pop('weight_decay')
            else:
                optimizer_grouped_parameters = model.parameters()
        else:
            optimizer_grouped_parameters = model.parameters()
        
        # Convert numeric parameters to proper types
        converted_kwargs = {}
        for key, value in kwargs.items():
            if key in ['lr', 'weight_decay', 'eps', 'betas']:
                if isinstance(value, str):
                    if key == 'betas':
                        # Handle betas as a list
                        converted_kwargs[key] = [float(x) for x in value]
                    else:
                        converted_kwargs[key] = float(value)
                else:
                    converted_kwargs[key] = value
            else:
                converted_kwargs[key] = value
        
        # Create optimizer
        if optimizer_name.lower() == 'adam':
            return Adam(optimizer_grouped_parameters, **converted_kwargs)
        elif optimizer_name.lower() == 'adamw':
            return AdamW(optimizer_grouped_parameters, **converted_kwargs)
        elif optimizer_name.lower() == 'sgd':
            return SGD(optimizer_grouped_parameters, **converted_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: adam, adamw, sgd")
    
    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = 'cosine',
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get learning rate scheduler by name.
        
        Args:
            optimizer: Optimizer instance
            scheduler_name: Name of the scheduler
            **kwargs: Scheduler parameters
            
        Returns:
            Scheduler instance
        """
                # Filter parameters based on scheduler type
        if scheduler_name.lower() == 'cosine':
            # CosineAnnealingLR parameters: T_max, eta_min
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['T_max', 'eta_min']}
            return CosineAnnealingLR(optimizer, **filtered_kwargs)
        elif scheduler_name.lower() == 'step':
            # StepLR parameters: step_size, gamma
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['step_size', 'gamma']}
            return StepLR(optimizer, **filtered_kwargs)
        elif scheduler_name.lower() == 'exponential':
            # ExponentialLR parameters: gamma
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['gamma']}
            return ExponentialLR(optimizer, **filtered_kwargs)
        elif scheduler_name.lower() == 'plateau':
            # ReduceLROnPlateau parameters: mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['mode', 'factor', 'patience', 'verbose', 'threshold', 'threshold_mode', 'cooldown', 'min_lr', 'eps']}
            return ReduceLROnPlateau(optimizer, **filtered_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: cosine, step, exponential, plateau")


class ModelInitializer:
    """
    Utilities for initializing model weights and parameters.
    
    Implements various weight initialization strategies and
    parameter initialization methods.
    """
    
    @staticmethod
    def initialize_weights(
        model: nn.Module,
        init_method: str = 'kaiming',
        **kwargs
    ):
        """
        Initialize model weights.
        
        Args:
            model: PyTorch model
            init_method: Initialization method
            **kwargs: Additional initialization parameters
        """
        for module in model.modules():
            try:
                if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                    if init_method.lower() == 'kaiming':
                        nn.init.kaiming_normal_(module.weight, **kwargs)
                    elif init_method.lower() == 'xavier':
                        nn.init.xavier_normal_(module.weight, **kwargs)
                    elif init_method.lower() == 'normal':
                        nn.init.normal_(module.weight, **kwargs)
                    
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                
                elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                
                elif isinstance(module, nn.Linear):
                    if init_method.lower() == 'kaiming':
                        nn.init.kaiming_normal_(module.weight, **kwargs)
                    elif init_method.lower() == 'xavier':
                        nn.init.xavier_normal_(module.weight, **kwargs)
                    elif init_method.lower() == 'normal':
                        nn.init.normal_(module.weight, **kwargs)
                    
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            except RuntimeError as e:
                # Skip initialization for modules with shared weights (e.g., ESCNN layers)
                if "more than one element of the written-to tensor refers to a single memory location" in str(e):
                    continue
                else:
                    raise e
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }


def create_dataloader_from_config(config: Dict[str, Any]) -> Union[MedMNIST3DDataModule, Any]:
    """
    Create dataloader from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DataModule
    """
    data_config = config.get('data', {})
    dataset_name = data_config.get('dataset_name', 'organmnist3d')
    
    if dataset_name == 'acdc':
        # Import ACDC modules
        try:
            from data.acdc_datamodule import ACDCDataModule
            from data.acdc_dataset import ACDCDataset
        except ImportError as e:
            raise ImportError(f"ACDC modules not available: {e}")
        
        return ACDCDataModule(
            root=data_config.get('root', './datasets/acdc'),
            batch_size=data_config.get('batch_size', 2),
            num_workers=data_config.get('num_workers', 8),
            pin_memory=data_config.get('pin_memory', True),
            persistent_workers=data_config.get('persistent_workers', True),
            train_val_split=data_config.get('train_val_split', 0.8),
            use_es_ed_only=data_config.get('use_es_ed_only', True),
            transforms_train=None,  # Will be set up in DataModule
            transforms_val=None
        )
    
    elif dataset_name in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']:
        # MedMNIST datasets
        if not MEDMNIST_AVAILABLE:
            raise ImportError("MedMNIST modules not available")
        
        return MedMNIST3DDataModule(
            dataset_name=dataset_name,
            data_dir=data_config.get('data_dir', './data'),
            batch_size=data_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            normalize=data_config.get('normalize', True),
            norm_method=data_config.get('norm_method', 'minmax'),
            augment=data_config.get('augment', True),
            task_type=data_config.get('task_type', 'auto'),
            size=data_config.get('size', 28)
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test the training utilities
    print("Testing Training Utilities...")
    
    try:
        # Test data augmentation
        aug = DataAugmentation3D()
        test_tensor = torch.randn(1, 64, 64, 64)  # Use 64x64x64 for testing
        augmented = aug(test_tensor)
        print(f"✅ Data augmentation: {test_tensor.shape} -> {augmented.shape}")
        
        # Test loss functions
        ce_loss = LossFunctions.get_loss_function('cross_entropy')
        focal_loss = LossFunctions.get_loss_function('focal')
        print("✅ Loss functions created successfully")
        
        # Test evaluation metrics
        metrics = EvaluationMetrics(num_classes=11)
        print("✅ Evaluation metrics created successfully")
        
        # Test optimizer factory
        test_model = nn.Linear(10, 5)
        optimizer = OptimizerFactory.get_optimizer(test_model, 'adam', lr=0.001)
        scheduler = OptimizerFactory.get_scheduler(optimizer, 'cosine', T_max=100)
        print("✅ Optimizer and scheduler created successfully")
        
        # Test model initializer
        ModelInitializer.initialize_weights(test_model)
        param_counts = ModelInitializer.count_parameters(test_model)
        print(f"✅ Model initialization: {param_counts}")
        
        print("✅ All training utilities tests passed!")
        
    except Exception as e:
        print(f"❌ Error testing training utilities: {e}")
        import traceback
        traceback.print_exc()

