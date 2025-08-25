"""
Training Utilities for MedMNIST Training Pipeline

This module provides comprehensive training utilities including:
- Data augmentation strategies for 3D data
- Loss functions for classification and segmentation
- Evaluation metrics and logging functions
- Data loading utilities and batch processing
- Model initialization and optimization strategies
- Training configuration helpers
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
    
    Includes standard losses and medical-specific losses with proper handling
    of class imbalance and multi-class scenarios.
    """
    
    @staticmethod
    def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
        """
        Get loss function by name.
        
        Args:
            loss_name: Name of the loss function
            **kwargs: Additional arguments for the loss function
            
        Returns:
            Loss function instance
        """
        loss_functions = {
            'cross_entropy': LossFunctions.cross_entropy_loss,
            'focal': LossFunctions.focal_loss,
            'dice': LossFunctions.dice_loss,
            'weighted_cross_entropy': LossFunctions.weighted_cross_entropy_loss,
            'focal_dice': LossFunctions.focal_dice_loss
        }
        
        if loss_name not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
        
        return loss_functions[loss_name](**kwargs)
    
    @staticmethod
    def cross_entropy_loss(**kwargs) -> nn.Module:
        """Standard cross-entropy loss."""
        return nn.CrossEntropyLoss(**kwargs)
    
    @staticmethod
    def focal_loss(alpha: float = 1.0, gamma: float = 2.0, **kwargs) -> nn.Module:
        """Focal loss for handling class imbalance."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce = nn.CrossEntropyLoss(reduction='none', **kwargs)
            
            def forward(self, inputs, targets):
                ce_loss = self.ce(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    @staticmethod
    def dice_loss(smooth: float = 1e-6, **kwargs) -> nn.Module:
        """Dice loss for segmentation tasks."""
        class DiceLoss(nn.Module):
            def __init__(self, smooth=smooth):
                super().__init__()
                self.smooth = smooth
            
            def forward(self, inputs, targets):
                # Convert to one-hot if needed
                if targets.dim() == 1:
                    targets = F.one_hot(targets, num_classes=inputs.size(1))
                
                # Apply softmax to inputs
                inputs = F.softmax(inputs, dim=1)
                
                # Flatten
                inputs = inputs.view(-1)
                targets = targets.view(-1)
                
                intersection = (inputs * targets).sum()
                dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
                return 1 - dice
        
        return DiceLoss()
    
    @staticmethod
    def weighted_cross_entropy_loss(class_weights: Optional[torch.Tensor] = None, **kwargs) -> nn.Module:
        """Weighted cross-entropy loss for class imbalance."""
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
        return nn.CrossEntropyLoss(**kwargs)
    
    @staticmethod
    def focal_dice_loss(alpha: float = 0.5, gamma: float = 2.0, smooth: float = 1e-6, **kwargs) -> nn.Module:
        """Combined focal and dice loss."""
        class FocalDiceLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma, smooth=smooth):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.smooth = smooth
                self.ce = nn.CrossEntropyLoss(reduction='none', **kwargs)
            
            def forward(self, inputs, targets):
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
                
                # Combine losses
                total_loss = focal_loss.mean() + (1 - self.alpha) * dice_loss
                return total_loss
        
        return FocalDiceLoss()


class EvaluationMetrics:
    """
    Collection of evaluation metrics for medical image classification.
    
    Implements standard classification metrics and medical-specific metrics
    with proper handling of multi-class scenarios.
    """
    
    def __init__(self, num_classes: int, device: torch.device = torch.device('cpu')):
        """
        Initialize evaluation metrics.
        
        Args:
            num_classes: Number of classes
            device: Device for computations
        """
        self.num_classes = num_classes
        self.device = device
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
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)
        
        # Compute metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (all_predictions == all_targets).float().mean().item()
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_mask = all_targets == i
            if class_mask.sum() > 0:
                # Precision
                tp = torch.sum((all_predictions == i) & class_mask).item()
                fp = torch.sum((all_predictions == i) & ~class_mask).item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics[f'precision_class_{i}'] = precision
                
                # Recall
                fn = torch.sum((all_predictions != i) & class_mask).item()
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


def create_dataloader_from_config(config: Dict[str, Any]) -> MedMNIST3DDataModule:
    """
    Create MedMNIST dataloader from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MedMNIST3DDataModule
    """
    if not MEDMNIST_AVAILABLE:
        raise ImportError("MedMNIST modules not available")
    
    data_config = config.get('data', {})
    
    return MedMNIST3DDataModule(
        dataset_name=data_config.get('dataset_name', 'organmnist3d'),
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

