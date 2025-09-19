"""
Main Training Script for Group Equivariant Convolutional Neural Networks

This script provides a complete training pipeline for both classification and segmentation
tasks using PyTorch Lightning with 3D Group Equivariant CNNs. The system supports:

CORE FEATURES:
- 3D Medical Image Classification (OrganMNIST3D dataset)
- 3D Medical Image Segmentation (ACDC dataset)
- Group Equivariance with C4 cyclic subgroup (90° rotations around z-axis)
- Anti-aliasing for group downsampling to prevent artifacts
- 4D U-Net architecture (3D spatial + group axis processing)

TRAINING FEATURES:
- Multi-GPU training support with DDP (Distributed Data Parallel)
- Mixed precision training (FP16, FP32, FP64)
- Comprehensive logging and monitoring (TensorBoard, CSV, Weights & Biases)
- Automatic checkpointing and model saving
- Early stopping and learning rate monitoring
- Test mode for quick pipeline validation

CONFIGURATION:
- YAML-based configuration management
- Dynamic dataset and model loading
- Loss function and metric configuration
- Hardware and optimization settings

MATHEMATICAL FOUNDATIONS:
- Group Theory: Octahedral group (24 elements) with C4 cyclic subgroup (4 elements)
- Fourier Analysis: Group Fourier transforms for efficient computation
- Anti-aliasing: Spectral anti-aliasing to prevent aliasing artifacts
- Equivariance: f(g·x) = g·f(x) for all group elements g

USAGE:
    # Classification training
    python main.py --config organmnist3d_config.yaml --train
    
    # Segmentation training  
    python main.py --config acdc.yaml --train
    
    # Testing
    python main.py --config acdc.yaml --test
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, ModelSummary
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import local modules
try:
    from config.config_loader import ConfigLoader, load_config
    from data.medmnist_loader import MedMNIST3DDataModule
    from train_utils import (
        LossFunctions, EvaluationMetrics, OptimizerFactory, 
        ModelInitializer, create_dataloader_from_config
    )
    LOCAL_MODULES_AVAILABLE = True
except ImportError as e:
    LOCAL_MODULES_AVAILABLE = False
    warnings.warn(f"Could not import local modules: {e}")

# Import GSampling models (only import, don't modify)
try:
    from models.g_cnn_3d import Gcnn3D
    from models.g_cnn_3d_seg import Gcnn3DSegmentation
    from models.model_handler import get_3d_model, get_3d_segmentation_model
    GCNN_AVAILABLE = True
except ImportError as e:
    GCNN_AVAILABLE = False
    warnings.warn(f"Could not import GCNN models: {e}")


class MedMNISTLightningModule(LightningModule):
    """
    PyTorch Lightning module for MedMNIST training with 3D GCNNs.
    
    Implements the training, validation, and testing logic with proper
    logging, checkpointing, and evaluation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Lightning module for Group Equivariant CNN training.
        
        This method sets up the complete training pipeline including:
        - Model initialization (3D GCNN for classification or 4D U-Net for segmentation)
        - Loss function configuration (Cross-entropy, Focal Dice, Soft Dice, Tversky)
        - Evaluation metrics (Accuracy, Dice, IoU)
        - Task type determination (classification vs segmentation)
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model: Model architecture parameters (layers, channels, group types)
                - training: Training parameters (loss, optimizer, learning rate)
                - hardware: Hardware settings (GPU, precision, batch size)
                - data: Dataset configuration (name, paths, splits)
                - antialiasing: Anti-aliasing settings for group downsampling
                
        Mathematical Context:
            The model implements group equivariance where f(g·x) = g·f(x) for all
            group elements g. The C4 cyclic subgroup provides 90° rotation equivariance
            around the z-axis, which is particularly useful for 3D medical images.
            
        Group Theory:
            - Octahedral group O: 24 elements representing cube rotations
            - C4 cyclic subgroup: 4 elements {e, r, r², r³} where r⁴ = e
            - Group downsampling: O → C4 reduces group order from 24 to 4
            - Channel calculation: total_channels = base_channels × group_order
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()  # Save hyperparameters for logging
        
        # Extract configuration sections for easier access
        self.model_config = config.get('model', {})        # Model architecture settings
        self.training_config = config.get('training', {})  # Training hyperparameters
        self.hardware_config = config.get('hardware', {})  # Hardware configuration
        self.data_config = config.get('data', {})          # Dataset configuration
        
        # Determine task type based on dataset name
        # ACDC dataset → segmentation, others → classification
        self.task_type = self._determine_task_type()
        
        # Initialize the 3D Group Equivariant CNN model
        # This creates either a classification model (Gcnn3D) or segmentation model (Gcnn3DSegmentation)
        self._init_model()
        
        # Initialize loss function based on task type
        # Classification: Cross-entropy, Segmentation: Focal Dice, Soft Dice, or Tversky
        self._init_loss_function()
        
        # Initialize evaluation metrics for monitoring training progress
        # Classification: Accuracy, Precision, Recall, F1
        # Segmentation: Dice coefficient, IoU, Hausdorff distance
        self._init_evaluation_metrics()
        
        # Counter for tracking training steps (used for learning rate scheduling)
        self.training_step_count = 0
        
        # Print initialization summary for debugging
        print(f"Initialized Lightning Module")
        print(f"  - Task type: {self.task_type}")
        print(f"  - Model type: {self.model_config.get('model_type', 'unknown')}")
        print(f"  - Number of classes: {self.model_config.get('num_classes', 'unknown')}")
        print(f"  - Model parameters: {self._count_parameters()}")
    
    def _determine_task_type(self) -> str:
        """Determine task type from configuration."""
        dataset_name = self.data_config.get('dataset_name', 'organmnist3d')
        if dataset_name == 'acdc':
            return 'segmentation'
        else:
            return 'classification'
    
    def _init_model(self):
        """Initialize the 3D GCNN model."""
        if not GCNN_AVAILABLE:
            raise ImportError("GCNN models not available. Please ensure gsampling is properly installed.")
        
        model_type = self.model_config.get('model_type', 'gcnn3d')
        
        if model_type == 'gcnn3d_seg':
            # Use GSampling GCNN3D Segmentation model
            self.model = self._create_gcnn3d_segmentation_model()
        elif model_type == 'gcnn3d':
            # Use GSampling GCNN3D model
            self.model = self._create_gcnn3d_model()
        elif model_type == 'simple_cnn3d':
            # Use simple 3D CNN model
            self.model = self._create_simple_cnn3d_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize model weights
        init_method = self.config.get('advanced', {}).get('initialization', {}).get('weight_init', 'kaiming')
        ModelInitializer.initialize_weights(self.model, init_method)
    
    def _create_gcnn3d_model(self) -> Gcnn3D:
        """
        Create 3D Group Equivariant CNN model for classification tasks.
        
        This method constructs a Gcnn3D model using the GSampling model handler.
        The model implements group equivariance with the following architecture:
        
        Architecture:
            Layer 0: Trivial → Regular representation (1 × |G| channels)
            Layer 1+: Hybrid layers with group convolution + group resampling
            Final: Global pooling + classification head
            
        Group Processing:
            - Input: (batch, 1, depth, height, width) - trivial representation
            - Layer 0: (batch, 1×24, depth, height, width) - regular representation
            - Group downsampling: (batch, channels×4, depth, height, width) - C4 subgroup
            - Group upsampling: (batch, channels×24, depth, height, width) - back to octahedral
            - Output: (batch, num_classes) - classification logits
            
        Mathematical Details:
            - Group convolution: f * ψ where f ∈ L²(G), ψ ∈ L²(G)
            - Channel calculation: total_channels = base_channels × group_order
            - Group downsampling: O(24) → C4(4) reduces group order by factor of 6
            - Anti-aliasing: Prevents aliasing artifacts during group downsampling
            
        Args:
            None (uses self.model_config for parameters)
            
        Returns:
            Gcnn3D: Configured 3D Group Equivariant CNN model
            
        Configuration Parameters:
            - input_channel (int): Number of input channels (always 1 for medical images)
            - num_layers (int): Number of convolutional layers
            - num_channels (List[int]): Channel progression [input, layer1, layer2, ..., output]
            - kernel_sizes (List[int]): Kernel size for each layer
            - num_classes (int): Number of output classes for classification
            - dwn_group_types (List[List[str]]): Group type transitions for each layer
            - init_group_order (int): Initial group order (24 for octahedral)
            - spatial_subsampling_factors (List[int]): Spatial downsampling factors
            - subsampling_factors (List[int]): Group downsampling factors
            - domain (int): Spatial domain dimension (3 for 3D)
            - pooling_type (str): Global pooling type ('max' or 'mean')
            - apply_antialiasing (bool): Whether to apply anti-aliasing
            - antialiasing_kwargs (Dict): Anti-aliasing parameters
            - dropout_rate (float): Dropout probability
            - fully_convolutional (bool): Whether to use fully convolutional mode
        """
        # Extract group configuration for group equivariance settings
        group_config = self.model_config.get('group_config', {})
        # Extract anti-aliasing configuration for spectral anti-aliasing
        antialiasing_config = self.model_config.get('antialiasing', {})
        
        # Create model using GSampling model handler
        # This constructs the complete 3D Group Equivariant CNN architecture
        model = get_3d_model(
            input_channel=1,  # Medical images typically have 1 channel (grayscale)
            num_layers=self.model_config.get('num_layers', 3),  # Number of conv layers
            num_channels=self.model_config.get('num_channels', [32, 64, 128]),  # Channel progression
            kernel_sizes=self.model_config.get('kernel_sizes', [3, 3, 3]),  # 3x3x3 kernels
            num_classes=self.model_config.get('num_classes', 11),  # Output classes
            # Group type transitions: [["octahedral", "octahedral"], ["octahedral", "cycle"], ...]
            dwn_group_types=group_config.get('dwn_group_types', [["octahedral", "octahedral"]]),
            init_group_order=group_config.get('init_group_order', 24),  # Octahedral group order
            # Spatial downsampling: [2, 2, 1] means 2x downsampling in first two layers
            spatial_subsampling_factors=group_config.get('spatial_subsampling_factors', [1, 1, 1]),
            # Group downsampling: [1, 1, 6] means 6x group downsampling in third layer
            subsampling_factors=group_config.get('subsampling_factors', [1, 1, 1]),
            domain=3,  # 3D spatial domain
            pooling_type=self.model_config.get('pooling_type', 'max'),  # Global pooling
            # Anti-aliasing settings for group downsampling
            apply_antialiasing=antialiasing_config.get('apply_antialiasing', True),
            antialiasing_kwargs=antialiasing_config,  # Pass all anti-aliasing parameters
            dropout_rate=self.model_config.get('dropout_rate', 0.1),  # Regularization
            fully_convolutional=self.model_config.get('fully_convolutional', False)  # Classification mode
        )
        
        return model
    
    def _create_gcnn3d_segmentation_model(self) -> Gcnn3DSegmentation:
        """
        Create 4D U-Net model for 3D medical image segmentation.
        
        This method constructs a Gcnn3DSegmentation model (4D U-Net) that combines:
        - 3D spatial processing (depth, height, width)
        - Group axis processing (group equivariance)
        - Encoder-decoder architecture with skip connections
        - Group downsampling and upsampling
        
        Architecture:
            Encoder: Feature extraction with group downsampling
            Bottleneck: Deepest features with group processing
            Decoder: Feature reconstruction with group upsampling + skip connections
            Final Conv: Output segmentation mask
            
        Group Processing:
            - Input: (batch, 1, depth, height, width) - trivial representation
            - Encoder: (batch, channels×|G|, depth/8, height/8, width/8) - regular representation
            - Decoder: (batch, channels×|G|, depth, height, width) - upsampled features
            - Output: (batch, num_classes, depth, height, width) - segmentation mask
            
        Mathematical Details:
            - 4D U-Net: Processes both spatial (3D) and group dimensions
            - Skip connections: Concatenate encoder and decoder features
            - Group equivariance: Maintains f(g·x) = g·f(x) property
            - Spatial upsampling: F.interpolate for size matching
            - Group pooling: Collapse group dimension for final output
            
        Args:
            None (uses self.model_config for parameters)
            
        Returns:
            Gcnn3DSegmentation: Configured 4D U-Net model for segmentation
            
        Configuration Parameters:
            - num_layers (int): Number of encoder/decoder layers
            - num_channels (List[int]): Channel progression [1, 32, 64, 128, 256]
            - dwn_group_types (List[List[str]]): Group type transitions
            - subsampling_factors (List[int]): Group downsampling factors
            - spatial_subsampling_factors (List[int]): Spatial downsampling factors
            - init_group_order (int): Initial group order (24 for octahedral)
            - num_classes (int): Number of segmentation classes
            - apply_antialiasing (bool): Whether to apply anti-aliasing
            - antialiasing_kwargs (Dict): Anti-aliasing parameters
        """
        # Extract group configuration for group equivariance settings
        group_config = self.model_config.get('group_config', {})
        # Extract anti-aliasing configuration for spectral anti-aliasing
        antialiasing_config = self.model_config.get('antialiasing', {})
        
        # Create model using GSampling model handler
        model = get_3d_segmentation_model(
            input_channel=1,  # Input has 1 channel
            num_layers=self.model_config.get('num_layers', 2),
            num_channels=self.model_config.get('num_channels', [1, 16, 32]),
            kernel_sizes=self.model_config.get('kernel_sizes', [3, 3]),
            num_classes=self.model_config.get('num_classes', 4),
            dwn_group_types=group_config.get('dwn_group_types', [["octahedral", "octahedral"], ["octahedral", "cycle"]]),
            init_group_order=group_config.get('init_group_order', 24),
            spatial_subsampling_factors=group_config.get('spatial_subsampling_factors', [1, 2]),
            subsampling_factors=group_config.get('subsampling_factors', [1, 6]),
            domain=3,
            apply_antialiasing=antialiasing_config.get('apply_antialiasing', True),
            antialiasing_kwargs=antialiasing_config,
            dropout_rate=self.model_config.get('dropout_rate', 0.1)
        )
        
        return model
    
    def _create_simple_cnn3d_model(self) -> nn.Module:
        """Create simple 3D CNN model for testing."""
        from testing_models.simple_models import create_simple_cnn3d_model
        
        return create_simple_cnn3d_model(
            num_channels=self.model_config.get('num_channels', [16, 32]),
            num_classes=self.model_config.get('num_classes', 11),
            dropout_rate=self.model_config.get('dropout_rate', 0.1)
        )
    
    def _init_loss_function(self):
        """Initialize the loss function."""
        loss_config = self.training_config.get('loss', {})
        loss_name = loss_config.get('name', 'cross_entropy')
        loss_params = loss_config.get('params', {})
        
        # Convert string parameters to appropriate types
        if 'smooth' in loss_params and isinstance(loss_params['smooth'], str):
            loss_params['smooth'] = float(loss_params['smooth'])
        if 'alpha' in loss_params and isinstance(loss_params['alpha'], str):
            loss_params['alpha'] = float(loss_params['alpha'])
        if 'gamma' in loss_params and isinstance(loss_params['gamma'], str):
            loss_params['gamma'] = float(loss_params['gamma'])
        
        # Convert scheduler parameters
        scheduler_config = self.training_config.get('scheduler', {})
        scheduler_params = scheduler_config.get('params', {})
        if 'eta_min' in scheduler_params and isinstance(scheduler_params['eta_min'], str):
            scheduler_params['eta_min'] = float(scheduler_params['eta_min'])
        if 'T_max' in scheduler_params and isinstance(scheduler_params['T_max'], str):
            scheduler_params['T_max'] = int(scheduler_params['T_max'])
        
        self.criterion = LossFunctions.get_loss_function(loss_name, **loss_params)
    
    def _init_evaluation_metrics(self):
        """Initialize evaluation metrics."""
        num_classes = self.model_config.get('num_classes', 11)
        self.eval_metrics = EvaluationMetrics(
            num_classes=num_classes, 
            device=self.device,
            task_type=self.task_type
        )
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return ModelInitializer.count_parameters(self.model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step with forward pass, loss computation, and metric calculation.
        
        This method handles both classification and segmentation tasks with appropriate
        data processing, loss computation, and metric calculation. It supports:
        
        Classification Tasks (OrganMNIST3D):
            - Input: (batch, 1, depth, height, width) - 3D medical images
            - Output: (batch, num_classes) - classification logits
            - Loss: Cross-entropy loss
            - Metrics: Accuracy, Precision, Recall, F1
            
        Segmentation Tasks (ACDC):
            - Input: (batch, 1, depth, height, width) - 3D cardiac MRI
            - Output: (batch, num_classes, depth, height, width) - segmentation mask
            - Loss: Focal Dice, Soft Dice, or Tversky loss
            - Metrics: Dice coefficient, IoU, Hausdorff distance
            
        Mathematical Details:
            - Forward pass: logits = model(x) where x is input tensor
            - Loss computation: L = loss_function(logits, targets)
            - Prediction: preds = argmax(logits, dim=1) for classification
            - Accuracy: acc = mean(preds == targets) for classification
            - Size interpolation: F.interpolate for spatial dimension matching
            
        Group Equivariance:
            The model maintains group equivariance f(g·x) = g·f(x) throughout
            the forward pass, ensuring that rotated inputs produce rotated outputs.
            
        Args:
            batch (Dict[str, Any]): Batch data containing:
                - For classification: (images, labels) tuple
                - For segmentation: {'image': List[Tensor], 'label': List[Tensor]}
            batch_idx (int): Index of current batch in epoch
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'loss': Computed loss value (scalar tensor)
                - 'acc': Computed accuracy (scalar tensor)
                
        Side Effects:
            - Updates self.training_step_count for learning rate scheduling
            - Logs metrics to PyTorch Lightning logger
            - Updates model gradients (via PyTorch Lightning)
        """
        # Handle different batch formats for classification vs segmentation
        if self.task_type == 'segmentation':
            # ACDC segmentation batch format: {'image': List[Tensor], 'label': List[Tensor]}
            # Each element in the list is a 3D tensor (depth, height, width)
            images = batch['image']  # List of 3D tensors with variable sizes
            labels = batch['label']  # List of 3D label tensors with variable sizes
            
            # Process first image in batch (TODO: implement proper batching for variable sizes)
            # This is a temporary solution - proper batching would handle all images
            if isinstance(images, list):
                image = images[0].unsqueeze(0)  # Add batch dimension: (1, depth, height, width)
                label = labels[0].unsqueeze(0)  # Add batch dimension: (1, depth, height, width)
            else:
                image = images  # Already has batch dimension
                label = labels  # Already has batch dimension
        else:
            # MedMNIST classification batch format: (images, labels) tuple
            # images: (batch, 1, depth, height, width), labels: (batch,) or (batch, num_classes)
            images, labels = batch
            image = images  # (batch, 1, depth, height, width)
            label = labels  # (batch,) for class indices or (batch, num_classes) for one-hot
        
        # Forward pass through the Group Equivariant CNN
        # This applies the complete model architecture including:
        # - Group convolution layers
        # - Group downsampling/upsampling
        # - Spatial downsampling/upsampling
        # - Anti-aliasing (if enabled)
        logits = self(image)  # Forward pass: x → model(x)
        
        # Process labels for loss computation based on task type
        if self.task_type == 'segmentation':
            # For segmentation, labels are integer class indices
            # Shape: (batch, depth, height, width) with values in [0, num_classes-1]
            loss_labels = label.long()  # Ensure integer type for segmentation loss
        else:
            # For classification, handle both integer labels and one-hot encoding
            loss_labels = label
            if label.dim() == 2:  # One-hot encoded labels (batch, num_classes)
                # Convert one-hot to class indices for loss computation
                loss_labels = torch.argmax(label, dim=1)  # (batch,)
        
        # Compute loss using the configured loss function
        # Classification: Cross-entropy loss
        # Segmentation: Focal Dice, Soft Dice, or Tversky loss
        loss = self.criterion(logits, loss_labels)
        
        # Compute predictions for accuracy calculation
        # Classification: argmax over class dimension
        # Segmentation: argmax over class dimension (per voxel)
        preds = torch.argmax(logits, dim=1)  # (batch, num_classes) → (batch,) or (batch, depth, height, width)
        
        # Handle size mismatch between predictions and labels (common in segmentation)
        # This can happen when the model output size doesn't exactly match the label size
        if preds.shape[1:] != loss_labels.shape[1:]:
            print(f"Training size mismatch: preds {preds.shape[1:]} vs labels {loss_labels.shape[1:]}, interpolating...")
            # Interpolate predictions to match label dimensions
            # Add channel dimension, interpolate, then remove channel dimension
            preds = F.interpolate(
                preds.unsqueeze(1).float(),  # Add channel dim: (batch, 1, depth, height, width)
                size=loss_labels.shape[1:],  # Target spatial size
                mode='nearest'  # Nearest neighbor for integer labels
            ).squeeze(1).long()  # Remove channel dim and convert back to long
        
        # Compute accuracy metric
        # For classification: mean(preds == targets)
        # For segmentation: mean(preds == targets) per voxel
        acc = (preds == loss_labels).float().mean()
        
        # Log metrics to PyTorch Lightning logger
        # These will be displayed in the progress bar and logged to files
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update step counter for learning rate scheduling
        # This is used by the learning rate scheduler to determine current step
        self.training_step_count += 1
        
        # Return metrics dictionary for PyTorch Lightning
        return {
            'loss': loss,  # Primary loss for backpropagation
            'acc': acc     # Accuracy metric for monitoring
        }
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Dictionary with 'image' and 'label' keys
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        if self.task_type == 'segmentation':
            # ACDC segmentation batch
            images = batch['image']  # List of tensors
            labels = batch['label']  # List of tensors
            
            # For now, process first image in batch (will be improved with proper batching)
            if isinstance(images, list):
                image = images[0].unsqueeze(0)  # Add batch dimension
                label = labels[0].unsqueeze(0)  # Add batch dimension
            else:
                image = images
                label = labels
        else:
            # MedMNIST classification batch
            images, labels = batch
            image = images
            label = labels
        
        # Forward pass
        logits = self(image)
        
        # Process labels for loss computation
        if self.task_type == 'segmentation':
            # For segmentation, labels are already in correct format
            loss_labels = label.long()
        else:
            # For classification, handle one-hot encoding
            loss_labels = label
            if label.dim() == 2:  # One-hot encoded
                loss_labels = torch.argmax(label, dim=1)
        
        # Compute loss
        loss = self.criterion(logits, loss_labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        
        # Handle size mismatch between predictions and labels
        if preds.shape[1:] != loss_labels.shape[1:]:
            print(f"Validation size mismatch: preds {preds.shape[1:]} vs labels {loss_labels.shape[1:]}, interpolating...")
            preds = F.interpolate(preds.unsqueeze(1).float(), size=loss_labels.shape[1:], mode='nearest').squeeze(1).long()
        
        acc = (preds == loss_labels).float().mean()
        
        # Update evaluation metrics - use interpolated logits if there was a size mismatch
        if logits.shape[2:] != loss_labels.shape[1:]:
            # Use the interpolated logits from the loss function
            eval_logits = F.interpolate(logits, size=loss_labels.shape[1:], mode='trilinear', align_corners=False)
        else:
            eval_logits = logits
        
        self.eval_metrics.update(eval_logits, loss_labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'val_acc': acc
        }
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Dictionary with 'image' and 'label' keys
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions and targets
        """
        if self.task_type == 'segmentation':
            # ACDC segmentation batch
            images = batch['image']  # List of tensors
            labels = batch['label']  # List of tensors
            
            # For now, process first image in batch (will be improved with proper batching)
            if isinstance(images, list):
                image = images[0].unsqueeze(0)  # Add batch dimension
                label = labels[0].unsqueeze(0)  # Add batch dimension
            else:
                image = images
                label = labels
        else:
            # MedMNIST classification batch
            images, labels = batch
            image = images
            label = labels
        
        # Forward pass
        logits = self(image)
        
        # Process labels for evaluation
        if self.task_type == 'segmentation':
            # For segmentation, labels are already in correct format
            eval_labels = label.long()
        else:
            # For classification, handle one-hot encoding
            eval_labels = label
            if label.dim() == 2:  # One-hot encoded
                eval_labels = torch.argmax(label, dim=1)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update evaluation metrics - use interpolated logits if there was a size mismatch
        if logits.shape[2:] != eval_labels.shape[1:]:
            # Use the interpolated logits from the loss function
            eval_logits = F.interpolate(logits, size=eval_labels.shape[1:], mode='trilinear', align_corners=False)
        else:
            eval_logits = logits
        
        self.eval_metrics.update(eval_logits, eval_labels)
        
        return {
            'preds': preds,
            'targets': eval_labels,
            'logits': logits
        }
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute comprehensive metrics
        metrics = self.eval_metrics.compute()
        
        # Log metrics
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                self.log(f'val_{name}', value, on_epoch=True, prog_bar=False)
        
        # Reset metrics for next epoch
        self.eval_metrics.reset()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute comprehensive metrics
        metrics = self.eval_metrics.compute()
        
        # Log metrics
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                self.log(f'test_{name}', value, on_epoch=True, prog_bar=False)
        
        # Print final test results
        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                print(f"{name}: {value:.4f}")
        
        # Save confusion matrix if requested
        if self.config.get('evaluation', {}).get('confusion_matrix', False):
            self._save_confusion_matrix(metrics.get('confusion_matrix'))
    
    def _save_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Save confusion matrix to file."""
        if confusion_matrix is not None:
            import numpy as np
            save_path = Path(self.config.get('logging', {}).get('save_dir', './checkpoints')) / 'confusion_matrix.npy'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, confusion_matrix)
            print(f"Confusion matrix saved to: {save_path}")
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer configuration
        opt_config = self.training_config.get('optimizer', {})
        opt_name = opt_config.get('name', 'adam')
        opt_params = opt_config.get('params', {})
        
        # Create optimizer
        optimizer = OptimizerFactory.get_optimizer(self, opt_name, **opt_params)
        
        # Get scheduler configuration
        sched_config = self.training_config.get('scheduler', {})
        sched_name = sched_config.get('name', 'cosine')
        sched_params = sched_config.get('params', {})
        
        # Create scheduler
        scheduler = OptimizerFactory.get_scheduler(optimizer, sched_name, **sched_params)
        
        # Configure scheduler
        if sched_name.lower() == 'plateau':
            scheduler_config = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'mode': 'min'
            }
        else:
            scheduler_config = {
                'scheduler': scheduler
            }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }


def create_trainer(config: Dict[str, Any]) -> Trainer:
    """
    Create PyTorch Lightning trainer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Trainer instance
    """
    # Get configuration sections
    hardware_config = config.get('hardware', {})
    logging_config = config.get('logging', {})
    training_config = config.get('training', {})
    test_mode_config = config.get('test_mode', {})
    
    # Callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=logging_config.get('save_dir', './checkpoints'),
        filename=logging_config.get('filename', '{epoch:02d}-{val_loss:.3f}'),
        monitor=logging_config.get('monitor', 'val_loss'),
        mode=logging_config.get('mode', 'min'),
        save_top_k=logging_config.get('save_top_k', 3),
        save_last=logging_config.get('save_last', True)
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min'),
            patience=early_stopping_config.get('patience', 15),
            min_delta=early_stopping_config.get('min_delta', 0.001)
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Model summary
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    # Loggers
    loggers = []
    logger_name = logging_config.get('logger', 'tensorboard')
    
    if logger_name.lower() == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir='./logs',
            name=logging_config.get('project_name', 'medmnist_gcnn'),
            version=logging_config.get('experiment_name', 'experiment')
        )
        loggers.append(logger)
    elif logger_name.lower() == 'csv':
        logger = CSVLogger(
            save_dir='./logs',
            name=logging_config.get('project_name', 'medmnist_gcnn'),
            version=logging_config.get('experiment_name', 'experiment')
        )
        loggers.append(logger)
    elif logger_name.lower() == 'wandb':
        logger = WandbLogger(
            project=logging_config.get('project_name', 'medmnist_gcnn'),
            name=logging_config.get('experiment_name', 'experiment'),
            tags=logging_config.get('tags', [])
        )
        loggers.append(logger)
    
    # Trainer configuration
    accelerator = hardware_config.get('accelerator', 'auto')
    devices = hardware_config.get('gpus', 'auto')
    
    # Handle CPU accelerator properly
    if accelerator == 'cpu':
        devices = 1  # CPU uses 1 core by default
    
    # Configure strategy to handle unused parameters for complex models
    strategy = hardware_config.get('strategy', 'auto')
    if strategy == 'auto' and devices != 1:
        # Use DDP with unused parameter detection for multi-GPU training
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    
    trainer_kwargs = {
        'max_epochs': training_config.get('max_epochs', 100),
        'accelerator': accelerator,
        'devices': devices,
        'precision': hardware_config.get('precision', 32),
        'strategy': strategy,
        'accumulate_grad_batches': hardware_config.get('accumulate_grad_batches', 1),
        'gradient_clip_val': hardware_config.get('gradient_clip_val', 1.0),
        'deterministic': hardware_config.get('deterministic', False),
        'benchmark': hardware_config.get('benchmark', True),
        'callbacks': callbacks,
        'logger': loggers,
        'log_every_n_steps': logging_config.get('log_every_n_steps', 50)
    }
    
    # Test mode configuration
    if test_mode_config.get('enabled', False):
        trainer_kwargs.update({
            'max_epochs': test_mode_config.get('max_epochs', 5),
            'fast_dev_run': test_mode_config.get('fast_dev_run', False),
            'limit_train_batches': test_mode_config.get('limit_train_batches', 0.1),
            'limit_val_batches': test_mode_config.get('limit_val_batches', 0.1),
            'limit_test_batches': test_mode_config.get('limit_test_batches', 0.1)
        })
    
    # Create trainer
    trainer = Trainer(**trainer_kwargs)
    
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MedMNIST with 3D GCNNs')
    parser.add_argument('--config', type=str, default='organmnist3d_config.yaml',
                       help='Configuration file to use')
    parser.add_argument('--test', action='store_true',
                       help='Enable test mode for quick validation')
    parser.add_argument('--config-dir', type=str, default='./config',
                       help='Directory containing configuration files')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to checkpoint file to load and evaluate')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate loaded model without training')
    
    args = parser.parse_args()
    
    print("MedMNIST Training Pipeline with 3D Group Equivariant CNNs")
    print("=" * 60)
    
    # Check dependencies
    if not LOCAL_MODULES_AVAILABLE:
        print("❌ Local modules not available. Please check imports.")
        return
    
    if not GCNN_AVAILABLE:
        print("❌ GCNN models not available. Please ensure gsampling is properly installed.")
        return
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config_loader = load_config(args.config, args.config_dir)
        
        # Enable test mode if requested
        if args.test:
            print("🔧 Enabling test mode for quick validation")
            config_loader.set_config('test_mode.enabled', True)
        
        # Validate configuration
        if not config_loader.validate_config():
            print("⚠️  Configuration validation failed. Proceeding with warnings.")
        
        # Print configuration
        config_loader.print_config()
        
        # Create dataloader
        print("\n📊 Creating dataloader...")
        datamodule = create_dataloader_from_config(config_loader.merged_config)
        datamodule.setup('fit')
        
        # Create model
        print("\n🧠 Creating model...")
        model = MedMNISTLightningModule(config_loader.merged_config)
        
        # Create trainer
        print("\n🚀 Creating trainer...")
        trainer = create_trainer(config_loader.merged_config)
        
        # Train model if not evaluating only
        if not args.evaluate_only:
            print("\n🎯 Starting training...")
            # Setup datamodule if it's ACDCDataModule
            if hasattr(datamodule, 'setup'):
                datamodule.setup('fit')
            trainer.fit(model, datamodule=datamodule)
            
            # Test model if requested
            if config_loader.get_config('evaluation.test_after_training', True):
                print("\n🧪 Testing model...")
                # Setup datamodule for testing if needed
                if hasattr(datamodule, 'setup'):
                    datamodule.setup('test')
                trainer.test(model, datamodule=datamodule)
            
            print("\n✅ Training completed successfully!")
        else:
            print("\n🔍 Evaluation only mode - skipping training")
            # Setup datamodule for evaluation if needed
            if hasattr(datamodule, 'setup'):
                datamodule.setup('test')
            trainer.test(model, datamodule=datamodule)
        
        # Load and evaluate checkpoint if provided
        if args.load_checkpoint:
            print(f"\n📂 Loading checkpoint: {args.load_checkpoint}")
            try:
                # Load the checkpoint
                checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
                
                # Handle shared weights issue by filtering out problematic layers
                state_dict = checkpoint['state_dict']
                cleaned_state_dict = {}
                
                # Skip problematic layers that have shared weights (pattern-based)
                problematic_patterns = [
                    'anti_aliaser',           # All anti-aliasing layers
                    'sample.sampling_matrix', # All sampling matrices
                    'sample.up_sampling_matrix', # All upsampling matrices
                    'blur.weight',           # All blur pool weights
                ]
                
                def is_problematic_key(key: str) -> bool:
                    """Check if a key matches any problematic pattern."""
                    return any(pattern in key for pattern in problematic_patterns)
                
                for key, value in state_dict.items():
                    if is_problematic_key(key):
                        print(f"⚠️  Skipping problematic layer: {key}")
                        continue
                    elif isinstance(value, torch.Tensor):
                        # Clone the tensor to avoid shared memory issues
                        # Convert inference tensors to regular tensors
                        if value.is_inference():
                            cleaned_state_dict[key] = value.clone().detach().requires_grad_(False)
                        else:
                            cleaned_state_dict[key] = value.clone().detach()
                    else:
                        cleaned_state_dict[key] = value
                
                # Load model weights with strict=False to handle missing keys
                missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
                
                if missing_keys:
                    print(f"⚠️  Missing keys (these will use random initialization): {len(missing_keys)}")
                    if len(missing_keys) <= 5:  # Show first few missing keys
                        for key in missing_keys[:5]:
                            print(f"    - {key}")
                
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys (these will be ignored): {len(unexpected_keys)}")
                    if len(unexpected_keys) <= 5:  # Show first few unexpected keys
                        for key in unexpected_keys[:5]:
                            print(f"    - {key}")
                
                print("✅ Checkpoint loaded successfully")
                
                # Move model to appropriate device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                print(f"✅ Model moved to {device}")
                
                # Evaluate on test set
                print("\n🧪 Evaluating loaded model on test set...")
                datamodule.setup('test')
                test_results = trainer.test(model, datamodule)
                
                print("\n📊 Test Results:")
                for metric, value in test_results[0].items():
                    print(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        print("\n✅ Pipeline completed successfully!")
        
        # Save final configuration
        config_loader.save_config('./final_config.yaml')
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

