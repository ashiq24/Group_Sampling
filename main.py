"""
Main Training Script for MedMNIST with 3D Group Equivariant Convolutional Neural Networks

This script provides a complete training pipeline using PyTorch Lightning with:
- Multi-GPU training support
- Customizable precision (FP16, FP32, FP64)
- Comprehensive logging and monitoring
- Test mode for quick pipeline validation
- Configuration management via YAML files
- Integration with GSampling 3D GCNN models
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
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
    from gsampling.models.g_cnn_3d import Gcnn3D
    from gsampling.models.model_handler import get_3d_model
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
        Initialize the Lightning module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Get configuration sections
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.hardware_config = config.get('hardware', {})
        
        # Initialize model
        self._init_model()
        
        # Initialize loss function
        self._init_loss_function()
        
        # Initialize evaluation metrics
        self._init_evaluation_metrics()
        
        # Training step counter
        self.training_step_count = 0
        
        print(f"Initialized MedMNIST Lightning Module")
        print(f"  - Model type: {self.model_config.get('model_type', 'unknown')}")
        print(f"  - Number of classes: {self.model_config.get('num_classes', 'unknown')}")
        print(f"  - Model parameters: {self._count_parameters()}")
    
    def _init_model(self):
        """Initialize the 3D GCNN model."""
        if not GCNN_AVAILABLE:
            raise ImportError("GCNN models not available. Please ensure gsampling is properly installed.")
        
        model_type = self.model_config.get('model_type', 'gcnn3d')
        
        if model_type == 'gcnn3d':
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
        """Create GCNN3D model from configuration."""
        group_config = self.model_config.get('group_config', {})
        antialiasing_config = self.model_config.get('antialiasing', {})
        
        # Create model using GSampling model handler
        model = get_3d_model(
            input_channel=1,  # Input has 1 channel
            num_layers=self.model_config.get('num_layers', 3),
            num_channels=self.model_config.get('num_channels', [32, 64, 128]),
            kernel_sizes=self.model_config.get('kernel_sizes', [3, 3, 3]),
            num_classes=self.model_config.get('num_classes', 11),
            dwn_group_types=group_config.get('dwn_group_types', [["octahedral", "octahedral"]]),
            init_group_order=group_config.get('init_group_order', 24),
            spatial_subsampling_factors=group_config.get('spatial_subsampling_factors', [1, 1, 1]),
            subsampling_factors=group_config.get('subsampling_factors', [1, 1, 1]),
            domain=3,
            pooling_type=self.model_config.get('pooling_type', 'max'),
            apply_antialiasing=antialiasing_config.get('apply_antialiasing', True),
            antialiasing_kwargs=antialiasing_config,
            dropout_rate=self.model_config.get('dropout_rate', 0.1),
            fully_convolutional=self.model_config.get('fully_convolutional', False)
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
        
        self.criterion = LossFunctions.get_loss_function(loss_name, **loss_params)
    
    def _init_evaluation_metrics(self):
        """Initialize evaluation metrics."""
        num_classes = self.model_config.get('num_classes', 11)
        self.eval_metrics = EvaluationMetrics(num_classes, device=self.device)
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        return ModelInitializer.count_parameters(self.model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Process labels for loss computation
        loss_labels = labels
        if labels.dim() == 2:  # One-hot encoded
            loss_labels = torch.argmax(labels, dim=1)
        
        # Compute loss
        loss = self.criterion(logits, loss_labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == loss_labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update step counter
        self.training_step_count += 1
        
        return {
            'loss': loss,
            'acc': acc
        }
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Process labels for loss computation
        loss_labels = labels
        if labels.dim() == 2:  # One-hot encoded
            loss_labels = torch.argmax(labels, dim=1)
        
        # Compute loss
        loss = self.criterion(logits, loss_labels)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == loss_labels).float().mean()
        
        # Update evaluation metrics
        self.eval_metrics.update(logits, loss_labels)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'val_acc': acc
        }
    
    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions and targets
        """
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        
        # Process labels for evaluation
        eval_labels = labels
        if labels.dim() == 2:  # One-hot encoded
            eval_labels = torch.argmax(labels, dim=1)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update evaluation metrics
        self.eval_metrics.update(logits, eval_labels)
        
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
    
    trainer_kwargs = {
        'max_epochs': training_config.get('max_epochs', 100),
        'accelerator': accelerator,
        'devices': devices,
        'precision': hardware_config.get('precision', 32),
        'strategy': hardware_config.get('strategy', 'auto'),
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
            trainer.fit(model, datamodule)
            
            # Test model if requested
            if config_loader.get_config('evaluation.test_after_training', True):
                print("\n🧪 Testing model...")
                trainer.test(model, datamodule)
            
            print("\n✅ Training completed successfully!")
        else:
            print("\n🔍 Evaluation only mode - skipping training")
        
        # Load and evaluate checkpoint if provided
        if args.load_checkpoint:
            print(f"\n📂 Loading checkpoint: {args.load_checkpoint}")
            try:
                # Load the checkpoint
                checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
                
                # Handle shared weights issue by filtering out problematic layers
                state_dict = checkpoint['state_dict']
                cleaned_state_dict = {}
                
                # Skip problematic layers that have shared weights
                problematic_keys = [
                    'model.spatial_sampling_layers.1.blur.weight',
                    'model.sampling_layers.1.sample.sampling_matrix',
                    'model.sampling_layers.1.sample.up_sampling_matrix'
                ]
                
                for key, value in state_dict.items():
                    if key in problematic_keys:
                        print(f"⚠️  Skipping problematic layer: {key}")
                        continue
                    elif isinstance(value, torch.Tensor):
                        # Clone the tensor to avoid shared memory issues
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

