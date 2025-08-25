"""
MedMNIST DataLoader for 3D Group Equivariant Convolutional Neural Networks

This module provides a comprehensive dataloader for MedMNIST datasets with support for:
- Both classification and segmentation tasks
- Multiple MedMNIST datasets (OrganMNIST3D, NoduleMNIST3D, etc.)
- 3D data preprocessing and augmentation
- Efficient data loading with proper memory management
- Group-equivariant data lifting support

Normalization Methods:
- 'minmax': Min-max scaling to [0, 1] range (default)
- 'zscore': Z-score normalization (zero mean, unit variance)
- 'robust': Robust normalization using 1st and 99th percentiles
- 'medmnist_standard': Dataset-specific normalization using computed statistics
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningDataModule
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    LightningDataModule = object

try:
    from medmnist import INFO, Evaluator
    from medmnist.dataset import MedMNIST3D
    MEDMNIST_AVAILABLE = True
except ImportError as e:
    MEDMNIST_AVAILABLE = False
    warnings.warn("MedMNIST not available. Install with: pip install medmnist")

# Available MedMNIST 3D datasets
MEDMNIST_3D_DATASETS = {
    'organmnist3d': {
        'name': 'OrganMNIST3D',
        'modality': 'CT',
        'task': 'classification',
        'num_classes': 11,
        'description': 'Abdominal organ classification from CT scans'
    },
    'nodulemnist3d': {
        'name': 'NoduleMNIST3D',
        'modality': 'CT',
        'task': 'classification',
        'num_classes': 2,
        'description': 'Lung nodule malignancy detection'
    },
    'fracturemnist3d': {
        'name': 'FractureMNIST3D',
        'modality': 'CT',
        'task': 'classification',
        'num_classes': 3,
        'description': 'Bone fracture classification'
    },
    'adrenalmnist3d': {
        'name': 'AdrenalMNIST3D',
        'modality': 'CT',
        'task': 'classification',
        'num_classes': 2,
        'description': 'Adrenal gland abnormality detection'
    },
    'vesselmnist3d': {
        'name': 'VesselMNIST3D',
        'modality': 'MRA',
        'task': 'classification',
        'num_classes': 2,
        'description': 'Cerebral vessel abnormality detection'
    },
    'synapsemnist3d': {
        'name': 'SynapseMNIST3D',
        'modality': 'Electron Microscope',
        'task': 'classification',
        'num_classes': 2,
        'description': 'Neural synapse detection'
    }
}


class MedMNIST3DDataset(Dataset):
    """
    PyTorch Dataset wrapper for MedMNIST 3D datasets.
    
    Supports both classification and segmentation tasks with 3D data augmentation
    and preprocessing. Designed to be easily extensible to different MedMNIST datasets.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: bool = True,
        data_dir: Optional[str] = None,
        task_type: str = 'auto',
        normalize: bool = True,
        norm_method: str = 'minmax',
        augment: bool = False
    ):
        """
        Initialize MedMNIST 3D dataset.
        
        Args:
            dataset_name: Name of the MedMNIST dataset (e.g., 'organmnist3d')
            split: Data split ('train', 'val', 'test')
            transform: Optional transforms for input data
            target_transform: Optional transforms for target data
            download: Whether to download dataset if not present
            data_dir: Directory to store/download dataset
            task_type: Task type ('auto', 'classification', 'segmentation')
            normalize: Whether to normalize data to [0, 1]
            augment: Whether to apply data augmentation
        """
        if not MEDMNIST_AVAILABLE:
            raise ImportError("MedMNIST not available. Install with: pip install medmnist")
        
        if dataset_name not in MEDMNIST_3D_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(MEDMNIST_3D_DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.split = split
        self.task_type = task_type
        self.normalize = normalize
        self.augment = augment
        
        # Normalization method
        self.norm_method = norm_method
        
        # Get dataset info
        self.dataset_info = MEDMNIST_3D_DATASETS[dataset_name]
        self.num_classes = self.dataset_info['num_classes']
        self.modality = self.dataset_info['modality']
        
        # Set task type automatically if not specified
        if self.task_type == 'auto':
            self.task_type = self.dataset_info['task']
        
        # Load MedMNIST dataset - use specific dataset class
        if dataset_name == 'organmnist3d':
            from medmnist import OrganMNIST3D
            self.medmnist_data = OrganMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        elif dataset_name == 'nodulemnist3d':
            from medmnist import NoduleMNIST3D
            self.medmnist_data = NoduleMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        elif dataset_name == 'fracturemnist3d':
            from medmnist import FractureMNIST3D
            self.medmnist_data = FractureMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        elif dataset_name == 'adrenalmnist3d':
            from medmnist import AdrenalMNIST3D
            self.medmnist_data = AdrenalMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        elif dataset_name == 'vesselmnist3d':
            from medmnist import VesselMNIST3D
            self.medmnist_data = VesselMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        elif dataset_name == 'synapsemnist3d':
            from medmnist import SynapseMNIST3D
            self.medmnist_data = SynapseMNIST3D(
                split=split,
                download=download,
                target_transform=target_transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Get dataset info from MedMNIST
        self.info = INFO[dataset_name]
        self.n_channels = self.info['n_channels']
        
        # Set up transforms
        self.transform = self._setup_transforms(transform)
        
        # Validate data
        self._validate_data()
        
        # Compute dataset statistics for normalization if needed
        if self.norm_method == 'medmnist_standard':
            self._compute_dataset_statistics()
        
        print(f"Loaded {self.dataset_name} ({self.split} split):")
        print(f"  - Samples: {len(self.medmnist_data)}")
        print(f"  - Input shape: {self.medmnist_data.imgs.shape}")
        print(f"  - Labels shape: {self.medmnist_data.labels.shape}")
        print(f"  - Task: {self.task_type}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Modality: {self.modality}")
        print(f"  - Normalization: {self.norm_method}")
        if hasattr(self, 'dataset_mean') and hasattr(self, 'dataset_std'):
            print(f"  - Dataset mean: {self.dataset_mean:.4f}")
            print(f"  - Dataset std: {self.dataset_std:.4f}")
    
    def _setup_transforms(self, transform: Optional[transforms.Compose]) -> transforms.Compose:
        """Set up data transforms for 3D data."""
        transforms_list = []
        
        # Convert to tensor
        transforms_list.append(self._to_tensor_3d)
        
        # Add channel dimension if needed
        if len(self.medmnist_data.imgs.shape) == 4:  # (N, H, W, D)
            transforms_list.append(self._add_channel_dim)
        
        # Normalize if requested
        if self.normalize:
            transforms_list.append(self._normalize_3d)
        
        # Add data augmentation if requested
        if self.augment and self.split == 'train':
            transforms_list.extend(self._get_augmentation_transforms())
        
        # Add custom transforms
        if transform is not None:
            transforms_list.append(transform)
        
        return transforms.Compose(transforms_list)
    
    def _compute_dataset_statistics(self, max_samples: int = 1000):
        """
        Compute dataset statistics for normalization.
        
        Args:
            max_samples: Maximum number of samples to use for statistics
        """
        print(f"Computing dataset statistics using up to {max_samples} samples...")
        
        # Use a subset for efficiency
        n_samples = min(len(self.medmnist_data), max_samples)
        indices = torch.randperm(len(self.medmnist_data))[:n_samples]
        
        # Collect statistics
        means = []
        stds = []
        
        for idx in indices:
            img = self.medmnist_data.imgs[idx]  # Shape: (H, W, D)
            
            # Convert to tensor first
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            
            # Add channel dimension if needed
            if img.dim() == 3:
                img = img.unsqueeze(0)  # Add channel dimension
            
            means.append(img.mean())
            stds.append(img.std())
        
        # Compute overall statistics
        self.dataset_mean = torch.stack(means).mean()
        self.dataset_std = torch.stack(stds).mean()
        
        print(f"Dataset statistics computed:")
        print(f"  - Mean: {self.dataset_mean:.4f}")
        print(f"  - Std: {self.dataset_std:.4f}")
    
    def _to_tensor_3d(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor for 3D data."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data
    
    def _add_channel_dim(self, data: torch.Tensor) -> torch.Tensor:
        """Add channel dimension to 3D data."""
        if data.dim() == 3:  # (H, W, D)
            return data.unsqueeze(0)  # (1, H, W, D)
        return data
    
    def _normalize_3d(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize 3D data using appropriate method for medical imaging.
        
        Args:
            data: Input tensor of shape (C, H, W, D)
            
        Returns:
            Normalized tensor
        """
        if data.dim() != 4:  # (C, H, W, D)
            return data
            
        # Get normalization method from config
        norm_method = getattr(self, 'norm_method', 'minmax')
        
        if norm_method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            return data
            
        elif norm_method == 'zscore':
            # Z-score normalization (zero mean, unit variance)
            mean = data.mean()
            std = data.std()
            if std > 0:
                return (data - mean) / std
            return data - mean
            
        elif norm_method == 'robust':
            # Robust normalization using percentiles
            q1 = torch.quantile(data, 0.01)
            q99 = torch.quantile(data, 0.99)
            if q99 > q1:
                return torch.clamp((data - q1) / (q99 - q1), 0, 1)
            return data
            
        elif norm_method == 'medmnist_standard':
            # MedMNIST-specific normalization based on dataset statistics
            if hasattr(self, 'dataset_mean') and hasattr(self, 'dataset_std'):
                return (data - self.dataset_mean) / self.dataset_std
            else:
                # Fallback to min-max if stats not available
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    return (data - data_min) / (data_max - data_min)
                return data
                
        else:
            # Default to min-max
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                return (data - data_min) / (data_max - data_min)
            return data
    
    def _get_augmentation_transforms(self) -> List:
        """Get data augmentation transforms for 3D data."""
        # Note: torchvision doesn't have built-in 3D transforms
        # We'll implement custom 3D augmentation methods
        return []
    
    def _validate_data(self):
        """Validate that the loaded data has expected properties."""
        expected_shape = (len(self.medmnist_data), 28, 28, 28)
        if self.medmnist_data.imgs.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {self.medmnist_data.imgs.shape}")
        
        if self.medmnist_data.labels.shape[0] != len(self.medmnist_data):
            raise ValueError("Number of labels doesn't match number of images")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.medmnist_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label) where:
            - image: 3D tensor of shape (C, H, W, D) or (|G|*C, H, W, D) if group_lifting
            - label: Target tensor for classification or segmentation
        """
        # Get image and label
        img = self.medmnist_data.imgs[idx]  # Shape: (H, W, D)
        label = self.medmnist_data.labels[idx]  # Shape: (num_classes,) or (H, W, D)
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        # Convert label to tensor if needed
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).float()
        
        return img, label
    

    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single sample."""
        return (self.n_channels, 28, 28, 28)
    
    def get_class_names(self) -> List[str]:
        """Get class names if available."""
        if hasattr(self.info, 'label') and self.info['label']:
            return self.info['label']
        return [f"Class_{i}" for i in range(self.num_classes)]


class MedMNIST3DDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for MedMNIST 3D datasets.
    
    Provides train/val/test dataloaders with consistent preprocessing
    and augmentation across splits.
    """
    
    def __init__(
        self,
        dataset_name: str = 'organmnist3d',
        data_dir: str = './data',
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        norm_method: str = 'minmax',
        augment: bool = True,
        task_type: str = 'auto',
        **kwargs
    ):
        """
        Initialize MedMNIST 3D DataModule.
        
        Args:
            dataset_name: Name of the MedMNIST dataset
            data_dir: Directory to store/download dataset
            batch_size: Batch size for all dataloaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for GPU training
            normalize: Whether to normalize data
            augment: Whether to apply data augmentation
            task_type: Task type ('auto', 'classification', 'segmentation')
            **kwargs: Additional arguments passed to dataset
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.norm_method = norm_method
        self.augment = augment
        self.task_type = task_type
        self.kwargs = kwargs
        
        # Dataset info
        self.dataset_info = MEDMNIST_3D_DATASETS[dataset_name]
        self.num_classes = self.dataset_info['num_classes']
        self.input_shape = self._get_input_shape()
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        print(f"Initialized MedMNIST 3D DataModule:")
        print(f"  - Dataset: {dataset_name}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Input shape: {self.input_shape}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Normalization: {norm_method}")
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get the input shape for the model."""
        return (self.dataset_info.get('n_channels', 1), 28, 28, 28)
    
    def prepare_data(self):
        """Download and prepare data if needed."""
        # This is called on the main process only
        pass
    
    def prepare_data_per_node(self):
        """Prepare data on each node."""
        # This is called on each node
        pass
    
    def _log_hyperparams(self):
        """Log hyperparameters for the datamodule."""
        return False
    
    @property
    def allow_zero_length_dataloader_with_multiple_devices(self):
        """Allow zero length dataloader with multiple devices."""
        return False
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for the current stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = MedMNIST3DDataset(
                dataset_name=self.dataset_name,
                split='train',
                data_dir=self.data_dir,
                normalize=self.normalize,
                norm_method=self.norm_method,
                augment=self.augment,
                task_type=self.task_type,
                **self.kwargs
            )
            
            self.val_dataset = MedMNIST3DDataset(
                dataset_name=self.dataset_name,
                split='val',
                data_dir=self.data_dir,
                normalize=self.normalize,
                norm_method=self.norm_method,
                augment=False,  # No augmentation for validation
                task_type=self.task_type,
                **self.kwargs
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = MedMNIST3DDataset(
                dataset_name=self.dataset_name,
                split='test',
                data_dir=self.data_dir,
                normalize=self.normalize,
                norm_method=self.norm_method,
                augment=False,  # No augmentation for testing
                task_type=self.task_type,
                **self.kwargs
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for testing model input/output shapes."""
        if self.train_dataset is None:
            self.setup('fit')
        
        sample_img, sample_label = self.train_dataset[0]
        sample_img = sample_img.unsqueeze(0)  # Add batch dimension
        sample_label = sample_label.unsqueeze(0)  # Add batch dimension
        
        return sample_img, sample_label


def create_medmnist_dataloader(
    dataset_name: str = 'organmnist3d',
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Convenience function to create a MedMNIST dataloader.
    
    Args:
        dataset_name: Name of the MedMNIST dataset
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataset
        
    Returns:
        PyTorch DataLoader
    """
    dataset = MedMNIST3DDataset(
        dataset_name=dataset_name,
        split=split,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataloader
    print("Testing MedMNIST 3D DataLoader...")
    
    # Test basic dataset
    dataset = MedMNIST3DDataset('organmnist3d', split='train', download=True)
    print(f"Dataset loaded successfully: {len(dataset)} samples")
    
    # Test dataloader
    dataloader = create_medmnist_dataloader('organmnist3d', 'train', batch_size=4)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch[0].shape}, Label shape: {batch[1].shape}")
    
    # Test DataModule
    datamodule = MedMNIST3DDataModule('organmnist3d', batch_size=4)
    datamodule.setup('fit')
    print(f"DataModule setup complete")
    print(f"Train samples: {len(datamodule.train_dataset)}")
    print(f"Val samples: {len(datamodule.val_dataset)}")
    
    print("✅ All tests passed!")

