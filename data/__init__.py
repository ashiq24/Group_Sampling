"""
Data loaders for Group Sampling library.

This package provides data loaders for various types of data to test
group equivariant neural networks.
"""

# Import MedMNIST dataloaders
try:
    from .medmnist_loader import (
        MedMNIST3DDataset,
        MedMNIST3DDataModule,
        MEDMNIST_3D_DATASETS,
        create_medmnist_dataloader
    )
    __all__ = [
        'MedMNIST3DDataset',
        'MedMNIST3DDataModule',
        'MEDMNIST_3D_DATASETS',
        'create_medmnist_dataloader'
    ]
except ImportError:
    # Fallback if MedMNIST is not available
    __all__ = []
