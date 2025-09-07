from typing import Optional
import os
import math

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from .acdc_dataset import ACDCDataset


def acdc_collate_fn(batch):
    """Custom collate function for ACDC dataset to handle variable image sizes."""
    # Separate images, labels, and other data
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    spacings = [item['spacing'] for item in batch]
    ids = [item['id'] for item in batch]
    
    # For now, we'll just return lists instead of stacked tensors
    # In practice, you might want to pad or crop to a common size
    return {
        'image': images,  # List of tensors instead of stacked tensor
        'label': labels,  # List of tensors instead of stacked tensor
        'spacing': spacings,
        'id': ids
    }


class ACDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 2,
        num_workers: int = 8,
        train_val_split: float = 0.8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_es_ed_only: bool = True,
        transforms_train=None,
        transforms_val=None,
    ) -> None:
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_es_ed_only = use_es_ed_only
        self.transforms_train = transforms_train
        self.transforms_val = transforms_val

        # PyTorch Lightning DataModule properties
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = False  # Disable hyperparameter logging for datamodule
        
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def setup(self, stage: Optional[str] = None):
        # Build full training dataset
        full_train = ACDCDataset(
            root=self.root,
            split="training",
            use_es_ed_only=self.use_es_ed_only,
            transforms=self.transforms_train,
        )
        n_total = len(full_train)
        n_train = int(self.train_val_split * n_total)
        n_val = n_total - n_train
        self.ds_train, self.ds_val = random_split(full_train, [n_train, n_val])
        # Override val transforms on subset
        if hasattr(self.ds_val, 'dataset'):
            self.ds_val.dataset.transforms = self.transforms_val

        # Build testing dataset
        self.ds_test = ACDCDataset(
            root=self.root,
            split="testing",
            use_es_ed_only=self.use_es_ed_only,
            transforms=self.transforms_val,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=acdc_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=acdc_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=acdc_collate_fn,
        )



