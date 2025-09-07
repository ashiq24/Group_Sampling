#!/usr/bin/env python3
"""
Test script for ACDC dataloader and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.acdc_dataset import ACDCDataset
from data.acdc_datamodule import ACDCDataModule

def visualize_3d_volume(volume, title="3D Volume", save_path=None):
    """Visualize 3D volume using orthogonal slices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Get middle slices
    d, h, w = volume.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    # Axial slice (xy plane)
    axes[0, 0].imshow(volume[mid_d, :, :], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={mid_d})')
    axes[0, 0].axis('off')
    
    # Coronal slice (xz plane)
    axes[0, 1].imshow(volume[:, mid_h, :], cmap='gray')
    axes[0, 1].set_title(f'Coronal (y={mid_h})')
    axes[0, 1].axis('off')
    
    # Sagittal slice (yz plane)
    axes[1, 0].imshow(volume[:, :, mid_w], cmap='gray')
    axes[1, 0].set_title(f'Sagittal (x={mid_w})')
    axes[1, 0].axis('off')
    
    # 3D projection (max intensity projection)
    axes[1, 1].imshow(np.max(volume, axis=0), cmap='gray')
    axes[1, 1].set_title('Max Intensity Projection')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def visualize_segmentation_overlay(image, mask, title="Segmentation Overlay", save_path=None):
    """Visualize segmentation mask overlaid on image."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Get middle slices
    d, h, w = image.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    # Define colors for different classes
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    # Axial slice
    axes[0, 0].imshow(image[mid_d, :, :], cmap='gray')
    mask_slice = mask[mid_d, :, :]
    if mask_slice.max() > 0:
        axes[0, 0].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), 
                         cmap='tab10', alpha=0.7, vmin=0, vmax=10)
    axes[0, 0].set_title(f'Axial (z={mid_d})')
    axes[0, 0].axis('off')
    
    # Coronal slice
    axes[0, 1].imshow(image[:, mid_h, :], cmap='gray')
    mask_slice = mask[:, mid_h, :]
    if mask_slice.max() > 0:
        axes[0, 1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), 
                         cmap='tab10', alpha=0.7, vmin=0, vmax=10)
    axes[0, 1].set_title(f'Coronal (y={mid_h})')
    axes[0, 1].axis('off')
    
    # Sagittal slice
    axes[1, 0].imshow(image[:, :, mid_w], cmap='gray')
    mask_slice = mask[:, :, mid_w]
    if mask_slice.max() > 0:
        axes[1, 0].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), 
                         cmap='tab10', alpha=0.7, vmin=0, vmax=10)
    axes[1, 0].set_title(f'Sagittal (x={mid_w})')
    axes[1, 0].axis('off')
    
    # Class distribution
    unique_classes, counts = np.unique(mask, return_counts=True)
    axes[1, 1].bar(unique_classes, counts)
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].set_xlabel('Class ID')
    axes[1, 1].set_ylabel('Voxel Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Segmentation visualization saved to: {save_path}")
    
    plt.show()

def test_acdc_dataset():
    """Test ACDC dataset loading."""
    print("Testing ACDC Dataset...")
    
    # Dataset path
    dataset_path = "datasets/acdc"
    
    # Test training dataset
    print("\n=== Testing Training Dataset ===")
    train_dataset = ACDCDataset(
        root=dataset_path,
        split="training",
        use_es_ed_only=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Test a few samples
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label shape: {sample['label'].shape if sample['label'] is not None else 'None'}")
        print(f"  Spacing: {sample['spacing']}")
        print(f"  ID: {sample['id']}")
        
        # Visualize first sample
        if i == 0:
            image = sample['image'].squeeze().numpy()  # Remove channel dimension
            if sample['label'] is not None:
                mask = sample['label'].numpy()
                
                # Create output directory
                os.makedirs("test_outputs", exist_ok=True)
                
                # Visualize image
                visualize_3d_volume(
                    image, 
                    title=f"ACDC Training Sample {i} - {sample['id']}",
                    save_path=f"test_outputs/acdc_sample_{i}_image.png"
                )
                
                # Visualize segmentation
                visualize_segmentation_overlay(
                    image, mask,
                    title=f"ACDC Training Sample {i} - Segmentation",
                    save_path=f"test_outputs/acdc_sample_{i}_segmentation.png"
                )
    
    # Test testing dataset
    print("\n=== Testing Testing Dataset ===")
    test_dataset = ACDCDataset(
        root=dataset_path,
        split="testing",
        use_es_ed_only=True
    )
    
    print(f"Testing dataset size: {len(test_dataset)}")
    
    # Test a few samples
    for i in range(min(2, len(test_dataset))):
        sample = test_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label shape: {sample['label'].shape if sample['label'] is not None else 'None'}")
        print(f"  Spacing: {sample['spacing']}")
        print(f"  ID: {sample['id']}")

def test_acdc_datamodule():
    """Test ACDC datamodule."""
    print("\n=== Testing ACDC DataModule ===")
    
    # Dataset path
    dataset_path = "datasets/acdc"
    
    # Create datamodule
    datamodule = ACDCDataModule(
        root=dataset_path,
        batch_size=2,
        num_workers=4,
        train_val_split=0.8,
        use_es_ed_only=True
    )
    
    # Setup
    datamodule.setup()
    
    # Test dataloaders
    print(f"Train dataset size: {len(datamodule.ds_train)}")
    print(f"Val dataset size: {len(datamodule.ds_val)}")
    print(f"Test dataset size: {len(datamodule.ds_test)}")
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch image count: {len(batch['image'])}")
    print(f"Batch image shapes: {[img.shape for img in batch['image']]}")
    print(f"Batch label count: {len(batch['label'])}")
    print(f"Batch label shapes: {[label.shape for label in batch['label']]}")
    print(f"Batch spacing count: {len(batch['spacing'])}")
    print(f"Batch IDs: {batch['id']}")
    
    # Test val dataloader
    print("\nTesting val dataloader...")
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    
    print(f"Batch image count: {len(batch['image'])}")
    print(f"Batch image shapes: {[img.shape for img in batch['image']]}")
    print(f"Batch label count: {len(batch['label'])}")
    print(f"Batch label shapes: {[label.shape for label in batch['label']]}")
    print(f"Batch spacing count: {len(batch['spacing'])}")
    print(f"Batch IDs: {batch['id']}")

def main():
    """Main test function."""
    print("ACDC Dataloader Test")
    print("=" * 50)
    
    try:
        # Test dataset
        test_acdc_dataset()
        
        # Test datamodule
        test_acdc_datamodule()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
