"""
MedMNIST Dataset Visualization and Validation Script

This script provides comprehensive visualization and exploration of MedMNIST 3D datasets
including:
- 3D volume visualization (slices, projections)
- Sample images from each class/dataset
- Data statistics and distributions
- Validation of data loading pipeline
- Sample plots for documentation and debugging
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MedMNISTVisualizer:
    """
    Comprehensive visualizer for MedMNIST 3D datasets.
    
    Provides methods for exploring data structure, visualizing samples,
    and validating the data loading pipeline.
    """
    
    def __init__(self, dataset_name: str = 'organmnist3d', data_dir: str = './data'):
        """
        Initialize the visualizer.
        
        Args:
            dataset_name: Name of the MedMNIST dataset to visualize
            data_dir: Directory containing the dataset
        """
        if not MEDMNIST_AVAILABLE:
            raise ImportError("MedMNIST modules not available")
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset_info = MEDMNIST_3D_DATASETS[dataset_name]
        
        # Load datasets for all splits
        self.datasets = {}
        self._load_datasets()
        
        print(f"Initialized visualizer for {dataset_name}")
        print(f"Dataset info: {self.dataset_info}")
    
    def _load_datasets(self):
        """Load datasets for all splits."""
        for split in ['train', 'val', 'test']:
            try:
                self.datasets[split] = MedMNIST3DDataset(
                    dataset_name=self.dataset_name,
                    split=split,
                    data_dir=self.data_dir,
                    download=True,
                    normalize=True,
                    augment=False
                )
                print(f"Loaded {split} split: {len(self.datasets[split])} samples")
            except Exception as e:
                print(f"Warning: Could not load {split} split: {e}")
                self.datasets[split] = None
    
    def visualize_sample_3d(self, split: str = 'train', sample_idx: int = 0, 
                           show_slices: bool = True, show_projections: bool = True):
        """
        Visualize a single 3D sample with slices and projections.
        
        Args:
            split: Data split to use ('train', 'val', 'test')
            sample_idx: Index of the sample to visualize
            show_slices: Whether to show orthogonal slices
            show_projections: Whether to show 2D projections
        """
        if split not in self.datasets or self.datasets[split] is None:
            print(f"Dataset for {split} split not available")
            return
        
        dataset = self.datasets[split]
        if sample_idx >= len(dataset):
            print(f"Sample index {sample_idx} out of range for {split} split")
            return
        
        # Get sample
        img, label = dataset[sample_idx]
        
        # Convert to numpy for visualization
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if hasattr(label, 'cpu'):
            label = label.cpu().numpy()
        
        # Ensure proper shape
        if img.ndim == 4:  # (C, H, W, D)
            img = img[0]  # Take first channel if multiple channels
        
        print(f"Sample {sample_idx} from {split} split:")
        print(f"  - Image shape: {img.shape}")
        print(f"  - Label: {label}")
        print(f"  - Label shape: {label.shape}")
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'{self.dataset_info["name"]} - Sample {sample_idx} ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        if show_slices:
            self._plot_orthogonal_slices(fig, img, sample_idx, split)
        
        if show_projections:
            self._plot_projections(fig, img, sample_idx, split)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_orthogonal_slices(self, fig, img: np.ndarray, sample_idx: int, split: str):
        """Plot orthogonal slices through the 3D volume."""
        H, W, D = img.shape
        
        # Create subplots for orthogonal slices
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        
        # XY slice (axial)
        ax1 = fig.add_subplot(gs[0, 0])
        mid_z = D // 2
        im1 = ax1.imshow(img[:, :, mid_z], cmap='gray', aspect='equal')
        ax1.set_title(f'XY Slice (Z={mid_z})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # XZ slice (sagittal)
        ax2 = fig.add_subplot(gs[0, 1])
        mid_y = H // 2
        im2 = ax2.imshow(img[mid_y, :, :], cmap='gray', aspect='equal')
        ax2.set_title(f'XZ Slice (Y={mid_y})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # YZ slice (coronal)
        ax3 = fig.add_subplot(gs[0, 2])
        mid_x = W // 2
        im3 = ax3.imshow(img[:, mid_x, :], cmap='gray', aspect='equal')
        ax3.set_title(f'YZ Slice (X={mid_x})')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Interactive slice selection
        ax4 = fig.add_subplot(gs[1, :])
        ax4.set_title('Interactive Slice Selection')
        ax4.set_xlabel('Slice Index')
        ax4.set_ylabel('Intensity')
        
        # Plot intensity profiles along each axis
        x_profile = img[mid_y, mid_x, :]
        y_profile = img[mid_y, :, mid_z]
        z_profile = img[:, mid_x, mid_z]
        
        ax4.plot(range(D), x_profile, 'r-', label='X-axis', linewidth=2)
        ax4.plot(range(W), y_profile, 'g-', label='Y-axis', linewidth=2)
        ax4.plot(range(H), z_profile, 'b-', label='Z-axis', linewidth=2)
        ax4.axvline(x=mid_z, color='r', linestyle='--', alpha=0.7)
        ax4.axvline(x=mid_y, color='g', linestyle='--', alpha=0.7)
        ax4.axvline(x=mid_x, color='b', linestyle='--', alpha=0.7)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    def _plot_projections(self, fig, img: np.ndarray, sample_idx: int, split: str):
        """Plot 2D projections of the 3D volume."""
        H, W, D = img.shape
        
        # Create subplots for projections
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        
        # Maximum intensity projection (MIP)
        ax1 = fig.add_subplot(gs[1, 0])
        mip_xy = np.max(img, axis=2)  # Max along Z-axis
        im1 = ax1.imshow(mip_xy, cmap='gray', aspect='equal')
        ax1.set_title('MIP XY (Max Z)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Average projection
        ax2 = fig.add_subplot(gs[1, 1])
        avg_xy = np.mean(img, axis=2)  # Average along Z-axis
        im2 = ax2.imshow(avg_xy, cmap='gray', aspect='equal')
        ax2.set_title('Average XY')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Volume rendering preview (sum projection)
        ax3 = fig.add_subplot(gs[1, 2])
        sum_xy = np.sum(img, axis=2)  # Sum along Z-axis
        im3 = ax3.imshow(sum_xy, cmap='hot', aspect='equal')
        ax3.set_title('Sum XY')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    def visualize_class_distribution(self, split: str = 'train'):
        """
        Visualize the distribution of classes in the dataset.
        
        Args:
            split: Data split to analyze ('train', 'val', 'test')
        """
        if split not in self.datasets or self.datasets[split] is None:
            print(f"Dataset for {split} split not available")
            return
        
        dataset = self.datasets[split]
        labels = []
        
        # Collect all labels
        for i in range(len(dataset)):
            _, label = dataset[i]
            if hasattr(label, 'cpu'):
                label = label.cpu().numpy()
            labels.append(label)
        
        labels = np.array(labels)
        
        # Handle different label formats
        if labels.ndim == 2:  # One-hot encoded
            class_counts = np.sum(labels, axis=0)
            class_names = [f'Class {i}' for i in range(len(class_counts))]
        else:  # Single label per sample
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = counts
            class_names = [f'Class {int(u)}' for u in unique]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Class Distribution - {self.dataset_info["name"]} ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        # Bar plot
        bars = ax1.bar(range(len(class_counts)), class_counts, 
                      color=sns.color_palette("husl", len(class_counts)))
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Class Distribution')
        ax1.set_xticks(range(len(class_counts)))
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(class_counts),
                    f'{int(count)}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette("husl", len(class_counts)))
        ax2.set_title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nClass Distribution Statistics ({split} split):")
        print(f"Total samples: {len(dataset)}")
        print(f"Number of classes: {len(class_counts)}")
        print(f"Class counts: {dict(zip(class_names, class_counts))}")
        print(f"Class balance: {np.min(class_counts) / np.max(class_counts):.3f}")
    
    def visualize_data_statistics(self, split: str = 'train'):
        """
        Visualize data statistics including intensity distributions and spatial properties.
        
        Args:
            split: Data split to analyze ('train', 'val', 'test')
        """
        if split not in self.datasets or self.datasets[split] is None:
            print(f"Dataset for {split} split not available")
            return
        
        dataset = self.datasets[split]
        
        # Collect statistics from a subset of samples
        n_samples = min(100, len(dataset))  # Limit for performance
        intensities = []
        volumes = []
        
        print(f"Analyzing {n_samples} samples from {split} split...")
        
        for i in range(n_samples):
            img, _ = dataset[i]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            
            if img.ndim == 4:  # (C, H, W, D)
                img = img[0]  # Take first channel
            
            intensities.extend(img.flatten())
            volumes.append(np.sum(img > 0.1))  # Count non-zero voxels
        
        intensities = np.array(intensities)
        volumes = np.array(volumes)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Data Statistics - {self.dataset_info["name"]} ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        # Intensity distribution
        ax1.hist(intensities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Intensity Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Intensity Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Volume distribution
        ax2.hist(volumes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Volume (voxels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Volume Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Intensity vs Volume scatter
        ax3.scatter(volumes, np.mean(intensities.reshape(n_samples, -1), axis=1), 
                   alpha=0.6, color='orange')
        ax3.set_xlabel('Volume (voxels)')
        ax3.set_ylabel('Mean Intensity')
        ax3.set_title('Volume vs Mean Intensity')
        ax3.grid(True, alpha=0.3)
        
        # Statistical summary
        stats_text = f"""
        Sample Statistics:
        - Mean intensity: {np.mean(intensities):.3f}
        - Std intensity: {np.std(intensities):.3f}
        - Min intensity: {np.min(intensities):.3f}
        - Max intensity: {np.max(intensities):.3f}
        - Mean volume: {np.mean(volumes):.1f}
        - Std volume: {np.std(volumes):.1f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title('Statistical Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nData Statistics ({split} split):")
        print(f"Intensity - Mean: {np.mean(intensities):.3f}, Std: {np.std(intensities):.3f}")
        print(f"Volume - Mean: {np.mean(volumes):.1f}, Std: {np.std(volumes):.1f}")
    
    def validate_data_loading(self, split: str = 'train', n_samples: int = 5):
        """
        Validate the data loading pipeline by checking multiple samples.
        
        Args:
            split: Data split to validate ('train', 'val', 'test')
            n_samples: Number of samples to check
        """
        if split not in self.datasets or self.datasets[split] is None:
            print(f"Dataset for {split} split not available")
            return
        
        dataset = self.datasets[split]
        n_samples = min(n_samples, len(dataset))
        
        print(f"\nValidating data loading for {split} split ({n_samples} samples):")
        print("=" * 60)
        
        for i in range(n_samples):
            try:
                img, label = dataset[i]
                
                # Check shapes
                img_shape = img.shape
                label_shape = label.shape
                
                # Check data types
                img_dtype = img.dtype
                label_dtype = label.dtype
                
                # Check value ranges
                img_min, img_max = img.min().item(), img.max().item()
                label_min, label_max = label.min().item(), label.max().item()
                
                print(f"Sample {i}:")
                print(f"  - Image: shape={img_shape}, dtype={img_dtype}, range=[{img_min:.3f}, {img_max:.3f}]")
                print(f"  - Label: shape={label_shape}, dtype={label_dtype}, range=[{label_min:.3f}, {label_max:.3f}]")
                
                # Validate shapes
                if img_shape != (1, 28, 28, 28):
                    print(f"  ⚠️  Warning: Unexpected image shape {img_shape}")
                
                if label_shape != (self.dataset_info['num_classes'],):
                    print(f"  ⚠️  Warning: Unexpected label shape {label_shape}")
                
                # Validate value ranges
                if img_min < 0 or img_max > 1:
                    print(f"  ⚠️  Warning: Image values outside [0,1] range")
                
                print()
                
            except Exception as e:
                print(f"Sample {i}: ❌ Error loading: {e}")
        
        print("=" * 60)
        print("Data loading validation complete!")
    
    def create_sample_gallery(self, split: str = 'train', samples_per_class: int = 2):
        """
        Create a gallery of sample images from each class.
        
        Args:
            split: Data split to use ('train', 'val', 'test')
            samples_per_class: Number of samples to show per class
        """
        if split not in self.datasets or self.datasets[split] is None:
            print(f"Dataset for {split} split not available")
            return
        
        dataset = self.datasets[split]
        num_classes = self.dataset_info['num_classes']
        
        # Collect samples for each class
        class_samples = {i: [] for i in range(num_classes)}
        
        for i in range(len(dataset)):
            if len(class_samples) >= num_classes * samples_per_class:
                break
            
            img, label = dataset[i]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if hasattr(label, 'cpu'):
                label = label.cpu().numpy()
            
            # Find class index
            if label.ndim == 1:  # One-hot encoded
                class_idx = np.argmax(label)
            else:  # Single label
                class_idx = int(label)
            
            if len(class_samples[class_idx]) < samples_per_class:
                class_samples[class_idx].append((i, img))
        
        # Create gallery
        fig, axes = plt.subplots(samples_per_class, num_classes, 
                                figsize=(3*num_classes, 3*samples_per_class))
        fig.suptitle(f'Sample Gallery - {self.dataset_info["name"]} ({split} split)', 
                    fontsize=16, fontweight='bold')
        
        if samples_per_class == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx in range(num_classes):
            for sample_idx in range(samples_per_class):
                ax = axes[sample_idx, class_idx]
                
                if sample_idx < len(class_samples[class_idx]):
                    img_idx, img = class_samples[class_idx][sample_idx]
                    
                    if img.ndim == 4:  # (C, H, W, D)
                        img = img[0]  # Take first channel
                    
                    # Show middle slice
                    mid_slice = img[:, :, img.shape[2]//2]
                    ax.imshow(mid_slice, cmap='gray', aspect='equal')
                    ax.set_title(f'Class {class_idx}, Sample {img_idx}')
                else:
                    ax.text(0.5, 0.5, 'No sample', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'Class {class_idx}')
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_visualization(self):
        """Run a comprehensive visualization of the dataset."""
        print(f"Running comprehensive visualization for {self.dataset_name}")
        print("=" * 60)
        
        # Visualize sample 3D data
        print("\n1. Visualizing 3D sample data...")
        self.visualize_sample_3d('train', 0)
        
        # Show class distribution
        print("\n2. Analyzing class distribution...")
        self.visualize_class_distribution('train')
        
        # Show data statistics
        print("\n3. Analyzing data statistics...")
        self.visualize_data_statistics('train')
        
        # Validate data loading
        print("\n4. Validating data loading pipeline...")
        self.validate_data_loading('train', 5)
        
        # Create sample gallery
        print("\n5. Creating sample gallery...")
        self.create_sample_gallery('train', 2)
        
        print("\n" + "=" * 60)
        print("Comprehensive visualization complete!")


def main():
    """Main function to run the visualization."""
    print("MedMNIST Dataset Visualization Script")
    print("=" * 50)
    
    if not MEDMNIST_AVAILABLE:
        print("❌ MedMNIST modules not available")
        print("Please install MedMNIST: pip install medmnist")
        return
    
    # Available datasets
    print("Available 3D datasets:")
    for name, info in MEDMNIST_3D_DATASETS.items():
        print(f"  - {name}: {info['name']} ({info['modality']})")
    
    # Create visualizer for OrganMNIST3D
    try:
        visualizer = MedMNISTVisualizer('organmnist3d')
        
        # Run comprehensive visualization
        visualizer.run_comprehensive_visualization()
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

