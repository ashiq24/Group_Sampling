# Group Sampling: 3D Group Equivariant Convolutional Neural Networks for Medical Image Analysis

## 🎯 **Project Overview**

This project implements **3D Group Equivariant Convolutional Neural Networks (GCNNs)** for medical image analysis, specifically designed for **classification** and **segmentation** tasks. The system leverages group theory principles to maintain **equivariance** under 3D rotations, providing robust and rotation-invariant features for 3D medical images.

## 🏗️ **Architecture Overview**

### **Core Components**

1. **Classification Architecture**: `Gcnn3D` - 3D Group Equivariant CNN for medical image classification
2. **Segmentation Architecture**: `Gcnn3DSegmentation` - 4D U-Net with group equivariance for 3D medical image segmentation
3. **Group Processing**: `gsampling/` library - Core group theory operations and anti-aliasing
4. **Training Pipeline**: PyTorch Lightning-based training with multi-GPU support

### **Mathematical Foundation**

- **Group Theory**: Octahedral group O (24 elements) with C4 cyclic subgroup (4 elements)
- **Equivariance**: f(g·x) = g·f(x) for all group elements g
- **Anti-Aliasing**: Spectral anti-aliasing to prevent artifacts during group downsampling
- **4D Processing**: Combines 3D spatial dimensions with group dimension

## 📁 **Project Structure**

```
Group_Sampling/
├── gsampling/                    # 🔒 Core library (DO NOT MODIFY)
│   ├── layers/                   # Core group equivariant layers
│   │   ├── anti_aliasing.py     # Spectral anti-aliasing for group downsampling
│   │   ├── downsampling.py      # Group downsampling operations
│   │   ├── rnconv.py            # Group equivariant convolutions
│   │   ├── sampling.py          # Subgroup sampling matrices
│   │   ├── helper.py            # Fourier operations and utilities
│   │   ├── cannonicalizer.py    # Group canonicalization (unused)
│   │   └── solvers.py           # Optimization solvers for anti-aliasing
│   ├── utils/                   # Group theory utilities
│   │   ├── group_utils.py       # Group registry and creation
│   │   └── graph_constructors.py # Graph factory for different groups
│   ├── core/                    # Core functionality
│   │   ├── graphs/              # Group graph implementations
│   │   │   ├── base.py          # Abstract group graph interface
│   │   │   ├── cyclic.py        # Cyclic group graphs
│   │   │   ├── dihedral.py      # Dihedral group graphs
│   │   │   ├── octahedral.py    # Octahedral group graphs
│   │   │   └── factory.py       # Graph factory
│   │   └── subsampling.py       # Subsampling strategies
│   └── thirdparty/              # External utilities
├── models/                       # 🚀 Application models (MODIFY HERE)
│   ├── g_cnn_3d.py              # 3D GCNN for classification
│   ├── g_cnn_3d_seg.py          # 4D U-Net for segmentation
│   ├── g_cnn.py                 # 2D GCNN (legacy)
│   ├── hybrid.py                # Hybrid convolution + group resampling
│   └── model_handler.py         # Model factory
├── data/                        # Data loaders and datasets
│   ├── medmnist_loader.py       # MedMNIST 3D dataset loader
│   ├── acdc_dataset.py          # ACDC cardiac MRI dataset
│   └── acdc_datamodule.py       # ACDC PyTorch Lightning datamodule
├── config/                      # Configuration files
│   ├── organmnist3d_config.yaml # OrganMNIST3D classification config
│   ├── acdc.yaml                # ACDC segmentation config
│   └── base_config.yaml         # Base configuration template
├── main.py                      # Main training script
├── train_utils.py               # Training utilities and metrics
└── tests/                       # Test suite
```

## 🧠 **Model Architectures**

### **1. Classification Model (`Gcnn3D`)**

**Purpose**: 3D medical image classification (e.g., OrganMNIST3D)

**Architecture**:
- **Input**: (batch, 1, depth, height, width) - 3D medical images
- **Layer 0**: Trivial → Regular representation (1 × 24 channels)
- **Layers 1+**: Hybrid layers with group convolution + group resampling
- **Spatial Pooling**: BlurPool3d for anti-aliased spatial downsampling
- **Global Pooling**: Collapse group and spatial dimensions
- **Output**: (batch, num_classes) - classification logits

**Group Processing Flow**:
1. Input: (batch, 1, 28, 28, 28) - trivial representation
2. Layer 0: (batch, 24, 28, 28, 28) - regular representation
3. Group downsampling: (batch, channels×4, 14, 14, 14) - C4 subgroup
4. Group upsampling: (batch, channels×24, 7, 7, 7) - back to octahedral
5. Output: (batch, 11) - classification logits

**Key Features**:
- **Dynamic Linear Layer**: Created on first forward pass to handle variable input sizes
- **Group Equivariance**: Maintains f(g·x) = g·f(x) throughout
- **Anti-Aliasing**: Prevents artifacts during group downsampling

### **2. Segmentation Model (`Gcnn3DSegmentation`)**

**Purpose**: 3D medical image segmentation (e.g., ACDC cardiac MRI)

**Architecture**:
- **Encoder**: 3D GCNN for feature extraction with group downsampling
- **Bottleneck**: Deepest features with group processing
- **Decoder**: Feature reconstruction with group upsampling + skip connections
- **Final Conv**: Output segmentation mask

**Group Processing Flow**:
1. Input: (batch, 1, depth, height, width) - trivial representation
2. Encoder: (batch, channels×24, depth/8, height/8, width/8) - regular representation
3. Decoder: (batch, channels×24, depth, height, width) - upsampled features
4. Output: (batch, num_classes, depth, height, width) - segmentation mask

**Key Features**:
- **4D U-Net**: Combines 3D spatial + group dimension processing
- **Skip Connections**: Concatenate encoder and decoder features
- **Group Pooling**: Collapse group dimension for final output

## 🔬 **Core Group Theory Operations**

### **1. Group Equivariant Convolutions (`rnconv.py`)**

**Mathematical Foundation**:
- **Equivariance**: f(g·x) = g·f(x) for all group elements g
- **Group Convolution**: (f * ψ)(g) = Σ_{h∈G} f(h)ψ(h^-1g)
- **Channel Calculation**: total_channels = base_channels × group_order

**Implementation**:
- Wraps ES-CNN's group equivariant convolutions
- Automatic tensor conversion (regular ↔ geometric)
- Support for 2D and 3D spatial domains

### **2. Group Downsampling (`downsampling.py`)**

**Mathematical Foundation**:
- **Subgroup Restriction**: S: L²(G) → L²(H) where H ⊆ G
- **Reynolds Projection**: R_G = (1/|G|) Σ_{g∈G} ρ(g) ⊗ ρ(g^-1)ᵀ
- **Spectral Subsampling**: S = Π_H ∘ R_G

**Implementation**:
- Supports octahedral → cyclic group transitions
- Maintains equivariance during downsampling
- Handles variable group orders

### **3. Anti-Aliasing (`anti_aliasing.py`)**

**Mathematical Foundation**:
- **Spectral Anti-Aliasing**: X̃ = L1_projector · X̂ before subsampling
- **L1 Projection**: Projects to invariant subspace of mapping matrix M
- **Smoothness Regularization**: tr(Mᵀ·F_Gᵀ·L·F_G·M)

**Implementation**:
- Prevents aliasing artifacts during group downsampling
- Multiple optimization modes (analytical, linear, GPU)
- Equivariance constraint enforcement

### **4. Group Graphs (`core/graphs/`)**

**Mathematical Foundation**:
- **Cayley Graphs**: Cay(G, S) where G is group, S is generating set
- **Fourier Basis**: Constructed from irreducible representations
- **Spectral Operators**: Laplacian, adjacency matrices for smoothness

**Supported Groups**:
- **Cyclic Groups**: C₄ (4 elements) for 90° rotations
- **Dihedral Groups**: D₄ (8 elements) for 2D symmetries
- **Octahedral Groups**: O (24 elements) for 3D cube rotations

## 🚀 **Training Pipeline**

### **PyTorch Lightning Integration**

**Features**:
- Multi-GPU training with DDP (Distributed Data Parallel)
- Mixed precision training (FP16, FP32, FP64)
- Comprehensive logging (TensorBoard, CSV, Weights & Biases)
- Automatic checkpointing and model saving
- Early stopping and learning rate monitoring

**Configuration**:
- YAML-based configuration management
- Dynamic dataset and model loading
- Loss function and metric configuration
- Hardware and optimization settings

### **Supported Datasets**

**Classification**:
- **OrganMNIST3D**: 11-class abdominal organ classification
- **Resolution**: 28×28×28 voxels
- **Modality**: CT (Computed Tomography)

**Segmentation**:
- **ACDC**: 4-class cardiac MRI segmentation
- **Resolution**: Variable (typically 64×64×64)
- **Modality**: MRI (Magnetic Resonance Imaging)

### **Loss Functions**

**Classification**:
- Cross-entropy loss
- Focal loss
- Weighted cross-entropy loss

**Segmentation**:
- Dice loss
- Soft Dice loss
- Tversky loss
- Focal Dice loss

### **Evaluation Metrics**

**Classification**:
- Accuracy, Precision, Recall, F1-score
- Macro-averaged metrics
- Confusion matrix

**Segmentation**:
- Dice coefficient, IoU (Intersection over Union)
- Hausdorff distance
- Per-class metrics

## 🔧 **Key Technical Features**

### **1. Group Equivariance**

**Mathematical Property**: f(g·x) = g·f(x)
- **Input Rotation**: Rotating input by 90° around z-axis
- **Output Rotation**: Produces correspondingly rotated output
- **Robustness**: Provides rotation-invariant features

### **2. Anti-Aliasing**

**Purpose**: Prevents aliasing artifacts during group downsampling
- **Spectral Approach**: Works in Fourier domain
- **Smoothness Regularization**: Ensures smooth transitions
- **Equivariance Preservation**: Maintains group properties

### **3. Dynamic Architecture**

**Current Issue**: Linear layer created dynamically on first forward pass
- **Problem**: Causes weight mismatch during checkpoint loading
- **Solution Needed**: Initialize all layers at construction time

### **4. Multi-Scale Processing**

**Spatial Downsampling**: BlurPool3d for anti-aliased spatial reduction
**Group Downsampling**: Spectral anti-aliasing for group order reduction
**Progressive Architecture**: Multiple scales for hierarchical feature learning

## 🐛 **Current Issues**

### **1. Dynamic Linear Layer Initialization**

**Problem**: Linear layer created dynamically during forward pass
- **Cause**: Variable input sizes require runtime layer creation
- **Impact**: Checkpoint loading fails due to weight mismatch
- **Solution**: Calculate expected output size and initialize linear layer at construction

### **2. Checkpoint Loading Issues**

**Problem**: Inference tensor errors during checkpoint loading
- **Cause**: PyTorch Lightning creates inference tensors during evaluation
- **Impact**: Cannot load saved models for evaluation
- **Solution**: Convert inference tensors to regular tensors before loading

### **3. Architecture Mismatch**

**Problem**: Saved checkpoints may have different architecture than current model
- **Cause**: Dynamic layer creation and configuration changes
- **Impact**: Strict loading fails, requires strict=False
- **Solution**: Ensure consistent architecture initialization

## 🎯 **Usage Instructions**

### **Classification Training**
```bash
# Activate environment
source activate groups

# Train OrganMNIST3D classification
python main.py --config organmnist3d_config.yaml --train

# Evaluate with saved checkpoint
python main.py --config organmnist3d_config.yaml --evaluate-only --load-checkpoint checkpoints/organmnist3d/last.ckpt
```

### **Segmentation Training**
```bash
# Train ACDC segmentation
python main.py --config acdc.yaml --train

# Evaluate with saved checkpoint
python main.py --config acdc.yaml --evaluate-only --load-checkpoint checkpoints/acdc/last.ckpt
```

### **Testing**
```bash
# Test classification pipeline
python main.py --config organmnist3d_config.yaml --test

# Test segmentation pipeline
python main.py --config acdc.yaml --test
```

## 🔬 **Mathematical Details**

### **Group Convolution**
```
(f * ψ)(g) = Σ_{h∈G} f(h)ψ(h^-1g)
```
Where f ∈ L²(G), ψ ∈ L²(G), and g ∈ G.

### **Spectral Subsampling**
```
S: L²(G) → L²(H) via S = Π_H ∘ R_G
```
Where R_G is Reynolds projection and Π_H is subgroup restriction.

### **Anti-Aliasing**
```
X̃ = L1_projector · X̂ before subsampling
```
Where L1_projector removes high-frequency components beyond Nyquist limit.

### **Channel Calculation**
```
total_channels = base_channels × group_order
```
Where group_order is |G| for the current group.

## 🚨 **Critical Notes for Future Development**

1. **DO NOT MODIFY** `gsampling/` directory - it's extensively tested and working
2. **MODIFY ONLY** `models/` directory for architecture changes
3. **Dynamic Initialization Issue**: Must be fixed to enable proper checkpoint loading
4. **Group Equivariance**: Must be preserved throughout all modifications
5. **Anti-Aliasing**: Essential for preventing artifacts during group downsampling
6. **Testing**: Run tests after any changes to ensure no regressions

## 📊 **Test Coverage**

- **Core Algorithms**: 161/161 tests passing
- **2D Groups**: 80/80 tests passing (Cyclic, Dihedral)
- **3D Groups**: 81/81 tests passing (Octahedral, Full Octahedral)
- **Anti-Aliasing**: 6/6 tests passing (All modes working)
- **Training Pipeline**: 100% accuracy in end-to-end validation

## 🎯 **Next Steps**

1. **Fix Dynamic Initialization**: Calculate expected output size and initialize linear layer at construction
2. **Fix Checkpoint Loading**: Handle inference tensor conversion properly
3. **Architecture Consistency**: Ensure saved and loaded models have identical architecture
4. **Documentation**: Complete inline documentation for all remaining modules
5. **Testing**: Comprehensive testing of fixed functionality

---

**Author**: Group Sampling Team  
**Last Updated**: Current Date  
**Status**: Active Development - Dynamic Initialization Issue Needs Resolution
