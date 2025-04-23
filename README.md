# Group Downsampling with Equivariant Anti-Aliasing 🌀📉

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv]( https://img.shields.io/badge/Openreview-red)](https://openreview.net/pdf?id=sOte83GogU)

**Bridging Signal Processing and Group Equivariant Deep Learning**  
*"Sampling theory meets symmetry preservation"*

![Teaser Image](https://raw.githubusercontent.com/ashiq24/Group_Sampling/refs/heads/main/figs/anti-al.png) 

## 🌟 What's New?
We present the first **group-equivariant downsampling** layer with built-in anti-aliasing that:
> Maintains equivariance guarantees
>
> Reduces feature map dimensions while preserving critical information
>
> Outperforms naive subsampling in G-CNNs

## 🚀 Key Features
- Uniform sub-group subsampling for arbitrary finite groups.

- Automatic subgroup selection based on group structure.

- Equivariant anti-aliasing filters learned via spectral optimization.

- Seamless integration with existing G-CNN architectures.

## 📦 Installation
```bash
git clone https://github.com/yourusername/Group-Sampling.git
cd Group-Sampling
pip install -e .
```
**External Dependencies:**  [ESCNN](https://github.com/QUVA-Lab/escnn) (for group operations)

## 🧪 Usage
Performs subgroup subsampling based on the corresponding Cayley graph and performs equivariant anti-aliasing operation.

![Teaser Image](https://raw.githubusercontent.com/ashiq24/Group_Sampling/refs/heads/main/figs/two-sub.png) 
1. Basic Integration

```python
from gsampling.layers.downsampling import SubgroupDownsample
d_layer = SubgroupDownsample(
    group_type="dihedral",         # Parent group type
    order=12,                     # Order of rotation elements
    sub_group_type="dihedral",    # Subgroup type to downsample to
    subsampling_factor=2,         # Factor to reduce group by
    num_features=10,              # Number of input feature channels
    generator="r-s",              # Generators for Cayley graph construction              # Computation device
    dtype=torch.float32,          # Data type
    apply_antialiasing=True,      # Enable equivariant anti-aliasing
    anti_aliasing_kwargs={        # Anti-aliasing optimization params
        "iterations": 100,        # Number of optimization steps
        "smoothness_loss_weight": 0.1, # Smoothness strength in optimization
    },
).to(device=device)

input_tensor = torch.randn( batch_size, 10 * 24, 32, 32, device=device, dtype=dtype)

out, canonicalization_element = d_layer(input_tensor)
print("Input shape:", input_tensor.shape)
print("Output shape:", out.shape)
```

2. Whole Model Integration
```python
from gsampling.models.model_handler import get_model
# Configure hierarchical group-equivariant architecture
model = get_model(
    input_channel=3,  # RGB input channels
    num_channels=[32, 64, 128],  # Feature channels per stage
    num_layers=3,     # Number of processing stages
    dwn_group_types=[
        ["dihedral", "dihedral"],  # [input_group, subgroup] for stage 1
        ["dihedral", "dihedral"],  # For stage 2
        ["dihedral", "dihedral"]   # For stage 3
    ],
    subsampling_factors=[2, 1, 1],  # Group reduction factors per stage
    spatial_subsampling_factors=[2, 1, 1],  # Spatial downsampling factors
    num_classes=10,    # STL-10 has 10 classes
    antialiasing_kwargs={
        "iterations": 100,  # Anti-aliasing optimization steps
        "smoothness_loss_weight": 0.5  # Trade-off between equivariance and smoothness
    }
)
```

