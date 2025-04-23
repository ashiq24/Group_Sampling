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
