# Group Sampling: Usage Guide

## Quick Start

### 1. Environment Setup
```bash
# Activate the environment
source activate groups

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import escnn; print('ESCNN available')"
```

### 2. Classification (OrganMNIST3D)

#### Training
```bash
python main.py --config organmnist3d_config.yaml --train
```

#### Testing
```bash
python main.py --config organmnist3d_config.yaml --test
```

#### Evaluation Only (using saved model)
```bash
python main.py --config organmnist3d_config.yaml --test --checkpoint_path checkpoints/organmnist3d/last.ckpt
```

### 3. Segmentation (ACDC)

#### Training
```bash
python main.py --config acdc.yaml --train
```

#### Testing
```bash
python main.py --config acdc.yaml --test
```

#### Evaluation Only (using saved model)
```bash
python main.py --config acdc.yaml --test --checkpoint_path checkpoints/acdc-epoch=10-val_dice_mean=0.850.ckpt
```

## Configuration Files

- `config/organmnist3d_config.yaml` - Classification configuration
- `config/acdc.yaml` - Segmentation configuration

## Key Features

- **Group Equivariance**: C4 cyclic subgroup for 90° rotations around z-axis
- **Anti-aliasing**: Prevents artifacts during group downsampling
- **4D U-Net**: 3D spatial + group axis processing
- **GPU Training**: Automatic GPU detection and usage
- **Checkpointing**: Automatic model saving and loading

## Output

- **Models**: Saved in `checkpoints/` directory
- **Logs**: Saved in `logs/` directory
- **Visualizations**: Saved in `test_outputs/` directory
