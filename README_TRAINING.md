# MedMNIST Training Pipeline with 3D Group Equivariant CNNs

This directory contains a complete training pipeline for MedMNIST datasets using 3D Group Equivariant Convolutional Neural Networks (GCNNs) with PyTorch Lightning.

## 🚀 Features

- **Complete Training Pipeline**: End-to-end training from data loading to model evaluation
- **PyTorch Lightning Integration**: Modern, scalable training framework
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Customizable Precision**: Support for FP16, FP32, and FP64 training
- **3D GCNN Models**: Integration with GSampling 3D GCNN implementations
- **Comprehensive Configuration**: YAML-based configuration management
- **Test Mode**: Quick validation of the entire pipeline
- **Medical Image Focus**: Optimized for 3D biomedical imaging datasets

## 📁 Directory Structure

```
├── data/
│   └── medmnist.py              # MedMNIST dataloader and DataModule
├── config/
│   ├── base_config.yaml         # Base configuration template
│   ├── organmnist3d_config.yaml # OrganMNIST3D specific config
│   ├── test_config.yaml         # Test mode configuration
│   └── config_loader.py         # Configuration management utilities
├── train_utils.py               # Training utilities and helpers
├── main.py                      # Main training script
├── dataset_viz.py               # Data visualization and validation
├── requirements_training.txt    # Training pipeline dependencies
└── README_TRAINING.md          # This file
```

## 🛠️ Installation

1. **Activate the groups environment**:
   ```bash
   source activate groups
   ```

2. **Install training pipeline dependencies**:
   ```bash
   pip install -r requirements_training.txt
   ```

3. **Verify GSampling installation**:
   ```bash
   python -c "from gsampling.models.g_cnn_3d import Gcnn3D; print('✅ GCNN models available')"
   ```

## 📊 Available Datasets

The pipeline supports all MedMNIST 3D datasets:

| Dataset | Modality | Classes | Description |
|---------|----------|---------|-------------|
| **OrganMNIST3D** | CT | 11 | Abdominal organ classification |
| **NoduleMNIST3D** | CT | 2 | Lung nodule malignancy detection |
| **FractureMNIST3D** | CT | 3 | Bone fracture classification |
| **AdrenalMNIST3D** | CT | 2 | Adrenal gland abnormality detection |
| **VesselMNIST3D** | MRA | 2 | Cerebral vessel abnormality detection |
| **SynapseMNIST3D** | Electron Microscope | 2 | Neural synapse detection |

## 🎯 Quick Start

### 1. Test Mode (Recommended for First Run)

Test the entire pipeline with minimal data:

```bash
python main.py --config test_config.yaml --test
```

This will:
- Train on 10% of the data
- Run for only 3 epochs
- Validate the complete pipeline
- Complete in 1-2 minutes

### 2. Full Training

Train on the complete OrganMNIST3D dataset:

```bash
python main.py --config organmnist3d_config.yaml
```

### 3. Custom Configuration

Use a custom configuration file:

```bash
python main.py --config my_config.yaml --config-dir ./my_configs
```

## ⚙️ Configuration

### Base Configuration

The `base_config.yaml` provides common settings for:
- Data loading parameters
- Model architecture
- Training hyperparameters
- Hardware configuration
- Logging and monitoring

### Dataset-Specific Configurations

Each dataset has optimized settings:
- **OrganMNIST3D**: 4-layer GCNN with octahedral group symmetry
- **Test Mode**: Simplified configuration for quick validation

### Key Configuration Options

```yaml
# Data Configuration
data:
  dataset_name: "organmnist3d"
  batch_size: 8
  group_lifting: true
  group_order: 24

# Model Configuration
model:
  model_type: "gcnn3d"
  num_layers: 4
  num_channels: [32, 64, 128, 256]

# Hardware Configuration
hardware:
  gpus: -1                    # Use all available GPUs
  precision: 16               # Mixed precision training
  strategy: "auto"            # Automatic distributed strategy
```

## 🔧 Training Pipeline Components

### 1. Data Loading (`data/medmnist.py`)

- **MedMNIST3DDataset**: PyTorch Dataset wrapper for MedMNIST
- **MedMNIST3DDataModule**: PyTorch Lightning DataModule
- **Group Lifting**: Support for group-equivariant data format
- **Data Augmentation**: 3D-specific augmentation strategies

### 2. Configuration Management (`config/`)

- **YAML-based**: Human-readable configuration files
- **Import System**: Base configs with dataset-specific overrides
- **Validation**: Automatic configuration validation
- **Flexible**: Easy modification without code changes

### 3. Training Utilities (`train_utils.py`)

- **Data Augmentation**: 3D rotation, scaling, flipping
- **Loss Functions**: Cross-entropy, focal, dice losses
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score
- **Optimizer Factory**: Adam, AdamW, SGD with schedulers
- **Model Initialization**: Weight initialization strategies

### 4. Main Training Script (`main.py`)

- **PyTorch Lightning Module**: Complete training logic
- **Multi-GPU Support**: Automatic distributed training
- **Comprehensive Logging**: TensorBoard, CSV, WandB support
- **Checkpointing**: Model saving and restoration
- **Test Mode**: Quick pipeline validation

## 🎨 Data Visualization

Use `dataset_viz.py` to explore and validate your data:

```bash
python dataset_viz.py
```

Features:
- **3D Volume Visualization**: Orthogonal slices and projections
- **Class Distribution**: Histograms and pie charts
- **Data Statistics**: Intensity and volume distributions
- **Sample Gallery**: Visual samples from each class
- **Pipeline Validation**: Verify data loading correctness

## 🚀 Advanced Usage

### Multi-GPU Training

```bash
# Use all available GPUs
python main.py --config organmnist3d_config.yaml

# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1
python main.py --config organmnist3d_config.yaml
```

### Mixed Precision Training

Enable in configuration:
```yaml
hardware:
  precision: 16  # Mixed precision (FP16)
```

### Custom Model Architectures

Modify the configuration:
```yaml
model:
  num_layers: 5
  num_channels: [16, 32, 64, 128, 256]
  group_config:
    dwn_group_types: [
      ["octahedral", "octahedral"],
      ["octahedral", "dihedral"],
      ["dihedral", "cycle"],
      ["cycle", "cycle"],
      ["cycle", "cycle"]
    ]
```

### Custom Loss Functions

```yaml
training:
  loss:
    name: "focal"
    params:
      alpha: 1.0
      gamma: 2.0
```

## 📈 Monitoring and Logging

### TensorBoard

```bash
tensorboard --logdir ./logs
```

### CSV Logging

Results are saved to `./logs/` for analysis.

### Weights & Biases

Enable in configuration:
```yaml
logging:
  logger: "wandb"
  project_name: "medmnist_gcnn"
```

## 🧪 Testing and Validation

### Test Mode

Quick validation of the entire pipeline:
```bash
python main.py --config test_config.yaml --test
```

### Individual Component Testing

Test specific components:
```bash
# Test dataloader
python data/medmnist.py

# Test configuration loader
python config/config_loader.py

# Test training utilities
python train_utils.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure `source activate groups` is activated
   - Check that GSampling is properly installed
   - Verify all dependencies are installed

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Enable mixed precision training
   - Use gradient accumulation

3. **Configuration Errors**:
   - Validate YAML syntax
   - Check required configuration keys
   - Use test mode for validation

### Debug Mode

Enable detailed logging:
```yaml
advanced:
  debugging:
    log_grad_norm: true
    log_memory_usage: true
    detect_anomaly: true
```

## 📚 Examples

### Basic Training

```bash
# Train on OrganMNIST3D
python main.py --config organmnist3d_config.yaml

# Train with test mode first
python main.py --config test_config.yaml --test
python main.py --config organmnist3d_config.yaml
```

### Custom Training

```bash
# Use custom configuration
python main.py --config my_experiment.yaml --config-dir ./experiments

# Override configuration
python main.py --config organmnist3d_config.yaml --test
```

## 🤝 Contributing

1. **Follow the existing code structure**
2. **Add comprehensive tests** for new features
3. **Update configuration files** for new options
4. **Document new functionality** in this README
5. **Ensure compatibility** with existing GSampling modules

## 📄 License

This training pipeline follows the same license as the main GSampling project.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration examples
3. Test with test mode first
4. Check GSampling compatibility

---

**Happy Training! 🎯🚀**


