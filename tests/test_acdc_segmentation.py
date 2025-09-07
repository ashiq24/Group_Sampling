#!/usr/bin/env python3
"""
Test script for ACDC segmentation pipeline.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.acdc_datamodule import ACDCDataModule
from models.model_handler import get_3d_segmentation_model
from train_utils import LossFunctions, EvaluationMetrics

def test_acdc_segmentation_pipeline():
    """Test the complete ACDC segmentation pipeline."""
    print("Testing ACDC Segmentation Pipeline")
    print("=" * 50)
    
    # Test data loading
    print("\n1. Testing ACDC DataModule...")
    try:
        datamodule = ACDCDataModule(
            root="./datasets/acdc",
            batch_size=1,
            num_workers=0,  # Use 0 for testing
            persistent_workers=False,  # Disable persistent workers for testing
            use_es_ed_only=True
        )
        
        # Setup data
        datamodule.setup('fit')
        
        # Test train dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        print(f"✅ DataModule loaded successfully")
        print(f"   - Batch keys: {list(batch.keys())}")
        print(f"   - Image type: {type(batch['image'])}")
        print(f"   - Label type: {type(batch['label'])}")
        
        if isinstance(batch['image'], list):
            print(f"   - Image shape: {batch['image'][0].shape}")
            print(f"   - Label shape: {batch['label'][0].shape}")
        else:
            print(f"   - Image shape: {batch['image'].shape}")
            print(f"   - Label shape: {batch['label'].shape}")
            
    except Exception as e:
        print(f"❌ DataModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model creation
    print("\n2. Testing Segmentation Model...")
    try:
        model = get_3d_segmentation_model(
            input_channel=1,
            num_layers=2,  # Reduce to 2 layers for testing
            num_channels=[1, 16, 32],
            kernel_sizes=[3, 3],
            num_classes=4,
            dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "cycle"]],
            init_group_order=24,
            spatial_subsampling_factors=[1, 2],
            subsampling_factors=[1, 6],
            domain=3,
            apply_antialiasing=True,
            antialiasing_kwargs={},
            dropout_rate=0.1
        )
        
        print(f"✅ Segmentation model created successfully")
        print(f"   - Model type: {type(model)}")
        
        # Test forward pass
        if isinstance(batch['image'], list):
            test_input = batch['image'][0].unsqueeze(0)  # Add batch dimension
        else:
            test_input = batch['image']
        
        print(f"   - Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   - Output shape: {output.shape}")
        print(f"   - Expected shape: (1, 4, D, H, W)")
        
        # Verify output shape
        expected_classes = 4
        if output.shape[1] == expected_classes:
            print(f"✅ Output shape correct: {output.shape}")
        else:
            print(f"❌ Output shape incorrect: expected {expected_classes} classes, got {output.shape[1]}")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loss functions
    print("\n3. Testing Loss Functions...")
    try:
        # Test focal dice loss
        focal_dice_loss = LossFunctions.get_loss_function('focal_dice', alpha=0.5, gamma=2.0)
        
        # Test with model output
        if isinstance(batch['label'], list):
            test_label = batch['label'][0].unsqueeze(0)  # Add batch dimension
        else:
            test_label = batch['label']
        
        loss = focal_dice_loss(output, test_label.long())
        print(f"✅ Focal Dice loss computed: {loss.item():.4f}")
        
        # Test soft dice loss
        soft_dice_loss = LossFunctions.get_loss_function('soft_dice')
        loss2 = soft_dice_loss(output, test_label.long())
        print(f"✅ Soft Dice loss computed: {loss2.item():.4f}")
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test evaluation metrics
    print("\n4. Testing Evaluation Metrics...")
    try:
        metrics = EvaluationMetrics(num_classes=4, device=torch.device('cpu'), task_type='segmentation')
        
        # Update with test data
        preds = torch.argmax(output, dim=1)
        metrics.update(output, test_label.long())
        
        # Compute metrics
        computed_metrics = metrics.compute()
        
        print(f"✅ Segmentation metrics computed:")
        for key, value in computed_metrics.items():
            if key != 'confusion_matrix':
                print(f"   - {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! ACDC segmentation pipeline is working correctly.")
    return True

def test_config_loading():
    """Test loading the ACDC configuration."""
    print("\n5. Testing Configuration Loading...")
    try:
        from config.config_loader import load_config
        
        config = load_config('acdc.yaml', './config')
        
        print(f"✅ ACDC config loaded successfully")
        print(f"   - Model type: {config.get_config('model.model_type')}")
        print(f"   - Dataset: {config.get_config('data.dataset_name')}")
        print(f"   - Classes: {config.get_config('model.num_classes')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ACDC Segmentation Pipeline Test")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists("./datasets/acdc"):
        print("❌ ACDC dataset not found at ./datasets/acdc")
        print("Please download the dataset first using the Hugging Face method.")
        sys.exit(1)
    
    # Run tests
    success = True
    
    # Test configuration
    success &= test_config_loading()
    
    # Test pipeline
    success &= test_acdc_segmentation_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Ready for ACDC segmentation training.")
        print("\nTo start training, run:")
        print("  source activate groups")
        print("  python main.py --config config/acdc.yaml")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
