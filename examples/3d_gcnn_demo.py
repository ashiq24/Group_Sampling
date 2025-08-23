#!/usr/bin/env python3
"""
3D GCNN Demo Script

This script demonstrates how to use the 3D Group Equivariant Convolutional Neural Network (GCNN)
with octahedral groups for 3D data processing.

The demo shows:
1. Creating a 3D GCNN model with octahedral groups
2. Processing 3D input tensors
3. Different group transition strategies
4. Anti-aliasing configuration
"""

import torch
import numpy as np
from gsampling.models.model_handler import get_3d_model


def demo_octahedral_gcnn():
    """Demo: Octahedral → Cycle 3D GCNN."""
    print("🚀 Demo: Octahedral → Cycle 3D GCNN")
    print("=" * 50)

    # Create model with octahedral → cycle transition
    model = get_3d_model(
        input_channel=1,           # Single channel 3D input
        num_channels=[32, 64],     # Two layers with 32, 64 channels
        num_layers=2,              # Two convolutional layers
        dwn_group_types=[          # Group transitions per layer
            ["octahedral", "cycle"],    # Layer 1: O(24) → C4
            ["cycle", "cycle"]          # Layer 2: C4 → C4
        ],
        init_group_order=24,       # Start with octahedral group
        spatial_subsampling_factors=[2, 1],  # 2x downsampling in first layer
        subsampling_factors=[6, 1],          # 24/4 = 6, then no subsampling
        apply_antialiasing=True,   # Enable anti-aliasing
        pooling_type="max"         # Max pooling for classification
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Domain: {model.domain}D")
    print(f"Layers: {model.num_layers}")
    print(f"Group transitions: {model.dwn_group_types}")
    print(f"Channels: {model.num_channels}")
    print(f"Spatial subsampling: {model.spatial_subsampling_factors}")
    print(f"Group subsampling: {model.subsampling_factors}")

    # Create 3D input tensor (e.g., volumetric data)
    batch_size = 4
    depth, height, width = 16, 16, 16
    x = torch.randn(batch_size, 1, depth, height, width)

    print(f"\nInput shape: {x.shape}")
    print("(B, C, D, H, W) = ({}, {}, {}, {}, {})".format(*x.shape))

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print("(B, Classes) = ({}, {})".format(*output.shape))

    # Show intermediate shapes by getting hidden features
    features_before, features_after, sampling_layers = model.get_hidden_feature(x)

    print("
Intermediate shapes:")
    for i, (feat_before, feat_after) in enumerate(zip(features_before, features_after)):
        print(f"  Layer {i+1}: {feat_before.shape} → {feat_after.shape}")

    print("✅ Octahedral 3D GCNN demo completed!\n")


def demo_full_octahedral_gcnn():
    """Demo: Full Octahedral → Dihedral 3D GCNN."""
    print("🚀 Demo: Full Octahedral → Dihedral 3D GCNN")
    print("=" * 55)

    # Create model with full octahedral → dihedral transition
    model = get_3d_model(
        input_channel=2,           # Two channel 3D input
        num_channels=[64, 128, 256],  # Three layers
        num_layers=3,
        dwn_group_types=[
            ["full_octahedral", "octahedral"],  # O_h(48) → O(24)
            ["octahedral", "dihedral"],          # O(24) → D4
            ["dihedral", "dihedral"]              # D4 → D4
        ],
        init_group_order=48,       # Start with full octahedral group
        spatial_subsampling_factors=[2, 2, 1],     # Progressive downsampling
        apply_antialiasing=True,
        pooling_type="mean"        # Mean pooling
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Complex group transitions: {model.dwn_group_types}")

    # Create input
    x = torch.randn(2, 2, 32, 32, 32)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    print("✅ Full octahedral 3D GCNN demo completed!\n")


def demo_fully_convolutional_3d():
    """Demo: Fully Convolutional 3D GCNN."""
    print("🚀 Demo: Fully Convolutional 3D GCNN")
    print("=" * 40)

    # Create fully convolutional model
    model = get_3d_model(
        input_channel=1,
        num_channels=[32, 64],
        num_layers=2,
        dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
        init_group_order=24,
        spatial_subsampling_factors=[2, 2],     # 4x total downsampling
        subsampling_factors=[6, 1],
        apply_antialiasing=True,
        fully_convolutional=True,   # Dense prediction mode
    )

    print("Fully Convolutional Mode: Enabled")
    print("This model outputs spatial feature maps instead of global classification")

    # Test with different input sizes
    test_sizes = [(16, 16, 16), (32, 32, 32), (24, 24, 24)]

    for depth, height, width in test_sizes:
        x = torch.randn(1, 1, depth, height, width)
        output = model(x)

        # Calculate expected output size after 2x2x2 downsampling
        expected_depth = depth // 4
        expected_height = height // 4
        expected_width = width // 4

        print(f"Input {depth}x{height}x{width} → Output {output.shape}")
        print(f"  Expected spatial: {expected_depth}x{expected_height}x{expected_width}")

    print("✅ Fully convolutional 3D GCNN demo completed!\n")


def demo_anti_aliasing_comparison():
    """Demo: Anti-aliasing vs No anti-aliasing comparison."""
    print("🚀 Demo: Anti-aliasing Comparison")
    print("=" * 35)

    # Create two models - one with anti-aliasing, one without
    model_with_aa = get_3d_model(
        input_channel=1,
        num_layers=1,
        num_channels=[32],
        apply_antialiasing=True,
        antialiasing_kwargs={
            "smooth_operator": "adjacency",
            "mode": "analytical",
            "iterations": 30,
            "smoothness_loss_weight": 1.0,
        }
    )

    model_without_aa = get_3d_model(
        input_channel=1,
        num_layers=1,
        num_channels=[32],
        apply_antialiasing=False,
    )

    # Same input
    x = torch.randn(1, 1, 8, 8, 8)

    print("With anti-aliasing:")
    out_with_aa = model_with_aa(x)
    print(f"  Output shape: {out_with_aa.shape}")
    print("  ✅ Anti-aliasing preserves group equivariance"
    print("Without anti-aliasing:")
    out_without_aa = model_without_aa(x)
    print(f"  Output shape: {out_without_aa.shape}")
    print("  ⚠️  May have aliasing artifacts"

    print("✅ Anti-aliasing comparison demo completed!\n")


if __name__ == "__main__":
    print("🎯 3D Group Equivariant CNN Demo")
    print("================================\n")

    # Run all demos
    demo_octahedral_gcnn()
    demo_full_octahedral_gcnn()
    demo_fully_convolutional_3d()
    demo_anti_aliasing_comparison()

    print("🎉 All 3D GCNN demos completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• 3D group equivariant convolutions")
    print("• Multiple group transition strategies")
    print("• Anti-aliased group subsampling")
    print("• 3D spatial subsampling with blur pooling")
    print("• Fully convolutional mode for dense prediction")
    print("• Support for octahedral and full octahedral groups")
