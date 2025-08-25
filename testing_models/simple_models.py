"""
Simple Model Implementations for Testing and Comparison

This module contains simple baseline models that can be used for:
- Testing the training pipeline
- Comparing performance with GCNN models
- Quick prototyping and debugging
"""

import torch
import torch.nn as nn


class SimpleCNN3D(nn.Module):
    """
    Simple 3D CNN model for testing and comparison.
    
    A basic 3D convolutional neural network with:
    - 3D convolutions with batch normalization
    - Max pooling for downsampling
    - Global average pooling
    - Dropout for regularization
    - Linear classification head
    """
    
    def __init__(self, num_channels, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # First conv layer
        self.conv1 = nn.Conv3d(1, num_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_channels[0])
        self.pool1 = nn.MaxPool3d(2)
        
        # Second conv layer
        self.conv2 = nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_channels[1])
        self.pool2 = nn.MaxPool3d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Dropout and linear layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_channels[1], num_classes)
        
    def forward(self, x):
        # Ensure input is 5D: (batch, channels, depth, height, width)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_simple_cnn3d_model(num_channels, num_classes, dropout_rate=0.1) -> nn.Module:
    """
    Factory function to create a SimpleCNN3D model.
    
    Args:
        num_channels: List of channel counts for each layer
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        
    Returns:
        Configured SimpleCNN3D model
    """
    return SimpleCNN3D(
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
