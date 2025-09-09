"""
Regular Group Equivariant Convolution Layer

This module provides a simplified interface for ES-CNN's group equivariant convolution
layers (R2Conv for 2D, R3Conv for 3D). It automatically handles the conversion between
regular PyTorch tensors and ES-CNN's geometric tensors, making it easier to integrate
group equivariant convolutions into standard neural network architectures.

Mathematical Foundation:
------------------------
Group equivariant convolutions satisfy the equivariance property:
f(g·x) = g·f(x) for all group elements g ∈ G

Where:
- f is the convolution operation
- g·x represents the group action on input x
- g·f(x) represents the group action on output f(x)

The convolution kernel is constrained to respect the group structure, ensuring that
rotations and other symmetries are preserved throughout the network.

Key Features:
- Automatic tensor conversion (regular ↔ geometric)
- Support for 2D and 3D spatial domains
- Flexible group and representation types
- Simplified interface for ES-CNN integration

Author: Group Sampling Team
"""

import torch.nn as nn
import escnn.nn as enn
from gsampling.utils.group_utils import get_group, get_gspace


class rnConv(nn.Module):
    def __init__(
        self,
        *,
        in_group_type: str,  # Input symmetry group type ("cyclic", "dihedral", "octahedral", etc.)
        in_order: int,  # Order of input group (e.g., 4 for C₄, 24 for octahedral)
        in_num_features: int,  # Number of input channels/features
        in_representation: str,  # Input representation type ("regular", "irreducible", "trivial")
        out_group_type: str,  # Output symmetry group type
        out_num_features: int,  # Number of output channels/features
        out_representation: str,  # Output representation type
        domain: int = 2,  # Spatial dimension (2 for 2D, 3 for 3D)
        kernel_size: int = 3,  # Convolution kernel size
        layer_kwargs: dict = {},  # Additional arguments for ES-CNN layer
    ):
        """
        This method wraps ESCNN convolution layers for both 2D and 3D domains.
        Makes it more convenient to use in the context of group equivariant CNNs. We can pass regular tensors
        and the layer will convert them to gspace tensors and perform the convolution.
        And return the regular tensor.

        Useful for the case when we want to use the layer as a part of the model that might operate mostly on regular tensors.

        Provides simplified interface for ES-CNN's R2Conv (2D) and R3Conv (3D) by automatically handling:
        - Geometric tensor conversion (regular tensor ↔ escnn.GeometricTensor)
        - Group space initialization
        - Representation type management
        - Domain-specific convolution layer selection

        Args:
            in_group_type (str): Input symmetry group type. Options: 'cyclic', 'dihedral', 'so2', etc.
            in_order (int): Order of finite subgroup (e.g., 4 for C₄ cyclic group)
            in_num_features (int): Number of input channels/features
            in_representation (str): Input representation type. Options: 'regular', 'irreducible', 'trivial'
            out_group_type (str): Output symmetry group type (same options as input)
            out_num_features (int): Number of output channels/features
            out_representation (str): Output representation type (same options as input)
            domain (int, optional): Spatial dimension of operation. Default: 2 (2D convolution), use 3 for 3D convolution
            kernel_size (int, optional): Convolution kernel size. Default: 3
            layer_kwargs (dict, optional): Additional arguments for ES-CNN R2Conv/R3Conv layer. Default: {}

        Shapes:
            - 2D Input: (batch, in_num_features * group_order, height, width)
            - 2D Output: (batch, out_num_features * group_order, height, width)
            - 3D Input: (batch, in_num_features * group_order, depth, height, width)
            - 3D Output: (batch, out_num_features * group_order, depth, height, width)

        Example:
            # 2D convolution
            >>> conv_2d = rnConv(in_group_type='cycle', in_order=4, in_num_features=32,
                                out_group_type='cycle', out_order=4, out_num_features=64,
                                in_representation='regular', out_representation='regular',
                                domain=2)
            >>> output_2d = conv_2d(input_tensor_2d)  # (B, C*4, H, W)
            
            # 3D convolution
            >>> conv_3d = rnConv(in_group_type='octahedral', in_order=24, in_num_features=16,
                                out_group_type='octahedral', out_order=24, out_num_features=32,
                                in_representation='regular', out_representation='regular',
                                domain=3)
            >>> output_3d = conv_3d(input_tensor_3d)  # (B, C*24, D, H, W)
        """
        super(rnConv, self).__init__()
        
        # Store layer configuration parameters
        self.in_group_type = in_group_type  # Input group type
        self.in_order = in_order  # Input group order
        self.in_num_features = in_num_features  # Input feature count
        self.in_representation = in_representation  # Input representation type
        self.out_group_type = out_group_type  # Output group type
        self.out_num_features = out_num_features  # Output feature count
        self.out_representation = out_representation  # Output representation type
        self.domain = domain  # Spatial domain (2D or 3D)

        # Initialize group objects for input and output
        # These represent the symmetry groups that the convolution respects
        self.G_in = get_group(in_group_type, in_order)  # Input group
        self.G_out = get_group(out_group_type, in_order)  # Output group
        self.out_order = in_order  # For now, assume same order as input
        
        # Initialize geometric spaces for input and output
        # These define how tensors are interpreted as group-equivariant features
        self.gspace_in = get_gspace(
            group_type=in_group_type,  # Input group type
            order=in_order,  # Input group order
            num_features=in_num_features,  # Input feature count
            representation=in_representation,  # Input representation
            domain=domain,  # Spatial domain
        )
        self.gspace_out = get_gspace(
            group_type=out_group_type,  # Output group type
            order=in_order,  # Output group order (same as input for now)
            num_features=out_num_features,  # Output feature count
            representation=out_representation,  # Output representation
            domain=domain,  # Spatial domain
        )

        # Initialize the appropriate ES-CNN convolution layer based on domain
        if domain == 2:
            # 2D group equivariant convolution
            # R2Conv operates on 2D spatial data with group structure
            self.conv = enn.R2Conv(
                in_type=self.gspace_in,  # Input geometric space
                out_type=self.gspace_out,  # Output geometric space
                kernel_size=kernel_size,  # Convolution kernel size
                padding=(kernel_size - 1) // 2,  # Same padding to preserve spatial dimensions
                **layer_kwargs,  # Additional layer arguments
            )
        elif domain == 3:
            # 3D group equivariant convolution
            # R3Conv operates on 3D spatial data with group structure
            self.conv = enn.R3Conv(
                in_type=self.gspace_in,  # Input geometric space
                out_type=self.gspace_out,  # Output geometric space
                kernel_size=kernel_size,  # Convolution kernel size
                padding=(kernel_size - 1) // 2,  # Same padding to preserve spatial dimensions
                **layer_kwargs,  # Additional layer arguments
            )
        else:
            raise ValueError(f"Domain {domain} not supported. Use domain=2 for 2D or domain=3 for 3D")

    def forward(self, x):
        """Forward pass of the group equivariant convolution.
        
        This method performs the following steps:
        1. Validates input tensor dimensions
        2. Converts regular tensor to geometric tensor
        3. Applies group equivariant convolution
        4. Converts result back to regular tensor
        
        Mathematical Operation:
        The convolution satisfies the equivariance property:
        f(g·x) = g·f(x) for all group elements g ∈ G
        
        Args:
            x: Input tensor with group structure
                - 2D: (batch, in_features * group_order, height, width)
                - 3D: (batch, in_features * group_order, depth, height, width)
                
        Returns:
            Output tensor with group structure
                - 2D: (batch, out_features * group_order, height, width)
                - 3D: (batch, out_features * group_order, depth, height, width)
        """
        # Validate input tensor dimensions based on spatial domain
        if self.domain == 2:
            if x.dim() != 4:
                raise ValueError(f"Expected 4D tensor for 2D convolution, got {x.dim()}D. "
                              f"Expected shape: {self.get_expected_input_shape()}")
        elif self.domain == 3:
            if x.dim() != 5:
                raise ValueError(f"Expected 5D tensor for 3D convolution, got {x.dim()}D. "
                              f"Expected shape: {self.get_expected_input_shape()}")
        
        # Step 1: Convert regular tensor to geometric tensor
        # This interprets the tensor as group-equivariant features
        # The geometric tensor knows about the group structure and representations
        f_x = self.gspace_in(x)
        
        # Step 2: Apply group equivariant convolution
        # The convolution kernel is constrained to respect the group structure
        # This ensures that rotations and other symmetries are preserved
        f_x_out = self.conv(f_x)
        
        # Step 3: Convert geometric tensor back to regular tensor
        # Extract the underlying tensor data while preserving the group structure
        return f_x_out.tensor

    def get_group(self):
        """Get the input group object.
        
        Returns:
            The input group object representing the symmetry group
        """
        return self.G_in

    def get_gspace(self):
        """Get the input geometric space object.
        
        Returns:
            The input geometric space object that defines tensor interpretation
        """
        return self.gspace_in
    
    def get_domain(self):
        """Get the spatial domain of this convolution layer.
        
        Returns:
            Integer representing the spatial dimension (2 for 2D, 3 for 3D)
        """
        return self.domain
    
    def get_expected_input_shape(self, batch_size: int = 1):
        """Get the expected input tensor shape for this convolution layer.
        
        The input shape includes the group structure, where the channel dimension
        is multiplied by the group order to account for all group elements.
        
        Args:
            batch_size: Batch size (default: 1)
            
        Returns:
            Tuple representing the expected input shape
                - 2D: (batch, in_features * group_order, height, width)
                - 3D: (batch, in_features * group_order, depth, height, width)
        """
        if self.domain == 2:
            # 2D case: (batch, channels * group_order, height, width)
            return (batch_size, self.in_num_features * self.in_order, None, None)
        elif self.domain == 3:
            # 3D case: (batch, channels * group_order, depth, height, width)
            return (batch_size, self.in_num_features * self.in_order, None, None, None)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
    
    def get_expected_output_shape(self, batch_size: int = 1):
        """Get the expected output tensor shape for this convolution layer.
        
        The output shape includes the group structure, where the channel dimension
        is multiplied by the group order to account for all group elements.
        
        Args:
            batch_size: Batch size (default: 1)
            
        Returns:
            Tuple representing the expected output shape
                - 2D: (batch, out_features * group_order, height, width)
                - 3D: (batch, out_features * group_order, depth, height, width)
        """
        if self.domain == 2:
            # 2D case: (batch, channels * group_order, height, width)
            return (batch_size, self.out_num_features * self.out_order, None, None)
        elif self.domain == 3:
            # 3D case: (batch, channels * group_order, depth, height, width)
            return (batch_size, self.out_num_features * self.out_order, None, None, None)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
