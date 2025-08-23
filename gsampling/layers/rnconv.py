from escnn import gspaces
import escnn.nn as enn
import torch.nn as nn
from gsampling.utils.group_utils import *


class rnConv(nn.Module):
    def __init__(
        self,
        *,
        in_group_type: str,
        in_order: int,
        in_num_features: int,
        in_representation: str,
        out_group_type: str,
        out_num_features: int,
        out_representation: str,
        domain: int = 2,
        kernel_size: int = 3,
        layer_kwargs: dict = {},
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
        self.in_group_type = in_group_type
        self.in_order = in_order
        self.in_num_features = in_num_features
        self.in_representation = in_representation
        self.out_group_type = out_group_type
        self.out_num_features = out_num_features
        self.out_representation = out_representation
        self.domain = domain

        self.G_in = get_group(in_group_type, in_order)
        self.G_out = get_group(out_group_type, in_order)
        self.out_order = in_order  # For now, assume same order as input
        self.gspace_in = get_gspace(
            group_type=in_group_type,
            order=in_order,
            num_features=in_num_features,
            representation=in_representation,
            domain=domain,
        )
        self.gspace_out = get_gspace(
            group_type=out_group_type,
            order=in_order,
            num_features=out_num_features,
            representation=out_representation,
            domain=domain,
        )

        if domain == 2:
            self.conv = enn.R2Conv(
                in_type=self.gspace_in,
                out_type=self.gspace_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                **layer_kwargs,
            )
        elif domain == 3:
            self.conv = enn.R3Conv(
                in_type=self.gspace_in,
                out_type=self.gspace_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                **layer_kwargs,
            )
        else:
            raise ValueError(f"Domain {domain} not supported. Use domain=2 for 2D or domain=3 for 3D")

    def forward(self, x):
        # Validate input tensor dimensions
        if self.domain == 2:
            if x.dim() != 4:
                raise ValueError(f"Expected 4D tensor for 2D convolution, got {x.dim()}D. "
                              f"Expected shape: {self.get_expected_input_shape()}")
        elif self.domain == 3:
            if x.dim() != 5:
                raise ValueError(f"Expected 5D tensor for 3D convolution, got {x.dim()}D. "
                              f"Expected shape: {self.get_expected_input_shape()}")
        
        # Convert to geometric tensor and perform convolution
        f_x = self.gspace_in(x)
        f_x_out = self.conv(f_x)
        return f_x_out.tensor

    def get_group(self):
        return self.G

    def get_gspace(self):
        return self.gspace_in
    
    def get_domain(self):
        """Get the spatial domain of this convolution layer."""
        return self.domain
    
    def get_expected_input_shape(self, batch_size: int = 1):
        """
        Get the expected input tensor shape for this convolution layer.
        
        Args:
            batch_size: Batch size (default: 1)
            
        Returns:
            Tuple representing the expected input shape
        """
        if self.domain == 2:
            return (batch_size, self.in_num_features * self.in_order, None, None)
        elif self.domain == 3:
            return (batch_size, self.in_num_features * self.in_order, None, None, None)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
    
    def get_expected_output_shape(self, batch_size: int = 1):
        """
        Get the expected output tensor shape for this convolution layer.
        
        Args:
            batch_size: Batch size (default: 1)
            
        Returns:
            Tuple representing the expected output shape
        """
        if self.domain == 2:
            return (batch_size, self.out_num_features * self.out_order, None, None)
        elif self.domain == 3:
            return (batch_size, self.out_num_features * self.out_order, None, None, None)
        else:
            raise ValueError(f"Unsupported domain: {self.domain}")
